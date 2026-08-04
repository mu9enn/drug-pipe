import dataclasses
import gc
import logging
import math
import os
from argparse import Namespace
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args
from megatron.training.training import get_model
from tqdm import tqdm

try:
    from megatron.core.pipeline_parallel.utils import unwrap_model
except ImportError:
    from megatron.core.utils import unwrap_model
from slime.utils import logging_utils
from slime.utils.memory_utils import clear_memory

from .checkpoint import load_checkpoint, save_checkpoint
from .cp_utils import reduce_train_step_metrics
from .data import DataIterator, get_batch
from .fp8_optimizer_state_offload import (
    bind_fp8_optimizer_state_offload,
    install_fp8_optimizer_state_offload_patch,
    split_fp8_optimizer_param_groups,
)
from .hybrid_optimizer_stream_patch import install_hybrid_optimizer_h2d_wait_patch
from .loss import loss_function
from .model_provider import get_model_provider_func

logger = logging.getLogger(__name__)
install_hybrid_optimizer_h2d_wait_patch()
install_fp8_optimizer_state_offload_patch()


def _optimizer_state_offload_instances(optimizer: MegatronOptimizer) -> list[MegatronOptimizer]:
    """Return MCore optimizers which implement the state-offload lifecycle."""
    return [
        instance
        for instance in getattr(optimizer, "chained_optimizers", ())
        if all(
            hasattr(instance, method)
            for method in ("offload_states", "reload_offloaded_states", "release_offloaded_gpu_states")
        )
    ]


def _offload_optimizer_states_if_needed(optimizer: MegatronOptimizer) -> None:
    """Start D2H state offload, avoiding a duplicate copy of already-offloaded state."""
    offload_master_weights = os.environ.get("OFFLOAD_OPTIMIZER_MASTER_WEIGHTS", "1") != "0"
    for instance in _optimizer_state_offload_instances(optimizer):
        state_offloader = getattr(instance, "_state_offloader", None)
        if state_offloader is not None and not getattr(state_offloader, "_offloaded", False):
            state_offloader.offload(offload_master_weights=offload_master_weights)


def _release_offloaded_optimizer_states(optimizer: MegatronOptimizer) -> None:
    for instance in _optimizer_state_offload_instances(optimizer):
        instance.release_offloaded_gpu_states()


def _drop_optimizer_cpu_state_before_weights_only_save(optimizer: MegatronOptimizer) -> int:
    """Release offloaded Adam state before a terminal weights-only checkpoint.

    The 122B FP8 profile streams its two Adam moments to CPU between steps.
    Those buffers are not part of a ``--no-save-optim`` checkpoint, but keeping
    them alive while torch.distributed.checkpoint stages model shards nearly
    doubles host memory.  This operation deliberately makes the optimizer
    unusable, so require an explicit terminal-save opt-in in addition to
    Megatron's weights-only flag.

    Returns the number of CPU tensor bytes released by this rank.
    """
    args = get_args()
    if not getattr(args, "no_save_optim", False):
        return 0
    if os.environ.get("SLIME_DROP_OPTIMIZER_STATE_BEFORE_WEIGHTS_ONLY_SAVE", "0") != "1":
        return 0

    released_bytes = 0
    for instance in _optimizer_state_offload_instances(optimizer):
        state_offloader = getattr(instance, "_state_offloader", None)
        cpu_buffers = getattr(state_offloader, "_opt_state_cpu_buffers", None)
        if not isinstance(cpu_buffers, dict):
            continue
        for param_buffers in cpu_buffers.values():
            for tensor in param_buffers.values():
                if isinstance(tensor, torch.Tensor):
                    released_bytes += tensor.untyped_storage().nbytes()
        cpu_buffers.clear()

    gc.collect()
    # Large pageable tensor allocations can remain in glibc arenas after the
    # final reference is dropped.  Return free arenas to the pod cgroup before
    # distributed checkpointing starts; failure to trim is non-fatal.
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (ImportError, OSError, AttributeError):
        pass
    logger.info(
        "Released %.2f GiB of offloaded optimizer CPU state before weights-only save",
        released_bytes / 1024**3,
    )
    return released_bytes


def _disable_tqdm_for_non_main_rank() -> bool:
    return not (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
    )


def _should_update_microbatch_pbar(model) -> bool:
    if _disable_tqdm_for_non_main_rank():
        return False

    while hasattr(model, "module"):
        model = model.module
    vp_stage = getattr(model, "vp_stage", None)
    if mpu.get_virtual_pipeline_model_parallel_world_size() is not None and vp_stage is not None:
        return mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
    return mpu.is_pipeline_last_stage(ignore_virtual=True)


def _wrap_forward_step_with_microbatch_pbar(forward_step_func, pbar):
    if pbar is None:
        return forward_step_func

    def wrapped_forward_step(*args, **kwargs):
        result = forward_step_func(*args, **kwargs)
        model = args[1] if len(args) > 1 else kwargs.get("model")
        if model is not None and _should_update_microbatch_pbar(model):
            pbar.update(1)
        return result

    return wrapped_forward_step


def _iter_critic_output_layers(model: Sequence[DDP]):
    for chunk_id, module in enumerate(unwrap_model(model)):
        output_layer = getattr(module, "output_layer", None)
        if output_layer is not None:
            yield chunk_id, output_layer


def _critic_output_layer_needs_reinit(args: Namespace, model: Sequence[DDP], role: str) -> bool:
    if role != "critic" or args.load is None:
        return False

    from megatron.core.dist_checkpointing.serialization import load_tensors_metadata
    from megatron.training.checkpointing import get_load_checkpoint_path_by_args

    checkpoint_path = Path(get_load_checkpoint_path_by_args(args))
    if not (checkpoint_path / ".metadata").is_file():
        return False

    checkpoint_metadata = load_tensors_metadata(str(checkpoint_path))
    for _chunk_id, output_layer in _iter_critic_output_layers(model):
        for name in ("weight", "bias"):
            param = getattr(output_layer, name, None)
            if param is None:
                continue

            param_name = f"output_layer.{name}"
            ckpt_tensor_metadata = next(
                (
                    tensor_metadata
                    for key, tensor_metadata in checkpoint_metadata.items()
                    if key == param_name or key.endswith(f".{param_name}")
                ),
                None,
            )
            expected_shape = tuple(param.shape)
            checkpoint_shape = tuple(ckpt_tensor_metadata.global_shape) if ckpt_tensor_metadata is not None else None
            if checkpoint_shape == expected_shape:
                continue

            reason = (
                "missing from checkpoint metadata"
                if checkpoint_shape is None
                else f"shape mismatch checkpoint={checkpoint_shape} runtime={expected_shape}"
            )
            logger.warning(
                "Will reinitialize critic %s after checkpoint load because it is %s",
                param_name,
                reason,
            )
            return True

    return False


@torch.no_grad()
def _reinitialize_critic_output_layer(model: Sequence[DDP]) -> None:
    for _chunk_id, output_layer in _iter_critic_output_layers(model):
        output_layer.weight.data.normal_(mean=0.0, std=0.02)
        if output_layer.bias is not None:
            output_layer.bias.data.zero_()


def get_optimizer_param_scheduler(args: Namespace, optimizer: MegatronOptimizer) -> OptimizerParamScheduler:
    """Create and configure the optimizer learning-rate/weight-decay scheduler.

    This configures iteration-based schedules derived from the global batch size
    and run-time arguments.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        optimizer (MegatronOptimizer): Megatron optimizer bound to the model.

    Returns:
        OptimizerParamScheduler: Initialized scheduler bound to ``optimizer``.
    """
    # Iteration-based training. ``train_iters`` is an estimate of the total
    # number of training steps — it's only used to size Megatron's LR decay
    # schedule (and ``lr_decay_iters`` defaults to it). With variable per-rollout
    # sample counts (dynamic sampling / filtering / custom step splitter) the
    # *actual* total can drift; the schedule still tracks the true progress via
    # ``opt_param_scheduler.num_steps`` (samples consumed, also persisted across
    # resume), so the worst case is the cosine/linear schedule reaches its
    # plateau slightly early or late. Pass ``--lr-decay-iters`` explicitly if you
    # need exact decay control.
    args.train_iters = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    wd_incr_steps = args.train_iters * args.global_batch_size
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters * args.global_batch_size
    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def setup_model_and_optimizer(
    args: Namespace,
    role: str = "actor",
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
    """Build model(s), wrap with DDP, and construct optimizer and scheduler.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        role (str): Logical role of the model (e.g., "actor", "critic").
        no_wd_decay_cond (Callable[..., bool] | None): Predicate to exclude
            parameters from weight decay.
        scale_lr_cond (Callable[..., bool] | None): Predicate to scale LR for
            selected parameter groups.
        lr_mult (float): Global learning-rate multiplier for the optimizer.

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
            - List of model chunks wrapped by ``DDP``.
            - The constructed ``MegatronOptimizer`` instance.
            - The learning-rate/weight-decay scheduler tied to the optimizer.
    """
    assert not args.moe_use_upcycling
    assert args.load is not None or args.pretrained_checkpoint is not None

    model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)

    # Optimizer
    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = None

    optimizer = get_megatron_optimizer(
        config=config,
        model_chunks=model,
        use_gloo_process_groups=args.enable_gloo_process_groups,
    )
    if args.use_precision_aware_optimizer and args.offload_optimizer_states:
        bound_offloaders = bind_fp8_optimizer_state_offload(optimizer)
        fp8_group_count = split_fp8_optimizer_param_groups(optimizer)
        if fp8_group_count:
            logger.info(
                "Split precision-aware FusedAdam into %d bounded parameter groups "
                "and bound %d state offloaders",
                fp8_group_count,
                bound_offloaders,
            )
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
    return model, optimizer, opt_param_scheduler


def enable_forward_pre_hook(model_chunks: Sequence[DDP]) -> None:
    """Enable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to enable hooks on.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks: Sequence[DDP], param_sync: bool = True) -> None:
    """Disable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to disable hooks on.
        param_sync (bool): Whether to synchronize parameters when disabling.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


@torch.no_grad()
def forward_only(
    f: Callable[..., dict[str, list[torch.Tensor]]],
    args: Namespace,
    model: Sequence[DDP],
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    store_prefix: str = "",
) -> dict[str, list[torch.Tensor]]:
    """Run forward passes only and collect non-loss outputs (e.g., logprobs).

    The model is put into evaluation mode, a forward-only pipeline pass is
    executed, and relevant outputs are aggregated and returned.

    Args:
        f (Callable[..., dict[str, list[torch.Tensor]]]): Post-forward callback used to
            compute and package outputs to collect. This should accept a logits
            tensor as its first positional argument and additional keyword-only
            arguments; see ``get_log_probs_and_entropy``/``get_values`` in
            ``megatron_utils.loss`` for examples. It will be partially applied
            so that the callable returned from the internal forward step only
            requires the logits tensor.
        args (Namespace): Runtime arguments.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding batches for inference.
        num_microbatches (Sequence[int]): Number of microbatches per rollout step.
        store_prefix (str): Prefix to prepend to stored output keys.

    Returns:
        dict[str, list[torch.Tensor]]: Aggregated outputs keyed by ``store_prefix + key``.
    """

    # reset data iterator
    for iterator in data_iterator:
        iterator.reset()

    config = get_model_config(model[0])

    def forward_step(
        data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
        """Forward step used by Megatron's pipeline engine.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
            Output tensor(s) and a callable that computes and packages results
            to be collected by the engine.
        """

        assert not return_schedule_plan, "forward_only step should never return schedule plan"

        # Get the batch.
        batch = get_batch(
            data_iterator,
            [
                "tokens",
                "loss_masks",
                "multimodal_train_inputs",
                "total_lengths",
                "response_lengths",
                "max_seq_lens",
            ],
            args.data_pad_size_multiplier,
            args.qkv_format,
            args.allgather_cp,
        )
        unconcat_tokens = batch["unconcat_tokens"]
        tokens = batch["tokens"]
        packed_seq_params = batch["packed_seq_params"]
        total_lengths = batch["total_lengths"]
        response_lengths = batch["response_lengths"]
        forward_kwargs = {
            "input_ids": tokens,
            "position_ids": None,
            "attention_mask": None,
            "labels": None,
            "packed_seq_params": packed_seq_params,
            "loss_mask": batch["full_loss_masks"],
        }
        if batch["multimodal_train_inputs"] is not None:
            forward_kwargs.update(batch["multimodal_train_inputs"])
        output_tensor = model(**forward_kwargs)

        return output_tensor, partial(
            f,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=args.use_rollout_entropy,
            max_seq_lens=batch.get("max_seq_lens", None),
        )

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if args.custom_megatron_before_log_prob_hook_path:
        from slime.utils.misc import load_function

        custom_before_log_prob_hook = load_function(args.custom_megatron_before_log_prob_hook_path)
        custom_before_log_prob_hook(args, model, store_prefix)

    forward_backward_func = get_forward_backward_func()
    # Don't care about timing during evaluation
    config.timers = None
    forward_data_store = []
    num_steps_per_rollout = len(num_microbatches)
    microbatch_pbar = tqdm(
        total=sum(num_microbatches),
        desc=f"{(store_prefix or getattr(model[0], 'role', 'actor')).rstrip('_')} forward",
        unit="microbatch",
        dynamic_ncols=True,
        leave=False,
        disable=_disable_tqdm_for_non_main_rank(),
    )
    forward_step_with_progress = _wrap_forward_step_with_microbatch_pbar(forward_step, microbatch_pbar)
    for step_id in range(num_steps_per_rollout):
        forward_data_store += forward_backward_func(
            forward_step_func=forward_step_with_progress,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches[step_id],
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=True,
        )
    microbatch_pbar.close()

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    rollout_data = {}
    # Store the results on the last stage
    if mpu.is_pipeline_last_stage():
        keys = forward_data_store[0].keys()
        for key in keys:
            values = []
            for value in forward_data_store:
                assert isinstance(value[key], list)
                values += value[key]

            if args.use_dynamic_batch_size:
                # TODO: This is ugly... Find a better way to make the data have the same order.
                # TODO: move this out of the loop.
                origin_values = [None] * len(values)
                origin_indices = sum(data_iterator[0].micro_batch_indices, [])
                for value, origin_index in zip(values, origin_indices, strict=False):
                    origin_values[origin_index] = value
                values = origin_values
            rollout_data[f"{store_prefix}{key}"] = values
    return rollout_data


def train_one_step(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    data_iterator: Sequence[DataIterator],
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    num_microbatches: int,
    step_global_batch_size: int,
    microbatch_pbar=None,
) -> tuple[dict[str, float], float]:
    """Execute a single pipeline-parallel training step.

    Runs forward/backward over ``num_microbatches``, applies optimizer step and
    one scheduler step when gradients are valid.

    Args:
        args (Namespace): Runtime arguments.
        rollout_id (int): Rollout identifier.
        step_id (int): Step index within the current rollout.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        num_microbatches (int): Number of microbatches to process.
        step_global_batch_size (int): Rollout count for this training step
            (total across DP; one "rollout" = one execution of one of the
            ``n_samples_per_prompt`` rollouts, which may emit >1 training
            sample under compact / subagent). Used both as the loss
            normalizer inside the closure and as the LR scheduler
            ``increment``. In the common case (1 rollout = 1 sample) this
            equals the per-step sample count, so behavior is unchanged.

    Returns:
        tuple[dict[str, float], float]: Reduced loss dictionary (last stage only)
        and gradient norm for logging.
    """
    args = get_args()

    verify_first_step_params = (
        os.environ.get("SLIME_VERIFY_FIRST_STEP_PARAMS", "0") == "1"
        and rollout_id == 0
        and step_id == 0
    )

    # A zero-LR first step is a useful integrity gate for finetuning from a
    # weights-only checkpoint: the live model must not jump when the optimizer
    # copies its FP32 master shards back into BF16 model parameters.  Keep a
    # small, deterministic sample on every rank so this check is cheap even for
    # very large models.  This is intentionally opt-in because it synchronizes
    # all training ranks once for diagnostics.
    first_step_param_samples = []
    if verify_first_step_params:
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if param.numel() == 0:
                    continue
                flat = param.detach().view(-1)
                # TransformerEngine/Megatron sharded parameter wrappers may
                # expose a logical ``numel`` larger than their locally
                # addressable storage.  A bounded contiguous slice is valid
                # for both ordinary tensors and those wrappers.
                first_step_param_samples.append((name, flat[:16].float().clone()))

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if args.custom_megatron_before_train_step_hook_path:
        from slime.utils.misc import load_function

        custom_before_train_step_hook = load_function(args.custom_megatron_before_train_step_hook_path)
        custom_before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)

    def forward_step(data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False) -> tuple[
        torch.Tensor,
        Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]],
    ]:
        """Forward step used by Megatron's pipeline engine during training.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]]]:
            Output tensor(s) and the loss function, which returns
            (loss, num_elems, {"keys": list[str], "values": torch.Tensor}).
        """

        # Get the batch.
        batch = get_batch(
            data_iterator,
            [
                "tokens",
                "multimodal_train_inputs",
                "packed_seq_params",
                "total_lengths",
                "response_lengths",
                "loss_masks",
                "log_probs",
                "ref_log_probs",
                "values",
                "advantages",
                "returns",
                "rollout_log_probs",
                "max_seq_lens",
                "teacher_log_probs",
                "rollout_mask_sums",
            ],
            args.data_pad_size_multiplier,
            args.qkv_format,
            args.allgather_cp,
        )

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            old_stage = os.environ["ROUTING_REPLAY_STAGE"]
            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"

        if return_schedule_plan:
            assert not args.enable_mtp_training, "MTP training should not be enabled when using combined 1f1b"
            position_ids = None
            output_tensor = model.build_schedule_plan(
                input_ids=batch["tokens"],
                position_ids=position_ids,
                attention_mask=None,
                labels=None,
                packed_seq_params=batch["packed_seq_params"],
                loss_mask=batch["full_loss_masks"],
            )
        else:
            forward_kwargs = {
                "input_ids": batch["tokens"],
                "position_ids": None,
                "attention_mask": None,
                "labels": None,
                "packed_seq_params": batch["packed_seq_params"],
                "loss_mask": batch["full_loss_masks"],
            }

            if batch["multimodal_train_inputs"] is not None:
                forward_kwargs.update(batch["multimodal_train_inputs"])

            if args.enable_mtp_training:
                forward_kwargs["mtp_kwargs"] = {"mtp_labels": batch["tokens"]}

            # Float16Module otherwise materializes the complete vocabulary
            # output in FP32 on the last pipeline stage.  For long sequences
            # that fixed upcast is several GiB (5.57 GiB for the measured
            # 122B batch).  The recomputed log-prob path performs numerically
            # stable FP32 work one bounded token tile at a time instead.
            if getattr(args, "recompute_vocab_log_probs", False):
                forward_kwargs["fp32_output"] = False

            output_tensor = model(**forward_kwargs)

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            os.environ["ROUTING_REPLAY_STAGE"] = old_stage

        return output_tensor, partial(loss_function, args, batch, num_microbatches, step_global_batch_size)

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=_wrap_forward_step_with_microbatch_pbar(forward_step, microbatch_pbar),
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )

    valid_step = True
    grad_norm = float("nan")
    if not getattr(args, "check_for_nan_in_loss_and_grad", True):
        found_inf_flag = optimizer.prepare_grads()
        if found_inf_flag:
            valid_step = False
        else:
            grad_norm = optimizer.get_grad_norm()
            if isinstance(grad_norm, torch.Tensor):
                valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
            else:
                valid_step = not (math.isnan(grad_norm) or math.isinf(grad_norm))

    # CI check: verify only MTP parameters have non-zero gradients when truncation happens
    # This check must happen before optimizer.step() as gradients may be modified during step
    if args.ci_test and args.enable_mtp_training:
        from slime.backends.megatron_utils.ci_utils import check_mtp_only_grad

        check_mtp_only_grad(model, step_id)

    if valid_step:
        if verify_first_step_params:
            learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
            logger.info(
                "First-step parameter integrity gate before optimizer.step: "
                "lr_min=%g lr_max=%g sampled_tensors=%d",
                min(learning_rates),
                max(learning_rates),
                len(first_step_param_samples),
            )

        # Update parameters.
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

        if verify_first_step_params:
            local_max_delta = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float32)
            sample_offset = 0
            for model_chunk in model:
                for _name, param in model_chunk.named_parameters():
                    if param.numel() == 0:
                        continue
                    flat = param.detach().view(-1)
                    before_name, before_values = first_step_param_samples[sample_offset]
                    del before_name
                    local_max_delta = torch.maximum(
                        local_max_delta,
                        (flat[:16].float() - before_values).abs().max(),
                    )
                    sample_offset += 1
            dist.all_reduce(local_max_delta, op=dist.ReduceOp.MAX)
            if dist.get_rank() == 0:
                logger.info(
                    "First-step parameter integrity gate after optimizer.step: global_sampled_max_abs_delta=%g",
                    local_max_delta.item(),
                )

        # Update learning rate. Use the per-step global_batch_size when dynamic
        # batching is on so the scheduler's samples-seen counter tracks reality.
        assert update_successful
        opt_param_scheduler.step(increment=step_global_batch_size)

        # Keep the steady state small for colocated training. The next
        # finalize_model_grads call starts the asynchronous H2D reload before
        # optimizer.step(); this D2H copy overlaps the gradient cleanup below.
        if args.offload_optimizer_states:
            _offload_optimizer_states_if_needed(optimizer)

    # release grad
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()
    if args.offload_optimizer_states:
        _release_offloaded_optimizer_states(optimizer)
        # FP8 group-wise Adam initialization leaves small cached segments on
        # every rank. On the measured 122B steady state GPU 0 had 1.68 GiB
        # reserved-but-unused yet only 0.95 GiB globally free, so FLA could not
        # obtain its contiguous 1.24 GiB recurrent-state workspace. Return
        # those segments once per global update, after D2H state release.
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        loss_reduced = reduce_train_step_metrics(
            losses_reduced,
            calculate_per_token_loss=args.calculate_per_token_loss,
            step_global_batch_size=step_global_batch_size,
            cp_size=mpu.get_context_parallel_world_size(),
            dp_with_cp_group=mpu.get_data_parallel_group(with_context_parallel=True),
        )
        return loss_reduced, grad_norm
    return {}, grad_norm


def should_disable_forward_pre_hook(args: Namespace) -> bool:
    """Block forward pre-hook for certain configurations."""
    return args.use_distributed_optimizer and args.overlap_param_gather


def train(
    rollout_id: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    global_batch_sizes: Sequence[int],
) -> None:
    """Run training over a rollout consisting of multiple steps.

    The model is switched to train mode, training hooks are configured, and
    ``train_one_step`` is invoked for each step in the rollout.

    Args:
        rollout_id (int): Rollout identifier.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        num_microbatches (Sequence[int]): Microbatches per step in the rollout.
        global_batch_sizes (Sequence[int]): Rollout count per step (total
            across DP; one "rollout" = one execution of one of the
            ``n_samples_per_prompt`` rollouts of a prompt). Same length as
            ``num_microbatches``; consumed by ``train_one_step`` for loss
            scaling and LR scheduler increments. Equals per-step sample count
            in the common case (1 rollout = 1 sample).
    """
    args = get_args()

    assert len(num_microbatches) == len(global_batch_sizes), (
        f"num_microbatches and global_batch_sizes must have the same length, "
        f"got {len(num_microbatches)} vs {len(global_batch_sizes)}"
    )

    for iterator in data_iterator:
        iterator.reset()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Setup some training config params.
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, (
            "When overlap_grad_reduce is True, config.no_sync_func must be None; "
            "a custom no_sync_func is not supported when overlapping grad-reduce"
        )
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    if args.offload_optimizer_states:

        def finalize_model_grads_with_state_reload(*fmg_args, **fmg_kwargs):
            # Long variable-length microbatches can leave hundreds of MiB in
            # inactive allocator segments.  The FP8 state reloader restores
            # many small storages and otherwise may fail even when reserved
            # but unused memory exceeds the requested allocation.
            torch.cuda.empty_cache()
            for optim_instance in _optimizer_state_offload_instances(optimizer):
                state_offloader = getattr(optim_instance, "_state_offloader", None)
                adam_optimizer = getattr(state_offloader, "adam_optimizer", None)
                # The patched FP8 FusedAdam reloads one bounded moment group
                # inside step() and evicts it immediately afterward.  A bulk
                # reload here recreates the full-state HBM peak and is not
                # needed for gradient finalization.  Master weights are kept
                # resident by the 122B profile; retain the stock path if a
                # different profile explicitly offloaded them.
                if (
                    state_offloader is not None
                    and getattr(adam_optimizer, "_slime_state_offloader", None)
                    is state_offloader
                    and not getattr(
                        state_offloader, "_offloaded_mcore_master_weights", False
                    )
                ):
                    continue
                optim_instance.reload_offloaded_states()
            return finalize_model_grads(*fmg_args, **fmg_kwargs)

        config.finalize_model_grads_func = finalize_model_grads_with_state_reload
    else:
        config.finalize_model_grads_func = finalize_model_grads

    # The first step has master weights but no Adam moments yet. Offload those
    # master weights before forward; later steps arrive here already offloaded
    # by train_one_step's post-step path.
    if args.offload_optimizer_states:
        _offload_optimizer_states_if_needed(optimizer)
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()
        _release_offloaded_optimizer_states(optimizer)

    pre_hook_enabled = False

    if args.reset_optimizer_states:
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            print("Reset optimizer states")
        for chained_optimizer in optimizer.chained_optimizers:
            for group in chained_optimizer.optimizer.param_groups:
                if "step" in group:
                    group["step"] = 0
            for state in chained_optimizer.optimizer.state.values():
                if "step" in state:
                    if isinstance(state["step"], torch.Tensor):
                        state["step"].zero_()
                    else:
                        state["step"] = 0
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, "Manual garbage collection interval should be larger than or equal to 0"
        gc.disable()
        gc.collect()

    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False

    num_steps_per_rollout = len(num_microbatches)
    microbatch_pbar = tqdm(
        total=sum(num_microbatches),
        desc=f"{getattr(model[0], 'role', 'actor')} train",
        unit="microbatch",
        dynamic_ncols=True,
        leave=False,
        disable=_disable_tqdm_for_non_main_rank(),
    )

    # Run training iterations till done.
    for step_id in range(num_steps_per_rollout):

        # Run training step.
        loss_dict, grad_norm = train_one_step(
            args,
            rollout_id,
            step_id,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            num_microbatches[step_id],
            global_batch_sizes[step_id],
            microbatch_pbar=microbatch_pbar,
        )

        if step_id == 0:
            # Enable forward pre-hook after training step has successfully run. All subsequent
            # forward passes will use the forward pre-hook / `param_sync_func` in
            # `forward_backward_func`.
            if should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                config.param_sync_func = param_sync_func
                pre_hook_enabled = True

        if args.enable_mtp_training:
            from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

            mtp_loss_scale = 1 / num_microbatches[step_id]
            tracker = MTPLossLoggingHelper.tracker
            if "values" in tracker:
                values = tracker["values"]
                if tracker.get("reduce_group") is not None:
                    torch.distributed.all_reduce(values, group=tracker.get("reduce_group"))
                if tracker.get("avg_group") is not None:
                    torch.distributed.all_reduce(values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.AVG)
                # here we assume only one mtp layer
                mtp_losses = (tracker["values"] * mtp_loss_scale).item()
                MTPLossLoggingHelper.clean_loss_in_tracker()

                # CI check: verify MTP loss is within expected bounds
                if args.ci_test:
                    from slime.backends.megatron_utils.ci_utils import check_mtp_loss

                    check_mtp_loss(mtp_losses)

        # per train step log.
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
            role = getattr(model[0], "role", "actor")
            role_tag = "" if role == "actor" else f"{role}-"
            log_dict = {
                f"train/{role_tag}{key}": val.mean().item() if isinstance(val, torch.Tensor) else val
                for key, val in loss_dict.items()
            }
            log_dict[f"train/{role_tag}grad_norm"] = grad_norm
            if args.enable_mtp_training:
                log_dict[f"train/{role_tag}mtp_loss"] = mtp_losses

            for param_group_id, param_group in enumerate(optimizer.param_groups):
                log_dict[f"train/{role_tag}lr-pg_{param_group_id}"] = opt_param_scheduler.get_lr(param_group)

            # Per-step gbs — uneven step sizes are easy to miss without this.
            log_dict[f"train/{role_tag}global_batch_size"] = global_batch_sizes[step_id]
            log_dict["train/step"] = accumulated_step_id
            logging_utils.log(args, log_dict, step_key="train/step")

            if args.ci_test and "train/train_rollout_logprob_abs_diff" in log_dict:
                assert log_dict["train/train_rollout_logprob_abs_diff"] <= 0.1, f"{log_dict=}"

            if args.ci_test and not args.ci_disable_kl_checker:
                if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
                    # TODO: figure out why KL is not exactly zero when using PPO loss with KL clipping, and whether this is expected behavior or a bug.
                    assert log_dict["train/ppo_kl"] < 1e-8, f"{log_dict=}"
                if accumulated_step_id == 0 and "train/kl_loss" in log_dict:
                    assert log_dict["train/kl_loss"] < 1e-8, f"{log_dict=}"

            logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict}")

            if args.ci_save_grad_norm is not None:
                ci_save_grad_norm_path = args.ci_save_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                torch.save(grad_norm, ci_save_grad_norm_path)
            elif args.ci_load_grad_norm is not None:
                ci_load_grad_norm_path = args.ci_load_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                expected_grad_norm = torch.load(ci_load_grad_norm_path)
                assert math.isclose(
                    grad_norm,
                    expected_grad_norm,
                    rel_tol=0.01,
                    abs_tol=0.01,
                ), f"grad norm mismatch: {grad_norm} != {expected_grad_norm}"
    microbatch_pbar.close()
    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)


def save(
    iteration: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
) -> None:
    """Persist a training checkpoint safely with forward hooks disabled.

    Args:
        iteration (int): Current global iteration number.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
    """
    args = get_args()
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model)
    _drop_optimizer_cpu_state_before_weights_only_save(optimizer)
    save_checkpoint(
        iteration,
        model,
        optimizer,
        opt_param_scheduler,
        num_floating_point_operations_so_far=0,
        checkpointing_context=None,
        train_data_iterator=None,
        preprocess_common_state_dict_fn=None,
    )
    if should_disable_forward_pre_hook(args):
        enable_forward_pre_hook(model)


def save_hf_model(args, rollout_id: int, model: Sequence[DDP]) -> None:
    """Save Megatron model in HuggingFace format.

    Args:
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        rollout_id (int): Rollout ID for path formatting.
    """
    if args.megatron_to_hf_mode != "bridge":
        try:
            from slime.backends.megatron_utils.hf_checkpoint_saver import save_hf_model_direct

            save_hf_model_direct(args, rollout_id, model)
        except Exception as e:
            if (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
            ):
                logger.error(f"Failed to save HuggingFace format: {e}")
        return

    should_log = (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
    )

    try:
        from megatron.bridge import AutoBridge

        from slime.utils.megatron_bridge_utils import patch_auto_bridge_hf_config, patch_megatron_model

        path = Path(args.save_hf.format(rollout_id=rollout_id))

        if should_log:
            logger.info(f"Saving model in HuggingFace format to {path}")

        bridge = patch_auto_bridge_hf_config(AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True))

        path.mkdir(parents=True, exist_ok=True)

        with patch_megatron_model(model):
            bridge.save_hf_pretrained(
                model,
                path=path,
            )

        if should_log:
            logger.info(f"Successfully saved HuggingFace model to {path}")
    except Exception as e:
        if should_log:
            logger.error(f"Failed to save HuggingFace format: {e}")


def initialize_model_and_optimizer(
    args: Namespace, role: str = "actor"
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
    """Initialize model(s), optimizer, scheduler, and load from checkpoint.

    Args:
        args (Namespace): Runtime arguments.
        role (str): Logical role of the model (e.g., "actor", "critic").

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
            DDP-wrapped model chunks, optimizer, scheduler, and iteration index.
    """

    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

        from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
    model[0].role = role
    reinit_critic_output_layer = _critic_output_layer_needs_reinit(args, model, role)
    clear_memory()
    iteration, _ = load_checkpoint(
        model,
        optimizer,
        opt_param_scheduler,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )
    if reinit_critic_output_layer:
        _reinitialize_critic_output_layer(model)
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()
    clear_memory()

    return model, optimizer, opt_param_scheduler, iteration
