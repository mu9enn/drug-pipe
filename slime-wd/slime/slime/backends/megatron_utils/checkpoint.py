import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path

# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args

from slime.utils import megatron_bridge_utils

try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # TODO: find a less hacky way to do this.
    import torch.distributed as dist
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass

logger = logging.getLogger(__name__)

__all__ = ["save_checkpoint"]


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()
    load_path = args.load

    assert Path(load_path).exists() and _is_dir_nonempty(
        load_path
    ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    if _is_megatron_checkpoint(load_path):
        with _allow_missing_lora_factory_keys(getattr(args, "megatron_lora", False)):
            return _load_checkpoint_megatron(
                ddp_model=ddp_model,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
                checkpointing_context=checkpointing_context,
                skip_load_to_model_and_opt=skip_load_to_model_and_opt,
            )
    else:
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )


@contextmanager
def _allow_missing_lora_factory_keys(enabled: bool):
    """Let a PEFT model restore a checkpoint created before LoRA injection.

    MCore strictness can filter ordinary missing adapter tensors, but tensor
    factories are merged afterwards and historically require identical dict
    keys.  Megatron-Bridge wraps several fused projections in factories, so a
    full-parameter SFT checkpoint has no ``.adapter.`` subtree to merge.  Skip
    only those missing LoRA factory leaves; retain the normal hard failure for
    every base-model key.
    """
    if not enabled:
        yield
        return

    from megatron.core.dist_checkpointing import mapping, serialization

    original = serialization.apply_factory_merges

    def drop_fp8_extra_state(tree):
        """Discard runtime FP8 calibration metadata when changing module topology."""
        removed = 0
        if isinstance(tree, dict):
            for child_key in list(tree):
                if isinstance(child_key, str) and "_extra_state" in child_key:
                    del tree[child_key]
                    removed += 1
                else:
                    removed += drop_fp8_extra_state(tree[child_key])
        elif isinstance(tree, list):
            for child in tree:
                removed += drop_fp8_extra_state(child)
        return removed

    def merge_allowing_missing_adapter(x1, x2, key=()):
        if isinstance(x2, mapping.ShardedTensorFactory):
            return x2.merge_fn(x1)
        if isinstance(x1, dict) and isinstance(x2, dict):
            for child_key, child_factory in x2.items():
                if child_key not in x1:
                    full_key = ".".join(str(part) for part in (*key, child_key))
                    if ".adapter." in f".{full_key}.":
                        logger.info("Keeping initialized LoRA tensor absent from base checkpoint: %s", full_key)
                        continue
                    raise ValueError(
                        "Different non-LoRA dict keys encountered in LoRA factory merge "
                        f"at {full_key}: checkpoint={x1.keys()} runtime={x2.keys()}"
                    )
                x1[child_key] = merge_allowing_missing_adapter(
                    x1[child_key], child_factory, key=(*key, child_key)
                )
            if not key:
                # TransformerEngine's delayed-scaling state is tied to the
                # exact linear-module topology.  Adapter injection changes the
                # runtime FP8 metadata layout (for example 3 vs 384 scale
                # entries) even though the frozen base weights are identical.
                # Fine-tuning should warm fresh scales instead of restoring
                # incompatible calibration history.
                removed = drop_fp8_extra_state(x1)
                logger.info("Dropped %d FP8 extra-state entries while restoring a LoRA finetune", removed)
            return x1
        if isinstance(x1, list) and isinstance(x2, list):
            if len(x1) != len(x2):
                raise ValueError(f"Different list lengths in LoRA factory merge at {key}: {len(x1)} != {len(x2)}")
            for index, child_factory in enumerate(x2):
                x1[index] = merge_allowing_missing_adapter(x1[index], child_factory, key=(*key, index))
            return x1
        if isinstance(x1, list) and isinstance(x2, dict):
            for index, child_factory in x2.items():
                if not isinstance(index, int) or index >= len(x1):
                    raise ValueError(f"Invalid list key in LoRA factory merge at {key}: {index}")
                x1[index] = merge_allowing_missing_adapter(x1[index], child_factory, key=(*key, index))
            return x1
        return x1

    serialization.apply_factory_merges = merge_allowing_missing_adapter
    try:
        yield
    finally:
        serialization.apply_factory_merges = original


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    import slime_plugins.megatron_bridge  # noqa: F401

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")

    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = megatron_bridge_utils.patch_auto_bridge_hf_config(
            AutoBridge.from_hf_pretrained(load_path, trust_remote_code=True)
        )
        bridge.load_hf_weights(ddp_model)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)
