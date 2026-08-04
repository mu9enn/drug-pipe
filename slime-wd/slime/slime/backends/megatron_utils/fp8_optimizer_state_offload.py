"""Compatibility patch for MCore optimizer-state offload with TE FP8 states.

The pinned MCore offloader treats Transformer Engine ``Float8Tensor.dtype``
as FP32.  It consequently dequantizes FP8 Adam moments into four-byte CPU
buffers and tries to resize an invalid wrapper storage on release.  Preserve
the tensor's one-byte ``_data`` representation instead; the tiny FP8 scales
remain resident on GPU and are reused when the raw bytes are restored.
"""

from __future__ import annotations

import os

import torch


# TE FusedAdam materializes FP32 copies of every low-precision state in a
# parameter group before launching its fused kernel. MCore's MoE groups can
# contain many billions of elements, so that transient defeats the memory
# saved by FP8 states. Keep each tensor intact but bound aggregate group size.
_DEFAULT_MAX_GROUP_NUMEL = int(
    os.environ.get("SLIME_FP8_OPTIMIZER_MAX_GROUP_NUMEL", 64 * 1024 * 1024)
)
if _DEFAULT_MAX_GROUP_NUMEL <= 0:
    raise ValueError("SLIME_FP8_OPTIMIZER_MAX_GROUP_NUMEL must be positive")


def _storage_tensor(tensor: torch.Tensor) -> torch.Tensor:
    raw = getattr(tensor, "_data", None)
    if isinstance(raw, torch.Tensor) and raw.dtype == torch.uint8 and raw.is_cuda:
        return raw
    return tensor


def _install_fused_adam_fragmentation_patch() -> bool:
    """Reclaim cached segments and stream initialized FP8 moments to CPU."""
    try:
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.optimizers import FusedAdam
        from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
    except ImportError:
        return False
    if getattr(FusedAdam, "_slime_fp8_fragmentation_patch", False):
        return True

    original_initialize_state = FusedAdam._initialize_state
    original_initialize_all_state = FusedAdam.initialize_state
    original_get_unscaled_state = FusedAdam.get_unscaled_state
    original_apply_scale = FusedAdam._apply_scale
    original_step = FusedAdam.step

    def _empty_cache_under_pressure(param):
        if not param.is_cuda:
            return
        free_bytes, _ = torch.cuda.mem_get_info(param.device)
        unused_reserved = torch.cuda.memory_reserved(param.device) - torch.cuda.memory_allocated(
            param.device
        )
        if free_bytes < 4 * 1024**3 and unused_reserved > 256 * 1024**2:
            torch.cuda.empty_cache()

    def _initialize_state_with_reclaim(self, param, *args, **kwargs):
        _empty_cache_under_pressure(param)
        state_name = args[0] if args else kwargs.get("state_name")
        zero_buffer = args[1] if len(args) > 1 else kwargs.get("zero_buffer", False)
        store_param_remainders = (
            args[2] if len(args) > 2 else kwargs.get("store_param_remainders", False)
        )
        if (
            zero_buffer
            and not store_param_remainders
            and self.name_to_dtype_map[state_name] == torch.uint8
        ):
            # TE's stock path creates a uint8 buffer and then calls
            # ``data.float()`` before quantizing an all-zero Adam moment.  A
            # single 6M-element parameter therefore needs a needless 24 MiB
            # FP32 transient exactly at the optimizer-state peak.  FP8 zero has
            # an all-zero bit pattern, so initialize the raw storage directly
            # while preserving the same quantizer and optimizer scale metadata.
            quantizer = Float8Quantizer(
                scale=torch.ones([1], dtype=torch.float32, device=param.device),
                amax=torch.zeros([1], dtype=torch.float32, device=param.device),
                fp8_dtype=tex.DType.kFloat8E4M3,
            )
            fp8_state = quantizer.make_empty(param.shape)
            _storage_tensor(fp8_state).zero_()
            self.state[param][state_name] = fp8_state
            if param not in self._scales:
                self._scales[param] = {}
            self._scales[param][state_name] = torch.ones(
                [1], dtype=torch.float32, device=param.device
            )
            _empty_cache_under_pressure(param)
            return None
        result = original_initialize_state(self, param, *args, **kwargs)
        _empty_cache_under_pressure(param)
        return result

    def _initialize_all_state_without_fp32_master_transient(
        self, param, store_param_remainders
    ):
        """Initialize an FP8 parameter's FP16 master copy at its final dtype.

        TE's stock path first allocates the final FP16 ``master_param`` and
        then dequantizes the FP8 model parameter into a full FP32 temporary
        before scaling it back to FP16.  On the 122B pipeline rank that final
        24 MiB temporary is the only allocation that does not fit.  With a
        unit optimizer scale, dequantizing directly to the final FP16 storage
        is numerically identical to that FP32-to-FP16 conversion and avoids
        both the temporary and a simultaneous duplicate FP16 allocation.
        """
        master_dtype = self.name_to_dtype_map.get("master_param")
        if (
            self.master_weights
            and isinstance(param, torch.Tensor)
            and hasattr(param, "dequantize")
            and hasattr(param, "_data")
            and not store_param_remainders
            and master_dtype in (torch.float16, torch.bfloat16)
        ):
            self._initialize_state(param, "exp_avg", zero_buffer=True)
            self._initialize_state(param, "exp_avg_sq", zero_buffer=True)

            # Float8Tensor.dequantize allocates exactly one tensor at the
            # requested dtype.  Store that allocation itself as the master
            # state; allocating a destination first would recreate a 12 MiB
            # transient at the same capacity boundary.
            master_param = param.dequantize(dtype=master_dtype).detach()
            self.state[param]["master_param"] = master_param
            if param not in self._scales:
                self._scales[param] = {}
            self._scales[param]["master_param"] = torch.ones(
                [1], dtype=torch.float32, device=param.device
            )
            _empty_cache_under_pressure(param)
            return None
        return original_initialize_all_state(self, param, store_param_remainders)

    def _get_unscaled_state_with_reclaim(self, param, *args, **kwargs):
        _empty_cache_under_pressure(param)
        unscaled = original_get_unscaled_state(self, param, *args, **kwargs)
        if getattr(self, "_slime_low_peak_unscale_active", False):
            state_name = args[0] if args else kwargs.get("state_name")
            scaled_state = self.state[param][state_name]
            storage_tensor = _storage_tensor(scaled_state)
            if storage_tensor.is_cuda and self.name_to_dtype_map[state_name] != torch.float32:
                # The fused kernel consumes the FP32 unscaled tensor, not the
                # low-precision state, until the post-kernel scaling pass.
                # Release the latter only after dequantization so the next two
                # FP32 states for this parameter have physical HBM to land in.
                storage_tensor.untyped_storage().resize_(0)
        return unscaled

    def _apply_scale_with_reallocated_destination(
        self, state_name, unscaled_state, scaled_state, scale
    ):
        storage_tensor = _storage_tensor(scaled_state)
        if (
            getattr(self, "_slime_low_peak_unscale_active", False)
            and storage_tensor.is_cuda
            and storage_tensor.untyped_storage().nbytes() == 0
        ):
            storage_tensor.untyped_storage().resize_(
                storage_tensor.numel() * storage_tensor.element_size()
            )
        if self.name_to_dtype_map[state_name] == torch.uint8:
            # TE 2.10's FP8 branch uses ``1 / scale`` without handling an
            # all-zero state.  In MoE training an expert can legitimately
            # receive no tokens, leaving its Adam moment exactly zero.  The
            # stock path then installs an infinite quantizer scale and encodes
            # zeros as 0x7f; the next dequantization returns NaNs and corrupts
            # the following optimizer update.  Keep a unit quantizer scale for
            # the zero case while preserving TE's behavior for every nonzero
            # (including non-finite) state.
            max_range = self.dtype_to_range_map[torch.uint8]
            if max_range.device != scaled_state.device:
                max_range = max_range.to(scaled_state.device)
                self.dtype_to_range_map[torch.uint8] = max_range
            if unscaled_state.device != scaled_state.device:
                unscaled_state = unscaled_state.to(scaled_state.device)
            min_val, max_val = torch.aminmax(unscaled_state)
            absmax = torch.maximum(-min_val, max_val).to(
                dtype=torch.float32, device=unscaled_state.device
            )
            torch.div(absmax, max_range, out=scale)
            safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            scaled_state._quantizer.scale.copy_(safe_scale.reciprocal())
            scaled_state.copy_(unscaled_state)
            result = None
        else:
            result = original_apply_scale(
                self, state_name, unscaled_state, scaled_state, scale
            )
        if getattr(self, "_slime_low_peak_unscale_active", False):
            # Scaling is the final consumer of this FP32 tensor.  Release it
            # immediately rather than retaining all three state temporaries
            # until TE leaves the parameter group.
            unscaled_state.untyped_storage().resize_(0)
        return result

    def _step_with_streamed_moment_offload(self, closure=None, grad_scaler=None):
        """Run bounded groups and evict updated moments before the next group.

        TE converts low-precision Adam states to FP32 group by group, but keeps
        every newly initialized FP8 moment resident until ``step`` returns.
        A 122B first step therefore accumulates roughly two extra bytes per
        local parameter and can OOM near the final groups.  When MCore state
        offload is bound to this optimizer, synchronously save each completed
        group's raw FP8 moment bytes into the offloader's canonical CPU
        buffers, then release only those moment storages.  Master weights stay
        resident until MCore finishes its post-step parameter synchronization.
        """
        offloader = getattr(self, "_slime_state_offloader", None)
        if offloader is None:
            return original_step(self, closure=closure, grad_scaler=grad_scaler)

        original_groups = self.param_groups
        loss = None
        try:
            for group_index, group in enumerate(original_groups):
                self.param_groups = [group]
                # Later optimizer steps arrive with the FP8 moments released
                # from HBM and their canonical bytes in the offloader's CPU
                # buffers.  MCore's stock reload restores *all* groups before
                # gradient finalization, which makes the 122B second step peak
                # at 139.98 GiB.  Restore only this bounded group immediately
                # before TE consumes it; the block below evicts it again before
                # advancing to the next group.
                for param in group["params"]:
                    param_state = self.state.get(param, {})
                    cpu_states = offloader._opt_state_cpu_buffers.get(param, {})
                    for state_key in ("exp_avg", "exp_avg_sq"):
                        state = param_state.get(state_key)
                        cpu_buffer = cpu_states.get(state_key)
                        if not isinstance(state, torch.Tensor) or not isinstance(
                            cpu_buffer, torch.Tensor
                        ):
                            continue
                        storage_tensor = _storage_tensor(state)
                        if (
                            not storage_tensor.is_cuda
                            or storage_tensor.untyped_storage().nbytes() != 0
                        ):
                            continue
                        storage_tensor.untyped_storage().resize_(
                            cpu_buffer.untyped_storage().size()
                        )
                        storage_tensor.copy_(cpu_buffer, non_blocking=False)

                self._slime_low_peak_unscale_active = True
                try:
                    group_loss = original_step(
                        self,
                        closure=closure if group_index == 0 else None,
                        grad_scaler=grad_scaler,
                    )
                finally:
                    self._slime_low_peak_unscale_active = False
                if group_loss is not None:
                    loss = group_loss

                for param in group["params"]:
                    param_state = self.state.get(param, {})
                    for state_key in ("exp_avg", "exp_avg_sq"):
                        state = param_state.get(state_key)
                        if not isinstance(state, torch.Tensor):
                            continue
                        storage_tensor = _storage_tensor(state)
                        if (
                            not storage_tensor.is_cuda
                            or storage_tensor.untyped_storage().nbytes() == 0
                        ):
                            continue
                        # Pageable host buffers are deliberate here.  The
                        # 122B first step creates hundreds of GiB of moment
                        # storage; pinning all of it makes that storage appear
                        # as shared/locked memory and can trip the pod's host
                        # memory limit even when ``free`` still looks healthy.
                        cpu_buffer = offloader._ensure_state_cpu_buffer(
                            param, state_key, storage_tensor, pin_memory=False
                        )
                        # A blocking copy is intentional: the GPU storage must
                        # be reusable by the following parameter group.
                        cpu_buffer.copy_(storage_tensor, non_blocking=False)
                        storage_tensor.untyped_storage().resize_(0)

                # TE's FP32 conversion temporaries die when the one-group call
                # returns. Return their cached segments to the allocator before
                # materializing the following group.
                torch.cuda.empty_cache()
        finally:
            self.param_groups = original_groups
        return loss

    FusedAdam._initialize_state = _initialize_state_with_reclaim
    FusedAdam.initialize_state = _initialize_all_state_without_fp32_master_transient
    FusedAdam.get_unscaled_state = _get_unscaled_state_with_reclaim
    FusedAdam._apply_scale = _apply_scale_with_reallocated_destination
    FusedAdam.step = _step_with_streamed_moment_offload
    FusedAdam._slime_fp8_fragmentation_patch = True
    return True


def install_fp8_optimizer_state_offload_patch() -> bool:
    # This installation is independent of the MCore offloader marker: Slime
    # can import the compatibility module more than once under Ray workers.
    _install_fused_adam_fragmentation_patch()
    try:
        from megatron.core.optimizer.cpu_offloading.optimizer_state_offloader import (
            OptimizerStateOffloader,
        )
    except ImportError:
        return False

    if getattr(OptimizerStateOffloader, "_slime_fp8_raw_state_patch", False):
        return True

    def _offload_states(self, offload_optimizer_states, offload_master_weights, use_pin_memory=True):
        # A FusedAdam instance bound by Slime uses streamed FP8 moments.  Keep
        # its remaining states and MCore master weights pageable as well, so a
        # later bulk offload cannot reintroduce a several-hundred-GiB pinned
        # allocation spike.  Copies are synchronous in this mode.
        if getattr(self.adam_optimizer, "_slime_state_offloader", None) is self:
            use_pin_memory = False
        self._offloaded_state_keys = self._get_state_keys_to_offload(
            offload_optimizer_states, offload_master_weights
        )
        for param, param_state in self.adam_optimizer.state.items():
            for state_key in self._offloaded_state_keys:
                if state_key not in param_state:
                    continue
                gpu_tensor = param_state[state_key]
                if not isinstance(gpu_tensor, torch.Tensor):
                    continue
                storage_tensor = _storage_tensor(gpu_tensor)
                if not storage_tensor.is_cuda:
                    continue
                existing_cpu_buffer = self._opt_state_cpu_buffers.get(param, {}).get(state_key)
                if (
                    storage_tensor.untyped_storage().nbytes() == 0
                    and existing_cpu_buffer is not None
                ):
                    # The group-wise FusedAdam patch already copied this FP8
                    # moment into the canonical offloader buffer.
                    continue
                cpu_buffer = self._ensure_state_cpu_buffer(
                    param, state_key, storage_tensor, use_pin_memory
                )
                cpu_buffer.copy_(storage_tensor, non_blocking=use_pin_memory)
                storage_tensor.record_stream(self._d2h_stream)

        if offload_master_weights and not self.optimizer_contains_master_weights:
            self._offload_shard_groups(
                self.dist_optimizer.shard_fp32_from_float16_groups,
                self._shard_fp32_from_float16_cpu_buffers,
                use_pin_memory,
            )
            self._offloaded_mcore_master_weights = True

    def _release_states(self):
        for param, param_state in self.adam_optimizer.state.items():
            if param not in self._opt_state_cpu_buffers:
                continue
            for state_key in self._offloaded_state_keys:
                if state_key not in self._opt_state_cpu_buffers[param] or state_key not in param_state:
                    continue
                _storage_tensor(param_state[state_key]).untyped_storage().resize_(0)

        if self._offloaded_mcore_master_weights:
            for group in self.dist_optimizer.shard_fp32_from_float16_groups:
                for gpu_tensor in group:
                    gpu_tensor.untyped_storage().resize_(0)

    def _reload_states(self, is_allocate_stage):
        for param, param_state in self.adam_optimizer.state.items():
            if param not in self._opt_state_cpu_buffers:
                continue
            for state_key in self._offloaded_state_keys:
                if state_key not in self._opt_state_cpu_buffers[param] or state_key not in param_state:
                    continue
                cpu_buffer = self._opt_state_cpu_buffers[param][state_key]
                storage_tensor = _storage_tensor(param_state[state_key])
                if is_allocate_stage:
                    storage_tensor.untyped_storage().resize_(cpu_buffer.untyped_storage().size())
                else:
                    storage_tensor.copy_(cpu_buffer, non_blocking=cpu_buffer.is_pinned())

        if self._offloaded_mcore_master_weights:
            self._reload_shard_groups(
                self.dist_optimizer.shard_fp32_from_float16_groups,
                self._shard_fp32_from_float16_cpu_buffers,
                is_allocate_stage,
            )

    OptimizerStateOffloader._offload_states = _offload_states
    OptimizerStateOffloader._release_states = _release_states
    OptimizerStateOffloader._reload_states = _reload_states
    OptimizerStateOffloader._slime_fp8_raw_state_patch = True

    return True


def split_fp8_optimizer_param_groups(optimizer, max_group_numel: int = _DEFAULT_MAX_GROUP_NUMEL) -> int:
    """Bound TE FusedAdam's low-precision-state dequantization transient.

    Returns the number of FusedAdam parameter groups after splitting. A
    single parameter larger than the cap remains a one-parameter group.
    """
    try:
        from transformer_engine.pytorch.optimizers import FusedAdam
    except ImportError:
        return 0

    if max_group_numel <= 0:
        raise ValueError("max_group_numel must be positive")

    instances = list(getattr(optimizer, "chained_optimizers", ())) or [optimizer]
    total_groups = 0
    for instance in instances:
        adam_optimizer = getattr(instance, "optimizer", None)
        if not isinstance(adam_optimizer, FusedAdam):
            continue
        if getattr(adam_optimizer, "_slime_fp8_groups_split", False):
            total_groups += len(adam_optimizer.param_groups)
            continue

        split_groups = []
        for original in adam_optimizer.param_groups:
            metadata = {key: value for key, value in original.items() if key != "params"}
            chunk = []
            chunk_numel = 0
            for param in original["params"]:
                param_numel = param.numel()
                if chunk and chunk_numel + param_numel > max_group_numel:
                    split_groups.append({**metadata, "params": chunk})
                    chunk = []
                    chunk_numel = 0
                chunk.append(param)
                chunk_numel += param_numel
            if chunk:
                split_groups.append({**metadata, "params": chunk})

        adam_optimizer.param_groups = split_groups
        adam_optimizer._slime_fp8_groups_split = True
        total_groups += len(split_groups)
    return total_groups


def bind_fp8_optimizer_state_offload(optimizer) -> int:
    """Give each TE FusedAdam access to its owning MCore state offloader."""
    bound = 0
    for instance in getattr(optimizer, "chained_optimizers", ()):
        state_offloader = getattr(instance, "_state_offloader", None)
        adam_optimizer = getattr(instance, "optimizer", None)
        if state_offloader is None or adam_optimizer is None:
            continue
        adam_optimizer._slime_state_offloader = state_offloader
        bound += 1
    return bound
