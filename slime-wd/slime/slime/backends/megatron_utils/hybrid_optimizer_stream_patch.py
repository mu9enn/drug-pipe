"""Compatibility fix for the pinned Megatron hybrid optimizer.

Megatron commit 1dcf0da records completion of CPU-to-GPU parameter copies on
the D2H stream.  The actor can therefore back up and publish parameters before
the H2D copies have finished.  Current upstream Megatron records the event on
the H2D stream instead.  Install that upstream behavior before any hybrid
optimizer is constructed.
"""

from __future__ import annotations

import torch


def install_hybrid_optimizer_h2d_wait_patch() -> bool:
    try:
        from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import (
            HybridDeviceOptimizer,
            _param_generator,
        )
    except ImportError:
        return False

    if getattr(HybridDeviceOptimizer, "_slime_h2d_wait_patch", False):
        return True

    def _register_param_copy_back_gpu_hook(self):
        def param_copy_back_gpu_hook_closure():
            def param_copy_back_gpu_hook(optimizer, args, kwargs):
                current_stream = torch.cuda.current_stream()
                self._h2d_stream.wait_stream(current_stream)
                with torch.cuda.stream(self._h2d_stream):
                    for param in _param_generator(optimizer):
                        gpu_param = self.cpu_copys_map_gpu_param[param]
                        gpu_param.data.copy_(param.data, non_blocking=True)
                # The consumer stream must wait for the copies it consumes.
                # The pinned Megatron version incorrectly records this event
                # on _d2h_stream, which does not order the H2D work above.
                self._h2d_stream.record_event().wait(current_stream)

            return param_copy_back_gpu_hook

        def fp32_param_copy_back_gpu_hook_closure():
            def fp32_param_copy_back_gpu_hook(optimizer, args, kwargs):
                for group in self.param_groups:
                    for param in group["params"]:
                        if param in self.gpu_params_map_cpu_copy:
                            continue
                        if param in self.param_to_fp32_param:
                            fp32_param = self.param_to_fp32_param[param]
                            param.data.copy_(fp32_param.data)

            return fp32_param_copy_back_gpu_hook

        for optimizer in self.sub_optimizers:
            if optimizer is not self.gpu_optimizer:
                optimizer.register_step_post_hook(param_copy_back_gpu_hook_closure())
            elif self.param_update_in_fp32:
                optimizer.register_step_post_hook(fp32_param_copy_back_gpu_hook_closure())

    HybridDeviceOptimizer._register_param_copy_back_gpu_hook = _register_param_copy_back_gpu_hook
    HybridDeviceOptimizer._slime_h2d_wait_patch = True
    return True
