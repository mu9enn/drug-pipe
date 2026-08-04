from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.utils import unwrap_model
from safetensors.torch import load_file, save_file

from slime.utils.megatron_bridge_utils import patch_auto_bridge_hf_config, patch_megatron_model

from .update_weight_from_tensor import UpdateWeightFromTensor


logger = logging.getLogger(__name__)


class UpdateWeightFromLoRA(UpdateWeightFromTensor):
    """One full SFT-base sync followed by lightweight HF LoRA hot reloads.

    The rollout engine starts from the official FP8 HuggingFace release, while
    the trainer starts from the completed full-parameter SFT checkpoint.  The
    first call therefore uses Slime's colocated CUDA-IPC tensor refit.  A
    cross-process NCCL refit is invalid here because trainer and rollout ranks
    intentionally share the same eight physical GPUs.
    Every later call exports only the trainable adapter and hot reloads it in
    SGLang, avoiding a 122B transfer after each policy update.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_synced = False
        self._adapter_loaded = False
        self._adapter_dir = Path(self.args.megatron_lora_sync_dir).resolve()
        self._skip_initial_base_sync = bool(
            getattr(self.args, "megatron_lora_skip_initial_base_sync", False)
            or os.environ.get("SLIME_LORA_SKIP_BASE_SYNC", "0") == "1"
        )

        import slime_plugins.megatron_bridge  # noqa: F401

        self._bridge = patch_auto_bridge_hf_config(
            AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
        )
        roots = unwrap_model(self.model)
        self._roots = roots
        self._peft_config = getattr(roots[0], "_slime_lora_config", None)
        if self._peft_config is None:
            raise RuntimeError("Megatron LoRA config was not attached to the actor model.")
        gdn_targets = {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}
        self._include_gdn_adapters = bool(gdn_targets.intersection(self.args.megatron_lora_target_modules))

    def update_weights(self) -> None:
        if not self._base_synced:
            if self._skip_initial_base_sync:
                logger.info(
                    "Skipping the initial full base sync because the rollout HF checkpoint "
                    "is declared aligned with the Megatron training checkpoint"
                )
            else:
                logger.info("LoRA rollout initialization: synchronizing the full SFT base once")
                super().update_weights()
            self._base_synced = True

        self._export_adapter()
        self._reload_adapter()

    def _export_adapter(self) -> None:
        # save_hf_adapter is collective across TP/PP/EP. Rank 0 writes a small
        # safetensors file only after all distributed shards have been gathered.
        with patch_megatron_model(self.model):
            self._bridge.save_hf_adapter(
                self.model,
                path=self._adapter_dir,
                peft_config=self._peft_config,
                base_model_name_or_path=self.args.hf_checkpoint,
                show_progress=False,
            )
        if self._include_gdn_adapters:
            self._append_language_gdn_adapters()

    def _append_language_gdn_adapters(self) -> None:
        """Append adapters for Slime's explicit Qwen3.5 GDN linears.

        Megatron Bridge exports adapters attached to Megatron parallel linear
        wrappers.  The language-only Qwen3.5 spec represents GDN projections
        as ordinary ``nn.Linear`` modules, whose ``LinearAdapter`` stores
        ``linear_in``/``linear_out`` directly rather than below ``adapter``.
        They are therefore trainable but absent from Bridge's generic PEFT
        stream.  These modules are replicated across TP ranks, so collect the
        local PP slices, verify duplicates, and append canonical PEFT names.
        """
        local_tensors: dict[str, torch.Tensor] = {}
        pattern = re.compile(
            r"(?:^|\.)decoder\.layers\.(\d+)\.self_attention\.linear_attn\."
            r"(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)\."
            r"(linear_in|linear_out)\.weight$"
        )
        for vp_stage, module in enumerate(self._roots):
            offset = get_transformer_layer_offset(module.config, vp_stage)
            for name, param in module.named_parameters():
                match = pattern.search(name)
                if match is None:
                    continue
                local_layer, projection, branch = match.groups()
                global_layer = int(local_layer) + offset
                suffix = "lora_A" if branch == "linear_in" else "lora_B"
                hf_name = (
                    "base_model.model.model.language_model.layers."
                    f"{global_layer}.linear_attn.{projection}.{suffix}.weight"
                )
                local_tensors[hf_name] = param.detach().cpu().contiguous()

        gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(local_tensors, gathered, dst=0)
        if dist.get_rank() == 0:
            merged: dict[str, torch.Tensor] = {}
            for shard in gathered:
                for name, tensor in shard.items():
                    if name in merged and not torch.equal(merged[name], tensor):
                        raise RuntimeError(f"Replicated GDN LoRA tensor differs across ranks: {name}")
                    merged[name] = tensor
            if not merged:
                raise RuntimeError("No Qwen3.5 GDN LoRA tensors were found for adapter export")

            adapter_path = self._adapter_dir / "adapter_model.safetensors"
            tensors = load_file(adapter_path, device="cpu")
            tensors.update(merged)
            temporary_path = adapter_path.with_suffix(".safetensors.tmp")
            save_file(tensors, temporary_path, metadata={"format": "pt"})
            os.replace(temporary_path, adapter_path)

            config_path = self._adapter_dir / "adapter_config.json"
            import json

            config = json.loads(config_path.read_text())
            target_modules = set(config.get("target_modules") or [])
            target_modules.update({"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"})
            config["target_modules"] = sorted(target_modules)
            config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            logger.info("Appended %d explicit Qwen3.5 GDN LoRA tensors", len(merged))
        dist.barrier()

    def _reload_adapter(self) -> None:
        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            if self._adapter_loaded:
                results = ray.get(
                    [engine.unload_lora_adapter.remote(self.args.sglang_lora_name) for engine in self.rollout_engines]
                )
                self._validate_results("unload", results)
            results = ray.get(
                [
                    engine.load_lora_adapter.remote(
                        self.args.sglang_lora_name,
                        str(self._adapter_dir),
                    )
                    for engine in self.rollout_engines
                ]
            )
            self._validate_results("load", results)
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier()
        self._adapter_loaded = True
        self.weight_version += 1

    @staticmethod
    def _validate_results(action: str, results) -> None:
        for result in results:
            if isinstance(result, dict) and not result.get("success", True):
                raise RuntimeError(f"SGLang LoRA {action} failed: {result}")
