"""Adapter-export dispatch for Slime's language-only Qwen3.5 Megatron model.

NVIDIA's Qwen3.5 bridge currently registers the HF conditional-generation
classes against ``Qwen3VLModel``.  Slime deliberately builds only their text
backbone as a ``GPTModel``.  Full weight conversion has a Slime-native path,
but PEFT export uses Megatron Bridge's adapter streamer and therefore needs the
equivalent language-only dispatch.
"""

from __future__ import annotations

import re
from typing import Union

from megatron.bridge.models.conversion.model_bridge import (
    HFWeightTuple,
    stream_adapter_weights_megatron_to_hf,
)
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.module import MegatronModule
from transformers import Qwen3_5MoeForConditionalGeneration


class _LanguageOnlyMappingRegistry:
    """Resolve GPTModel names through the official VLM language mappings."""

    def __init__(self, delegate):
        self._delegate = delegate

    def megatron_to_hf_lookup(self, name: str):
        mapping = self._delegate.megatron_to_hf_lookup(name)
        if mapping is None and not name.startswith("language_model."):
            mapping = self._delegate.megatron_to_hf_lookup(f"language_model.{name}")
        if mapping is None:
            # Slime's language-only Qwen3.5 spec keeps the six GDN projection
            # matrices as explicit nn.Linear children below ``linear_attn``;
            # NVIDIA's VLM spec represents them through two fused ``in_proj``
            # mappings.  PEFT export must retain the canonical HF leaf names.
            match = re.fullmatch(
                r"decoder\.layers\.(\d+)\.self_attention\.linear_attn\."
                r"(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)\.weight",
                name,
            )
            if match:
                layer, module = match.groups()
                mapping = AutoMapping(
                    megatron_param=name,
                    hf_param=(
                        f"model.language_model.layers.{layer}.linear_attn."
                        f"{module}.weight"
                    ),
                )
        return mapping


class Qwen35LanguageOnlyAdapterBridge(Qwen35VLMoEBridge):
    """Reuse NVIDIA's Qwen3.5 HF PEFT names for a bare GPTModel backbone."""

    def mapping_registry(self):
        return _LanguageOnlyMappingRegistry(super().mapping_registry())


@stream_adapter_weights_megatron_to_hf.impl(
    (Qwen3_5MoeForConditionalGeneration, GPTModel)
)
def _stream_qwen35_moe_language_adapter(
    _,
    megatron_model: Union[MegatronModule, list[MegatronModule]],
    cpu: bool = True,
    show_progress: bool = True,
) -> list[HFWeightTuple]:
    bridge = Qwen35LanguageOnlyAdapterBridge()
    return bridge.stream_adapter_weights_megatron_to_hf(
        megatron_model,
        cpu=cpu,
        show_progress=show_progress,
    )
