import inspect
import os
from collections import defaultdict

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from safetensors import safe_open

from mbridge.core import register_model
from mbridge.core.safetensor_io import SafeTensorIO
from mbridge.models import Qwen2MoEBridge


class Qwen35DequantFP8SafeTensorIO(SafeTensorIO):
    """Load official block-FP8 weights on the conversion rank's CUDA device."""

    def load_some_hf_weight(self, hf_weight_names: list[str]) -> dict[str, torch.Tensor]:
        from mbridge.models.ext.deepseek_v3.kernel import weight_dequant

        file_to_weight_map = defaultdict(list)
        for name in hf_weight_names:
            file_to_weight_map[self.index[name]].append(name)

        device = f"cuda:{torch.cuda.current_device()}"
        ret = {}
        old_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            for filename, weight_names in file_to_weight_map.items():
                with safe_open(os.path.join(self.hf_dir, filename), framework="pt", device=device) as handle:
                    for name in weight_names:
                        weight = handle.get_tensor(name)
                        scale_name = f"{name}_scale_inv"
                        if weight.element_size() == 1 and scale_name in self.index:
                            scale_filename = self.index[scale_name]
                            if scale_filename == filename:
                                scale_inv = handle.get_tensor(scale_name)
                            else:
                                with safe_open(
                                    os.path.join(self.hf_dir, scale_filename), framework="pt", device=device
                                ) as scale_handle:
                                    scale_inv = scale_handle.get_tensor(scale_name)
                            ret[name] = weight_dequant(weight, scale_inv)
                        else:
                            ret[name] = weight
        finally:
            torch.set_default_dtype(old_default_dtype)
        return ret


@register_model(["qwen3_5", "qwen3_5_moe"])
class Qwen3_5Bridge(Qwen2MoEBridge):
    """
    Bridge for Qwen3.5 models (both dense and MoE variants).
    Qwen3.5 is a VLM model with weights under model.language_model.layers prefix,
    separate in_proj_qkv + in_proj_z for linear attention, and nested text_config.
    """

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": ["model.language_model.layers.{layer_number}.self_attn.o_proj.weight"],
        "self_attention.linear_qkv.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.input_layernorm.weight"
        ],
        "self_attention.q_layernorm.weight": ["model.language_model.layers.{layer_number}.self_attn.q_norm.weight"],
        "self_attention.k_layernorm.weight": ["model.language_model.layers.{layer_number}.self_attn.k_norm.weight"],
        "self_attention.linear_qkv.weight": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
    } | {
        f"self_attention.{weight_name}": ["model.language_model.layers.{layer_number}." + weight_name]
        for weight_name in [
            "input_layernorm.weight",
            # linear attn
            "linear_attn.A_log",
            "linear_attn.conv1d.weight",
            "linear_attn.dt_bias",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            # gated attn (full attention layers)
            "self_attn.k_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        ]
    }

    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.language_model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc1.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.linear_fc2.weight": ["model.language_model.layers.{layer_number}.mlp.down_proj.weight"],
        # MoE mappings
        "shared_experts.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight",
            "model.language_model.layers.{layer_number}.mlp.shared_expert.up_proj.weight",
        ],
        "pre_mlp_layernorm": ["model.language_model.layers.{layer_number}.post_attention_layernorm.weight"],
        "shared_experts.linear_fc2.weight": [
            "model.language_model.layers.{layer_number}.mlp.shared_expert.down_proj.weight"
        ],
        "mlp.router.weight": ["model.language_model.layers.{layer_number}.mlp.gate.weight"],
        "shared_experts.gate_weight": ["model.language_model.layers.{layer_number}.mlp.shared_expert_gate.weight"],
        # Fused expert format: single 3D tensor for all experts
        "mlp.experts.linear_fc1": [
            "model.language_model.layers.{layer_number}.mlp.experts.gate_up_proj",
        ],
        "mlp.experts.linear_fc2": ["model.language_model.layers.{layer_number}.mlp.experts.down_proj"],
    }

    # MTP layer uses individual expert format (not fused)
    _MTP_MLP_MAPPING = {
        "mlp.experts.linear_fc1": [
            "mtp.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
            "mtp.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
        ],
        "mlp.experts.linear_fc2": ["mtp.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"],
    }

    def _uses_hf_block_fp8_weights(self) -> bool:
        """Return whether the source checkpoint is Hugging Face block FP8.

        Qwen's official FP8 checkpoints use per-expert tensors plus companion
        ``weight_scale_inv`` tensors.  The BF16 checkpoints supported by the
        original bridge use fused expert tensors, so the two formats require
        different name mappings as well as FP8 dequantization while importing.
        """
        quantization_config = getattr(self.hf_config, "quantization_config", None)
        if isinstance(quantization_config, dict):
            quant_method = quantization_config.get("quant_method")
        else:
            quant_method = getattr(quantization_config, "quant_method", None)
        return quant_method == "fp8"

    def _get_safetensor_io(self, weights_path: str):
        if self._uses_hf_block_fp8_weights():
            # This loader consumes each weight_scale_inv tensor and materializes
            # the corresponding BF16 value on the current conversion rank.  The
            # saved torch_dist checkpoint therefore starts from the official
            # FP8 checkpoint's dequantized values instead of its raw E4M3 codes.
            return Qwen35DequantFP8SafeTensorIO(self._get_actual_hf_path(weights_path))
        return super()._get_safetensor_io(weights_path)

    def _global_expert_id(self, mcore_weights_name: str) -> int:
        local_expert_id = int(mcore_weights_name.split("weight")[-1])
        from megatron.core import mpu

        ep_size = mpu.get_expert_model_parallel_world_size()
        if ep_size == 1:
            return local_expert_id
        num_experts = self._get_text_config().num_experts
        assert num_experts % ep_size == 0, (num_experts, ep_size)
        num_local_experts = num_experts // ep_size
        return mpu.get_expert_model_parallel_rank() * num_local_experts + local_expert_id

    def _weight_name_mapping_fp8_expert(self, name: str, prefix: str) -> list[str]:
        layer_number = name.split(".")[2]
        expert_id = self._global_expert_id(name)
        hf_prefix = prefix.format(layer_number=layer_number, expert_id=expert_id)
        if "linear_fc1" in name:
            return [f"{hf_prefix}.gate_proj.weight", f"{hf_prefix}.up_proj.weight"]
        if "linear_fc2" in name:
            return [f"{hf_prefix}.down_proj.weight"]
        raise NotImplementedError(f"Unsupported FP8 expert parameter name: {name}")

    # Override to make ffn_hidden_size optional (Qwen3.5 MoE has no intermediate_size)
    _CONFIG_MAPPING = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_query_groups": "num_key_value_heads",
        "ffn_hidden_size": ("intermediate_size", None),
        "attention_dropout": "attention_dropout",
        "layernorm_epsilon": "rms_norm_eps",
        "hidden_dropout": ("hidden_dropout", 0.0),
        "kv_channels": ("head_dim", None),
    }

    def _get_text_config(self):
        """Get the text config, handling VLM nesting."""
        if hasattr(self.hf_config, "text_config"):
            return self.hf_config.text_config
        return self.hf_config

    def _adjust_mapping_for_shared_weights(self):
        text_config = self._get_text_config()
        tie_word_embeddings = getattr(text_config, "tie_word_embeddings", False) or getattr(
            self.hf_config, "tie_word_embeddings", False
        )
        if tie_word_embeddings:
            self._DIRECT_MAPPING = dict(self._DIRECT_MAPPING)
            self._DIRECT_MAPPING["output_layer.weight"] = "model.language_model.embed_tokens.weight"

    def _supports_transformer_config_kwarg(self, kwarg_name: str) -> bool:
        """Check whether the current TransformerConfig accepts a given kwarg."""
        transformer_config_class = getattr(self, "TransformerConfigClass", None)
        if transformer_config_class is None:
            return True

        dataclass_fields = getattr(transformer_config_class, "__dataclass_fields__", None)
        if dataclass_fields is not None:
            return kwarg_name in dataclass_fields

        try:
            signature = inspect.signature(transformer_config_class)
        except (TypeError, ValueError):
            return True
        return kwarg_name in signature.parameters

    def _get_transformer_layer_spec(self, vp_stage=None):
        transformer_layer_spec = super()._get_transformer_layer_spec(vp_stage)
        self._last_transformer_layer_spec = transformer_layer_spec
        return transformer_layer_spec

    def _get_gptmodel_args(self) -> dict:
        """Override to add MTP block spec if needed."""
        ret = super()._get_gptmodel_args()
        text_config = self._get_text_config()
        if getattr(text_config, "mtp_num_hidden_layers", None) is not None:
            transformer_layer_spec = getattr(self, "_last_transformer_layer_spec", None)
            if transformer_layer_spec is None:
                transformer_layer_spec = self._get_transformer_layer_spec()
            mtp_block_spec = get_gpt_mtp_block_spec(self.config, transformer_layer_spec, use_transformer_engine=True)
            ret["mtp_block_spec"] = mtp_block_spec
        return ret

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        """Override to handle fused expert weights.
        For regular layers: experts use fused 3D format (all experts in one tensor).
        For MTP layers: experts use individual format (per-expert tensors).
        """
        if self._uses_hf_block_fp8_weights() and "mlp.experts.linear_fc" in name:
            return self._weight_name_mapping_fp8_expert(
                name,
                "model.language_model.layers.{layer_number}.mlp.experts.{expert_id}",
            )

        layer_number = name.split(".")[2]
        convert_names = []
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                if "{expert_id}" in mapping_names[0]:
                    expert_id = name.split("weight")[-1]
                    convert_names.extend(
                        [x.format(layer_number=layer_number, expert_id=expert_id) for x in mapping_names]
                    )
                else:
                    convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_name_mapping_mtp_mlp(self, name: str) -> list[str]:
        """Handle MTP MLP mappings, keeping per-expert tensors unfused for MoE layers."""
        if self._uses_hf_block_fp8_weights() and "mlp.experts.linear_fc" in name:
            return self._weight_name_mapping_fp8_expert(
                name,
                "mtp.layers.{layer_number}.mlp.experts.{expert_id}",
            )

        layer_number = name.split(".")[2]
        mapping = self._MTP_MLP_MAPPING if "mlp.experts.linear_fc" in name else self._MLP_MAPPING
        convert_names = []
        for keyword, mapping_names in mapping.items():
            if keyword in name:
                if "{expert_id}" in mapping_names[0]:
                    expert_id = name.split("weight")[-1]
                    convert_names.extend(
                        [x.format(layer_number=layer_number, expert_id=expert_id) for x in mapping_names]
                    )
                else:
                    convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        """Override to handle MTP layer mappings."""
        if "mtp" in mcore_weights_name:
            return self._convert_mtp_param(mcore_weights_name)
        return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)

    def _convert_mtp_param(self, name: str) -> list[str]:
        """Convert MTP layer parameters from MCore to HF format."""
        if "mtp.layers." not in name:
            raise NotImplementedError(f"Invalid MTP parameter name: {name}")

        parts = name.split(".")
        mtp_layer_idx = parts[2]  # mtp.layers.{idx}

        direct_name_mapping = {
            f"mtp.layers.{mtp_layer_idx}.eh_proj.weight": "mtp.fc.weight",
            f"mtp.layers.{mtp_layer_idx}.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            f"mtp.layers.{mtp_layer_idx}.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            f"mtp.layers.{mtp_layer_idx}.final_layernorm.weight": "mtp.norm.weight",
        }

        if name in direct_name_mapping:
            return [direct_name_mapping[name]]

        if "transformer_layer" in name:
            proxy_name = name.replace(
                f"mtp.layers.{mtp_layer_idx}.transformer_layer",
                f"decoder.layers.{mtp_layer_idx}",
            )

            if "self_attention" in proxy_name or "input_layernorm.weight" in proxy_name:
                convert_names = super()._weight_name_mapping_attention(proxy_name)
            elif "mlp" in proxy_name or "pre_mlp_layernorm" in proxy_name:
                convert_names = self._weight_name_mapping_mtp_mlp(proxy_name)
            else:
                raise NotImplementedError(f"Unsupported transformer component in MTP: {name}")

            # MTP weights use model.language_model prefix in regular layers,
            # but mtp.layers.{idx} directly for MTP layers
            convert_names = [
                cn.replace(f"model.language_model.layers.{mtp_layer_idx}", f"mtp.layers.{mtp_layer_idx}")
                for cn in convert_names
            ]
            return convert_names

        raise NotImplementedError(f"Unsupported MTP parameter name: {name}")

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> tuple[list[str], list[torch.Tensor]]:
        if "self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            # merge qkv
            assert len(hf_weights) == 3
            text_config = self._get_text_config()
            num_key_value_heads = text_config.num_key_value_heads
            hidden_dim = text_config.hidden_size
            num_attention_heads = text_config.num_attention_heads
            num_querys_per_group = num_attention_heads // text_config.num_key_value_heads
            head_dim = getattr(text_config, "head_dim", hidden_dim // num_attention_heads)
            group_dim = head_dim * num_attention_heads // num_key_value_heads
            q, k, v = hf_weights
            # q k v might be tp split
            real_num_key_value_heads = q.shape[0] // (2 * group_dim)
            q = (
                q.view(
                    [
                        real_num_key_value_heads,
                        num_querys_per_group,
                        2,
                        head_dim,
                        -1,
                    ]
                )
                .transpose(1, 2)
                .flatten(1, 3)
            )
            k = k.view([real_num_key_value_heads, head_dim, -1])
            v = v.view([real_num_key_value_heads, head_dim, -1])
            out_shape = [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]

            qgkv = torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()
            return qgkv

        # Handle fused expert weights: extract single expert from 3D fused tensor
        if "mlp.experts.linear_fc" in mcore_weights_name and len(hf_weights) == 1:
            w = hf_weights[0]
            if w.dim() == 3:
                # Extract local expert_id from name like "...linear_fc1.weight42"
                local_expert_id = int(mcore_weights_name.split("weight")[-1])
                # When using Expert Parallelism (EP), the local expert_id is relative
                # to this EP rank. We need to convert to global expert_id to index
                # into the full HF fused tensor [num_experts, ...].
                from megatron.core import mpu

                ep_size = mpu.get_expert_model_parallel_world_size()
                if ep_size > 1:
                    ep_rank = mpu.get_expert_model_parallel_rank()
                    num_local_experts = w.shape[0] // ep_size
                    global_expert_id = ep_rank * num_local_experts + local_expert_id
                else:
                    global_expert_id = local_expert_id
                expert_w = w[global_expert_id]  # (out_features, in_features)
                return expert_w.contiguous()

        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)

    def _build_config(self):
        text_config = self._get_text_config()

        mtp_args = {}
        if hasattr(text_config, "mtp_num_hidden_layers"):
            mtp_args["mtp_num_layers"] = text_config.mtp_num_hidden_layers

        base_kwargs = dict(
            text_config_key="text_config" if hasattr(self.hf_config, "text_config") else None,
            use_cpu_initialization=False,
            # Other optimizations
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            # Qwen3.5 specific
            moe_router_pre_softmax=False,
            qk_layernorm=True,
            attention_output_gate=True,
            **mtp_args,
        )

        if self._supports_transformer_config_kwarg("use_gated_attention"):
            base_kwargs["use_gated_attention"] = True

        # Handle MoE-specific config
        if hasattr(text_config, "num_experts"):
            base_kwargs.update(
                moe_ffn_hidden_size=text_config.moe_intermediate_size,
                moe_shared_expert_intermediate_size=getattr(text_config, "shared_expert_intermediate_size", None),
                moe_router_bias_update_rate=0.001,
                moe_router_topk=text_config.num_experts_per_tok,
                num_moe_experts=text_config.num_experts,
                moe_aux_loss_coeff=text_config.router_aux_loss_coef,
                moe_router_load_balancing_type="none",
                moe_grouped_gemm=True,
                moe_router_score_function="softmax",
                moe_shared_expert_gate=True,
            )
            # For MoE models without intermediate_size, use shared_expert_intermediate_size
            if not hasattr(text_config, "intermediate_size"):
                base_kwargs["ffn_hidden_size"] = text_config.shared_expert_intermediate_size

        return self._build_base_config(**base_kwargs)
