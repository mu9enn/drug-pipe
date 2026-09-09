"""Reload an exported Qwen3.5 checkpoint and run its official CPU reference path."""
import json
import sys
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

torch.set_num_threads(16)
# Installed CUDA extensions cannot run on CPU. Use the model's own reference
# implementations for this serialization check; serving retains its GPU kernels.
for name in ['FusedRMSNormGated', 'causal_conv1d_fn', 'causal_conv1d_update',
             'chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule']:
    setattr(qwen, name, None)
qwen.is_fast_path_available = False
model, info = AutoModelForImageTextToText.from_pretrained(
    sys.argv[1], dtype=torch.bfloat16, attn_implementation='eager',
    local_files_only=True, output_loading_info=True,
)
assert not any(info.get(key) for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), info
model.eval()
with torch.no_grad():
    logits = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits
assert torch.isfinite(logits).all()
report = {'passed': True, 'model': sys.argv[1], 'device': 'cpu', 'dtype': str(logits.dtype),
          'loading_info': info, 'logits_shape': list(logits.shape),
          'validation': 'official torch reference forward after full checkpoint deserialization'}
Path(sys.argv[2]).write_text(json.dumps(report, indent=2, default=sorted) + '\n')
