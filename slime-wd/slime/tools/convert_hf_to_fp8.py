"""
python tools/convert_hf_to_fp8.py [-h] [--model-dir MODEL_DIR] [--save-dir SAVE_DIR] [--strategy {block,channel,tensor}] [--block-size [BLOCK_SIZE ...]]
                           [--max-workers MAX_WORKERS]

options:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
                        Path to the directory of the HF safetensors model.
  --save-dir SAVE_DIR   Path to the directory to save the converted model.
  --strategy {block,channel,tensor}
  --block-size [BLOCK_SIZE ...]
                        eg. --block-size 32 32
  --max-workers MAX_WORKERS
                        Number of worker threads for parallel processing
"""

import argparse
import gc
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

import safetensors
import safetensors.torch
import torch
import torch.nn.functional as F
from tqdm import tqdm

FP8_INFO = torch.finfo(torch.float8_e4m3fn)
FP8_MAX, FP8_MIN = FP8_INFO.max, FP8_INFO.min


def ceildiv(a, b):
    return -(-a // b)


def block_fp8(weight, block_size, scale_dtype=torch.float32):

    # per block quant
    block_n, block_k = block_size[0], block_size[1]

    shape_0, shape_1 = weight.shape

    n_tiles = ceildiv(shape_0, block_n)
    k_tiles = ceildiv(shape_1, block_k)

    q_weight = F.pad(
        weight,
        (0, k_tiles * block_k - shape_1, 0, n_tiles * block_n - shape_0),
        mode="constant",
        value=0.0,
    )

    qweight = q_weight.reshape(n_tiles, block_n, k_tiles, block_k)
    block_max = torch.max(torch.abs(qweight), dim=1, keepdim=True)[0]
    block_max = torch.max(block_max, dim=3, keepdim=True)[0]

    # Some official checkpoints (including Qwen3.5 FP8) round the scale to
    # BF16 *before* quantizing. Keep this selectable because other model
    # families publish FP32 scales.
    scale = (block_max.clamp(min=1e-12).to(torch.float32) / FP8_MAX).to(scale_dtype)
    qweight = (
        (qweight / scale)
        .clamp(min=FP8_MIN, max=FP8_MAX)
        .reshape((n_tiles * block_n, k_tiles * block_k))
        .to(torch.float8_e4m3fn)
    )
    qweight = qweight[:shape_0, :shape_1].clone().detach()
    scale = scale.reshape(n_tiles, k_tiles)

    return qweight, scale


def channel_fp8(weight):
    channel_max = torch.max(weight.abs(), dim=-1, keepdim=True)[0]
    scale = channel_max.clamp(min=1e-12).to(torch.float32) / FP8_MAX
    qweight = (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX)
    qweight = qweight.to(torch.float8_e4m3fn)
    return qweight, scale


def tensor_fp8(weight):
    scale = weight.abs().max().clamp(min=1e-12).to(torch.float32) / FP8_MAX
    qweight = (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX)
    qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.view(1)
    return qweight, scale


def quant_fp8(weight, strategy, block_size=None, scale_dtype=torch.float32):
    if strategy == "tensor":
        return tensor_fp8(weight)
    elif strategy == "channel":
        return channel_fp8(weight)
    else:
        return block_fp8(weight, block_size, scale_dtype)


class ConversionResult:
    def __init__(self):
        self.lock = threading.Lock()
        self.weight_map = {}
        self.param_count = 0
        self.modules_to_not_convert = []

    def add_result(self, filename, q_weights, module_names):
        with self.lock:
            for k, v in q_weights.items():
                self.weight_map[k] = filename
                # HF's index metadata is the serialized tensor byte count,
                # not the number of first-dimension entries.
                self.param_count += v.numel() * v.element_size()
            self.modules_to_not_convert.extend(module_names)


def quantize_qwen35_fused_experts(key, weight, q_weights, strategy, block_size, scale_dtype):
    """Expand Qwen3.5's fused MoE tensors into SGLang's HF layout.

    The BF16 Qwen3.5 checkpoints store all experts in two tensors named
    ``experts.gate_up_proj`` and ``experts.down_proj`` (without a ``weight``
    suffix).  The official FP8 checkpoint stores one block-quantized tensor
    per expert/projection.  The generic converter used to skip the two fused
    keys entirely, leaving almost all 122B parameters in BF16.
    """
    gate_up_suffix = ".experts.gate_up_proj"
    down_suffix = ".experts.down_proj"
    if key.endswith(gate_up_suffix):
        if weight.ndim != 3 or weight.shape[1] % 2:
            raise ValueError(f"Unexpected fused gate/up shape for {key}: {tuple(weight.shape)}")
        prefix = key[: -len(gate_up_suffix)]
        for expert_index, expert_weight in enumerate(weight):
            gate_weight, up_weight = expert_weight.chunk(2, dim=0)
            for projection, projection_weight in (("gate_proj", gate_weight), ("up_proj", up_weight)):
                output_key = f"{prefix}.experts.{expert_index}.{projection}.weight"
                q_weights[output_key], q_weights[f"{output_key}_scale_inv"] = quant_fp8(
                    projection_weight, strategy, block_size, scale_dtype
                )
        return True
    if key.endswith(down_suffix):
        if weight.ndim != 3:
            raise ValueError(f"Unexpected fused down shape for {key}: {tuple(weight.shape)}")
        prefix = key[: -len(down_suffix)]
        for expert_index, expert_weight in enumerate(weight):
            output_key = f"{prefix}.experts.{expert_index}.down_proj.weight"
            q_weights[output_key], q_weights[f"{output_key}_scale_inv"] = quant_fp8(
                expert_weight, strategy, block_size, scale_dtype
            )
        return True
    return False


def process_file(input_path, output_path, filename, strategy, block_size, scale_dtype, result_collector):
    if not filename.endswith(".safetensors"):
        return

    print(f"Processing {filename}, memory usage: {torch.cuda.memory_allocated()}")
    weights = {}
    q_weights = {}

    with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device="cuda") as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)

    modules_to_not_convert = []
    for key in weights.keys():
        if quantize_qwen35_fused_experts(key, weights[key], q_weights, strategy, block_size, scale_dtype):
            continue
        if (
            "weight" in key
            and "layernorm" not in key
            and "embed" not in key
            and "router" not in key
            and "mlp.gate." not in key
            and "shared_expert_gate" not in key
            and "norm" not in key
            and "lm_head" not in key
            and "eh_proj" not in key
            and "weights_proj" not in key
            and "conv1d" not in key
            and "A_log" not in key
            and "dt_bias" not in key
            and "in_proj_a" not in key
            and "in_proj_b" not in key
        ):
            qw, s = quant_fp8(weights[key], strategy, block_size, scale_dtype)
            q_weights[key] = qw
            if block_size:
                scale_name = key.replace(".weight", ".weight_scale_inv")
            else:
                scale_name = key.replace(".weight", ".weight_scale")
            q_weights[scale_name] = s
        else:
            modules_to_not_convert.append(key.replace(".weight", ""))
            q_weights[key] = weights[key]

    safetensors.torch.save_file(q_weights, os.path.join(output_path, filename), metadata={"format": "pt"})

    result_collector.add_result(filename, q_weights, modules_to_not_convert)


def convert_fp8(
    input_path,
    output_path,
    strategy,
    block_size=None,
    max_workers=4,
    scale_fmt=None,
    scale_dtype="float32",
):
    input_path = os.path.abspath(input_path)
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(os.path.join(input_path, filename)):
            shutil.copyfile(os.path.join(input_path, filename), os.path.join(output_path, filename))

    safetensors_files = [f for f in os.listdir(input_path) if f.endswith(".safetensors")]

    result_collector = ConversionResult()
    scale_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[scale_dtype]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for filename in safetensors_files:
            future = executor.submit(
                process_file,
                input_path,
                output_path,
                filename,
                strategy,
                block_size,
                scale_dtype,
                result_collector,
            )
            futures.append(future)

        for future in tqdm(futures, desc="Processing files"):
            future.result()

    if strategy == "block" or strategy == "tensor":
        quantization_config = {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
        }
        if block_size:
            quantization_config["weight_block_size"] = block_size
            if scale_fmt is not None:
                quantization_config["scale_fmt"] = scale_fmt
        if len(result_collector.modules_to_not_convert) > 0:
            quantization_config["modules_to_not_convert"] = list(set(result_collector.modules_to_not_convert))
    else:
        quant_group = {
            "group_0": {
                "input_activations": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": None,
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "float",
                },
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": strategy,
                    "symmetric": True,
                    "type": "float",
                },
            },
        }
        quantization_config = {
            "config_groups": quant_group,
            "format": "float-quantized",
            "ignore": list(set(result_collector.modules_to_not_convert)),
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
        }

    config_path = os.path.join(input_path, "config.json")
    if os.path.exists(config_path):
        cfg = json.load(open(config_path))
        cfg["quantization_config"] = quantization_config
        json.dump(cfg, open(os.path.join(output_path, "config.json"), "w"), indent=2)

    index_dict = {"weight_map": result_collector.weight_map, "metadata": {"total_size": result_collector.param_count}}
    json.dump(index_dict, open(os.path.join(output_path, "model.safetensors.index.json"), "w"), indent=2)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, help="Path to the directory of the HF safetensors model.")
    parser.add_argument("--save-dir", type=str, help="Path to the directory to save the converted model.")
    parser.add_argument("--strategy", type=str, default="block", choices=["block", "channel", "tensor"])
    parser.add_argument("--block-size", type=int, nargs="*", default=None, help="eg. --block-size 128 128")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of worker threads for parallel processing")
    parser.add_argument("--scale-fmt", type=str, default=None, choices=["ue8m0"])
    parser.add_argument(
        "--scale-dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
        help="Stored block-scale dtype. Qwen3.5 official FP8 checkpoints use bfloat16.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        print(f"Creating directory {args.save_dir}")
        os.makedirs(args.save_dir)
    elif not os.path.isdir(args.save_dir):
        raise ValueError("The save_dir should be a directory.")

    convert_fp8(
        args.model_dir,
        args.save_dir,
        args.strategy,
        args.block_size,
        args.max_workers,
        args.scale_fmt,
        args.scale_dtype,
    )
