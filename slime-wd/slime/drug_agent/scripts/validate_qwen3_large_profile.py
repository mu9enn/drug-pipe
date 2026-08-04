#!/usr/bin/env python3
"""Static validation for a sourced qwen3_large_profile.sh profile.

This deliberately avoids importing Megatron/Transformers so it can run on the
login host.  Runtime validation still belongs to the worker preflight and the
one-step probes.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
from typing import Any


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def env_int(name: str) -> int:
    try:
        value = int(os.environ[name])
    except (KeyError, ValueError):
        fail(f"{name} must be an exported integer")
    if value <= 0:
        fail(f"{name} must be positive; got {value}")
    return value


def parse_model_args(items: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    index = 0
    while index < len(items):
        item = items[index]
        if not item.startswith("--"):
            fail(f"unexpected model argument token: {item}")
        if item == "--spec":
            if index + 2 >= len(items):
                fail("--spec requires a module and function name")
            parsed[item] = items[index + 1 : index + 3]
            index += 3
            continue
        if index + 1 < len(items) and not items[index + 1].startswith("--"):
            parsed[item] = items[index + 1]
            index += 2
        else:
            parsed[item] = True
            index += 1
    return parsed


def require_arg(args: dict[str, Any], name: str, expected: Any) -> None:
    if name not in args:
        fail(f"model args are missing {name}")
    actual = args[name]
    if isinstance(expected, int):
        try:
            actual = int(actual)
        except (TypeError, ValueError):
            fail(f"{name} must be an integer; got {actual!r}")
    if actual != expected:
        fail(f"{name}={actual!r}, but HF config requires {expected!r}")


def validate_hf_checkpoint(root: pathlib.Path, model_args: dict[str, Any]) -> dict[str, Any]:
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        fail(f"HF checkpoint lacks config/index: {root}")
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)
    mapping = {
        "--num-layers": "num_hidden_layers",
        "--hidden-size": "hidden_size",
        "--num-attention-heads": "num_attention_heads",
        "--num-query-groups": "num_key_value_heads",
        "--kv-channels": "head_dim",
        "--vocab-size": "vocab_size",
    }
    for cli_name, config_name in mapping.items():
        require_arg(model_args, cli_name, int(text_config[config_name]))

    is_moe = text_config.get("num_experts") is not None
    if is_moe:
        require_arg(model_args, "--num-experts", int(text_config["num_experts"]))
        require_arg(model_args, "--moe-router-topk", int(text_config["num_experts_per_tok"]))
        require_arg(model_args, "--moe-ffn-hidden-size", int(text_config["moe_intermediate_size"]))
        require_arg(
            model_args,
            "--moe-shared-expert-intermediate-size",
            int(text_config["shared_expert_intermediate_size"]),
        )
    else:
        require_arg(model_args, "--ffn-hidden-size", int(text_config["intermediate_size"]))
        if "--num-experts" in model_args:
            fail("dense HF config must not receive --num-experts")

    index = json.loads(index_path.read_text())
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards:
        fail(f"empty weight_map: {index_path}")
    missing = [name for name in shards if not (root / name).is_file()]
    if missing:
        fail(f"{len(missing)} indexed HF shards are absent: {missing[:8]}")
    indexed_bytes = sum((root / name).stat().st_size for name in shards)
    return {
        "model_type": text_config.get("model_type"),
        "is_moe": is_moe,
        "num_experts": text_config.get("num_experts"),
        "num_query_groups": int(text_config["num_key_value_heads"]),
        "num_attention_heads": int(text_config["num_attention_heads"]),
        "mtp_layers_in_hf": int(text_config.get("mtp_num_hidden_layers", 0)),
        "weight_shards": len(shards),
        "indexed_bytes": indexed_bytes,
    }


def count_and_validate_data(path: pathlib.Path, kind: str) -> int:
    if not path.is_file():
        fail(f"{kind} data is missing: {path}")
    count = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"{kind} line {line_number} is invalid JSON: {exc}")
            if kind == "sft":
                messages = record.get("messages")
                if not isinstance(messages, list) or not messages:
                    fail(f"SFT line {line_number} has no messages")
                if not any(m.get("role") == "assistant" and m.get("step_loss_mask", 1) == 1 for m in messages):
                    fail(f"SFT line {line_number} has no supervised assistant turn")
            else:
                if not isinstance(record.get("prompt"), list):
                    fail(f"{kind} line {line_number} prompt must be a message list")
                if not isinstance(record.get("label"), dict) or not isinstance(record.get("metadata"), dict):
                    fail(f"{kind} line {line_number} must contain dict label/metadata")
            count += 1
    return count


def main() -> None:
    separator = sys.argv.index("--") if "--" in sys.argv else 1
    model_args = parse_model_args(sys.argv[separator + 1 :])
    profile = os.environ.get("MODEL_PROFILE") or fail("MODEL_PROFILE is not exported")
    num_gpus = env_int("NUM_GPUS")
    tp = env_int("TENSOR_MODEL_PARALLEL_SIZE")
    pp = env_int("PIPELINE_MODEL_PARALLEL_SIZE")
    cp = env_int("CONTEXT_PARALLEL_SIZE")
    ep = env_int("EXPERT_MODEL_PARALLEL_SIZE")
    etp = env_int("EXPERT_TENSOR_PARALLEL_SIZE")
    rollout_tp = env_int("ROLLOUT_NUM_GPUS_PER_ENGINE")

    if num_gpus % (tp * pp * cp):
        fail(f"world={num_gpus} is not divisible by TP*PP*CP={tp * pp * cp}")
    if num_gpus % (etp * ep * pp):
        fail(f"world={num_gpus} is not divisible by ETP*EP*PP={etp * ep * pp}")
    if num_gpus % rollout_tp:
        fail(f"world={num_gpus} is not divisible by rollout TP={rollout_tp}")

    hf = validate_hf_checkpoint(pathlib.Path(os.environ["HF_CHECKPOINT"]), model_args)
    if hf["num_attention_heads"] % tp:
        fail(f"attention heads={hf['num_attention_heads']} are not divisible by TP={tp}")
    if hf["num_query_groups"] % tp:
        fail(f"KV query groups={hf['num_query_groups']} are not divisible by TP={tp}")
    if hf["is_moe"]:
        if int(hf["num_experts"]) % ep:
            fail(f"experts={hf['num_experts']} are not divisible by EP={ep}")
    elif ep != 1 or etp != 1:
        fail("dense model must use EP=ETP=1")

    first_layers_raw = os.environ.get("NUM_LAYERS_IN_FIRST_PIPELINE_STAGE")
    last_layers_raw = os.environ.get("NUM_LAYERS_IN_LAST_PIPELINE_STAGE")
    layout_raw = os.environ.get("PIPELINE_MODEL_PARALLEL_LAYOUT")
    layout_stage_layers: list[int] | None = None
    if layout_raw:
        layout = layout_raw.replace(r"\|", "|")
        stages = layout.split("|")
        if len(stages) != pp:
            fail(f"pipeline layout must contain PP={pp} stages; got {len(stages)}: {layout}")
        if layout.count("E") != 1 or not layout.startswith("E"):
            fail("pipeline layout must start with exactly one embedding layer E")
        if layout.count("L") != 1 or not layout.endswith("L"):
            fail("pipeline layout must end with exactly one loss layer L")
        layout_stage_layers = []
        for stage in stages:
            decoder_layers = sum(int(count or "1") for count in re.findall(r"t(?:\*(\d+))?", stage))
            layout_stage_layers.append(decoder_layers)
        total_layers = int(model_args["--num-layers"])
        if sum(layout_stage_layers) != total_layers:
            fail(
                f"pipeline layout decoder count must equal {total_layers}; "
                f"got stages={layout_stage_layers}"
            )
        if first_layers_raw and int(first_layers_raw) != layout_stage_layers[0]:
            fail("profile first-stage metadata disagrees with pipeline layout")
        if last_layers_raw and int(last_layers_raw) != layout_stage_layers[-1]:
            fail("profile last-stage metadata disagrees with pipeline layout")
    elif bool(first_layers_raw) != bool(last_layers_raw):
        fail("uneven pipeline requires both first- and last-stage layer counts")
    elif first_layers_raw:
        if pp < 2:
            fail("uneven pipeline requires PP>=2")
        first_layers = int(first_layers_raw)
        last_layers = int(last_layers_raw)
        total_layers = int(model_args["--num-layers"])
        middle_stages = pp - 2
        middle_layers = total_layers - first_layers - last_layers
        invalid_middle = (
            middle_layers != 0
            if middle_stages == 0
            else middle_layers <= 0 or middle_layers % middle_stages != 0
        )
        if first_layers <= 0 or last_layers <= 0 or invalid_middle:
            fail(
                f"uneven PP layer counts must leave an equal positive share for {middle_stages} "
                f"middle stages out of {total_layers}; got first={first_layers}, last={last_layers}"
            )

    prompt_len = env_int("ROLLOUT_MAX_PROMPT_LEN")
    response_len = env_int("ROLLOUT_MAX_RESPONSE_LEN")
    context_len = env_int("ROLLOUT_MAX_CONTEXT_LEN")
    if context_len != prompt_len + response_len:
        fail(f"context length {context_len} must equal prompt+response={prompt_len + response_len}")

    fractions = {
        "sft": float(os.environ["OPTIMIZER_OFFLOAD_FRACTION"]),
        "toolrl": float(os.environ.get("TOOLRL_OPTIMIZER_OFFLOAD_FRACTION", os.environ["OPTIMIZER_OFFLOAD_FRACTION"])),
        "gad": float(os.environ.get("GAD_OPTIMIZER_OFFLOAD_FRACTION", os.environ["OPTIMIZER_OFFLOAD_FRACTION"])),
    }
    for method, fraction in fractions.items():
        if not 0.0 <= fraction <= 1.0:
            fail(f"{method} optimizer offload fraction must be in [0,1]; got {fraction}")
    dtype_bytes = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}
    dtypes = {
        "main_grad": os.environ["MAIN_GRADS_DTYPE"],
        "main_param": os.environ["MAIN_PARAMS_DTYPE"],
        "exp_avg": os.environ["EXP_AVG_DTYPE"],
        "exp_avg_sq": os.environ["EXP_AVG_SQ_DTYPE"],
    }
    unknown = {name: value for name, value in dtypes.items() if value not in dtype_bytes}
    if unknown:
        fail(f"unsupported optimizer dtype(s): {unknown}")
    fp32_accum = os.environ.get("ACCUMULATE_ALLREDUCE_GRADS_IN_FP32") == "1"
    if fp32_accum != (dtypes["main_grad"] == "fp32"):
        fail("FP32 accumulation flag and MAIN_GRADS_DTYPE disagree")
    cpu_offloads = {
        "sft": os.environ.get("OPTIMIZER_CPU_OFFLOAD") == "1",
        "toolrl": os.environ.get(
            "TOOLRL_OPTIMIZER_CPU_OFFLOAD", os.environ.get("OPTIMIZER_CPU_OFFLOAD", "0")
        )
        == "1",
        "gad": os.environ.get(
            "GAD_OPTIMIZER_CPU_OFFLOAD", os.environ.get("OPTIMIZER_CPU_OFFLOAD", "0")
        )
        == "1",
    }
    optimizer_cpu_offload = cpu_offloads["sft"]
    if any(cpu_offloads.values()) and any(value != "fp32" for value in dtypes.values()):
        fail(
            "this pinned Megatron HybridDeviceOptimizer CPUAdam path keeps the "
            "offloaded main parameter, gradient and Adam moments in FP32; "
            "non-FP32 dtype declarations would understate host memory"
        )

    counts: dict[str, int] = {}
    if os.environ.get("VALIDATE_LARGE_PROFILE_DATA", "1") == "1":
        counts = {
            "sft": count_and_validate_data(pathlib.Path(os.environ["CANONICAL_DATA"]), "sft"),
            "toolrl": count_and_validate_data(pathlib.Path(os.environ["TOOLRL_DATA"]), "toolrl"),
            "gad": count_and_validate_data(pathlib.Path(os.environ["GAD_DATA"]), "gad"),
        }
        expected = {"sft": 364, "toolrl": 3182, "gad": 3147}
        if counts != expected:
            fail(f"dataset counts changed: actual={counts}, expected={expected}; re-audit schedules before training")

    param_billions = float(os.environ["TRAINABLE_PARAM_BILLIONS"])
    # CPUAdam in the exact image does not consume the precision-aware moment
    # dtype arguments. TE's state-offloader, in contrast, stores the main
    # parameter and two moments but not the gradient on host.
    state_bytes = 16 if any(cpu_offloads.values()) else sum(dtype_bytes[value] for value in dtypes.values())
    state_offload = os.environ.get("OFFLOAD_OPTIMIZER_STATES") == "1"
    fp8_param_gather = os.environ.get("FP8_PARAM_GATHER") == "1"
    fp8_recipe = os.environ.get("FP8_RECIPE")
    if state_offload and any(cpu_offloads.values()):
        fail("OFFLOAD_OPTIMIZER_STATES and OPTIMIZER_CPU_OFFLOAD are mutually exclusive")
    if fp8_param_gather and not state_offload:
        fail("this low-memory FP8 profile requires OFFLOAD_OPTIMIZER_STATES=1")
    if fp8_param_gather and fp8_recipe == "blockwise" and dtypes["main_param"] != "fp32":
        fail(
            "this pinned MCore cannot create FP16 optimizer shards from "
            "BlockwiseQTensor; use FP8_RECIPE=delayed or MAIN_PARAMS_DTYPE=fp32"
        )
    peak_fraction = max(fractions.values())
    if state_offload:
        host_state_bytes = sum(dtype_bytes[dtypes[key]] for key in ("main_param", "exp_avg", "exp_avg_sq"))
        cpu_offload_gib = param_billions * 1e9 * host_state_bytes / 1024**3
    else:
        cpu_offload_gib = param_billions * 1e9 * state_bytes * peak_fraction / 1024**3
    minimum_host_gib = env_int("MIN_HOST_MEMORY_GIB")
    gad_minimum_host_gib = env_int("GAD_MIN_HOST_MEMORY_GIB")
    host_reserve_gib = env_int("HOST_MEMORY_RESERVE_GIB")
    if gad_minimum_host_gib < minimum_host_gib:
        fail(
            "GAD host-memory floor cannot be below the general profile floor: "
            f"gad={gad_minimum_host_gib} GiB general={minimum_host_gib} GiB"
        )
    if cpu_offload_gib + host_reserve_gib > minimum_host_gib:
        fail(
            "profile host-memory floor is internally inconsistent: "
            f"offload upper bound {cpu_offload_gib:.1f} GiB + reserve {host_reserve_gib} GiB "
            f"> minimum {minimum_host_gib} GiB"
        )
    result = {
        "profile": profile,
        "topology": {
            "world": num_gpus,
            "tp": tp,
            "pp": pp,
            "cp": cp,
            "ep": ep,
            "etp": etp,
            "first_stage_layers": int(first_layers_raw) if first_layers_raw else None,
            "last_stage_layers": int(last_layers_raw) if last_layers_raw else None,
            "pipeline_layout": layout_raw.replace(r"\|", "|") if layout_raw else None,
            "pipeline_stage_layers": layout_stage_layers,
        },
        "hf": hf,
        "datasets": counts,
        "optimizer": {
            "offload_fractions": fractions,
            "host_memory_bound_fraction": peak_fraction,
            "cpu_offload": cpu_offloads,
            "state_offload": state_offload,
            "fp8_param_gather": fp8_param_gather,
            "fp8_recipe": fp8_recipe,
            "cpu_state_bytes_per_parameter": state_bytes,
            "dtypes": dtypes,
            "upper_bound_cpu_offload_gib": round(cpu_offload_gib, 1),
            "host_memory_reserve_gib": host_reserve_gib,
            "minimum_host_memory_gib": minimum_host_gib,
            "gad_minimum_host_memory_gib": gad_minimum_host_gib,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    print("PASS: large model profile is statically consistent")


if __name__ == "__main__":
    main()
