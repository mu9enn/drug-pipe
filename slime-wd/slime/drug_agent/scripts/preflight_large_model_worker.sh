#!/usr/bin/env bash
# Read-only preflight for the large-model drug-agent profiles.
set -euo pipefail

SLIME_ENV=${SLIME_ENV:-/root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh}
if [[ ! -f "$SLIME_ENV" ]]; then
  SLIME_ENV=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi
if [[ ! -f "$SLIME_ENV" ]]; then
  echo "FAIL: slime_env.sh is not visible; check the GPFS mount target and SSH user permissions" >&2
  exit 2
fi
source "$SLIME_ENV"
cd "$SLIME"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-/root/Megatron-LM}

# SSH sessions in rjob pods do not always inherit the container entrypoint's
# NVIDIA paths even though the devices and driver mount are present.
export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

EXPECTED_GPUS=${EXPECTED_GPUS:-4}
MIN_GPU_MEMORY_GIB=${MIN_GPU_MEMORY_GIB:-130}
MIN_HOST_MEMORY_GIB=${MIN_HOST_MEMORY_GIB:-0}
if [[ ! "$EXPECTED_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_GPUS must be a positive integer; got: $EXPECTED_GPUS" >&2
  exit 2
fi

command -v python >/dev/null
command -v ray >/dev/null
command -v torchrun >/dev/null
command -v nvidia-smi >/dev/null

python - "$EXPECTED_GPUS" "$MIN_GPU_MEMORY_GIB" <<'PY'
import sys
import torch

expected = int(sys.argv[1])
minimum_bytes = float(sys.argv[2]) * 1024**3
if not torch.cuda.is_available():
    raise SystemExit("FAIL: torch.cuda.is_available() is false")
actual = torch.cuda.device_count()
if actual != expected:
    raise SystemExit(f"FAIL: expected {expected} visible GPUs, found {actual}")
for idx in range(actual):
    props = torch.cuda.get_device_properties(idx)
    print(f"gpu={idx} name={props.name} bytes={props.total_memory}")
    if props.total_memory < minimum_bytes:
        raise SystemExit(
            f"FAIL: gpu {idx} has {props.total_memory / 1024**3:.1f} GiB; "
            f"profile requires at least {minimum_bytes / 1024**3:.1f} GiB"
        )
PY

HOST_MEMORY_GIB=$(awk '/^MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)
echo "host_memory_gib=$HOST_MEMORY_GIB required_gib=$MIN_HOST_MEMORY_GIB"
if (( HOST_MEMORY_GIB < MIN_HOST_MEMORY_GIB )); then
  echo "FAIL: host memory is below the profile requirement: actual=${HOST_MEMORY_GIB}GiB required=${MIN_HOST_MEMORY_GIB}GiB" >&2
  exit 2
fi

REQUIRED_PATHS=(
  "$MEGATRON_LM_PATH/megatron"
  "$SLIME/train.py"
  "$SLIME/tools/convert_hf_to_torch_dist.py"
  "$SLIME/scripts/models/qwen3.5-27B.sh"
  "$SLIME/scripts/models/qwen3.5-35B-A3B.sh"
  "$SLIME/scripts/models/qwen3.5-122B-A10B.sh"
)
if [[ -n "${HF_CHECKPOINT:-}" ]]; then
  REQUIRED_PATHS+=("$HF_CHECKPOINT/config.json" "$HF_CHECKPOINT/model.safetensors.index.json")
fi
if [[ -n "${PROMPT_DATA:-}" ]]; then
  REQUIRED_PATHS+=("$PROMPT_DATA")
fi
for path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "FAIL: required path is missing: $path" >&2
    exit 2
  fi
done

if [[ -n "${HF_CHECKPOINT:-}" ]]; then
  python - "$HF_CHECKPOINT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted(set(index.get("weight_map", {}).values()))
if not shards:
    raise SystemExit("FAIL: model.safetensors.index.json has an empty weight_map")
missing = [name for name in shards if not (root / name).is_file()]
if missing:
    raise SystemExit(f"FAIL: {len(missing)} indexed weight shards are missing: {missing[:8]}")
size = sum((root / name).stat().st_size for name in shards)
print(f"hf_weight_shards={len(shards)} indexed_bytes={size}")
PY
fi

python - <<'PY'
import fla
import mbridge
import sglang
import torch
import transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("sglang", sglang.__version__)
print("mbridge", getattr(mbridge, "__version__", "unknown"))
print("fla", getattr(fla, "__version__", "unknown"))
PY

echo "disk:"
df -h "$GROUP_SPACE" | tail -1
echo "PASS: large-model worker preflight"
