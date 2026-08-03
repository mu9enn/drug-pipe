#!/usr/bin/env bash
# Convert a supported Qwen3.5/Qwen3.6 HF checkpoint to Megatron torch_dist.
# This script never deletes or overwrites a partial output directory.
set -euo pipefail

SLIME_ENV=${SLIME_ENV:-/root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh}
if [[ ! -f "$SLIME_ENV" ]]; then
  SLIME_ENV=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi
if [[ ! -f "$SLIME_ENV" ]]; then
  echo "SLIME environment file not found: $SLIME_ENV" >&2
  exit 2
fi
source "$SLIME_ENV"
cd "$SLIME"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-/root/Megatron-LM}

: "${MODEL_ARGS_FILE:?Set MODEL_ARGS_FILE to a scripts/models/*.sh file}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the source Hugging Face checkpoint}"
: "${SAVE_DIR:?Set SAVE_DIR to a new torch_dist output directory}"
NUM_GPUS=${NUM_GPUS:-1}

for path in "$MODEL_ARGS_FILE" "$HF_CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "Required conversion input does not exist: $path" >&2
    exit 2
  fi
done
if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer; got: $NUM_GPUS" >&2
  exit 2
fi

SOURCE_REAL=$(readlink -f "$HF_CHECKPOINT")
SAVE_REAL=$(readlink -m "$SAVE_DIR")
if [[ "$SOURCE_REAL" == "$SAVE_REAL" ]]; then
  echo "Refusing to overwrite the Hugging Face source checkpoint: $SOURCE_REAL" >&2
  exit 2
fi
case "$SAVE_REAL/" in
  "$SOURCE_REAL/"*)
    echo "Refusing to place converted files inside the Hugging Face source checkpoint: $SAVE_REAL" >&2
    exit 2
    ;;
esac
if [[ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]]; then
  if [[ "$(<"$SAVE_DIR/latest_checkpointed_iteration.txt")" == "release" ]] \
    && find "$SAVE_DIR" -type f -name '*.distcp' -print -quit | grep -q .; then
    echo "torch_dist checkpoint already complete: $SAVE_DIR"
    exit 0
  fi
  echo "Checkpoint tracker exists but the release/distcp contract is incomplete: $SAVE_DIR" >&2
  exit 2
fi
if [[ -e "$SAVE_DIR" ]] && find "$SAVE_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite non-empty partial output: $SAVE_DIR" >&2
  echo "Move it aside or choose a fresh SAVE_DIR after inspecting the partial conversion." >&2
  exit 2
fi

# Fail before allocating GPUs if the copied/downloaded HF checkpoint is
# incomplete.  The index is authoritative for which safetensor shards must be
# present; unindexed auxiliary files do not affect conversion.
SOURCE_BYTES=$(python - "$HF_CHECKPOINT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
index = root / "model.safetensors.index.json"
if not index.is_file():
    raise SystemExit(f"Missing Hugging Face weight index: {index}")
payload = json.loads(index.read_text())
shards = sorted(set(payload.get("weight_map", {}).values()))
if not shards:
    raise SystemExit(f"Hugging Face weight index has no weight_map entries: {index}")
missing = [name for name in shards if not (root / name).is_file()]
if missing:
    preview = ", ".join(missing[:8])
    raise SystemExit(f"Missing {len(missing)} indexed safetensor shard(s): {preview}")
print(sum((root / name).stat().st_size for name in shards))
PY
)

SAVE_PARENT=$(dirname "$SAVE_REAL")
while [[ ! -e "$SAVE_PARENT" ]]; do
  SAVE_PARENT=$(dirname "$SAVE_PARENT")
done
FREE_BYTES=$(df --output=avail -B1 "$SAVE_PARENT" | tail -1 | tr -d ' ')
REQUIRED_FREE_BYTES=$(python - "$SOURCE_BYTES" "${CONVERSION_FREE_SPACE_FACTOR:-1.15}" <<'PY'
import math
import sys
print(math.ceil(int(sys.argv[1]) * float(sys.argv[2])))
PY
)
if (( FREE_BYTES < REQUIRED_FREE_BYTES )); then
  echo "Insufficient free space for safe conversion: free=$FREE_BYTES required=$REQUIRED_FREE_BYTES parent=$SAVE_PARENT" >&2
  exit 2
fi

VISIBLE_GPU_COUNT=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
if (( VISIBLE_GPU_COUNT < NUM_GPUS )); then
  echo "Requested NUM_GPUS=$NUM_GPUS but only $VISIBLE_GPU_COUNT CUDA devices are visible" >&2
  exit 2
fi

LOCK_DIR="${SAVE_REAL}.convert.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Conversion lock already exists: $LOCK_DIR" >&2
  echo "Another conversion may be running; inspect it before removing a stale lock." >&2
  exit 2
fi
cleanup_lock() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT INT TERM

mkdir -p "$SAVE_DIR"
source "$MODEL_ARGS_FILE"

echo "[torch_dist] source=$SOURCE_REAL"
echo "[torch_dist] output=$SAVE_REAL"
echo "[torch_dist] model_args=$MODEL_ARGS_FILE gpus=$NUM_GPUS"

PYTHONPATH="${MEGATRON_LM_PATH}:${SLIME}:${PYTHONPATH:-}" \
  torchrun --standalone --nproc_per_node "$NUM_GPUS" \
  tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --save "$SAVE_DIR"

if [[ "$(<"$SAVE_DIR/latest_checkpointed_iteration.txt")" != "release" ]]; then
  echo "Conversion did not produce a release checkpoint tracker in $SAVE_DIR" >&2
  exit 2
fi
DISTCP_COUNT=$(find "$SAVE_DIR" -type f -name '*.distcp' | wc -l)
if (( DISTCP_COUNT == 0 )); then
  echo "Conversion tracker exists but no .distcp files were found in $SAVE_DIR" >&2
  exit 2
fi
echo "[torch_dist] complete: $DISTCP_COUNT distcp files in $SAVE_DIR"
