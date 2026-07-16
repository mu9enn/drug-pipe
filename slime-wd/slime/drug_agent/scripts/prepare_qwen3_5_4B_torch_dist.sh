#!/bin/bash

set -euo pipefail
set -x

# Re-use the same environment bootstrap pattern as the other drug_agent scripts.
if [ -f /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh ]; then
  # GPU worker / RJob environment.
  source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
elif [ -f /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh ]; then
  # Local workstation fallback.
  source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi

if [ -z "${SLIME:-}" ]; then
  echo "SLIME is not set. Please source slime_env.sh first."
  exit 1
fi

cd "$SLIME"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-4B.sh}
if [[ "$MODEL_ARGS_FILE" != /* ]]; then
  MODEL_ARGS_FILE="$SLIME/$MODEL_ARGS_FILE"
fi

if [ ! -f "$MODEL_ARGS_FILE" ]; then
  echo "MODEL_ARGS_FILE not found: $MODEL_ARGS_FILE"
  exit 1
fi

source "$MODEL_ARGS_FILE"

if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
  DETECTED_GPUS=0
fi
NUM_GPUS=${NUM_GPUS:-${DETECTED_GPUS}}
if [ -z "${NUM_GPUS}" ] || [ "${NUM_GPUS}" -le 0 ]; then
  NUM_GPUS=1
fi

MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-/root/Megatron-LM}
HF_CHECKPOINT=${HF_CHECKPOINT:-$DATA/Qwen3.5-4B}
SAVE_DIR=${SAVE_DIR:-$DATA/Qwen3.5-4B_torch_dist}

if [ ! -d "$HF_CHECKPOINT" ]; then
  echo "HF checkpoint directory not found: $HF_CHECKPOINT"
  exit 1
fi

mkdir -p "$SAVE_DIR"

echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "MODEL_ARGS_FILE=$MODEL_ARGS_FILE"
echo "NUM_GPUS=$NUM_GPUS"
echo "HF_CHECKPOINT=$HF_CHECKPOINT"
echo "SAVE_DIR=$SAVE_DIR"
echo "MEGATRON_LM_PATH=$MEGATRON_LM_PATH"

export PYTHONPATH="$MEGATRON_LM_PATH${PYTHONPATH:+:$PYTHONPATH}"

# `convert_hf_to_torch_dist.py` initializes torch.distributed internally.
# `torchrun --standalone` gives it the expected rank/world-size env vars and
# works for both 1-GPU and multi-GPU conversion.
torchrun --standalone --nproc_per_node="$NUM_GPUS" \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --save "$SAVE_DIR"
