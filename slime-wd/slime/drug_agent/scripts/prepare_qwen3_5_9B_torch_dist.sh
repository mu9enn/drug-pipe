#!/usr/bin/env bash
set -euo pipefail

if [[ -f /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh ]]; then
  source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
else
  source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi

export MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-scripts/models/qwen3.5-9B.sh}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-$DATA/Qwen3.5-9B}"
export SAVE_DIR="${SAVE_DIR:-$DATA/Qwen3.5-9B_torch_dist}"

# A 9B BF16 model fits comfortably on one H200.  A one-rank release
# checkpoint is the least surprising conversion source; torch_dist reshards
# it deterministically when the formal TP4/PP2 jobs load it.
export NUM_GPUS="${NUM_GPUS:-1}"

exec bash "$SLIME/drug_agent/scripts/prepare_qwen3_5_4B_torch_dist.sh"
