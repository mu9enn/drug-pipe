#!/bin/bash
set -euo pipefail
SLIME_ENV=${SLIME_ENV:-/root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh}
if [ ! -f "$SLIME_ENV" ]; then
  SLIME_ENV=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi
source "$SLIME_ENV"
cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh

MODEL_PATH=${DISCRIMINATOR_MODEL_PATH:-$DATA/Qwen3.5-0.8B}
OUTPUT_DIR=${DISCRIMINATOR_OUTPUT_DIR:-$DRUG_AGENT_RUNS_ROOT/gad_discriminator_online}
EXTRA_ARGS=()
if [ -n "${DISCRIMINATOR_RESUME:-}" ]; then
  EXTRA_ARGS+=(--resume "$DISCRIMINATOR_RESUME")
fi
if [ ! -e "$MODEL_PATH" ]; then
  echo "Discriminator model does not exist: $MODEL_PATH" >&2
  exit 2
fi

python -m drug_agent.gad.service \
  --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR" \
  --host "${GAD_DISCRIMINATOR_HOST:-0.0.0.0}" --port "${GAD_DISCRIMINATOR_PORT:-8100}" \
  --lr "${DISCRIMINATOR_LR:-1e-5}" --max-length "${DISCRIMINATOR_MAX_LENGTH:-4096}" \
  --update-steps "${DISCRIMINATOR_UPDATE_STEPS:-1}" --reward-clip "${DISCRIMINATOR_REWARD_CLIP:-2.0}" \
  --save-interval "${DISCRIMINATOR_SAVE_INTERVAL:-50}" \
  "${EXTRA_ARGS[@]}"
