#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../scripts/resolve_slime_env.sh"
source "$SLIME_ENV"
cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh

MODEL_PATH=${DISCRIMINATOR_MODEL_PATH:-$DATA/Qwen3.5-4B}
OUTPUT_DIR=${DISCRIMINATOR_OUTPUT_DIR:-$DRUG_AGENT_RUNS_ROOT/gad_discriminator_online}
EXTRA_ARGS=()
: "${DISCRIMINATOR_RESUME:?Set DISCRIMINATOR_RESUME to a Stage 2 discriminator warmup checkpoint}"
EXTRA_ARGS+=(--resume "$DISCRIMINATOR_RESUME")
if [ "${DISCRIMINATOR_OFFLOAD_AFTER_REQUEST:-0}" = "1" ]; then
  EXTRA_ARGS+=(--offload-after-request)
elif [ "${DISCRIMINATOR_OFFLOAD_AFTER_REQUEST:-0}" != "0" ]; then
  echo "DISCRIMINATOR_OFFLOAD_AFTER_REQUEST must be 0 or 1" >&2
  exit 2
fi
if [ ! -e "$MODEL_PATH" ]; then
  echo "Discriminator model does not exist: $MODEL_PATH" >&2
  exit 2
fi

python -m drug_agent.gad.service \
  --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR" \
  --host "${GAD_DISCRIMINATOR_HOST:-0.0.0.0}" --port "${GAD_DISCRIMINATOR_PORT:-8100}" \
  --device "${GAD_DISCRIMINATOR_DEVICE:-cuda}" \
  --lr "${DISCRIMINATOR_LR:-1e-5}" --max-length "${DISCRIMINATOR_MAX_LENGTH:-4096}" \
  --update-steps "${DISCRIMINATOR_UPDATE_STEPS:-1}" --reward-clip "${DISCRIMINATOR_REWARD_CLIP:-2.0}" \
  --clip-grad "${DISCRIMINATOR_CLIP_GRAD:-1.0}" \
  --save-interval "${DISCRIMINATOR_SAVE_INTERVAL:-50}" \
  --keep-last-checkpoints "${DISCRIMINATOR_KEEP_LAST:-2}" \
  "${EXTRA_ARGS[@]}"
