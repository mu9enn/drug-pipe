#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../scripts/resolve_slime_env.sh"
source "$SLIME_ENV"
cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh

PAIRS=${PAIRS:-$DRUG_AGENT_DATA_ROOT/gad/stage2_negatives.jsonl}
MODEL_PATH=${DISCRIMINATOR_MODEL_PATH:-$DATA/Qwen3.5-4B}
OUTPUT_DIR=${DISCRIMINATOR_OUTPUT_DIR:-$DRUG_AGENT_RUNS_ROOT/gad_discriminator_warmup}
: "${GENERATOR_WARMUP_LOAD:?Set GENERATOR_WARMUP_LOAD to the completed generator SFT checkpoint}"
EXTRA_ARGS=()
if [ -n "${DISCRIMINATOR_RESUME:-}" ]; then
  EXTRA_ARGS+=(--resume "$DISCRIMINATOR_RESUME")
fi
for path in "$PAIRS" "$MODEL_PATH" "$GENERATOR_WARMUP_LOAD"; do
  if [ ! -e "$path" ]; then
    echo "Required discriminator warmup input does not exist: $path" >&2
    exit 2
  fi
done

python -m drug_agent.gad.train_discriminator \
  --pairs "$PAIRS" --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR" \
  --generator-warmup-checkpoint "$GENERATOR_WARMUP_LOAD" \
  --epochs "${DISCRIMINATOR_EPOCHS:-1}" --batch-size "${DISCRIMINATOR_BATCH_SIZE:-2}" \
  --lr "${DISCRIMINATOR_LR:-1e-5}" --max-length "${DISCRIMINATOR_MAX_LENGTH:-4096}" \
  --clip-grad "${DISCRIMINATOR_CLIP_GRAD:-1.0}" \
  --save-interval "${DISCRIMINATOR_SAVE_INTERVAL:-50}" \
  --keep-last-checkpoints "${DISCRIMINATOR_KEEP_LAST:-2}" \
  "${EXTRA_ARGS[@]}"
