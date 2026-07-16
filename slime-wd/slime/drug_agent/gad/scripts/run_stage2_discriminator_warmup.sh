#!/bin/bash
set -euo pipefail
SLIME_ENV=${SLIME_ENV:-/root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh}
if [ ! -f "$SLIME_ENV" ]; then
  SLIME_ENV=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi
source "$SLIME_ENV"
cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh

PAIRS=${PAIRS:-$DRUG_AGENT_DATA_ROOT/gad/stage2_negatives.jsonl}
MODEL_PATH=${DISCRIMINATOR_MODEL_PATH:-$DATA/Qwen3.5-0.8B}
OUTPUT_DIR=${DISCRIMINATOR_OUTPUT_DIR:-$DRUG_AGENT_RUNS_ROOT/gad_discriminator_warmup}
EXTRA_ARGS=()
if [ -n "${DISCRIMINATOR_RESUME:-}" ]; then
  EXTRA_ARGS+=(--resume "$DISCRIMINATOR_RESUME")
fi
for path in "$PAIRS" "$MODEL_PATH"; do
  if [ ! -e "$path" ]; then
    echo "Required discriminator warmup input does not exist: $path" >&2
    exit 2
  fi
done

python -m drug_agent.gad.train_discriminator \
  --pairs "$PAIRS" --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR" \
  --epochs "${DISCRIMINATOR_EPOCHS:-1}" --batch-size "${DISCRIMINATOR_BATCH_SIZE:-2}" \
  --lr "${DISCRIMINATOR_LR:-1e-5}" --max-length "${DISCRIMINATOR_MAX_LENGTH:-4096}" \
  --save-interval "${DISCRIMINATOR_SAVE_INTERVAL:-50}" \
  "${EXTRA_ARGS[@]}"
