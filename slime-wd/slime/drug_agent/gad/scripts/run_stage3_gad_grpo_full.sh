#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../scripts/resolve_slime_env.sh"
source "$SLIME_ENV"

PROMPT_DATA=${PROMPT_DATA:-$DRUG_AGENT_DATA_ROOT/gad/gad_steps.jsonl}
if [ ! -f "$PROMPT_DATA" ]; then
  echo "GAD step data does not exist: $PROMPT_DATA" >&2
  exit 2
fi

export PROMPT_DATA
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
DATASET_SIZE=$(wc -l < "$PROMPT_DATA")
export NUM_ROLLOUT=${NUM_ROLLOUT:-$(((DATASET_SIZE + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE))}
export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}
export ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-6144}
export ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-512}
export ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-6656}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-4096}
export SAVE_INTERVAL=${SAVE_INTERVAL:-100}

echo "[4B GAD full] dataset_size=$DATASET_SIZE prompt_batches=$NUM_ROLLOUT"
exec bash drug_agent/gad/scripts/run_stage3_gad_grpo.sh
