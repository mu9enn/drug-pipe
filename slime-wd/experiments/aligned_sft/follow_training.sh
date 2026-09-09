#!/usr/bin/env bash
set -Eeuo pipefail
: "${TRAINING_RUN_ROOT:?}" "${TRAINING_JOB_NAME:?}" "${PAIR_STAMP:?}"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
trap 'rc=$?; printf "%s\n" "$rc" > "$TRAINING_RUN_ROOT/paired_orchestrator.exit"' EXIT
echo "Waiting for successful training and exported checkpoint: $TRAINING_JOB_NAME"
for ((attempt=0; attempt<2880; attempt++)); do
 if [[ -f "$TRAINING_RUN_ROOT/worker.exit" ]]; then
  [[ "$(cat "$TRAINING_RUN_ROOT/worker.exit")" == 0 ]]
  [[ -f "$TRAINING_RUN_ROOT/training.complete" && -s "$TRAINING_RUN_ROOT/hf/model.safetensors.index.json" ]]
  break
 fi
 rjob get "$TRAINING_JOB_NAME" --namespace=ailab-ma4agismall > "$TRAINING_RUN_ROOT/paired_wait_state.txt" 2>&1
 if rg -q 'Failed|Stopped' "$TRAINING_RUN_ROOT/paired_wait_state.txt"; then
  cat "$TRAINING_RUN_ROOT/paired_wait_state.txt"
  exit 1
 fi
 sleep 60
done
[[ -f "$TRAINING_RUN_ROOT/training.complete" && -f "$TRAINING_RUN_ROOT/probe_reload_validation.json" ]]
SFT_MODEL_DIR="$TRAINING_RUN_ROOT/hf" bash "$script_dir/evaluate_pair.sh"
touch "$TRAINING_RUN_ROOT/paired_evaluation.complete"
