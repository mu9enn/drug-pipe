#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export DRUG_AGENT_EVAL_PROFILE=slime-raw
if [[ -n "${MODEL_CHECKPOINT:-}" ]]; then
  export RUN_NAME=${RUN_NAME:-slime_raw_$(basename "$MODEL_CHECKPOINT")_$(date +%Y%m%d_%H%M%S)}
fi
exec bash "$SCRIPT_DIR/run_molbench_eval.sh" "$@"
