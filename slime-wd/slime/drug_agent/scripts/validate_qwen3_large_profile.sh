#!/usr/bin/env bash
set -euo pipefail

SLIME_ENV=${SLIME_ENV:-/root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh}
if [[ ! -f "$SLIME_ENV" ]]; then
  SLIME_ENV=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi
source "$SLIME_ENV" >/dev/null
cd "$SLIME"

: "${MODEL_PROFILE:?Set MODEL_PROFILE before running profile validation}"
source drug_agent/scripts/qwen3_large_profile.sh
source "$MODEL_ARGS_FILE"

python drug_agent/scripts/validate_qwen3_large_profile.py -- "${MODEL_ARGS[@]}"
