#!/bin/bash
set -euo pipefail
SLIME_ENV=${SLIME_ENV:-/root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh}
if [ ! -f "$SLIME_ENV" ]; then
  SLIME_ENV=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi
source "$SLIME_ENV"
cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh

INPUT=${INPUT:-$GROUP_SPACE/slime-wd/data/mcp_sft_all.train.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-$DRUG_AGENT_DATA_ROOT/gad}
mkdir -p "$OUTPUT_ROOT"
if [ ! -f "$INPUT" ]; then
  echo "GAD source JSONL does not exist: $INPUT" >&2
  exit 2
fi

python -m drug_agent.gad.data \
  --input "$INPUT" \
  --output "$OUTPUT_ROOT/gad_steps.jsonl" \
  --skipped-report "$OUTPUT_ROOT/gad_steps.skipped.jsonl" \
  --report "$OUTPUT_ROOT/gad_steps.report.json"
