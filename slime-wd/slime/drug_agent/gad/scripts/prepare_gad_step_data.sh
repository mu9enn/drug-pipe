#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../scripts/resolve_slime_env.sh"
source "$SLIME_ENV"
cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh

INPUT=${INPUT:-$DRUG_AGENT_DATA_ROOT/react_trajectories.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-$DRUG_AGENT_DATA_ROOT/gad}
CONTEXT_BUDGET_MODE=${CONTEXT_BUDGET_MODE:-none}
RAW_OUTPUT="$OUTPUT_ROOT/gad_steps.raw.jsonl"
FINAL_OUTPUT="$OUTPUT_ROOT/gad_steps.jsonl"
mkdir -p "$OUTPUT_ROOT"
if [ ! -f "$INPUT" ]; then
  echo "GAD source JSONL does not exist: $INPUT" >&2
  exit 2
fi

python -m drug_agent.gad.data \
  --input "$INPUT" \
  --output "$RAW_OUTPUT" \
  --skipped-report "$OUTPUT_ROOT/gad_steps.skipped.jsonl" \
  --report "$OUTPUT_ROOT/gad_steps.report.json"

if [ "$CONTEXT_BUDGET_MODE" = "none" ]; then
  cp "$RAW_OUTPUT" "$FINAL_OUTPUT"
elif [ "$CONTEXT_BUDGET_MODE" = "claude" ]; then
  : "${HF_CHECKPOINT:?HF_CHECKPOINT is required for CONTEXT_BUDGET_MODE=claude}"
  SUMMARY_CACHE_ROOT=${SUMMARY_CACHE_ROOT:-$OUTPUT_ROOT/summary_cache}
  python drug_agent/scripts/compact_rl_context.py \
    --input "$RAW_OUTPUT" \
    --output "$FINAL_OUTPUT" \
    --audit "$OUTPUT_ROOT/context_manifest.json" \
    --excluded "$OUTPUT_ROOT/context_excluded.jsonl" \
    --tokenizer "$HF_CHECKPOINT" \
    --max-tokens "${ROLLOUT_MAX_PROMPT_LEN:-245760}" \
    --max-response-tokens "${ROLLOUT_MAX_RESPONSE_LEN:-16384}" \
    --max-context-tokens "${ROLLOUT_MAX_CONTEXT_LEN:-262144}" \
    --summary-max-tokens "${SUMMARY_MAX_TOKENS:-32768}" \
    --semantic-summarizer claude \
    --summary-cache-root "$SUMMARY_CACHE_ROOT" \
    --claude-bin "${CLAUDE_BIN:-claude}" \
    --llm-timeout-sec "${LLM_TIMEOUT_SEC:-600}" \
    --llm-max-attempts "${LLM_MAX_ATTEMPTS:-3}"
else
  echo "CONTEXT_BUDGET_MODE must be none or claude; got $CONTEXT_BUDGET_MODE" >&2
  exit 2
fi
