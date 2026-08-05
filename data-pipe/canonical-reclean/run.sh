#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$DATA_PIPE_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
INPUT="${INPUT:-$REPO_ROOT/slime-wd/outputs/slime_drug_agent_data/live_tool_catalog_v2/react_trajectories.jsonl}"
TOOL_CATALOG="${TOOL_CATALOG:-$REPO_ROOT/slime-wd/outputs/slime_drug_agent_data/live_tool_catalog_v2/tool_catalog.json}"
OUTPUT="${OUTPUT:-$REPO_ROOT/slime-wd/outputs/slime_drug_agent_data/live_tool_catalog_v3}"
TOKENIZER="${TOKENIZER:-$REPO_ROOT/slime-wd/data/Qwen3.5-9B}"
MAX_WORKERS="${MAX_WORKERS:-2}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
LIMIT="${LIMIT:-0}"
RECORD_ID="${RECORD_ID:-}"
RESUME="${RESUME:-}"

if [[ -n "${RUN_ROOT:-}" ]]; then
  RUN_ROOT="$RUN_ROOT"
elif [[ -n "$RESUME" ]]; then
  RUN_ROOT="$RESUME"
else
  RUN_ROOT="$DATA_PIPE_DIR/results/canonical_reclean/run_$(date +%Y%m%d_%H%M%S)"
fi

args=(
  --input "$INPUT"
  --tool-catalog "$TOOL_CATALOG"
  --output "$OUTPUT"
  --run-root "$RUN_ROOT"
  --tokenizer "$TOKENIZER"
  --claude-bin "$CLAUDE_BIN"
  --max-workers "$MAX_WORKERS"
  --timeout-sec "$TIMEOUT_SEC"
  --max-attempts "$MAX_ATTEMPTS"
  --limit "$LIMIT"
)
if [[ -n "$RECORD_ID" ]]; then
  args+=(--record-id "$RECORD_ID")
fi
if [[ -n "$RESUME" ]]; then
  args+=(--resume)
fi

cd "$DATA_PIPE_DIR"
PYTHONPATH="$DATA_PIPE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" "$SCRIPT_DIR/reclean.py" "${args[@]}"
