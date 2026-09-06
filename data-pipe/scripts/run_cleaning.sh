#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DRUG_PIPE_DIR="$(cd "$DATA_PIPE_DIR/.." && pwd)"
SLIME_DIR="$DRUG_PIPE_DIR/slime-wd/slime"
PYTHON_BIN="${PYTHON_BIN:-python}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
RESULTS_ROOT="${RESULTS_ROOT:-$DATA_PIPE_DIR/results}"
PRECLEAN_ROOT="${PRECLEAN_ROOT:-}"
WORK_ROOT="${WORK_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-$DATA_PIPE_DIR/pipeline/cleaning/prompts/qwen35_system.md}"
DEPLOYMENT_TOOL_SET="${DEPLOYMENT_TOOL_SET:-}"
TOOL_VISIBILITY="${TOOL_VISIBILITY:-all}"
TIMEOUT_SEC="${LLM_CLEAN_TIMEOUT_SEC:-300}"
LIMIT=0
MAX_WORKERS="${MAX_WORKERS:-1}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_cleaning.sh --deployment-tool-set PATH [options]

Builds the structured data path:
  Claude raw -> uncleaned Qwen-native audit projection
             -> cleaned semantic mother dataset -> mandatory reasoning clean
             -> Qwen3.5 structured SFT view

Options:
  --results-root PATH
  --preclean-root PATH          Uncleaned raw-event native-message audit view
  --work-root PATH
  --output-root PATH
  --deployment-tool-set PATH   Exact tools visible to the student at deployment
  --tool-visibility MODE       all (default) or trajectory-plus-distractors
  --system-prompt-file PATH    Qwen adapter system prompt
  --claude-bin PATH
  --timeout-sec SECONDS
  --limit N
  --max-workers N
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-root) RESULTS_ROOT="${2:-}"; shift 2 ;;
    --preclean-root) PRECLEAN_ROOT="${2:-}"; shift 2 ;;
    --work-root) WORK_ROOT="${2:-}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --deployment-tool-set) DEPLOYMENT_TOOL_SET="${2:-}"; shift 2 ;;
    --tool-visibility) TOOL_VISIBILITY="${2:-}"; shift 2 ;;
    --system-prompt-file) SYSTEM_PROMPT_FILE="${2:-}"; shift 2 ;;
    --claude-bin) CLAUDE_BIN="${2:-}"; shift 2 ;;
    --timeout-sec) TIMEOUT_SEC="${2:-}"; shift 2 ;;
    --limit) LIMIT="${2:-}"; shift 2 ;;
    --max-workers) MAX_WORKERS="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

PRECLEAN_ROOT="${PRECLEAN_ROOT:-$RESULTS_ROOT/qwen35_native_raw}"
WORK_ROOT="${WORK_ROOT:-$RESULTS_ROOT/semantic_work}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RESULTS_ROOT/cleaned}"

if [[ -z "$DEPLOYMENT_TOOL_SET" || ! -f "$DEPLOYMENT_TOOL_SET" ]]; then
  echo "[error] --deployment-tool-set must name the exact deployment-visible tool manifest" >&2
  exit 1
fi
if [[ ! -f "$SYSTEM_PROMPT_FILE" ]]; then
  echo "[error] Qwen adapter system prompt not found: $SYSTEM_PROMPT_FILE" >&2
  exit 1
fi
if [[ "$TOOL_VISIBILITY" != "all" && "$TOOL_VISIBILITY" != "trajectory-plus-distractors" ]]; then
  echo "[error] unsupported --tool-visibility: $TOOL_VISIBILITY" >&2
  exit 1
fi

RESULTS_ROOT="$(realpath -m "$RESULTS_ROOT")"
PRECLEAN_ROOT="$(realpath -m "$PRECLEAN_ROOT")"
WORK_ROOT="$(realpath -m "$WORK_ROOT")"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
DEPLOYMENT_TOOL_SET="$(realpath "$DEPLOYMENT_TOOL_SET")"
SYSTEM_PROMPT_FILE="$(realpath "$SYSTEM_PROMPT_FILE")"

export PYTHONPATH="$DATA_PIPE_DIR:$SLIME_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$PRECLEAN_ROOT" "$WORK_ROOT" "$OUTPUT_ROOT"
cd "$DATA_PIPE_DIR"

echo "[cleaning] raw events -> uncleaned Qwen3.5 native-message audit view"
"$PYTHON_BIN" -m pipeline.cleaning.preclean_native \
  --results-root "$RESULTS_ROOT" \
  --output-root "$PRECLEAN_ROOT"

echo "[cleaning] raw events -> semantic mother dataset"
"$PYTHON_BIN" -m pipeline.cleaning.python_clean \
  --results-root "$RESULTS_ROOT" \
  --output-root "$WORK_ROOT"

VIEW_INPUT="$WORK_ROOT/semantic_trajectories.jsonl"
echo "[cleaning] semantic reasoning -> immutable checked mandatory LLM materialization"
"$PYTHON_BIN" -m pipeline.cleaning.llm_clean \
  --input "$VIEW_INPUT" \
  --output-root "$OUTPUT_ROOT" \
  --claude-bin "$CLAUDE_BIN" \
  --timeout-sec "$TIMEOUT_SEC" \
  --limit "$LIMIT" \
  --max-workers "$MAX_WORKERS"
VIEW_INPUT="$OUTPUT_ROOT/semantic_trajectories.jsonl"

echo "[cleaning] semantic -> Qwen3.5 structured SFT view"
"$PYTHON_BIN" -m pipeline.cleaning.materialize_sft \
  --input "$VIEW_INPUT" \
  --output-root "$OUTPUT_ROOT" \
  --deployment-tool-set "$DEPLOYMENT_TOOL_SET" \
  --tool-visibility "$TOOL_VISIBILITY" \
  --system-prompt-file "$SYSTEM_PROMPT_FILE"

count=$(wc -l < "$OUTPUT_ROOT/qwen35_sft.jsonl")
if [[ "$count" -eq 0 ]]; then
  echo "[error] qwen35_sft.jsonl is empty; semantic mother data remains in $WORK_ROOT" >&2
  exit 1
fi
echo "[done] structured Qwen3.5 SFT samples: $count"
echo "  semantic_mother: $WORK_ROOT/semantic_trajectories.jsonl"
echo "  preclean_native_audit: $PRECLEAN_ROOT/qwen35_native_raw.jsonl"
echo "  materialized: $OUTPUT_ROOT"
