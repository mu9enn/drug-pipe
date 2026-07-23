#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
RESULTS_ROOT="${RESULTS_ROOT:-$DATA_PIPE_DIR/results}"
WORK_ROOT="${WORK_ROOT:-$RESULTS_ROOT/cleaning_work}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RESULTS_ROOT/cleaned}"
TIMEOUT_SEC="${LLM_CLEAN_TIMEOUT_SEC:-300}"
LIMIT=0
ONLY_MOLCLAW_TOOL=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_cleaning.sh [options]

Runs three logical steps through two commands:
  1. deterministic Python filtering
  2. deterministic Python ReAct structuring
  3. restricted-patch LLM prose cleaning

Options:
  --results-root PATH
  --work-root PATH
  --output-root PATH
  --claude-bin PATH
  --timeout-sec SECONDS
  --limit N
  --only-molclaw-tool
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-root) RESULTS_ROOT="${2:-}"; shift 2 ;;
    --work-root) WORK_ROOT="${2:-}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --claude-bin) CLAUDE_BIN="${2:-}"; shift 2 ;;
    --timeout-sec) TIMEOUT_SEC="${2:-}"; shift 2 ;;
    --limit) LIMIT="${2:-}"; shift 2 ;;
    --only-molclaw-tool) ONLY_MOLCLAW_TOOL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -f "$DATA_PIPE_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$DATA_PIPE_DIR/.env"
  set +a
fi

RESULTS_ROOT="$(realpath -m "$RESULTS_ROOT")"
WORK_ROOT="$(realpath -m "$WORK_ROOT")"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
if [[ ! -d "$RESULTS_ROOT" ]]; then
  echo "[error] results root not found: $RESULTS_ROOT" >&2
  exit 1
fi

cd "$DATA_PIPE_DIR"
echo "[cleaning] steps 1-2/3: Python filter then canonical ReAct structuring"
python_clean_args=(
  --results-root "$RESULTS_ROOT"
  --output-root "$WORK_ROOT"
)
if [[ "$ONLY_MOLCLAW_TOOL" -eq 1 ]]; then
  python_clean_args+=(--only-molclaw-tool)
fi
"$PYTHON_BIN" -m pipeline.cleaning.python_clean \
  "${python_clean_args[@]}"

echo "[cleaning] step 3/3: restricted-patch LLM prose clean"
"$PYTHON_BIN" -m pipeline.cleaning.llm_clean \
  --input "$WORK_ROOT/python_drafts.jsonl" \
  --python-audit "$WORK_ROOT/python_audit.jsonl" \
  --output-root "$OUTPUT_ROOT" \
  --claude-bin "$CLAUDE_BIN" \
  --timeout-sec "$TIMEOUT_SEC" \
  --limit "$LIMIT"

count=$(wc -l < "$OUTPUT_ROOT/react_trajectories.jsonl")
if [[ "$count" -eq 0 ]]; then
  echo "[error] final react_trajectories.jsonl is empty; inspect rejected/audit outputs" >&2
  exit 1
fi
echo "[done] accepted canonical ReAct: $count"
echo "  python_work: $WORK_ROOT"
echo "  final_output: $OUTPUT_ROOT"
