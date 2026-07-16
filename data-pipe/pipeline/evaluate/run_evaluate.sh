#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="$SCRIPT_DIR/run_eval_bench.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  bash pipeline/evaluate/run_evaluate.sh <RESULTS_DIR> [vs|ac|pf]

Example:
  bash pipeline/evaluate/run_evaluate.sh results/molbench_vs_manual_run_YYYYMMDD_HHMMSS vs
EOF
  exit 0
fi

RESULTS_DIR="$1"
TASK="${2:-}"

if [[ -n "$TASK" ]]; then
  exec "$PYTHON_BIN" "$EVAL_PY" "$RESULTS_DIR" --task "$TASK"
fi
exec "$PYTHON_BIN" "$EVAL_PY" "$RESULTS_DIR"
