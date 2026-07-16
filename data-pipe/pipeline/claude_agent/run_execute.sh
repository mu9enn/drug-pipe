#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="$SCRIPT_DIR/launch_claude.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash pipeline/claude_agent/run_execute.sh [launch_claude args...]

Examples:
  bash pipeline/claude_agent/run_execute.sh \
    --run-dataset --task vs --dataset-csv molbench/molbench-vs-30.csv

  bash pipeline/claude_agent/run_execute.sh \
    --run-dataset --task kg --dataset-csv pipeline/kg/data/<run_id>/kg_tasks_exec.csv
EOF
  exit 0
fi

exec bash "$LAUNCH_SCRIPT" "$@"
