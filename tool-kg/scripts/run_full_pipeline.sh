#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
MAX_ALERT_RERUN_ROUNDS=3
MAX_WORKERS=1
RESUME=0

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [run_id] [--max-alert-rerun-rounds <n>] [--max-workers <n>] [--resume]"
  exit 0
fi

if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  RUN_ID="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-alert-rerun-rounds)
      MAX_ALERT_RERUN_ROUNDS="${2:-3}"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="${2:-1}"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [run_id] [--max-alert-rerun-rounds <n>] [--max-workers <n>] [--resume]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

echo "[full-pipeline] run_id=$RUN_ID"
echo "[full-pipeline] stage1: alert-rerun enabled, max_rounds=$MAX_ALERT_RERUN_ROUNDS"
stage1_args=(
  "$RUN_ID"
  --alert-rerun
  --max-alert-rerun-rounds "$MAX_ALERT_RERUN_ROUNDS"
  --max-workers "$MAX_WORKERS"
)
if [[ "$RESUME" -eq 1 ]]; then
  stage1_args+=(--resume)
fi
bash "$PROJECT_ROOT/scripts/run_pipeline_stage1_toolcards.sh" "${stage1_args[@]}"

echo "[full-pipeline] stage2: alert-rerun enabled, max_rounds=$MAX_ALERT_RERUN_ROUNDS"
stage2_args=(
  "$RUN_ID"
  --alert-rerun
  --max-alert-rerun-rounds "$MAX_ALERT_RERUN_ROUNDS"
  --max-workers "$MAX_WORKERS"
)
if [[ "$RESUME" -eq 1 ]]; then
  stage2_args+=(--resume)
fi
bash "$PROJECT_ROOT/scripts/run_pipeline_stage2_graph.sh" "${stage2_args[@]}"

echo "[full-pipeline] complete: $PROJECT_ROOT/runs/$RUN_ID"
