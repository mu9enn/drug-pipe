#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
TOOL_IDS_FILE=""
ALERT_RERUN=0
MAX_ALERT_RERUN_ROUNDS=3
MAX_WORKERS=1
RESUME=0

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [run_id] [--tool-ids-file <path>] [--alert-rerun] [--max-alert-rerun-rounds <n>] [--max-workers <n>] [--resume]"
  exit 0
fi

if [[ $# -gt 0 ]]; then
  RUN_ID="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tool-ids-file)
      TOOL_IDS_FILE="${2:-}"
      shift 2
      ;;
    --alert-rerun)
      ALERT_RERUN=1
      shift
      ;;
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
      echo "Usage: $0 [run_id] [--tool-ids-file <path>] [--alert-rerun] [--max-alert-rerun-rounds <n>] [--max-workers <n>] [--resume]"
      exit 0
      ;;
    *)
      if [[ -z "$TOOL_IDS_FILE" && "$1" != --* ]]; then
        TOOL_IDS_FILE="$1"
        shift
      else
        echo "Unknown argument: $1" >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

API_KEY="${MOLCLAW_SCP_API_KEY:-}"
if [[ -z "$API_KEY" ]]; then
  echo "ERROR: MOLCLAW_SCP_API_KEY is required" >&2
  exit 1
fi

RUN_DIR="$PROJECT_ROOT/runs/$RUN_ID"

run_cli() {
  local cmd="$1"
  shift || true
  local resume_args=()
  if [[ "$RESUME" -eq 1 ]]; then
    resume_args+=(--resume)
  fi
  PYTHONPATH="$PROJECT_ROOT/src" python3 -m molclaw_kg.cli \
    --project-root "$PROJECT_ROOT" \
    --run-id "$RUN_ID" \
    --api-key "$API_KEY" \
    --mode claude_cc \
    --max-workers "$MAX_WORKERS" \
    "${resume_args[@]}" \
    "$cmd" \
    "$@"
}

read_alert_count() {
  local meta="$RUN_DIR/tool_card_alerts_meta.json"
  if [[ ! -f "$meta" ]]; then
    echo 0
    return
  fi
  python3 - "$meta" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
obj = json.loads(p.read_text(encoding='utf-8'))
print(int(obj.get('alert_count', 0)))
PY
}

print_alert_summary() {
  local meta="$RUN_DIR/tool_card_alerts_meta.json"
  if [[ ! -f "$meta" ]]; then
    echo "[stage1-alert] tool-card alerts meta missing"
    return
  fi
  python3 - "$meta" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
obj = json.loads(p.read_text(encoding='utf-8'))
cnt = int(obj.get('alert_count', 0))
print(f"[stage1-alert] tool-card alerts: {cnt}")
print(f"[stage1-alert] rerun targets: {obj.get('rerun_targets_path')}")
print(f"[stage1-alert] alerts jsonl: {obj.get('alerts_path')}")
PY
}

# stage1 main
if [[ "$RESUME" -eq 1 && -f "$RUN_DIR/tool_snapshot.jsonl" ]]; then
  echo "[stage1-resume] snapshot already exists, skipping"
else
  run_cli snapshot
fi

if [[ "$RESUME" -eq 1 && -f "$RUN_DIR/doc_chunks.jsonl" ]]; then
  echo "[stage1-resume] doc-chunks already exists, skipping"
else
  run_cli doc-chunks
fi

if [[ -n "$TOOL_IDS_FILE" ]]; then
  run_cli tool-cards --tool-ids-file "$TOOL_IDS_FILE"
else
  run_cli tool-cards
fi

print_alert_summary

if [[ "$ALERT_RERUN" -eq 1 ]]; then
  round=1
  alert_count="$(read_alert_count)"
  while [[ "$alert_count" -gt 0 && "$round" -le "$MAX_ALERT_RERUN_ROUNDS" ]]; do
    targets_file="$RUN_DIR/tool_card_rerun_targets.txt"
    if [[ ! -s "$targets_file" ]]; then
      echo "[stage1-alert-rerun] stop: rerun target file missing or empty: $targets_file"
      break
    fi

    echo "[stage1-alert-rerun] round=$round alert_count=$alert_count"
    run_cli tool-cards \
      --tool-ids-file "$targets_file" \
      --merge-into-existing \
      --rerun-round "$round"

    print_alert_summary
    alert_count="$(read_alert_count)"
    round=$((round + 1))
  done

  if [[ "$alert_count" -gt 0 ]]; then
    echo "[stage1-alert-rerun] residual alerts after max rounds: $alert_count"
  else
    echo "[stage1-alert-rerun] alerts cleared"
  fi
fi

echo "stage1 complete: $RUN_DIR"
