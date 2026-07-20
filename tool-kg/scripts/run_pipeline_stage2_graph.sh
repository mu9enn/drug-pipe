#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="${1:-}"
ALERT_RERUN=0
MAX_ALERT_RERUN_ROUNDS=3
MAX_WORKERS=1
RESUME=0

if [[ "$RUN_ID" == "--help" || "$RUN_ID" == "-h" ]]; then
  echo "Usage: $0 <run_id> [--alert-rerun] [--max-alert-rerun-rounds <n>] [--max-workers <n>] [--resume]"
  exit 0
fi

if [[ -z "$RUN_ID" ]]; then
  echo "Usage: $0 <run_id> [--alert-rerun] [--max-alert-rerun-rounds <n>] [--max-workers <n>] [--resume]" >&2
  exit 1
fi
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
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
      echo "Usage: $0 <run_id> [--alert-rerun] [--max-alert-rerun-rounds <n>] [--max-workers <n>] [--resume]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
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
for f in tool_snapshot.jsonl tool_cards.jsonl; do
  if [[ ! -f "$RUN_DIR/$f" ]]; then
    echo "ERROR: missing stage1 artifact: $RUN_DIR/$f" >&2
    exit 2
  fi
done

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
  local meta="$RUN_DIR/pair_adjudication_alerts_meta.json"
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
  local meta="$RUN_DIR/pair_adjudication_alerts_meta.json"
  if [[ ! -f "$meta" ]]; then
    echo "[stage2-alert] adjudication alerts meta missing"
    return
  fi
  python3 - "$meta" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
obj = json.loads(p.read_text(encoding='utf-8'))
cnt = int(obj.get('alert_count', 0))
pair_cnt = int(obj.get('alert_pair_count', 0))
print(f"[stage2-alert] adjudication alerts: {cnt} (pairs: {pair_cnt})")
print(f"[stage2-alert] rerun targets: {obj.get('rerun_targets_path')}")
print(f"[stage2-alert] alerts jsonl: {obj.get('alerts_path')}")
PY
}

if [[ "$RESUME" -eq 1 && -f "$RUN_DIR/candidate_pairs.jsonl" ]]; then
  echo "[stage2-resume] candidate_pairs already exists, skipping candidates"
else
  run_cli candidates
fi

run_cli adjudicate

print_alert_summary

if [[ "$ALERT_RERUN" -eq 1 ]]; then
  round=1
  alert_count="$(read_alert_count)"
  while [[ "$alert_count" -gt 0 && "$round" -le "$MAX_ALERT_RERUN_ROUNDS" ]]; do
    targets_file="$RUN_DIR/pair_adjudication_rerun_targets.txt"
    if [[ ! -s "$targets_file" ]]; then
      echo "[stage2-alert-rerun] stop: rerun target file missing or empty: $targets_file"
      break
    fi

    echo "[stage2-alert-rerun] round=$round alert_count=$alert_count"
    run_cli adjudicate \
      --pair-ids-file "$targets_file" \
      --merge-into-existing \
      --bypass-cache-for-targets \
      --rerun-round "$round"

    print_alert_summary
    alert_count="$(read_alert_count)"
    round=$((round + 1))
  done

  if [[ "$alert_count" -gt 0 ]]; then
    echo "[stage2-alert-rerun] residual alerts after max rounds: $alert_count"
    echo "[stage2-alert-rerun] review: $RUN_DIR/pair_adjudication_alerts.jsonl"
  else
    echo "[stage2-alert-rerun] alerts cleared"
  fi
fi

run_cli canonical-edges
run_cli finalize

echo "stage2 complete: $RUN_DIR"
echo "canonical results: $RUN_DIR/results"
