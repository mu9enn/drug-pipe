#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PIPELINE_DIR/.." && pwd)"
cd "$REPO_DIR"

PROVIDER="${1:-manual}"
CLAUDE_BIN="${2:-claude}"
LIMIT="${3:-0}"
NUM_ROLLOUTS="${4:-1}"
PARALLEL_ROLLOUTS="${5:-1}"
DATASET_DIR="${6:-$REPO_DIR/molbench}"
MAX_WORKERS="${MAX_WORKERS:-$PARALLEL_ROLLOUTS}"
VS_CSV=""
AC_CSV=""
PF_CSV=""

if (( $# > 6 )); then
  EXTRA=("${@:7}")
  i=0
  while (( i < ${#EXTRA[@]} )); do
    arg="${EXTRA[$i]}"
    case "$arg" in
      --vs-csv=*)
        VS_CSV="${arg#*=}"
        ;;
      --ac-csv=*)
        AC_CSV="${arg#*=}"
        ;;
      --pf-csv=*)
        PF_CSV="${arg#*=}"
        ;;
      --vs-csv)
        i=$((i + 1))
        VS_CSV="${EXTRA[$i]:-}"
        if [[ -z "$VS_CSV" ]]; then
          echo "[error] --vs-csv requires a path value" >&2
          exit 1
        fi
        ;;
      --ac-csv)
        i=$((i + 1))
        AC_CSV="${EXTRA[$i]:-}"
        if [[ -z "$AC_CSV" ]]; then
          echo "[error] --ac-csv requires a path value" >&2
          exit 1
        fi
        ;;
      --pf-csv)
        i=$((i + 1))
        PF_CSV="${EXTRA[$i]:-}"
        if [[ -z "$PF_CSV" ]]; then
          echo "[error] --pf-csv requires a path value" >&2
          exit 1
        fi
        ;;
      --max-workers=*)
        MAX_WORKERS="${arg#*=}"
        ;;
      --max-workers)
        i=$((i + 1))
        MAX_WORKERS="${EXTRA[$i]:-}"
        if [[ -z "$MAX_WORKERS" ]]; then
          echo "[error] --max-workers requires a value" >&2
          exit 1
        fi
        ;;
      *)
        echo "[error] unknown extra arg: $arg" >&2
        exit 1
        ;;
    esac
    i=$((i + 1))
  done
fi

if ! [[ "$MAX_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] --max-workers must be a positive integer" >&2
  exit 1
fi

FLOW="$PIPELINE_DIR/claude_agent/test_flow_claude.sh"
LOG_DIR="$REPO_DIR/results/parallel_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Preflight: RDKit is required for AC/PF evaluation and trajectory export.
if ! python -c "from rdkit import Chem" >/dev/null 2>&1; then
  echo "[error] RDKit is not available in current Python env. Please install/activate env with rdkit before running parallel tests." >&2
  exit 1
fi

# if ! command -v cc-switch >/dev/null 2>&1; then
#   echo "[error] cc-switch not found in PATH, but is required for provider switch." >&2
#   exit 1
# fi
#
# if ! cc-switch provider switch "$PROVIDER" >/dev/null; then
#   echo "[error] failed to switch provider via cc-switch: $PROVIDER" >&2
#   exit 1
# fi
# echo "[run] provider switched once via cc-switch: $PROVIDER"
echo "[run] provider switch step disabled in script (expect external cc-switch before run)"

declare -A PIDS
declare -A LOGS
declare -A TOTAL_EVENTS
declare -A EXIT_CODES
declare -A DONE_COUNTS
declare -A LOG_SIZES
declare -A LAST_AGE
declare -A TAIL_STATE

count_csv_rows() {
  local csv_path="$1"
  python - "$csv_path" <<'PY'
import csv
import sys

path = sys.argv[1]
count = 0
with open(path, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    next(reader, None)  # header
    for _ in reader:
        count += 1
print(count)
PY
}

make_bar() {
  local current="$1"
  local total="$2"
  local width=30
  local pct=0
  local filled=0
  local empty=0
  local bar_fill=""
  local bar_empty=""

  if (( total > 0 )); then
    pct=$(( current * 100 / total ))
    if (( pct > 100 )); then
      pct=100
    fi
    filled=$(( pct * width / 100 ))
  fi
  empty=$(( width - filled ))

  if (( filled > 0 )); then
    bar_fill="$(printf '%*s' "$filled" '' | tr ' ' '#')"
  fi
  if (( empty > 0 )); then
    bar_empty="$(printf '%*s' "$empty" '' | tr ' ' '-')"
  fi

  printf "[%s%s] %3d%% (%d/%d)" "$bar_fill" "$bar_empty" "$pct" "$current" "$total"
}

cleanup_children() {
  for task in vs ac pf; do
    if [[ -n "${PIDS[$task]:-}" ]]; then
      local pid="${PIDS[$task]}"
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
      fi
    fi
  done
  # Reap children to avoid zombies.
  for task in vs ac pf; do
    if [[ -n "${PIDS[$task]:-}" ]]; then
      wait "${PIDS[$task]}" 2>/dev/null || true
    fi
  done
}

on_interrupt() {
  echo
  echo "[run] interrupted, stopping background tasks..."
  cleanup_children
  exit 130
}

trap on_interrupt INT TERM

for TASK in vs ac pf; do
  LOG_FILE="$LOG_DIR/${TASK}.log"
  case "$TASK" in
    vs)
      DATASET_CSV="${VS_CSV:-$DATASET_DIR/molbench-vs-900.csv}"
      ;;
    ac)
      DATASET_CSV="${AC_CSV:-$DATASET_DIR/molbench-ac-900.csv}"
      ;;
    pf)
      DATASET_CSV="${PF_CSV:-$DATASET_DIR/molbench-pf-900.csv}"
      ;;
  esac
  if [[ ! -f "$DATASET_CSV" ]]; then
    echo "[error] dataset csv not found for task=${TASK}: $DATASET_CSV" >&2
    exit 1
  fi
  ROWS="$(count_csv_rows "$DATASET_CSV")"
  if (( LIMIT > 0 && LIMIT < ROWS )); then
    ROWS="$LIMIT"
  fi
  TOTAL_EVENTS["$TASK"]=$(( ROWS * NUM_ROLLOUTS ))

  echo "[run] start task=${TASK}, log=${LOG_FILE}"
  bash "$FLOW" "$PROVIDER" "$CLAUDE_BIN" "$LIMIT" "$NUM_ROLLOUTS" "$PARALLEL_ROLLOUTS" "$TASK" "$DATASET_CSV" 1 "$MAX_WORKERS" >"$LOG_FILE" 2>&1 &
  PIDS["$TASK"]=$!
  LOGS["$TASK"]="$LOG_FILE"
  DONE_COUNTS["$TASK"]=0
  LOG_SIZES["$TASK"]=0
  LAST_AGE["$TASK"]="-"
  TAIL_STATE["$TASK"]="RUN"
done

print_progress_once() {
  local task="$1"
  local pid="${PIDS[$task]}"
  local log_file="${LOGS[$task]}"
  local total="${TOTAL_EVENTS[$task]}"
  local done="${DONE_COUNTS[$task]:-0}"
  local status="RUN"
  local tail_state="${TAIL_STATE[$task]:-RUN}"
  local age="${LAST_AGE[$task]:--}"
  local now_ts
  now_ts="$(date +%s)"

  if [[ -f "$log_file" ]]; then
    local size=0
    local old_size=0
    local append_from=1
    local chunk=""
    local inc=0
    local mtime=0
    local last_line=""

    size="$(stat -c%s "$log_file" 2>/dev/null || echo 0)"
    mtime="$(stat -c%Y "$log_file" 2>/dev/null || echo 0)"
    if (( mtime > 0 )); then
      age=$(( now_ts - mtime ))
    else
      age="-"
    fi

    old_size="${LOG_SIZES[$task]:-0}"
    if (( size > old_size )); then
      append_from=$(( old_size + 1 ))
      chunk="$(tail -c +"$append_from" "$log_file" 2>/dev/null || true)"
      inc="$(printf "%s" "$chunk" | grep -c "^\[run\] task=${task} row=" || true)"
      done=$(( done + inc ))

      if [[ -n "$chunk" ]]; then
        last_line="$(printf "%s" "$chunk" | tail -n 1)"
        if [[ "$last_line" == *"[done] full pipeline completed"* || "$last_line" == RESULTS_DIR=* ]]; then
          tail_state="OK"
        elif [[ "$last_line" == *"[error]"* || "$last_line" == *"Traceback (most recent call last)"* ]]; then
          tail_state="ERR"
        elif [[ "$last_line" == *"[run]"* || "$last_line" == *"MolBench-"* ]]; then
          tail_state="RUN"
        fi
      fi

      LOG_SIZES["$task"]="$size"
    fi
  fi
  if (( done > total )); then
    done="$total"
  fi

  if [[ -n "${EXIT_CODES[$task]+x}" ]]; then
    if [[ "${EXIT_CODES[$task]}" -eq 0 ]]; then
      status="OK "
    else
      status="ERR"
    fi
  elif ! kill -0 "$pid" 2>/dev/null; then
    if wait "$pid"; then
      EXIT_CODES["$task"]=0
      status="OK "
      done="$total"
      tail_state="OK"
    else
      EXIT_CODES["$task"]=$?
      status="ERR"
      tail_state="ERR"
    fi
  fi

  DONE_COUNTS["$task"]="$done"
  LAST_AGE["$task"]="$age"
  TAIL_STATE["$task"]="$tail_state"

  printf "%s | %s | %s | age=%ss tail=%s\n" "$task" "$(make_bar "$done" "$total")" "$status" "$age" "$tail_state"
}

printed=0
while true; do
  if (( printed == 1 )); then
    printf '\033[3F'
  fi
  print_progress_once vs
  print_progress_once ac
  print_progress_once pf
  printed=1

  if [[ -n "${EXIT_CODES[vs]+x}" && -n "${EXIT_CODES[ac]+x}" && -n "${EXIT_CODES[pf]+x}" ]]; then
    break
  fi
  sleep 1
done

FAIL=0
for TASK in vs ac pf; do
  if [[ "${EXIT_CODES[$TASK]}" -eq 0 ]]; then
    echo "[ok] task=${TASK} completed"
  else
    echo "[error] task=${TASK} failed (see $LOG_DIR/${TASK}.log)" >&2
    FAIL=1
  fi
done

echo "LOG_DIR=$LOG_DIR"
exit "$FAIL"
