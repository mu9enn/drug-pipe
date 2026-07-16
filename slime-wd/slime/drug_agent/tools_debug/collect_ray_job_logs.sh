#!/bin/bash

set -euo pipefail

JOB_ID=${1:-}
if [ -z "$JOB_ID" ]; then
  echo "usage: $0 raysubmit_<job_id> [output_dir]" >&2
  exit 2
fi

OUTPUT_DIR=${2:-/tmp}
mkdir -p "$OUTPUT_DIR"

RAY_ADDRESS=${RAY_ADDRESS:-http://127.0.0.1:8265}
FULL_LOG="$OUTPUT_DIR/${JOB_ID}.full.log"
ERROR_LOG="$OUTPUT_DIR/${JOB_ID}.first_error.log"
WORKER_GREP="$OUTPUT_DIR/${JOB_ID}.ray_worker_error_grep.log"
RECENT_LOGS="$OUTPUT_DIR/${JOB_ID}.recent_ray_logs.txt"

ray job logs "$JOB_ID" --address="$RAY_ADDRESS" > "$FULL_LOG" 2>&1 || true

LINE=$(grep -nEi 'traceback|runtimeerror|assertionerror|outofmemoryerror|raytaskerror|exception|sigkill|sigterm|killed|nccl|failed' "$FULL_LOG" | head -1 | cut -d: -f1 || true)
if [ -n "$LINE" ]; then
  START=$((LINE > 40 ? LINE - 40 : 1))
  END=$((LINE + 160))
  sed -n "${START},${END}p" "$FULL_LOG" > "$ERROR_LOG" || true
else
  : > "$ERROR_LOG"
fi

grep -RniE 'traceback|runtimeerror|assertionerror|outofmemoryerror|raytaskerror|nccl|sigkill|sigterm|killed|failed' \
  /tmp/ray/session_latest/logs 2>/dev/null | head -100 > "$WORKER_GREP" || true

find /tmp/ray/session_latest/logs -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -30 > "$RECENT_LOGS" || true

echo "full_log=$FULL_LOG"
echo "first_error_log=$ERROR_LOG"
echo "worker_error_grep=$WORKER_GREP"
echo "recent_ray_logs=$RECENT_LOGS"

if [ -s "$ERROR_LOG" ]; then
  echo "---- first error context ----"
  cat "$ERROR_LOG"
elif [ -s "$WORKER_GREP" ]; then
  echo "---- worker log grep ----"
  cat "$WORKER_GREP"
else
  echo "No traceback-like lines found. Inspect $FULL_LOG and $RECENT_LOGS."
fi
