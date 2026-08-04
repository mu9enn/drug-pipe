#!/usr/bin/env bash
# Read-only health snapshot for a Slime serial run. Run inside the worker, or
# stream this file over SSH as documented in SKILL.md.
set -euo pipefail

RUN_ROOT=${1:?Usage: health_snapshot.sh RUN_ROOT [RAY_JOB_ID]}
RAY_JOB_ID=${2:-}

if [[ ! -d "$RUN_ROOT" ]]; then
  echo "FAIL: run root does not exist: $RUN_ROOT" >&2
  exit 2
fi

echo "timestamp=$(date --iso-8601=seconds)"
echo "run_root=$RUN_ROOT"

echo "gpu_snapshot:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
    --format=csv,noheader
else
  echo "nvidia-smi=missing"
fi

echo "host_memory:"
free -h || true
echo "filesystem:"
df -h "$RUN_ROOT" | tail -1 || true

echo "training_processes:"
ps -eo pid,ppid,etime,stat,rss,args --sort=-rss \
  | grep -E 'run_.*(serial|sft|toolrl|gad)|ray::JobSupervisor|ray::SGLangEngine|MegatronTrainRayActor|RolloutManager|drug_agent.gad.service' \
  | grep -vE 'grep -E|health_snapshot.sh' \
  | head -40 || true

if [[ -n "$RAY_JOB_ID" ]] && command -v ray >/dev/null 2>&1; then
  echo "ray_job:"
  ray job status "$RAY_JOB_ID" 2>&1 | tail -20 || true
fi

echo "stage_markers:"
find "$RUN_ROOT" -maxdepth 2 -type f \
  \( -name '*.complete' -o -name '*_DONE' -o -name 'PASS' -o -name 'ALL_COMPLETE' \
     -o -name 'WAITING_FOR_EXTERNAL_RL' -o -name 'latest_checkpointed_iteration.txt' \) \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort || true

LATEST_LOG=$(find "$RUN_ROOT" -maxdepth 3 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null \
  | sort -n | tail -1 | cut -d' ' -f2- || true)
if [[ -z "$LATEST_LOG" ]]; then
  echo "latest_log=missing"
  exit 3
fi

NOW=$(date +%s)
MTIME=$(stat -c %Y "$LATEST_LOG")
echo "latest_log=$LATEST_LOG"
echo "latest_log_age_seconds=$((NOW - MTIME))"
echo "latest_metrics:"
grep -aE 'model.py:[0-9]+ - step [0-9]+:|perf [0-9]+:|rollout.py:[0-9]+ - perf [0-9]+:' "$LATEST_LOG" \
  | tail -20 | cut -c1-1600 || true

START_LINE=$(grep -an 'Running entrypoint for job' "$LATEST_LOG" | tail -1 | cut -d: -f1 || true)
RUNTIME_TMP=$(mktemp)
trap 'rm -f "$RUNTIME_TMP"' EXIT
if [[ -n "$START_LINE" ]]; then
  tail -n +"$START_LINE" "$LATEST_LOG"
else
  tail -n 4000 "$LATEST_LOG"
fi | grep -avE 'sglang_rollout.py:.* - (First rollout sample|Finish rollout):' > "$RUNTIME_TMP" || true

ERROR_PATTERN="traceback|runtimeerror|outofmemory|cuda out of memory|actorunavailable|sigkill|killed|nccl[[:space:]:_-]*(error|timeout|abort|watchdog)|((train/)?loss|(train/)?grad_norm)['\"]?[[:space:]]*[:=][[:space:]]*(nan|[-+]?inf)([^[:alnum:]_]|$)"
echo "runtime_errors:"
ERROR_LINES=$(grep -aiE "$ERROR_PATTERN" "$RUNTIME_TMP" || true)
if [[ -n "$ERROR_LINES" ]]; then
  printf '%s\n' "$ERROR_LINES" | tail -40 | cut -c1-1600
fi
ERROR_COUNT=$(printf '%s\n' "$ERROR_LINES" | sed '/^[[:space:]]*$/d' | wc -l)
echo "runtime_error_count=$ERROR_COUNT"

RECENT_STEPS=$(grep -aE 'model.py:[0-9]+ - step [0-9]+:' "$RUNTIME_TMP" | tail -6 || true)
STEP_COUNT=$(printf '%s\n' "$RECENT_STEPS" | sed '/^[[:space:]]*$/d' | wc -l)
ZERO_GRAD_COUNT=$(printf '%s\n' "$RECENT_STEPS" | grep -c "'train/grad_norm': 0.0" || true)
echo "recent_step_count=$STEP_COUNT"
echo "recent_zero_grad_count=$ZERO_GRAD_COUNT"

if (( ERROR_COUNT > 0 )); then
  echo "health=NEEDS_DIAGNOSIS"
  exit 1
fi
if (( STEP_COUNT >= 6 && ZERO_GRAD_COUNT == STEP_COUNT )); then
  echo "health=NEEDS_SIGNAL_REVIEW"
  exit 4
fi
echo "health=NO_RUNTIME_ERROR_DETECTED"
