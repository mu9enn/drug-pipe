#!/usr/bin/env bash
# Append a compact health snapshot once per hour. This watchdog reports; it
# deliberately does not kill/restart training without diagnosing the error.
set -euo pipefail
RUN_ROOT=${1:?Usage: monitor_qwen3_large_serial.sh RUN_ROOT [interval_seconds]}
INTERVAL=${2:-3600}
while [[ ! -d "$RUN_ROOT" ]]; do
  sleep 2
done
mkdir -p "$RUN_ROOT/monitor"

# A stage log is intentionally reused across retries. Return only the current
# Ray submission segment so repaired failures from older submissions cannot
# poison the live health decision even when the new 122B step emits few lines.
live_log_segment() {
  local log=$1 start_line
  start_line=$(grep -an 'Running entrypoint for job' "$log" | tail -1 | cut -d: -f1 || true)
  if [[ -n "$start_line" ]]; then
    tail -n +"$start_line" "$log"
  else
    tail -400 "$log"
  fi
}

# Rollout samples are logged as one (potentially very large) line and may quote
# failures from the supervised trajectory itself, including strings such as
# "Traceback" or "RuntimeError".  Those are training data, not runtime errors.
# Exclude only the two full-sample log records from fatal-error detection while
# leaving optimizer metrics and all ordinary SGLang/Ray diagnostics intact.
live_runtime_log_segment() {
  local log=$1
  live_log_segment "$log" | grep -avE 'sglang_rollout.py:.* - (First rollout sample|Finish rollout):' || true
}

# Keep metric matching assignment-shaped.  A broad "loss ... nan" expression
# also matches SGLang's very long server_args line because it contains both
# loss-related option names and `enable_nan_detection=False`.
ERROR_PATTERN="traceback|runtimeerror|outofmemory|cuda out of memory|actorunavailable|sigkill|killed|nccl[[:space:]:_-]*(error|timeout|abort|watchdog)|nccl[^[:space:]]*[[:space:]]+(error|timeout|abort|watchdog)|((train/)?loss|(train/)?grad_norm)['\"]?[[:space:]]*[:=][[:space:]]*(nan|[-+]?inf)([^[:alnum:]_]|$)"

while true; do
  stamp=$(date +%Y%m%d_%H%M%S)
  out=$RUN_ROOT/monitor/$stamp.log
  stale_training=0
  signal_warning=0
  {
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "run_root=$RUN_ROOT"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
      --format=csv,noheader
    free -h
    df -h "$RUN_ROOT"
    # With pipefail, `head` intentionally closing the pipe can make `ps`
    # return SIGPIPE/141 and terminate the whole hourly watchdog after its
    # first snapshot.  The process table is diagnostic-only, so accept that
    # normal short-read status.
    ps -eo pid,etime,stat,rss,cmd --sort=-rss | head -35 | cut -c1-1200 || true
    gpu_process_count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | sort -u | wc -l)
    echo "gpu_process_count=$gpu_process_count"
    if [[ -f "$RUN_ROOT/serial_status.log" ]]; then
      tail -30 "$RUN_ROOT/serial_status.log"
    fi
    latest_log=$(find "$RUN_ROOT/logs" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
    if [[ -n "$latest_log" ]]; then
      echo "latest_log=$latest_log"
      log_mtime=$(stat -c %Y "$latest_log")
      log_age_seconds=$(($(date +%s) - log_mtime))
      echo "latest_log_age_seconds=$log_age_seconds"
      tail -120 "$latest_log" | cut -c1-1200
      live_runtime_log_segment "$latest_log" | grep -niE "$ERROR_PATTERN" | tail -40 | cut -c1-1200 || true

      # A live training stage should not go silent for half an hour.  This is
      # deliberately much longer than measured 122B microbatches/optimizer
      # steps, while still detecting a wedged collective before the next
      # hourly visit loses another full interval.
      if [[ -f "$RUN_ROOT/serial_status.log" ]] && \
         tail -1 "$RUN_ROOT/serial_status.log" | grep -q 'START ' && \
         (( log_age_seconds > 1800 )); then
        stale_training=1
        echo "health_error=active_stage_log_stale"
      fi

      # GRPO can legitimately produce an occasional all-equal reward group,
      # but six consecutive zero-gradient updates mean GPUs are busy without
      # a learning signal.  Flag this separately for reward/sampling review;
      # do not kill or mutate a healthy process automatically.
      recent_steps=$(live_log_segment "$latest_log" | grep -aE 'model.py:.* - step [0-9]+:' | tail -6 || true)
      recent_step_count=$(printf '%s\n' "$recent_steps" | sed '/^[[:space:]]*$/d' | wc -l)
      recent_zero_grad_count=$(printf '%s\n' "$recent_steps" | grep -c "'train/grad_norm': 0.0" || true)
      echo "recent_step_count=$recent_step_count"
      echo "recent_zero_grad_count=$recent_zero_grad_count"
      if (( recent_step_count >= 6 && recent_zero_grad_count == recent_step_count )); then
        signal_warning=1
        echo "health_warning=consecutive_zero_gradient_updates"
      fi
    fi
    echo "checkpoint_pointers:"
    find "$RUN_ROOT" -mindepth 2 -maxdepth 3 -name latest_checkpointed_iteration.txt \
      -type f -print -exec tail -1 {} \; 2>/dev/null | tail -30 || true
    if curl -fsS http://127.0.0.1:8265/api/version >/dev/null 2>&1; then
      ray job list --address=http://127.0.0.1:8265 2>&1 | tail -80 || true
    else
      echo "ray_dashboard=not_ready"
    fi
  } > "$out" 2>&1
  ln -sfn "$out" "$RUN_ROOT/monitor/latest.log"

  if { [[ -f "$RUN_ROOT/serial_status.log" ]] && tail -1 "$RUN_ROOT/serial_status.log" | grep -qi 'FAILED'; } || \
     { [[ -n "${latest_log:-}" ]] && live_runtime_log_segment "$latest_log" | grep -qiE "$ERROR_PATTERN"; } || \
     (( stale_training == 1 )); then
    touch "$RUN_ROOT/monitor/NEEDS_DIAGNOSIS"
  else
    rm -f "$RUN_ROOT/monitor/NEEDS_DIAGNOSIS"
  fi
  if (( signal_warning == 1 )); then
    touch "$RUN_ROOT/monitor/NEEDS_SIGNAL_REVIEW"
  else
    rm -f "$RUN_ROOT/monitor/NEEDS_SIGNAL_REVIEW"
  fi
  sleep "$INTERVAL"
done
