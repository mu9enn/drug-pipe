#!/usr/bin/env bash
# Serve the official Qwen3.5-122B-A10B-FP8 checkpoint on a dedicated 8xH200
# rollout worker.  The trainer connects with Slime --rollout-external and
# updates these weights after every optimizer step.
set -euo pipefail

SLIME_ENV=${SLIME_ENV:-/root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh}
if [[ ! -f "$SLIME_ENV" ]]; then
  SLIME_ENV=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi
source "$SLIME_ENV"
cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh

MODEL_PROFILE=qwen35-122b-8xh200
source drug_agent/scripts/qwen3_large_profile.sh

EXTERNAL_ROLLOUT_HOST=${EXTERNAL_ROLLOUT_HOST:?Set EXTERNAL_ROLLOUT_HOST to the Ray-reachable pod IP of this worker}
EXTERNAL_ROLLOUT_PORT=${EXTERNAL_ROLLOUT_PORT:-10090}
EXTERNAL_ROLLOUT_MODEL=${EXTERNAL_ROLLOUT_MODEL:-$ROLLOUT_HF_CHECKPOINT}
EXTERNAL_ROLLOUT_CONTEXT_LENGTH=${EXTERNAL_ROLLOUT_CONTEXT_LENGTH:-$ROLLOUT_MAX_CONTEXT_LEN}
EXTERNAL_ROLLOUT_MEM_FRACTION=${EXTERNAL_ROLLOUT_MEM_FRACTION:-$SGLANG_MEM_FRACTION_STATIC}
EXTERNAL_ROLLOUT_KV_CACHE_DTYPE=${EXTERNAL_ROLLOUT_KV_CACHE_DTYPE:-$SGLANG_KV_CACHE_DTYPE}

[[ -d "$EXTERNAL_ROLLOUT_MODEL" ]] || {
  echo "External rollout model directory is missing: $EXTERNAL_ROLLOUT_MODEL" >&2
  exit 2
}
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
H200_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c H200 || true)
if (( GPU_COUNT != 8 || H200_COUNT != 8 )); then
  echo "Qwen122 external rollout requires exactly 8 visible H200 GPUs; found GPUs=$GPU_COUNT H200=$H200_COUNT" >&2
  exit 2
fi
if [[ ${ALLOW_BUSY_EXTERNAL_ROLLOUT_GPUS:-0} != 1 ]]; then
  BUSY_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  [[ -z "$BUSY_PIDS" ]] || {
    echo "External rollout GPUs already have compute processes: $BUSY_PIDS" >&2
    exit 2
  }
fi
if ss -ltnH | awk '{print $4}' | grep -qE "(^|:)$EXTERNAL_ROLLOUT_PORT$"; then
  echo "External rollout port is already listening: $EXTERNAL_ROLLOUT_PORT" >&2
  exit 2
fi

export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=${SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK:-1}
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=${SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE:-false}
# SGLang enters a TorchMemorySaver region during model load even for an
# external, non-offloaded server.  The trainer uses expandable segments for
# its separate Megatron actor, but the native rollout process must not inherit
# that allocator mode.
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

echo "[qwen122-external-rollout] model=$EXTERNAL_ROLLOUT_MODEL host=$EXTERNAL_ROLLOUT_HOST port=$EXTERNAL_ROLLOUT_PORT tp=8 context=$EXTERNAL_ROLLOUT_CONTEXT_LENGTH mem_fraction=$EXTERNAL_ROLLOUT_MEM_FRACTION kv=$EXTERNAL_ROLLOUT_KV_CACHE_DTYPE"
exec python3 -m sglang.launch_server \
  --model-path "$EXTERNAL_ROLLOUT_MODEL" \
  --host "$EXTERNAL_ROLLOUT_HOST" \
  --port "$EXTERNAL_ROLLOUT_PORT" \
  --tp-size 8 \
  --trust-remote-code \
  --context-length "$EXTERNAL_ROLLOUT_CONTEXT_LENGTH" \
  --mem-fraction-static "$EXTERNAL_ROLLOUT_MEM_FRACTION" \
  --kv-cache-dtype "$EXTERNAL_ROLLOUT_KV_CACHE_DTYPE" \
  --disable-custom-all-reduce \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --skip-server-warmup \
  --enable-draft-weights-cpu-backup \
  --enable-metrics
