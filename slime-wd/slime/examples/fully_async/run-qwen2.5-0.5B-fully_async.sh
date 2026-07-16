#!/bin/bash
# Tiny end-to-end fully-async GRPO example using Qwen2.5-0.5B-Instruct on the
# dapo-math-17k dataset. Designed to run on a single 4-GPU node in a few
# minutes — the same script the CI uses for ``test_qwen2.5_0.5B_fully_async_short``.
#
# Cluster notes:
# 1. In this RJob/SSH environment, Python may incorrectly see os.cpu_count() == 1
#    while nproc / cpuset actually expose many CPUs. We inject a persistent
#    sitecustomize.py from group-space to fix os.cpu_count() for all Python
#    subprocesses, including Ray workers and SGLang engines.
# 2. RJob injects CUDA/NVIDIA variables into PID 1, but SSH login shells may not
#    inherit them. We recover PATH / LD_LIBRARY_PATH / CUDA_HOME /
#    NVIDIA_VISIBLE_DEVICES / NVIDIA_DRIVER_CAPABILITIES from /proc/1/environ
#    and explicitly pass them into Ray runtime_env.

set -ex

# ---------------------------------------------------------------------------
# Clean any leftover ray/sglang.
# ---------------------------------------------------------------------------
unset RAY_ADDRESS || true
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray python 2>/dev/null || true
sleep 3

export PYTHONBUFFERED=16

# ---------------------------------------------------------------------------
# Locate persistent workspace paths.
# Supports both rjob and rlaunch style mounts.
# ---------------------------------------------------------------------------
for cand in \
  "${GROUP_SPACE:-}" \
  "$HOME/slime_sxy/group-space/sunxiangyu" \
  "/root/slime_sxy/group-space/sunxiangyu" \
  "/home/sunxiangyu/slime_sxy/group-space/sunxiangyu"
do
  if [ -n "$cand" ] && [ -d "$cand/slime_wd/slime" ]; then
    export GROUP_SPACE="$cand"
    break
  fi
done

if [ -z "${GROUP_SPACE:-}" ]; then
  echo "[ERROR] Cannot find group-space/sunxiangyu with slime_wd/slime"
  exit 1
fi

export WD="${WD:-$GROUP_SPACE/slime_wd}"
export SLIME="${SLIME:-$WD/slime}"
export DATA="${DATA:-$WD/data}"

# ---------------------------------------------------------------------------
# Persistent Python CPU-count fix.
# Expected file:
#   $GROUP_SPACE/slime_env/python_cpu_fix/sitecustomize.py
# ---------------------------------------------------------------------------
export PYTHON_CPU_FIX_DIR="${PYTHON_CPU_FIX_DIR:-$GROUP_SPACE/slime_env/python_cpu_fix}"

if [ ! -f "$PYTHON_CPU_FIX_DIR/sitecustomize.py" ]; then
  echo "[ERROR] Missing $PYTHON_CPU_FIX_DIR/sitecustomize.py"
  echo "Create it once under group-space before running this script."
  exit 1
fi

export REAL_CPU="${REAL_CPU:-$(nproc)}"
export PYTHON_CPU_COUNT="$REAL_CPU"

# ---------------------------------------------------------------------------
# Recover CUDA/NVIDIA environment from PID 1.
# RJob sshd login shell may not inherit these variables.
# ---------------------------------------------------------------------------
if [ -r /proc/1/environ ]; then
  _P1_PATH="$(tr '\0' '\n' < /proc/1/environ | grep '^PATH=' | cut -d= -f2- || true)"
  _P1_LD_LIBRARY_PATH="$(tr '\0' '\n' < /proc/1/environ | grep '^LD_LIBRARY_PATH=' | cut -d= -f2- || true)"
  _P1_CUDA_HOME="$(tr '\0' '\n' < /proc/1/environ | grep '^CUDA_HOME=' | cut -d= -f2- || true)"
  _P1_NVIDIA_VISIBLE_DEVICES="$(tr '\0' '\n' < /proc/1/environ | grep '^NVIDIA_VISIBLE_DEVICES=' | cut -d= -f2- || true)"
  _P1_NVIDIA_DRIVER_CAPABILITIES="$(tr '\0' '\n' < /proc/1/environ | grep '^NVIDIA_DRIVER_CAPABILITIES=' | cut -d= -f2- || true)"

  if [ -n "$_P1_PATH" ]; then
    export PATH="$_P1_PATH:$PATH"
  fi

  if [ -n "$_P1_LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH="$_P1_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
  fi

  if [ -n "$_P1_CUDA_HOME" ]; then
    export CUDA_HOME="$_P1_CUDA_HOME"
  fi

  if [ -n "$_P1_NVIDIA_VISIBLE_DEVICES" ]; then
    export NVIDIA_VISIBLE_DEVICES="$_P1_NVIDIA_VISIBLE_DEVICES"
  fi

  if [ -n "$_P1_NVIDIA_DRIVER_CAPABILITIES" ]; then
    export NVIDIA_DRIVER_CAPABILITIES="$_P1_NVIDIA_DRIVER_CAPABILITIES"
  fi
fi

# Fallbacks.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"

# CPU fix must be first so every Python process imports sitecustomize.py.
export PYTHONPATH="$PYTHON_CPU_FIX_DIR:/root/Megatron-LM:$SLIME:${PYTHONPATH:-}"

# Cluster defaults.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# ---------------------------------------------------------------------------
# Environment checks.
# ---------------------------------------------------------------------------
python - <<'PY'
import os, multiprocessing
print("[cpu-check] os.cpu_count():", os.cpu_count())
print("[cpu-check] multiprocessing.cpu_count():", multiprocessing.cpu_count())
try:
    print("[cpu-check] sched affinity:", len(os.sched_getaffinity(0)))
except Exception as e:
    print("[cpu-check] no sched_getaffinity:", e)

try:
    import torch
    print("[cuda-check] torch:", torch.__version__)
    print("[cuda-check] torch.version.cuda:", torch.version.cuda)
    print("[cuda-check] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("[cuda-check] NVIDIA_VISIBLE_DEVICES:", os.environ.get("NVIDIA_VISIBLE_DEVICES"))
    print("[cuda-check] NVIDIA_DRIVER_CAPABILITIES:", os.environ.get("NVIDIA_DRIVER_CAPABILITIES"))
    print("[cuda-check] LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
    print("[cuda-check] torch.cuda.is_available():", torch.cuda.is_available())
    print("[cuda-check] torch.cuda.device_count():", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print("[cuda-check]", i, torch.cuda.get_device_name(i))
except Exception as e:
    print("[cuda-check] failed:", repr(e))
    raise
PY

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-0.5B.sh"

MODEL_DIR=${MODEL_DIR:-$DATA/Qwen2.5-0.5B-Instruct}
DATA_PATH=${DATA_PATH:-$DATA/dapo-math-17k/dapo-math-17k.jsonl}

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --ref-load "${MODEL_DIR}_torch_dist"
   --save /tmp/slime_fully_async_demo/
   --save-interval 9999
)

ROLLOUT_ARGS=(
   # ↓↓↓ This is the only knob you need to flip to go fully-async ↓↓↓
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async

   --prompt-data "${DATA_PATH}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout 3
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len 1024
   --rollout-temperature 1

   --global-batch-size 32
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.55
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ---------------------------------------------------------------------------
# Launch Ray head.
# ---------------------------------------------------------------------------
NUM_GPUS=${NUM_GPUS:-4}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

ray start --head \
   --node-ip-address "${MASTER_ADDR}" \
   --num-gpus "${NUM_GPUS}" \
   --num-cpus "${REAL_CPU}" \
   --disable-usage-stats \
   --dashboard-host=0.0.0.0 \
   --dashboard-port=8265

# Explicitly pass CPU fix + CUDA/NVIDIA environment to Ray workers and SGLang.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHON_CPU_FIX_DIR}:/root/Megatron-LM/:${SLIME}:${SCRIPT_DIR}:${PYTHONPATH:-}\",
    \"PYTHON_CPU_COUNT\": \"${REAL_CPU}\",
    \"PATH\": \"${PATH}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"CUDA_HOME\": \"${CUDA_HOME:-/usr/local/cuda}\",
    \"NVIDIA_VISIBLE_DEVICES\": \"${NVIDIA_VISIBLE_DEVICES:-all}\",
    \"NVIDIA_DRIVER_CAPABILITIES\": \"${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_IB_DISABLE\": \"${NCCL_IB_DISABLE:-1}\"
  }
}"

# fully-async splits actor / rollout onto disjoint GPUs (no colocation).
ACTOR_GPUS=${ACTOR_GPUS:-1}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}

if [ "${ROLLOUT_GPUS}" -le 0 ]; then
  echo "[ERROR] ROLLOUT_GPUS must be > 0, got ${ROLLOUT_GPUS}."
  echo "For fully_async, use at least NUM_GPUS=2 with ACTOR_GPUS=1 ROLLOUT_GPUS=1."
  exit 1
fi

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   ${MODEL_ARGS[@]} \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
