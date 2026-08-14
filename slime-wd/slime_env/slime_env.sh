#!/usr/bin/env bash

# Locate group-space.
for cand in \
  "$HOME/slime_sxy/group-space/sunxiangyu" \
  "/root/slime_sxy/group-space/sunxiangyu" \
  "/home/sunxiangyu/slime_sxy/group-space/sunxiangyu"
do
  if [ -d "$cand/drug-pipe/slime-wd/slime" ]; then
    export GROUP_SPACE="$cand"
    _SLIME_WD="$cand/drug-pipe/slime-wd"
    break
  fi
  # Compatibility for machines that still use the pre-migration layout.
  if [ -d "$cand/slime_wd/slime" ]; then
    export GROUP_SPACE="$cand"
    _SLIME_WD="$cand/slime_wd"
    break
  fi
done

if [ -z "${GROUP_SPACE:-}" ] || [ -z "${_SLIME_WD:-}" ]; then
  echo "[slime_env] ERROR: cannot find drug-pipe/slime-wd/slime or legacy slime_wd/slime"
  return 1 2>/dev/null || exit 1
fi

export WD="$_SLIME_WD"
export SLIME="$WD/slime"
export DATA="$WD/data"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-$WD/outputs}"
export DRUG_AGENT_DATA_ROOT="${DRUG_AGENT_DATA_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_data}"
export DRUG_AGENT_RUNS_ROOT="${DRUG_AGENT_RUNS_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_runs}"
unset _SLIME_WD

export PYTHON_CPU_FIX_DIR="$GROUP_SPACE/slime_env/python_cpu_fix"

# CPU fix.
export REAL_CPU="${REAL_CPU:-$(nproc)}"
export PYTHON_CPU_COUNT="$REAL_CPU"

# Recover CUDA/NVIDIA environment from container init process.
# RJob sshd login shell may not inherit these variables, but PID 1 usually has them.
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

# Python CPU fix must be first in PYTHONPATH.
export PYTHONPATH="$PYTHON_CPU_FIX_DIR:/root/Megatron-LM:$SLIME:${PYTHONPATH:-}"

# Cluster defaults.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

echo "[slime_env] GROUP_SPACE=$GROUP_SPACE"
echo "[slime_env] WD=$WD"
echo "[slime_env] SLIME=$SLIME"
echo "[slime_env] DATA=$DATA"
echo "[slime_env] OUTPUTS_ROOT=$OUTPUTS_ROOT"
echo "[slime_env] DRUG_AGENT_DATA_ROOT=$DRUG_AGENT_DATA_ROOT"
echo "[slime_env] DRUG_AGENT_RUNS_ROOT=$DRUG_AGENT_RUNS_ROOT"
echo "[slime_env] REAL_CPU=$REAL_CPU"
echo "[slime_env] PYTHON_CPU_FIX_DIR=$PYTHON_CPU_FIX_DIR"
echo "[slime_env] CUDA_HOME=${CUDA_HOME:-}"
echo "[slime_env] NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-}"
echo "[slime_env] NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-}"
echo "[slime_env] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"

python - <<'PY'
import os, multiprocessing
print("[slime_env] os.cpu_count():", os.cpu_count())
print("[slime_env] multiprocessing.cpu_count():", multiprocessing.cpu_count())
try:
    print("[slime_env] sched affinity:", len(os.sched_getaffinity(0)))
except Exception as e:
    print("[slime_env] no sched_getaffinity:", e)

try:
    import torch
    print("[slime_env] torch:", torch.__version__)
    print("[slime_env] torch.version.cuda:", torch.version.cuda)
    print("[slime_env] torch.cuda.is_available():", torch.cuda.is_available())
    print("[slime_env] torch.cuda.device_count():", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print("[slime_env] cuda", i, torch.cuda.get_device_name(i))
except Exception as e:
    print("[slime_env] torch cuda check failed:", repr(e))
PY
