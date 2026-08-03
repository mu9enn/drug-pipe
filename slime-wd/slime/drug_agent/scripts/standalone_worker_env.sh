#!/usr/bin/env bash
# Fallback environment for an rjob whose GPFS mount is hidden below /root.
# Source this only after deploying Slime and Megatron-LM under RUNTIME_ROOT.

export RUNTIME_ROOT=${RUNTIME_ROOT:-/home/sunxiangyu/runtime}
export GROUP_SPACE="$RUNTIME_ROOT"
export WD="$RUNTIME_ROOT"
export SLIME="$RUNTIME_ROOT/slime"
export MEGATRON_LM_PATH="$RUNTIME_ROOT/Megatron-LM"
export DATA="$RUNTIME_ROOT/data"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-$RUNTIME_ROOT/outputs}"
export DRUG_AGENT_DATA_ROOT="${DRUG_AGENT_DATA_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_data}"
export DRUG_AGENT_RUNS_ROOT="${DRUG_AGENT_RUNS_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_runs}"
export PYTHON_CPU_FIX_DIR="$RUNTIME_ROOT/python_cpu_fix"

export REAL_CPU=${REAL_CPU:-$(nproc)}
export PYTHON_CPU_COUNT="$REAL_CPU"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="/usr/local/nvidia/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$MEGATRON_LM_PATH:$SLIME:${PYTHONPATH:-}"
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

for required in "$SLIME/train.py" "$MEGATRON_LM_PATH/megatron/core"; do
  if [[ ! -e "$required" ]]; then
    echo "[standalone-worker-env] missing runtime component: $required" >&2
    return 2 2>/dev/null || exit 2
  fi
done

mkdir -p "$DATA" "$OUTPUTS_ROOT" "$DRUG_AGENT_DATA_ROOT" "$DRUG_AGENT_RUNS_ROOT"
echo "[standalone-worker-env] RUNTIME_ROOT=$RUNTIME_ROOT"
echo "[standalone-worker-env] SLIME=$SLIME"
echo "[standalone-worker-env] DATA=$DATA"
