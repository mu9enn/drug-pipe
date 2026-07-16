#!/bin/bash

set -euo pipefail

if [ -f /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh ]; then
  source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
else
  source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi

cd "$SLIME"

# Conservative 4xH200 smoke profile for the current long ReAct SFT samples.
# TP=4 shards the large vocabulary logits and their temporary clone across all
# four GPUs; DP=1 then permits the smallest valid RBS=GBS=1 smoke step.
export MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-4B.sh}
export HF_CHECKPOINT=${HF_CHECKPOINT:-$DATA/Qwen3.5-4B}
export REF_LOAD=${REF_LOAD:-$DATA/Qwen3.5-4B_torch_dist}

export NUM_GPUS=${NUM_GPUS:-4}
export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-4}
export PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
export CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
export EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
export EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}

export NUM_ROLLOUT=${NUM_ROLLOUT:-1}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1}
export NUM_EPOCH=${NUM_EPOCH:-1}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-4096}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1}

# The colocated SGLang path requires TorchMemorySaver, which is incompatible
# with expandable CUDA allocator segments. The generic SFT entrypoint repeats
# this guard before Ray starts so externally inherited values cannot leak in.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
  echo "[4B SFT] Unsetting incompatible PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" >&2
  unset PYTORCH_CUDA_ALLOC_CONF
fi
if [[ "${PYTORCH_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
  echo "[4B SFT] Unsetting incompatible PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF}" >&2
  unset PYTORCH_ALLOC_CONF
fi

export DRUG_AGENT_RUNS_ROOT=${DRUG_AGENT_RUNS_ROOT:-$DRUG_AGENT_RUNS_ROOT}
export SAVE_DIR=${SAVE_DIR:-$DRUG_AGENT_RUNS_ROOT/Qwen3.5-4B_drug_sft_smoke_tp4}

if [ ! -d "$HF_CHECKPOINT" ]; then
  echo "HF_CHECKPOINT not found: $HF_CHECKPOINT" >&2
  exit 2
fi
if [ ! -d "$REF_LOAD" ]; then
  echo "REF_LOAD torch_dist checkpoint not found: $REF_LOAD" >&2
  echo "Prepare it first with: bash drug_agent/scripts/prepare_qwen3_5_4B_torch_dist.sh" >&2
  exit 2
fi

exec bash drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh
