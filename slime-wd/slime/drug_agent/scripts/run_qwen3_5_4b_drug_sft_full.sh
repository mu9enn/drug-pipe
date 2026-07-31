#!/bin/bash

set -euo pipefail

if [ -f /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh ]; then
  source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
else
  source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi

cd "$SLIME"

# Full one-epoch Qwen3.5-4B ReAct SFT profile. TP=4 retains the memory-safe
# topology validated with the long ReAct trajectories. In train-only SFT,
# RBS controls how often Slime crosses the rollout/train boundary, while GBS
# controls optimizer updates. The default therefore loads one complete epoch
# per rollout (RBS=dataset size) but keeps GBS=1. This preserves one optimizer
# update per record without paying model sleep/wake overhead for every record.
export MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-4B.sh}
export HF_CHECKPOINT=${HF_CHECKPOINT:-$DATA/Qwen3.5-4B}
export REF_LOAD=${REF_LOAD:-$DATA/Qwen3.5-4B_torch_dist}
export PROMPT_DATA=${PROMPT_DATA:-$DRUG_AGENT_DATA_ROOT/react_trajectories.jsonl}

export NUM_GPUS=${NUM_GPUS:-4}
export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-4}
export PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
export CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
export EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
export EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}

export SFT_EPOCH_ONLY=1
export NUM_EPOCH=${NUM_EPOCH:-1}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
export RECOMPUTE_FULL=${RECOMPUTE_FULL:-1}
export RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}

# This is an offline SFT run: no SGLang rollout model is instantiated, so
# colocating/offloading the actor only creates repeated CPU<->GPU transfers.
export SFT_DEBUG_TRAIN_ONLY=${SFT_DEBUG_TRAIN_ONLY:-1}
export SFT_DISABLE_OFFLOAD=${SFT_DISABLE_OFFLOAD:-1}

export LR=${LR:-1e-5}
export MIN_LR=${MIN_LR:-1e-6}
export LR_WARMUP_FRACTION=${LR_WARMUP_FRACTION:-0.1}
export SAVE_INTERVAL=${SAVE_INTERVAL:-10}

export DRUG_AGENT_RUNS_ROOT=${DRUG_AGENT_RUNS_ROOT:-$DRUG_AGENT_RUNS_ROOT}
RUN_NAME=${RUN_NAME:-Qwen3.5-4B_drug_sft_full_$(date +%Y%m%d_%H%M%S)}
export SAVE_DIR=${SAVE_DIR:-$DRUG_AGENT_RUNS_ROOT/$RUN_NAME}
if [ -n "${RESUME_DIR:-}" ]; then
  if [ ! -f "$RESUME_DIR/latest_checkpointed_iteration.txt" ]; then
    echo "RESUME_DIR is not a valid slime checkpoint directory: $RESUME_DIR" >&2
    exit 2
  fi
  export SAVE_DIR="$RESUME_DIR"
  export LOAD="$RESUME_DIR"
fi
export RAY_SUBMIT_LOG=${RAY_SUBMIT_LOG:-$SAVE_DIR/ray_submit.log}
mkdir -p "$SAVE_DIR"

if [ ! -f "$PROMPT_DATA" ]; then
  echo "PROMPT_DATA not found: $PROMPT_DATA" >&2
  exit 2
fi
if [ ! -d "$HF_CHECKPOINT" ]; then
  echo "HF_CHECKPOINT not found: $HF_CHECKPOINT" >&2
  exit 2
fi
if [ ! -d "$REF_LOAD" ]; then
  echo "REF_LOAD torch_dist checkpoint not found: $REF_LOAD" >&2
  echo "Prepare it first with: bash drug_agent/scripts/prepare_qwen3_5_4B_torch_dist.sh" >&2
  exit 2
fi

DATASET_SIZE=$(wc -l < "$PROMPT_DATA")
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-$DATASET_SIZE}
if [ "$ROLLOUT_BATCH_SIZE" -lt "$GLOBAL_BATCH_SIZE" ]; then
  echo "ROLLOUT_BATCH_SIZE must be >= GLOBAL_BATCH_SIZE: RBS=$ROLLOUT_BATCH_SIZE GBS=$GLOBAL_BATCH_SIZE" >&2
  exit 2
fi
if [ $((ROLLOUT_BATCH_SIZE % GLOBAL_BATCH_SIZE)) -ne 0 ]; then
  echo "ROLLOUT_BATCH_SIZE must be divisible by GLOBAL_BATCH_SIZE: RBS=$ROLLOUT_BATCH_SIZE GBS=$GLOBAL_BATCH_SIZE" >&2
  exit 2
fi
if [ $((DATASET_SIZE % ROLLOUT_BATCH_SIZE)) -ne 0 ]; then
  echo "Full epoch would drop a tail because dataset_size is not divisible by ROLLOUT_BATCH_SIZE: dataset_size=$DATASET_SIZE RBS=$ROLLOUT_BATCH_SIZE" >&2
  echo "Choose a divisor of dataset_size or regenerate the materialized dataset." >&2
  exit 2
fi

echo "[4B SFT full] dataset_size=$DATASET_SIZE epochs=$NUM_EPOCH rollout_steps_per_epoch=$((DATASET_SIZE / ROLLOUT_BATCH_SIZE)) optimizer_steps_per_epoch=$((DATASET_SIZE / GLOBAL_BATCH_SIZE))"
echo "[4B SFT full] RBS=$ROLLOUT_BATCH_SIZE GBS=$GLOBAL_BATCH_SIZE max_tokens_per_gpu=$MAX_TOKENS_PER_GPU train_only=$SFT_DEBUG_TRAIN_ONLY disable_offload=$SFT_DISABLE_OFFLOAD"
echo "[4B SFT full] save_dir=$SAVE_DIR"

exec bash drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh
