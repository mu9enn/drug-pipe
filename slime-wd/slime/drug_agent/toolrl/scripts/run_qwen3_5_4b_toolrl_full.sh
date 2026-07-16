#!/bin/bash
set -euo pipefail
SLIME_ENV=${SLIME_ENV:-/root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh}
if [ ! -f "$SLIME_ENV" ]; then
  SLIME_ENV=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi
source "$SLIME_ENV"
cd "$SLIME"
export PROMPT_DATA=${PROMPT_DATA:-$DRUG_AGENT_DATA_ROOT/toolrl/mcp_sft_all.toolrl_steps.jsonl}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-4}
DATASET_SIZE=$(wc -l < "$PROMPT_DATA")
export NUM_ROLLOUT=${NUM_ROLLOUT:-$(((DATASET_SIZE + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE))}
export MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-4B.sh}
export HF_CHECKPOINT=${HF_CHECKPOINT:-$DATA/Qwen3.5-4B}
export REF_LOAD=${REF_LOAD:-$DATA/Qwen3.5-4B_torch_dist}
# Default to the original model; STUDENT_LOAD remains an optional override.
export LOAD=${LOAD:-${STUDENT_LOAD:-$REF_LOAD}}
RUN_NAME=${RUN_NAME:-Qwen3.5-4B_toolrl_full_$(date +%Y%m%d_%H%M%S)}
export SAVE_DIR=${SAVE_DIR:-$DRUG_AGENT_RUNS_ROOT/$RUN_NAME}
export NUM_GPUS=${NUM_GPUS:-4}
export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-4}
export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
export ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-6144}
export ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-512}
export ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-6656}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-4096}
export RECOMPUTE_FULL=${RECOMPUTE_FULL:-1}
export RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
export SAVE_INTERVAL=${SAVE_INTERVAL:-20}
echo "[4B ToolRL full] dataset_size=$DATASET_SIZE prompt_batches=$NUM_ROLLOUT"
exec bash drug_agent/toolrl/scripts/run_toolrl_grpo.sh
