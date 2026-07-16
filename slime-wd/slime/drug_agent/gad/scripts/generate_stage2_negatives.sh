#!/bin/bash
set -euo pipefail
SLIME_ENV=${SLIME_ENV:-/root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh}
if [ ! -f "$SLIME_ENV" ]; then
  SLIME_ENV=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi
source "$SLIME_ENV"
cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh

PROMPT_DATA=${PROMPT_DATA:-$DRUG_AGENT_DATA_ROOT/gad/gad_steps.jsonl}
GAD_NEGATIVE_CACHE=${GAD_NEGATIVE_CACHE:-$DRUG_AGENT_DATA_ROOT/gad/stage2_negatives.jsonl}
MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-4B.sh}
HF_CHECKPOINT=${HF_CHECKPOINT:-$DATA/Qwen3.5-4B}
REF_LOAD=${REF_LOAD:-$DATA/Qwen3.5-4B_torch_dist}
STUDENT_LOAD=${STUDENT_LOAD:-$REF_LOAD}
NUM_GPUS=${NUM_GPUS:-4}
TP=${TENSOR_MODEL_PARALLEL_SIZE:-4}
NUM_ROLLOUT=${NUM_ROLLOUT:-}
RBS=${ROLLOUT_BATCH_SIZE:-1}
MAX_PROMPT=${ROLLOUT_MAX_PROMPT_LEN:-6144}
MAX_RESPONSE=${ROLLOUT_MAX_RESPONSE_LEN:-512}
MAX_CONTEXT=${ROLLOUT_MAX_CONTEXT_LEN:-6656}
ROLLOUT_TP=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
export GAD_NEGATIVE_CACHE
rm -f "$GAD_NEGATIVE_CACHE"
for path in "$PROMPT_DATA" "$STUDENT_LOAD" "$HF_CHECKPOINT" "$REF_LOAD" "$MODEL_ARGS_FILE"; do
  if [ ! -e "$path" ]; then
    echo "Required Stage 2 input does not exist: $path" >&2
    exit 2
  fi
done
DATASET_SIZE=$(wc -l < "$PROMPT_DATA")
if [ -z "$NUM_ROLLOUT" ]; then
  NUM_ROLLOUT=$(((DATASET_SIZE + RBS - 1) / RBS))
fi
echo "[GAD Stage 2] dataset_size=$DATASET_SIZE rollout_batch_size=$RBS num_rollout=$NUM_ROLLOUT"

# Colocated SGLang uses TorchMemorySaver, which does not support expandable
# allocator segments. Do not inherit this setting into Ray workers.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
  unset PYTORCH_CUDA_ALLOC_CONF
fi
if [[ "${PYTORCH_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
  unset PYTORCH_ALLOC_CONF
fi

source "$MODEL_ARGS_FILE"
ray stop --force 2>/dev/null || true
pkill -9 sglang 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
ray start --head --node-ip-address=127.0.0.1 --num-gpus "$NUM_GPUS" --disable-usage-stats --dashboard-host=0.0.0.0

ray job submit --address=http://127.0.0.1:8265 \
  --runtime-env-json="{\"env_vars\":{\"PYTHONPATH\":\"${PYTHON_CPU_FIX_DIR}:/root/Megatron-LM/:${SLIME}:${PYTHONPATH:-}\",\"GAD_NEGATIVE_CACHE\":\"${GAD_NEGATIVE_CACHE}\",\"DRUG_AGENT_TRAINING_OFFLINE\":\"1\",\"DRUG_AGENT_ALLOW_TOOL_ENV\":\"0\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\"}}" \
  -- python3 train.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node "$NUM_GPUS" --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_CHECKPOINT" --ref-load "$REF_LOAD" --load "$STUDENT_LOAD" \
  --finetune --no-load-optim --no-load-rng --start-rollout-id 0 \
  --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label --metadata-key metadata --apply-chat-template --rollout-shuffle \
  --custom-rm-path drug_agent.gad.negative_cache.zero_reward \
  --custom-rollout-log-function-path drug_agent.gad.negative_cache.log_negative_cache \
  --advantage-estimator grpo \
  --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$RBS" --n-samples-per-prompt 1 \
  --rollout-max-prompt-len "$MAX_PROMPT" --rollout-max-response-len "$MAX_RESPONSE" \
  --rollout-max-context-len "$MAX_CONTEXT" --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.8}" \
  --rollout-num-gpus-per-engine "$ROLLOUT_TP" \
  --global-batch-size "$RBS" --tensor-model-parallel-size "$TP" --sequence-parallel \
  --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}" \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --optimizer adam --lr 0 --lr-decay-style constant --weight-decay 0 \
  --attention-dropout 0.0 --hidden-dropout 0.0 --attention-backend flash
