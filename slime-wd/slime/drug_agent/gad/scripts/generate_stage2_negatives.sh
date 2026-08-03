#!/bin/bash
set -euo pipefail
SLIME_ENV=${SLIME_ENV:-/root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh}
if [ ! -f "$SLIME_ENV" ]; then
  SLIME_ENV=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi
source "$SLIME_ENV"
cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-/root/Megatron-LM}

PROMPT_DATA=${PROMPT_DATA:-$DRUG_AGENT_DATA_ROOT/gad/gad_steps.jsonl}
GAD_NEGATIVE_CACHE=${GAD_NEGATIVE_CACHE:-$DRUG_AGENT_DATA_ROOT/gad/stage2_negatives.jsonl}
MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-4B.sh}
HF_CHECKPOINT=${HF_CHECKPOINT:-$DATA/Qwen3.5-4B}
REF_LOAD=${REF_LOAD:-$DATA/Qwen3.5-4B_torch_dist}
STUDENT_LOAD=${STUDENT_LOAD:-$REF_LOAD}
NUM_GPUS=${NUM_GPUS:-4}
TP=${TENSOR_MODEL_PARALLEL_SIZE:-4}
PP=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
CP=${CONTEXT_PARALLEL_SIZE:-1}
EP=${EXPERT_MODEL_PARALLEL_SIZE:-1}
ETP=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
NUM_ROLLOUT=${NUM_ROLLOUT:-}
RBS=${ROLLOUT_BATCH_SIZE:-1}
MAX_PROMPT=${ROLLOUT_MAX_PROMPT_LEN:-6144}
MAX_RESPONSE=${ROLLOUT_MAX_RESPONSE_LEN:-512}
MAX_CONTEXT=${ROLLOUT_MAX_CONTEXT_LEN:-6656}
ROLLOUT_TP=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
SGLANG_MEM_FRACTION_STATIC=${GAD_SGLANG_MEM_FRACTION_STATIC:-${SGLANG_MEM_FRACTION_STATIC:-0.75}}
COLOCATE_OFFLOAD_TRAIN=${COLOCATE_OFFLOAD_TRAIN:-1}
OPTIMIZER_OFFLOAD_FRACTION=${GAD_OPTIMIZER_OFFLOAD_FRACTION:-${OPTIMIZER_OFFLOAD_FRACTION:-}}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-2048}
APPLY_CHAT_TEMPLATE_KWARGS=${APPLY_CHAT_TEMPLATE_KWARGS:-'{"enable_thinking":false}'}
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
MODEL_PARALLEL_SIZE=$((TP * PP * CP))
if [ "$MODEL_PARALLEL_SIZE" -le 0 ] || [ $((NUM_GPUS % MODEL_PARALLEL_SIZE)) -ne 0 ]; then
  echo "NUM_GPUS must be divisible by TP*PP*CP: NUM_GPUS=$NUM_GPUS TP=$TP PP=$PP CP=$CP" >&2
  exit 2
fi
EXPERT_MODEL_SIZE=$((ETP * EP * PP))
if [ "$EXPERT_MODEL_SIZE" -le 0 ] || [ $((NUM_GPUS % EXPERT_MODEL_SIZE)) -ne 0 ]; then
  echo "NUM_GPUS must be divisible by ETP*EP*PP: NUM_GPUS=$NUM_GPUS ETP=$ETP EP=$EP PP=$PP" >&2
  exit 2
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
SEQUENCE_PARALLEL_ARGS=()
if [ "$TP" -gt 1 ]; then
  SEQUENCE_PARALLEL_ARGS+=(--sequence-parallel)
fi
OPTIMIZER_EXTRA_ARGS=()
if [ "${OPTIMIZER_CPU_OFFLOAD:-0}" = "1" ]; then
  OPTIMIZER_EXTRA_ARGS+=(--optimizer-cpu-offload)
  if [ -n "${OPTIMIZER_OFFLOAD_FRACTION:-}" ]; then
    OPTIMIZER_EXTRA_ARGS+=(--optimizer-offload-fraction "$OPTIMIZER_OFFLOAD_FRACTION")
  fi
  if [ "${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-1}" = "1" ]; then
    OPTIMIZER_EXTRA_ARGS+=(--overlap-cpu-optimizer-d2h-h2d)
  fi
fi
if [ "${USE_PRECISION_AWARE_OPTIMIZER:-0}" = "1" ]; then
  OPTIMIZER_EXTRA_ARGS+=(--use-precision-aware-optimizer)
fi
if [ "${OFFLOAD_OPTIMIZER_STATES:-0}" = "1" ]; then
  OPTIMIZER_EXTRA_ARGS+=(--offload-optimizer-states)
fi
if [ -n "${MAIN_GRADS_DTYPE:-}" ]; then
  OPTIMIZER_EXTRA_ARGS+=(--main-grads-dtype "$MAIN_GRADS_DTYPE")
fi
if [ -n "${MAIN_PARAMS_DTYPE:-}" ]; then
  OPTIMIZER_EXTRA_ARGS+=(--main-params-dtype "$MAIN_PARAMS_DTYPE")
fi
if [ -n "${EXP_AVG_DTYPE:-}" ]; then
  OPTIMIZER_EXTRA_ARGS+=(--exp-avg-dtype "$EXP_AVG_DTYPE")
fi
if [ -n "${EXP_AVG_SQ_DTYPE:-}" ]; then
  OPTIMIZER_EXTRA_ARGS+=(--exp-avg-sq-dtype "$EXP_AVG_SQ_DTYPE")
fi
if [ "${FP8_PARAM_GATHER:-0}" = "1" ]; then
  OPTIMIZER_EXTRA_ARGS+=(
    --fp8-format "${FP8_FORMAT:-e4m3}"
    --fp8-recipe "${FP8_RECIPE:-blockwise}"
    --fp8-param-gather
  )
fi
MOE_EXTRA_ARGS=()
if [ "${MOE_ENABLE_DEEPEP:-0}" = "1" ]; then
  MOE_EXTRA_ARGS+=(--moe-token-dispatcher-type flex --moe-enable-deepep)
fi
GRAD_PRECISION_ARGS=()
if [ "${ACCUMULATE_ALLREDUCE_GRADS_IN_FP32:-1}" = "1" ]; then
  GRAD_PRECISION_ARGS+=(--accumulate-allreduce-grads-in-fp32)
fi
PLACEMENT_ARGS=(--colocate --offload-rollout)
if [ "$COLOCATE_OFFLOAD_TRAIN" = "1" ]; then
  PLACEMENT_ARGS+=(--offload-train)
else
  PLACEMENT_ARGS+=(--no-offload-train)
fi
bash drug_agent/scripts/guard_ray_restart.sh
ray stop --force 2>/dev/null || true
pkill -9 sglang 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
ray start --head --node-ip-address=127.0.0.1 --num-gpus "$NUM_GPUS" --disable-usage-stats --dashboard-host=0.0.0.0

ray job submit --address=http://127.0.0.1:8265 \
  --runtime-env-json="{\"env_vars\":{\"PYTHONPATH\":\"${PYTHON_CPU_FIX_DIR}:${MEGATRON_LM_PATH}:${SLIME}:${PYTHONPATH:-}\",\"GAD_NEGATIVE_CACHE\":\"${GAD_NEGATIVE_CACHE}\",\"DRUG_AGENT_TRAINING_OFFLINE\":\"1\",\"DRUG_AGENT_ALLOW_TOOL_ENV\":\"0\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NVSHMEM_DISABLE_NCCL\":\"1\"}}" \
  -- python3 train.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node "$NUM_GPUS" --num-gpus-per-node "$NUM_GPUS" "${PLACEMENT_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_CHECKPOINT" --ref-load "$REF_LOAD" --load "$STUDENT_LOAD" \
  --finetune --no-load-optim --no-load-rng --start-rollout-id 0 \
  --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label --metadata-key metadata --apply-chat-template \
  --apply-chat-template-kwargs "$APPLY_CHAT_TEMPLATE_KWARGS" --rollout-shuffle \
  --custom-rm-path drug_agent.gad.negative_cache.zero_reward \
  --custom-rollout-log-function-path drug_agent.gad.negative_cache.log_negative_cache \
  --advantage-estimator grpo \
  --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$RBS" --n-samples-per-prompt 1 \
  --rollout-max-prompt-len "$MAX_PROMPT" --rollout-max-response-len "$MAX_RESPONSE" \
  --rollout-max-context-len "$MAX_CONTEXT" --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.8}" \
  --rollout-num-gpus-per-engine "$ROLLOUT_TP" \
  --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
  --global-batch-size "$RBS" --tensor-model-parallel-size "$TP" --pipeline-model-parallel-size "$PP" \
  --context-parallel-size "$CP" --expert-model-parallel-size "$EP" --expert-tensor-parallel-size "$ETP" \
  "${SEQUENCE_PARALLEL_ARGS[@]}" \
  --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}" \
  --log-probs-chunk-size "$LOG_PROBS_CHUNK_SIZE" --recompute-loss-function \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --optimizer adam --lr 0 --lr-decay-style constant --weight-decay 0 "${OPTIMIZER_EXTRA_ARGS[@]}" \
  --attention-dropout 0.0 --hidden-dropout 0.0 "${GRAD_PRECISION_ARGS[@]}" --attention-backend flash \
  "${MOE_EXTRA_ARGS[@]}"
