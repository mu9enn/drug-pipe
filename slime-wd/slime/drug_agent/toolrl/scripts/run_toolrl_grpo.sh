#!/bin/bash
set -ex

if [ -f /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh ]; then
  source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
else
  source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi

cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
  unset PYTORCH_CUDA_ALLOC_CONF
fi
if [[ "${PYTORCH_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
  unset PYTORCH_ALLOC_CONF
fi

unset RAY_ADDRESS || true
pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
pkill -9 ray python 2>/dev/null || true
sleep 2

export PYTHONBUFFERED=16
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

NUM_GPUS=${NUM_GPUS:-2}
REAL_CPU=${REAL_CPU:-$(nproc)}

MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-0.8B.sh}
if [ ! -f "$MODEL_ARGS_FILE" ]; then
  echo "MODEL_ARGS_FILE not found: $MODEL_ARGS_FILE" >&2
  exit 2
fi
source "$MODEL_ARGS_FILE"

PROMPT_DATA=${PROMPT_DATA:?PROMPT_DATA must point to a step-level ToolRL JSONL file}
if [ ! -f "$PROMPT_DATA" ]; then
  echo "PROMPT_DATA not found: $PROMPT_DATA" >&2
  exit 2
fi

HF_CHECKPOINT=${HF_CHECKPOINT:-$DATA/Qwen3.5-0.8B}
REF_LOAD=${REF_LOAD:-$DATA/Qwen3.5-0.8B_torch_dist}
SAVE_DIR=${SAVE_DIR:-$DRUG_AGENT_RUNS_ROOT/Qwen3.5-0.8B_toolrl_grpo}
SAVE_INTERVAL=${SAVE_INTERVAL:-1}
LOAD=${LOAD:-}
TOOLRL_RESUME=${TOOLRL_RESUME:-0}

ROLLOUT_FUNCTION_PATH=slime.rollout.sglang_rollout.generate_rollout
CUSTOM_RM_PATH=drug_agent.toolrl.molclaw_reward.reward_func
REWARD_KEY=${REWARD_KEY:-score}

NUM_ROLLOUT=${NUM_ROLLOUT:-2}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-2048}
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
NUM_EPOCH=${NUM_EPOCH:-1}
LR=${LR:-1e-6}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
RECOMPUTE_FULL=${RECOMPUTE_FULL:-0}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}

MODEL_PARALLEL_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE * EXPERT_MODEL_PARALLEL_SIZE))
if [ "$MODEL_PARALLEL_SIZE" -le 0 ] || [ $((NUM_GPUS % MODEL_PARALLEL_SIZE)) -ne 0 ]; then
  echo "NUM_GPUS must be divisible by TP*PP*CP*EP: NUM_GPUS=$NUM_GPUS TP=$TENSOR_MODEL_PARALLEL_SIZE PP=$PIPELINE_MODEL_PARALLEL_SIZE CP=$CONTEXT_PARALLEL_SIZE EP=$EXPERT_MODEL_PARALLEL_SIZE" >&2
  exit 2
fi
DATA_PARALLEL_SIZE=$((NUM_GPUS / MODEL_PARALLEL_SIZE))
if [ $((GLOBAL_BATCH_SIZE % DATA_PARALLEL_SIZE)) -ne 0 ]; then
  echo "GLOBAL_BATCH_SIZE must be divisible by DATA_PARALLEL_SIZE: GBS=$GLOBAL_BATCH_SIZE DP=$DATA_PARALLEL_SIZE" >&2
  exit 2
fi

BATCHES_PER_ROLLOUT=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
if [ "$BATCHES_PER_ROLLOUT" -le 0 ]; then
  echo "ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT must be positive: ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT=$N_SAMPLES_PER_PROMPT" >&2
  exit 2
fi

DERIVED_TRAIN_ITERS=$((NUM_ROLLOUT * BATCHES_PER_ROLLOUT / GLOBAL_BATCH_SIZE))
if [ "$DERIVED_TRAIN_ITERS" -lt 1 ]; then
  echo "[drug_agent/toolrl] Derived train_iters=$DERIVED_TRAIN_ITERS from NUM_ROLLOUT=$NUM_ROLLOUT, ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE, N_SAMPLES_PER_PROMPT=$N_SAMPLES_PER_PROMPT, GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE; bumping LR_DECAY_ITERS to 1 so Megatron scheduler stays valid." >&2
  LR_DECAY_ITERS=${LR_DECAY_ITERS:-1}
else
  LR_DECAY_ITERS=${LR_DECAY_ITERS:-$DERIVED_TRAIN_ITERS}
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

CKPT_ARGS=(
  --hf-checkpoint "$HF_CHECKPOINT"
  --ref-load "$REF_LOAD"
  --save "$SAVE_DIR"
  --save-interval "$SAVE_INTERVAL"
)
if [ -n "$LOAD" ]; then
  CKPT_ARGS+=(--load "$LOAD")
  if [ "$TOOLRL_RESUME" != "1" ]; then
    CKPT_ARGS+=(--finetune --no-load-optim --no-load-rng --start-rollout-id 0)
  fi
fi

TOOLRL_ARGS=(
  --rollout-function-path "$ROLLOUT_FUNCTION_PATH"
  --custom-rm-path "$CUSTOM_RM_PATH"
  --reward-key "$REWARD_KEY"

  --prompt-data "$PROMPT_DATA"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --apply-chat-template
  --rollout-shuffle

  --advantage-estimator grpo
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28

  --num-rollout "$NUM_ROLLOUT"
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
  --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
  --rollout-num-gpus-per-engine "$ROLLOUT_NUM_GPUS_PER_ENGINE"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --balance-data
)
if [ -n "$ROLLOUT_MAX_PROMPT_LEN" ]; then
  TOOLRL_ARGS+=(--rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN")
fi
if [ -n "$ROLLOUT_MAX_CONTEXT_LEN" ]; then
  TOOLRL_ARGS+=(--rollout-max-context-len "$ROLLOUT_MAX_CONTEXT_LEN")
fi

PERF_ARGS=(
  --tensor-model-parallel-size "$TENSOR_MODEL_PARALLEL_SIZE"
  --pipeline-model-parallel-size "$PIPELINE_MODEL_PARALLEL_SIZE"
  --context-parallel-size "$CONTEXT_PARALLEL_SIZE"
  --expert-model-parallel-size "$EXPERT_MODEL_PARALLEL_SIZE"
  --expert-tensor-parallel-size "$EXPERT_TENSOR_PARALLEL_SIZE"
  --use-dynamic-batch-size
  --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
)
if [ "$TENSOR_MODEL_PARALLEL_SIZE" -gt 1 ]; then
  PERF_ARGS+=(--sequence-parallel)
fi
if [ "$RECOMPUTE_FULL" = "1" ]; then
  PERF_ARGS+=(
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers "$RECOMPUTE_NUM_LAYERS"
  )
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "$LR"
  --lr-decay-style "${LR_DECAY_STYLE:-constant}"
  --lr-decay-iters "$LR_DECAY_ITERS"
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --adam-beta1 "${ADAM_BETA1:-0.9}"
  --adam-beta2 "${ADAM_BETA2:-0.95}"
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

ray start --head \
  --node-ip-address "$MASTER_ADDR" \
  --num-gpus "$NUM_GPUS" \
  --num-cpus "$REAL_CPU" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHON_CPU_FIX_DIR}:/root/Megatron-LM/:${SLIME}:${PYTHONPATH:-}\",
    \"PYTHON_CPU_COUNT\": \"${REAL_CPU}\",
    \"PATH\": \"${PATH}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"CUDA_HOME\": \"${CUDA_HOME:-/usr/local/cuda}\",
    \"NVIDIA_VISIBLE_DEVICES\": \"${NVIDIA_VISIBLE_DEVICES:-all}\",
    \"NVIDIA_DRIVER_CAPABILITIES\": \"${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_IB_DISABLE\": \"${NCCL_IB_DISABLE:-1}\"
    ,\"DRUG_AGENT_TRAINING_OFFLINE\": \"1\"
    ,\"DRUG_AGENT_ALLOW_TOOL_ENV\": \"0\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$NUM_GPUS" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${TOOLRL_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${MISC_ARGS[@]}"
