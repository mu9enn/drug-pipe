#!/bin/bash
set -euo pipefail
SLIME_ENV=${SLIME_ENV:-/root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh}
if [ ! -f "$SLIME_ENV" ]; then
  SLIME_ENV=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi
source "$SLIME_ENV"
cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh

: "${GAD_DISCRIMINATOR_URL:?Set GAD_DISCRIMINATOR_URL to the independent discriminator service}"
PROMPT_DATA=${PROMPT_DATA:-$DRUG_AGENT_DATA_ROOT/gad/gad_steps.jsonl}
MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-4B.sh}
HF_CHECKPOINT=${HF_CHECKPOINT:-$DATA/Qwen3.5-4B}
REF_LOAD=${REF_LOAD:-$DATA/Qwen3.5-4B_torch_dist}
STUDENT_LOAD=${STUDENT_LOAD:-$REF_LOAD}
SAVE_DIR=${SAVE_DIR:-$DRUG_AGENT_RUNS_ROOT/Qwen3.5-4B_gad_grpo}
NUM_GPUS=${NUM_GPUS:-4}
TP=${TENSOR_MODEL_PARALLEL_SIZE:-4}
NUM_ROLLOUT=${NUM_ROLLOUT:-20}
RBS=${ROLLOUT_BATCH_SIZE:-1}
N_SAMPLES=${N_SAMPLES_PER_PROMPT:-2}
GBS=${GLOBAL_BATCH_SIZE:-2}
ROLLOUT_TP=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
MAX_PROMPT=${ROLLOUT_MAX_PROMPT_LEN:-6144}
MAX_RESPONSE=${ROLLOUT_MAX_RESPONSE_LEN:-512}
MAX_CONTEXT=${ROLLOUT_MAX_CONTEXT_LEN:-6656}
mkdir -p "$SAVE_DIR"
for path in "$PROMPT_DATA" "$STUDENT_LOAD" "$HF_CHECKPOINT" "$REF_LOAD" "$MODEL_ARGS_FILE"; do
  if [ ! -e "$path" ]; then
    echo "Required Stage 3 input does not exist: $path" >&2
    exit 2
  fi
done
TOTAL_SAMPLES=$((RBS * N_SAMPLES))
if [ "$TOTAL_SAMPLES" -lt "$GBS" ] || [ $((TOTAL_SAMPLES % GBS)) -ne 0 ]; then
  echo "RBS*N_SAMPLES must be >= and divisible by GBS: RBS=$RBS N_SAMPLES=$N_SAMPLES GBS=$GBS" >&2
  exit 2
fi
if [ $((NUM_GPUS % TP)) -ne 0 ]; then
  echo "NUM_GPUS must be divisible by TP: NUM_GPUS=$NUM_GPUS TP=$TP" >&2
  exit 2
fi
if [ $((NUM_GPUS % ROLLOUT_TP)) -ne 0 ]; then
  echo "NUM_GPUS must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE: NUM_GPUS=$NUM_GPUS rollout_tp=$ROLLOUT_TP" >&2
  exit 2
fi
if [ "${GAD_SKIP_SERVICE_HEALTHCHECK:-0}" != "1" ]; then
  curl -fsS "${GAD_DISCRIMINATOR_URL%/}/health" >/dev/null || {
    echo "GAD discriminator is not reachable: ${GAD_DISCRIMINATOR_URL%/}/health" >&2
    exit 2
  }
fi

# TorchMemorySaver used by colocated SGLang is incompatible with expandable
# allocator segments.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
  unset PYTORCH_CUDA_ALLOC_CONF
fi
if [[ "${PYTORCH_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
  unset PYTORCH_ALLOC_CONF
fi

LOAD_ARGS=(--load "$STUDENT_LOAD")
if [ "${STUDENT_RESUME:-0}" != "1" ]; then
  # Starting GAD from SFT means loading weights, not resuming SFT optimizer,
  # RNG, or rollout counters.
  LOAD_ARGS+=(--finetune --no-load-optim --no-load-rng --start-rollout-id 0)
fi

source "$MODEL_ARGS_FILE"
ray stop --force 2>/dev/null || true
pkill -9 sglang 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
ray start --head --node-ip-address=127.0.0.1 --num-gpus "$NUM_GPUS" --disable-usage-stats --dashboard-host=0.0.0.0

RUNTIME_ENV="{\"env_vars\":{\"PYTHONPATH\":\"${PYTHON_CPU_FIX_DIR}:/root/Megatron-LM/:${SLIME}:${PYTHONPATH:-}\",\"GAD_DISCRIMINATOR_URL\":\"${GAD_DISCRIMINATOR_URL}\",\"GAD_REWARD_COEF\":\"${GAD_REWARD_COEF:-0.8}\",\"GAD_FORMAT_REWARD_COEF\":\"${GAD_FORMAT_REWARD_COEF:-0.1}\",\"GAD_TOOL_REWARD_COEF\":\"${GAD_TOOL_REWARD_COEF:-0.1}\",\"GAD_FINAL_REWARD_CLIP\":\"${GAD_FINAL_REWARD_CLIP:-2.0}\",\"GAD_TRAJECTORY_LOG\":\"${SAVE_DIR}/gad_trajectories.jsonl\",\"DRUG_AGENT_TRAINING_OFFLINE\":\"1\",\"DRUG_AGENT_ALLOW_TOOL_ENV\":\"0\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\"}}"

ray job submit --address=http://127.0.0.1:8265 --runtime-env-json="$RUNTIME_ENV" \
  -- python3 train.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node "$NUM_GPUS" --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_CHECKPOINT" --ref-load "$REF_LOAD" "${LOAD_ARGS[@]}" --save "$SAVE_DIR" --save-interval "${SAVE_INTERVAL:-5}" \
  --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label --metadata-key metadata --apply-chat-template --rollout-shuffle \
  --custom-rm-path drug_agent.gad.reward.reward_func --group-rm --reward-key score \
  --custom-rollout-log-function-path drug_agent.gad.trajectory_logger.log_rollout_data \
  --advantage-estimator grpo --use-kl-loss --kl-loss-coef "${KL_LOSS_COEF:-0.001}" --kl-loss-type low_var_kl \
  --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$RBS" --n-samples-per-prompt "$N_SAMPLES" \
  --rollout-max-prompt-len "$MAX_PROMPT" --rollout-max-response-len "$MAX_RESPONSE" \
  --rollout-max-context-len "$MAX_CONTEXT" --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.8}" \
  --rollout-num-gpus-per-engine "$ROLLOUT_TP" \
  --global-batch-size "$GBS" --balance-data \
  --tensor-model-parallel-size "$TP" --sequence-parallel --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}" \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --optimizer adam --lr "${STUDENT_LR:-1e-6}" --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 \
  --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash
