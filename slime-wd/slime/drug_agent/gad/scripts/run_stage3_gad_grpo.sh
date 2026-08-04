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

GAD_REWARD_MODE=${GAD_REWARD_MODE:-pure}
if [ "$GAD_REWARD_MODE" != "pure" ] && [ "$GAD_REWARD_MODE" != "rule" ] && [ "$GAD_REWARD_MODE" != "hybrid" ]; then
  echo "GAD_REWARD_MODE must be pure, rule, or hybrid; got $GAD_REWARD_MODE" >&2
  exit 2
fi
if [ "$GAD_REWARD_MODE" != "rule" ]; then
  : "${GAD_DISCRIMINATOR_URL:?Set GAD_DISCRIMINATOR_URL for pure/hybrid GAD reward}"
fi
: "${STUDENT_WARMUP_LOAD:?Set STUDENT_WARMUP_LOAD to the generator SFT warmup checkpoint}"
: "${DISCRIMINATOR_WARMUP_LOAD:?Set DISCRIMINATOR_WARMUP_LOAD to the paired discriminator warmup checkpoint}"
: "${GAD_WARMUP_MANIFEST:?Set GAD_WARMUP_MANIFEST to Stage 2 warmup_manifest.json}"
PROMPT_DATA=${PROMPT_DATA:-$DRUG_AGENT_DATA_ROOT/gad/gad_steps.jsonl}
MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-4B.sh}
HF_CHECKPOINT=${HF_CHECKPOINT:-$DATA/Qwen3.5-4B}
ROLLOUT_HF_CHECKPOINT=${ROLLOUT_HF_CHECKPOINT:-$HF_CHECKPOINT}
REF_LOAD=${REF_LOAD:-$DATA/Qwen3.5-4B_torch_dist}
STUDENT_LOAD=$STUDENT_WARMUP_LOAD
SAVE_DIR=${SAVE_DIR:-$DRUG_AGENT_RUNS_ROOT/Qwen3.5-4B_gad_grpo}
CHECKPOINT_KEEP_LAST=${CHECKPOINT_KEEP_LAST:-2}
NUM_GPUS=${NUM_GPUS:-4}
TP=${TENSOR_MODEL_PARALLEL_SIZE:-4}
PP=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
CP=${CONTEXT_PARALLEL_SIZE:-1}
EP=${EXPERT_MODEL_PARALLEL_SIZE:-1}
ETP=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
NUM_ROLLOUT=${NUM_ROLLOUT:-20}
RBS=${ROLLOUT_BATCH_SIZE:-1}
N_SAMPLES=${N_SAMPLES_PER_PROMPT:-2}
GBS=${GLOBAL_BATCH_SIZE:-2}
ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-gspo}
ROLLOUT_TP=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
ROLLOUT_EXTERNAL=${ROLLOUT_EXTERNAL:-0}
ROLLOUT_EXTERNAL_NUM_GPUS=${ROLLOUT_EXTERNAL_NUM_GPUS:-$NUM_GPUS}
ROLLOUT_EXTERNAL_ENGINE_ADDRS=${ROLLOUT_EXTERNAL_ENGINE_ADDRS:-}
SKIP_RAY_RESTART=${SKIP_RAY_RESTART:-0}
RAY_DASHBOARD_ADDRESS=${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}
MAX_PROMPT=${ROLLOUT_MAX_PROMPT_LEN:-6144}
MAX_RESPONSE=${ROLLOUT_MAX_RESPONSE_LEN:-512}
MAX_CONTEXT=${ROLLOUT_MAX_CONTEXT_LEN:-6656}
SGLANG_MEM_FRACTION_STATIC=${GAD_SGLANG_MEM_FRACTION_STATIC:-${SGLANG_MEM_FRACTION_STATIC:-0.75}}
COLOCATE_OFFLOAD_TRAIN=${COLOCATE_OFFLOAD_TRAIN:-1}
COLOCATE_OFFLOAD_ROLLOUT=${COLOCATE_OFFLOAD_ROLLOUT:-1}
OPTIMIZER_CPU_OFFLOAD=${GAD_OPTIMIZER_CPU_OFFLOAD:-${OPTIMIZER_CPU_OFFLOAD:-0}}
OPTIMIZER_OFFLOAD_FRACTION=${GAD_OPTIMIZER_OFFLOAD_FRACTION:-${OPTIMIZER_OFFLOAD_FRACTION:-}}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-2048}
RECOMPUTE_VOCAB_LOG_PROBS=${RECOMPUTE_VOCAB_LOG_PROBS:-0}
DYNAMIC_SAMPLING_FILTER_PATH=${DYNAMIC_SAMPLING_FILTER_PATH:-}
USE_ROLLOUT_LOGPROBS=${USE_ROLLOUT_LOGPROBS:-0}
APPLY_CHAT_TEMPLATE_KWARGS=${APPLY_CHAT_TEMPLATE_KWARGS:-'{"enable_thinking":false}'}
USE_KL_LOSS=${USE_KL_LOSS:-1}
MEGATRON_LORA=${MEGATRON_LORA:-0}
MEGATRON_LORA_RANK=${MEGATRON_LORA_RANK:-32}
MEGATRON_LORA_ALPHA=${MEGATRON_LORA_ALPHA:-64}
MEGATRON_LORA_DROPOUT=${MEGATRON_LORA_DROPOUT:-0.0}
MEGATRON_LORA_SYNC_DIR=${MEGATRON_LORA_SYNC_DIR:-$SAVE_DIR/adapter_current}
SGLANG_LORA_NAME=${SGLANG_LORA_NAME:-slime_actor}
mkdir -p "$SAVE_DIR"
for path in "$PROMPT_DATA" "$STUDENT_LOAD" "$DISCRIMINATOR_WARMUP_LOAD" "$GAD_WARMUP_MANIFEST" "$ROLLOUT_HF_CHECKPOINT" "$REF_LOAD" "$MODEL_ARGS_FILE"; do
  if [ ! -e "$path" ]; then
    echo "Required Stage 3 input does not exist: $path" >&2
    exit 2
  fi
done
python -m drug_agent.gad.checkpoint_contract \
  --manifest "$GAD_WARMUP_MANIFEST" \
  --student-checkpoint "$STUDENT_LOAD" \
  --discriminator-checkpoint "$DISCRIMINATOR_WARMUP_LOAD"
TOTAL_SAMPLES=$((RBS * N_SAMPLES))
if [ "$TOTAL_SAMPLES" -lt "$GBS" ] || [ $((TOTAL_SAMPLES % GBS)) -ne 0 ]; then
  echo "RBS*N_SAMPLES must be >= and divisible by GBS: RBS=$RBS N_SAMPLES=$N_SAMPLES GBS=$GBS" >&2
  exit 2
fi
MODEL_PARALLEL_SIZE=$((TP * PP * CP))
if [ $((NUM_GPUS % MODEL_PARALLEL_SIZE)) -ne 0 ]; then
  echo "NUM_GPUS must be divisible by TP*PP*CP: NUM_GPUS=$NUM_GPUS TP=$TP PP=$PP CP=$CP" >&2
  exit 2
fi
DATA_PARALLEL_SIZE=$((NUM_GPUS / MODEL_PARALLEL_SIZE))
EXPERT_MODEL_SIZE=$((ETP * EP * PP))
if [ "$EXPERT_MODEL_SIZE" -le 0 ] || [ $((NUM_GPUS % EXPERT_MODEL_SIZE)) -ne 0 ]; then
  echo "NUM_GPUS must be divisible by ETP*EP*PP: NUM_GPUS=$NUM_GPUS ETP=$ETP EP=$EP PP=$PP" >&2
  exit 2
fi
if [ $((GBS % DATA_PARALLEL_SIZE)) -ne 0 ]; then
  echo "GLOBAL_BATCH_SIZE must be divisible by data parallel size: GBS=$GBS DP=$DATA_PARALLEL_SIZE" >&2
  exit 2
fi
if [ $((NUM_GPUS % ROLLOUT_TP)) -ne 0 ]; then
  echo "NUM_GPUS must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE: NUM_GPUS=$NUM_GPUS rollout_tp=$ROLLOUT_TP" >&2
  exit 2
fi
EXTERNAL_ENGINE_ADDRS=()
if [ "$ROLLOUT_EXTERNAL" = "1" ]; then
  if [ "$SKIP_RAY_RESTART" != "1" ]; then
    echo "ROLLOUT_EXTERNAL=1 requires SKIP_RAY_RESTART=1 and a pre-formed actor+rollout Ray cluster" >&2
    exit 2
  fi
  read -r -a EXTERNAL_ENGINE_ADDRS <<< "$ROLLOUT_EXTERNAL_ENGINE_ADDRS"
  if [ "$ROLLOUT_EXTERNAL_NUM_GPUS" -le 0 ] || [ $((ROLLOUT_EXTERNAL_NUM_GPUS % ROLLOUT_TP)) -ne 0 ]; then
    echo "External rollout GPUs must be positive and divisible by rollout TP: gpus=$ROLLOUT_EXTERNAL_NUM_GPUS tp=$ROLLOUT_TP" >&2
    exit 2
  fi
  EXPECTED_EXTERNAL_ENGINES=$((ROLLOUT_EXTERNAL_NUM_GPUS / ROLLOUT_TP))
  if [ "${#EXTERNAL_ENGINE_ADDRS[@]}" -ne "$EXPECTED_EXTERNAL_ENGINES" ]; then
    echo "Expected $EXPECTED_EXTERNAL_ENGINES external engine addresses, got ${#EXTERNAL_ENGINE_ADDRS[@]}: $ROLLOUT_EXTERNAL_ENGINE_ADDRS" >&2
    exit 2
  fi
  for addr in "${EXTERNAL_ENGINE_ADDRS[@]}"; do
    curl -fsS --max-time 10 "http://$addr/health_generate" >/dev/null || {
      echo "External rollout engine is not healthy: http://$addr/health_generate" >&2
      exit 2
    }
  done
fi
if [ "$USE_ROLLOUT_LOGPROBS" != "0" ] && [ "$USE_ROLLOUT_LOGPROBS" != "1" ]; then
  echo "USE_ROLLOUT_LOGPROBS must be 0 or 1; got $USE_ROLLOUT_LOGPROBS" >&2
  exit 2
fi
if [ "$GAD_REWARD_MODE" != "rule" ] && [ "${GAD_SKIP_SERVICE_HEALTHCHECK:-0}" != "1" ]; then
  HEALTH_JSON=$(curl -fsS "${GAD_DISCRIMINATOR_URL%/}/health") || {
    echo "GAD discriminator is not reachable: ${GAD_DISCRIMINATOR_URL%/}/health" >&2
    exit 2
  }
  python - "$HEALTH_JSON" "$DISCRIMINATOR_WARMUP_LOAD" <<'PY'
import json, pathlib, sys
health = json.loads(sys.argv[1])
expected = pathlib.Path(sys.argv[2]).resolve()
actual_raw = health.get("resume_checkpoint")
if not actual_raw or pathlib.Path(actual_raw).resolve() != expected:
    raise SystemExit(f"GAD discriminator service checkpoint mismatch: {actual_raw} != {expected}")
PY
fi

# TorchMemorySaver used by colocated SGLang is incompatible with expandable
# allocator segments.
USES_COLOCATED_MEMORY_SAVER=0
if [ "$ROLLOUT_EXTERNAL" != "1" ] && { [ "$COLOCATE_OFFLOAD_ROLLOUT" = "1" ] || [ "$COLOCATE_OFFLOAD_TRAIN" = "1" ]; }; then
  USES_COLOCATED_MEMORY_SAVER=1
fi
if [ "$USES_COLOCATED_MEMORY_SAVER" = "1" ] && [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
  unset PYTORCH_CUDA_ALLOC_CONF
fi
if [ "$USES_COLOCATED_MEMORY_SAVER" = "1" ] && [[ "${PYTORCH_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
  unset PYTORCH_ALLOC_CONF
fi

LOAD_ARGS=(--load "$STUDENT_LOAD")
if [ "${STUDENT_RESUME:-0}" != "1" ]; then
  # Starting GAD from SFT means loading weights, not resuming SFT optimizer,
  # RNG, or rollout counters.
  LOAD_ARGS+=(--finetune --no-load-optim --no-load-rng --start-rollout-id 0)
fi
SAVE_EXTRA_ARGS=()
if [ "${NO_SAVE_OPTIM:-0}" = "1" ]; then
  SAVE_EXTRA_ARGS+=(--no-save-optim --no-save-rng)
fi
if [ "${DISABLE_CHECKPOINT_SAVE:-0}" != "1" ]; then
  SAVE_EXTRA_ARGS+=(--save "$SAVE_DIR" --save-interval "${SAVE_INTERVAL:-5}" --save-retain-last "$CHECKPOINT_KEEP_LAST")
fi

source "$MODEL_ARGS_FILE"
SEQUENCE_PARALLEL_ARGS=()
if [ "$TP" -gt 1 ]; then
  SEQUENCE_PARALLEL_ARGS+=(--sequence-parallel)
fi
PIPELINE_LAYOUT_ARGS=()
if [ -n "${PIPELINE_MODEL_PARALLEL_LAYOUT:-}" ]; then
  PIPELINE_LAYOUT_ARGS+=(--pipeline-model-parallel-layout "$PIPELINE_MODEL_PARALLEL_LAYOUT")
else
  if [ -n "${NUM_LAYERS_IN_FIRST_PIPELINE_STAGE:-}" ]; then
    PIPELINE_LAYOUT_ARGS+=(--decoder-first-pipeline-num-layers "$NUM_LAYERS_IN_FIRST_PIPELINE_STAGE")
  fi
  if [ -n "${NUM_LAYERS_IN_LAST_PIPELINE_STAGE:-}" ]; then
    PIPELINE_LAYOUT_ARGS+=(--decoder-last-pipeline-num-layers "$NUM_LAYERS_IN_LAST_PIPELINE_STAGE")
  fi
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
if [ "${OVERLAP_GRAD_REDUCE:-0}" = "1" ]; then
  GRAD_PRECISION_ARGS+=(--overlap-grad-reduce)
fi
if [ "${OVERLAP_PARAM_GATHER:-0}" = "1" ]; then
  GRAD_PRECISION_ARGS+=(--overlap-param-gather)
fi
if [ "$ROLLOUT_EXTERNAL" = "1" ]; then
  PLACEMENT_ARGS=(
    --no-offload-train
    --rollout-external
    --rollout-num-gpus "$ROLLOUT_EXTERNAL_NUM_GPUS"
    --rollout-external-engine-addrs "${EXTERNAL_ENGINE_ADDRS[@]}"
  )
else
  PLACEMENT_ARGS=(--colocate)
  if [ "$COLOCATE_OFFLOAD_ROLLOUT" = "1" ]; then
    PLACEMENT_ARGS+=(--offload-rollout)
  else
    PLACEMENT_ARGS+=(--no-offload-rollout)
  fi
  if [ "$COLOCATE_OFFLOAD_TRAIN" = "1" ]; then
    PLACEMENT_ARGS+=(--offload-train)
  else
    PLACEMENT_ARGS+=(--no-offload-train)
  fi
fi
if [ -n "${TRAIN_MEMORY_MARGIN_BYTES:-}" ]; then
  PLACEMENT_ARGS+=(--train-memory-margin-bytes "$TRAIN_MEMORY_MARGIN_BYTES")
fi
SGLANG_EXTRA_ARGS=()
if [ -n "${SGLANG_KV_CACHE_DTYPE:-}" ]; then
  SGLANG_EXTRA_ARGS+=(--sglang-kv-cache-dtype "$SGLANG_KV_CACHE_DTYPE")
fi
if [ "${SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-0}" = "1" ]; then
  SGLANG_EXTRA_ARGS+=(--sglang-disable-custom-all-reduce)
fi
if [ "${SGLANG_DISABLE_CUDA_GRAPH:-0}" = "1" ]; then
  SGLANG_EXTRA_ARGS+=(--sglang-disable-cuda-graph)
fi
if [ "${SGLANG_DISABLE_OVERLAP_SCHEDULE:-0}" = "1" ]; then
  SGLANG_EXTRA_ARGS+=(--sglang-disable-overlap-schedule)
fi
if [ -n "$DYNAMIC_SAMPLING_FILTER_PATH" ]; then
  SGLANG_EXTRA_ARGS+=(--dynamic-sampling-filter-path "$DYNAMIC_SAMPLING_FILTER_PATH")
fi
if [ -n "${DYNAMIC_SAMPLING_MAX_DROPPED_GROUPS:-}" ]; then
  SGLANG_EXTRA_ARGS+=(--dynamic-sampling-max-dropped-groups "$DYNAMIC_SAMPLING_MAX_DROPPED_GROUPS")
fi
if [ "$USE_ROLLOUT_LOGPROBS" = "1" ]; then
  SGLANG_EXTRA_ARGS+=(--use-rollout-logprobs)
fi
if [ "$USE_KL_LOSS" = "1" ]; then
  SGLANG_EXTRA_ARGS+=(--use-kl-loss --kl-loss-coef "${KL_LOSS_COEF:-0.001}" --kl-loss-type low_var_kl)
fi
LORA_ARGS=()
if [ "$MEGATRON_LORA" = "1" ]; then
  LORA_ARGS+=(
    --megatron-lora
    --megatron-lora-rank "$MEGATRON_LORA_RANK"
    --megatron-lora-alpha "$MEGATRON_LORA_ALPHA"
    --megatron-lora-dropout "$MEGATRON_LORA_DROPOUT"
    --megatron-lora-sync-dir "$MEGATRON_LORA_SYNC_DIR"
    --sglang-lora-name "$SGLANG_LORA_NAME"
    --megatron-lora-target-modules
      linear_qkv linear_proj
      '*.shared_experts.linear_fc1' '*.shared_experts.linear_fc2'
    --sglang-enable-lora
    --sglang-max-lora-rank "$MEGATRON_LORA_RANK"
    --sglang-lora-target-modules all
    --sglang-max-loras-per-batch 1
    --sglang-max-loaded-loras 1
  )
  if [ "${MEGATRON_LORA_SKIP_INITIAL_BASE_SYNC:-0}" = "1" ]; then
    LORA_ARGS+=(--megatron-lora-skip-initial-base-sync)
  fi
fi
LOG_PROB_EXTRA_ARGS=()
if [ "$RECOMPUTE_VOCAB_LOG_PROBS" = "1" ]; then
  LOG_PROB_EXTRA_ARGS+=(--recompute-vocab-log-probs)
fi
if [ "$SKIP_RAY_RESTART" = "1" ]; then
  ray status >/dev/null
  if [ "$ROLLOUT_EXTERNAL" = "1" ]; then
    REQUIRED_CLUSTER_GPUS=$((NUM_GPUS + ROLLOUT_EXTERNAL_NUM_GPUS))
    python3 - "$REQUIRED_CLUSTER_GPUS" <<'PY'
import ray
import sys

required = float(sys.argv[1])
ray.init(address="auto", logging_level="ERROR")
available = float(ray.cluster_resources().get("GPU", 0.0))
ray.shutdown()
if available < required:
    raise SystemExit(f"External rollout Ray cluster has {available:g} GPUs; need at least {required:g}")
print(f"[external-rollout] Ray cluster GPU slots: {available:g}/{required:g}")
PY
  fi
else
  bash drug_agent/scripts/guard_ray_restart.sh
  ray stop --force 2>/dev/null || true
  pkill -9 sglang 2>/dev/null || true
  pkill -9 -x raylet 2>/dev/null || true
  pkill -9 -x gcs_server 2>/dev/null || true
  ray start --head --node-ip-address=127.0.0.1 --num-gpus "$NUM_GPUS" --disable-usage-stats --dashboard-host=0.0.0.0
fi

RUNTIME_ENV="{\"env_vars\":{\"PYTHONPATH\":\"${PYTHON_CPU_FIX_DIR}:${MEGATRON_LM_PATH}:${SLIME}:${PYTHONPATH:-}\",\"GAD_REWARD_MODE\":\"${GAD_REWARD_MODE}\",\"GAD_DISCRIMINATOR_URL\":\"${GAD_DISCRIMINATOR_URL:-}\",\"GAD_REWARD_COEF\":\"${GAD_REWARD_COEF:-0.8}\",\"GAD_FORMAT_REWARD_COEF\":\"${GAD_FORMAT_REWARD_COEF:-0.1}\",\"GAD_TOOL_REWARD_COEF\":\"${GAD_TOOL_REWARD_COEF:-0.1}\",\"GAD_FINAL_REWARD_CLIP\":\"${GAD_FINAL_REWARD_CLIP:-2.0}\",\"GAD_TRAJECTORY_LOG\":\"${SAVE_DIR}/gad_trajectories.jsonl\",\"DRUG_AGENT_TRAINING_OFFLINE\":\"1\",\"DRUG_AGENT_ALLOW_TOOL_ENV\":\"0\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NVSHMEM_DISABLE_NCCL\":\"1\",\"NCCL_IB_DISABLE\":\"${NCCL_IB_DISABLE:-1}\"}}"

ray job submit --address="$RAY_DASHBOARD_ADDRESS" --runtime-env-json="$RUNTIME_ENV" \
  -- python3 train.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node "$NUM_GPUS" --num-gpus-per-node "$NUM_GPUS" "${PLACEMENT_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$ROLLOUT_HF_CHECKPOINT" --ref-load "$REF_LOAD" "${LOAD_ARGS[@]}" "${SAVE_EXTRA_ARGS[@]}" \
  --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label --metadata-key metadata --apply-chat-template \
  --apply-chat-template-kwargs "$APPLY_CHAT_TEMPLATE_KWARGS" --rollout-shuffle \
  --custom-rm-path drug_agent.gad.reward.reward_func --group-rm --reward-key score \
  --custom-rollout-log-function-path drug_agent.gad.trajectory_logger.log_rollout_data \
  --advantage-estimator "$ADVANTAGE_ESTIMATOR" \
  --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$RBS" --n-samples-per-prompt "$N_SAMPLES" \
  --rollout-max-prompt-len "$MAX_PROMPT" --rollout-max-response-len "$MAX_RESPONSE" \
  --rollout-max-context-len "$MAX_CONTEXT" --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.8}" \
  --rollout-num-gpus-per-engine "$ROLLOUT_TP" \
  --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
  "${SGLANG_EXTRA_ARGS[@]}" \
  --global-batch-size "$GBS" --balance-data \
  --tensor-model-parallel-size "$TP" --pipeline-model-parallel-size "$PP" --context-parallel-size "$CP" \
  --expert-model-parallel-size "$EP" --expert-tensor-parallel-size "$ETP" \
  "${PIPELINE_LAYOUT_ARGS[@]}" \
  "${SEQUENCE_PARALLEL_ARGS[@]}" --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}" \
  --log-probs-chunk-size "$LOG_PROBS_CHUNK_SIZE" --recompute-loss-function "${LOG_PROB_EXTRA_ARGS[@]}" \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --optimizer adam --lr "${STUDENT_LR:-1e-6}" --min-lr "${STUDENT_MIN_LR:-0.0}" \
  --lr-decay-style "${LR_DECAY_STYLE:-constant}" --lr-warmup-fraction "${LR_WARMUP_FRACTION:-0.0}" \
  --weight-decay "${WEIGHT_DECAY:-0.1}" --adam-beta1 0.9 --adam-beta2 0.95 \
  "${OPTIMIZER_EXTRA_ARGS[@]}" \
  --attention-dropout 0.0 --hidden-dropout 0.0 "${GRAD_PRECISION_ARGS[@]}" --attention-softmax-in-fp32 --attention-backend flash \
  "${MOE_EXTRA_ARGS[@]}" \
  "${LORA_ARGS[@]}"
