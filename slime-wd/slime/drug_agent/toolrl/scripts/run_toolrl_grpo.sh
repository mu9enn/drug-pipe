#!/bin/bash
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../scripts/resolve_slime_env.sh"
source "$SLIME_ENV"

cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-/root/Megatron-LM}
ROLLOUT_EXTERNAL=${ROLLOUT_EXTERNAL:-0}

# Preserve expandable segments for the Megatron actors: long Qwen3.5 GDN
# batches otherwise strand memory in split reserved blocks.  SGLang's
# TorchMemorySaver adapter is incompatible with expandable segments, so local
# SGLang Ray actors explicitly receive allocator-clean environment variables in
# slime/ray/rollout.py.  Do not unset the parent value here: that would also
# remove the fragmentation protection from Megatron.

unset RAY_ADDRESS || true
SKIP_RAY_RESTART=${SKIP_RAY_RESTART:-0}
if [ "$ROLLOUT_EXTERNAL" = "1" ] && [ "$SKIP_RAY_RESTART" != "1" ]; then
  echo "ROLLOUT_EXTERNAL=1 requires SKIP_RAY_RESTART=1 and a pre-formed actor+rollout Ray cluster" >&2
  exit 2
fi
if [ "$SKIP_RAY_RESTART" != "1" ]; then
  bash drug_agent/scripts/guard_ray_restart.sh
  pkill -9 sglang 2>/dev/null || true
  sleep 2
  ray stop --force 2>/dev/null || true
  pkill -9 -x raylet 2>/dev/null || true
  pkill -9 -x gcs_server 2>/dev/null || true
  sleep 2
fi

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
# Slime trains from REF_LOAD but uses the HF checkpoint for SGLang rollout and
# rollout-weight quantization metadata.  Large profiles can therefore keep a
# BF16 torch_dist actor while serving an official FP8 HF rollout checkpoint.
ROLLOUT_HF_CHECKPOINT=${ROLLOUT_HF_CHECKPOINT:-$HF_CHECKPOINT}
REF_LOAD=${REF_LOAD:-$DATA/Qwen3.5-0.8B_torch_dist}
SAVE_DIR=${SAVE_DIR:-$DRUG_AGENT_RUNS_ROOT/Qwen3.5-0.8B_toolrl_grpo}
SAVE_INTERVAL=${SAVE_INTERVAL:-1}
CHECKPOINT_KEEP_LAST=${CHECKPOINT_KEEP_LAST:-2}
DISTRIBUTED_TIMEOUT_MINUTES=${DISTRIBUTED_TIMEOUT_MINUTES:-10}
LOAD=${LOAD:-}
TOOLRL_RESUME=${TOOLRL_RESUME:-0}

ROLLOUT_FUNCTION_PATH=slime.rollout.sglang_rollout.generate_rollout
CUSTOM_RM_PATH=drug_agent.toolrl.molclaw_reward.reward_func
REWARD_KEY=${REWARD_KEY:-score}

NUM_ROLLOUT=${NUM_ROLLOUT:-2}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
TOOLRL_REWARD_MODE=${TOOLRL_REWARD_MODE:-official}
ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-grpo}
NORMALIZE_ADVANTAGES=${NORMALIZE_ADVANTAGES:-0}
DYNAMIC_SAMPLING_FILTER_PATH=${DYNAMIC_SAMPLING_FILTER_PATH:-}
USE_ROLLOUT_LOGPROBS=${USE_ROLLOUT_LOGPROBS:-0}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-2048}
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-}
ROLLOUT_LONG_RESPONSE_LEN=${ROLLOUT_LONG_RESPONSE_LEN:-}
ROLLOUT_LONG_TASK_TYPES=${ROLLOUT_LONG_TASK_TYPES:-}
CUSTOM_GENERATE_FUNCTION_PATH=${CUSTOM_GENERATE_FUNCTION_PATH:-}
CUSTOM_ROLLOUT_LOG_FUNCTION_PATH=${CUSTOM_ROLLOUT_LOG_FUNCTION_PATH:-}
ROLLOUT_ALL_SAMPLES_PROCESS_PATH=${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.75}
# Slime's colocate default pauses the complete Megatron CUDA allocator into
# host RAM between rollout and training.  That is independent from Megatron's
# optimizer CPU offload and can create a second, transient model-sized host
# copy.  Large single-node profiles can instead keep the actor resident while
# only SGLang sleeps, provided the combined resident GPU budget was validated.
COLOCATE_OFFLOAD_TRAIN=${COLOCATE_OFFLOAD_TRAIN:-1}
# Qwen3.5 hybrid GDN/Mamba rollout engines can preserve all model weights yet
# produce corrupted generations after a release/resume cycle.  Large H200
# profiles whose actor + low-fraction SGLang footprint fits can keep rollout
# resident and avoid that stateful path entirely.
COLOCATE_OFFLOAD_ROLLOUT=${COLOCATE_OFFLOAD_ROLLOUT:-1}
# A colocated large actor may need a higher optimizer offload fraction than
# train-only SFT so SGLang can restore its weights/cache beside the actor.
OPTIMIZER_CPU_OFFLOAD=${TOOLRL_OPTIMIZER_CPU_OFFLOAD:-${OPTIMIZER_CPU_OFFLOAD:-0}}
OPTIMIZER_OFFLOAD_FRACTION=${TOOLRL_OPTIMIZER_OFFLOAD_FRACTION:-${OPTIMIZER_OFFLOAD_FRACTION:-}}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
NUM_EPOCH=${NUM_EPOCH:-1}
LR=${LR:-1e-6}
MIN_LR=${MIN_LR:-0.0}
LR_WARMUP_FRACTION=${LR_WARMUP_FRACTION:-}
LR_WARMUP_INIT=${LR_WARMUP_INIT:-}
USE_KL_LOSS=${USE_KL_LOSS:-0}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-low_var_kl}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
# External rollout keeps the Megatron actor and SGLang engines on different
# workers.  Slime still reserves logical rollout GPU slots in the shared Ray
# placement group, so the caller must pre-form the two-node Ray cluster before
# invoking this launcher.
ROLLOUT_EXTERNAL_NUM_GPUS=${ROLLOUT_EXTERNAL_NUM_GPUS:-$NUM_GPUS}
ROLLOUT_EXTERNAL_ENGINE_ADDRS=${ROLLOUT_EXTERNAL_ENGINE_ADDRS:-}
RAY_DASHBOARD_ADDRESS=${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}
RECOMPUTE_FULL=${RECOMPUTE_FULL:-0}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
RECOMPUTE_LOSS_FUNCTION=${RECOMPUTE_LOSS_FUNCTION:-1}
RECOMPUTE_VOCAB_LOG_PROBS=${RECOMPUTE_VOCAB_LOG_PROBS:-0}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-2048}
APPLY_CHAT_TEMPLATE_KWARGS=${APPLY_CHAT_TEMPLATE_KWARGS:-'{"enable_thinking":false}'}
MEGATRON_LORA=${MEGATRON_LORA:-0}
MEGATRON_LORA_RANK=${MEGATRON_LORA_RANK:-32}
MEGATRON_LORA_ALPHA=${MEGATRON_LORA_ALPHA:-64}
MEGATRON_LORA_DROPOUT=${MEGATRON_LORA_DROPOUT:-0.0}
MEGATRON_LORA_SYNC_DIR=${MEGATRON_LORA_SYNC_DIR:-$SAVE_DIR/adapter_current}
SGLANG_LORA_NAME=${SGLANG_LORA_NAME:-slime_actor}

MODEL_PARALLEL_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
if [ "$MODEL_PARALLEL_SIZE" -le 0 ] || [ $((NUM_GPUS % MODEL_PARALLEL_SIZE)) -ne 0 ]; then
  echo "NUM_GPUS must be divisible by TP*PP*CP: NUM_GPUS=$NUM_GPUS TP=$TENSOR_MODEL_PARALLEL_SIZE PP=$PIPELINE_MODEL_PARALLEL_SIZE CP=$CONTEXT_PARALLEL_SIZE" >&2
  exit 2
fi
DATA_PARALLEL_SIZE=$((NUM_GPUS / MODEL_PARALLEL_SIZE))
EXPERT_MODEL_SIZE=$((EXPERT_TENSOR_PARALLEL_SIZE * EXPERT_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE))
if [ "$EXPERT_MODEL_SIZE" -le 0 ] || [ $((NUM_GPUS % EXPERT_MODEL_SIZE)) -ne 0 ]; then
  echo "NUM_GPUS must be divisible by ETP*EP*PP: NUM_GPUS=$NUM_GPUS ETP=$EXPERT_TENSOR_PARALLEL_SIZE EP=$EXPERT_MODEL_PARALLEL_SIZE PP=$PIPELINE_MODEL_PARALLEL_SIZE" >&2
  exit 2
fi
if [ $((GLOBAL_BATCH_SIZE % DATA_PARALLEL_SIZE)) -ne 0 ]; then
  echo "GLOBAL_BATCH_SIZE must be divisible by DATA_PARALLEL_SIZE: GBS=$GLOBAL_BATCH_SIZE DP=$DATA_PARALLEL_SIZE" >&2
  exit 2
fi
EXTERNAL_ENGINE_ADDRS=()
if [ "$ROLLOUT_EXTERNAL" = "1" ]; then
  read -r -a EXTERNAL_ENGINE_ADDRS <<< "$ROLLOUT_EXTERNAL_ENGINE_ADDRS"
  EXPECTED_EXTERNAL_ENGINES=$((ROLLOUT_EXTERNAL_NUM_GPUS / ROLLOUT_NUM_GPUS_PER_ENGINE))
  if [ "$ROLLOUT_EXTERNAL_NUM_GPUS" -le 0 ] || [ $((ROLLOUT_EXTERNAL_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE)) -ne 0 ]; then
    echo "External rollout GPUs must be positive and divisible by rollout TP: gpus=$ROLLOUT_EXTERNAL_NUM_GPUS tp=$ROLLOUT_NUM_GPUS_PER_ENGINE" >&2
    exit 2
  fi
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

BATCHES_PER_ROLLOUT=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
if [[ "$ADVANTAGE_ESTIMATOR" =~ ^(grpo|gspo|reinforce_plus_plus_baseline)$ ]] && [ "$N_SAMPLES_PER_PROMPT" -lt 2 ]; then
  echo "Group-baseline ToolRL requires N_SAMPLES_PER_PROMPT >= 2 for $ADVANTAGE_ESTIMATOR; got $N_SAMPLES_PER_PROMPT" >&2
  exit 2
fi
if [ "$TOOLRL_REWARD_MODE" != "official" ] && [ "$TOOLRL_REWARD_MODE" != "molclaw" ] && [ "$TOOLRL_REWARD_MODE" != "decision_aware" ] && [ "$TOOLRL_REWARD_MODE" != "hierarchical" ]; then
  echo "TOOLRL_REWARD_MODE must be official, molclaw, decision_aware, or hierarchical; got $TOOLRL_REWARD_MODE" >&2
  exit 2
fi
if [ "$USE_ROLLOUT_LOGPROBS" != "0" ] && [ "$USE_ROLLOUT_LOGPROBS" != "1" ]; then
  echo "USE_ROLLOUT_LOGPROBS must be 0 or 1; got $USE_ROLLOUT_LOGPROBS" >&2
  exit 2
fi
if [ "$BATCHES_PER_ROLLOUT" -le 0 ]; then
  echo "ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT must be positive: ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT=$N_SAMPLES_PER_PROMPT" >&2
  exit 2
fi
if [ -n "$ROLLOUT_LONG_RESPONSE_LEN" ]; then
  if [ "$ROLLOUT_LONG_RESPONSE_LEN" -lt "$ROLLOUT_MAX_RESPONSE_LEN" ]; then
    echo "ROLLOUT_LONG_RESPONSE_LEN must be >= ROLLOUT_MAX_RESPONSE_LEN" >&2
    exit 2
  fi
  if [ -z "$CUSTOM_GENERATE_FUNCTION_PATH" ] || [ -z "$ROLLOUT_LONG_TASK_TYPES" ]; then
    echo "Long-response tier requires CUSTOM_GENERATE_FUNCTION_PATH and ROLLOUT_LONG_TASK_TYPES" >&2
    exit 2
  fi
  if [ -n "$ROLLOUT_MAX_PROMPT_LEN" ] && [ -n "$ROLLOUT_MAX_CONTEXT_LEN" ] \
      && [ $((ROLLOUT_MAX_PROMPT_LEN + ROLLOUT_LONG_RESPONSE_LEN)) -gt "$ROLLOUT_MAX_CONTEXT_LEN" ]; then
    echo "Prompt plus long response exceeds rollout context: prompt=$ROLLOUT_MAX_PROMPT_LEN long=$ROLLOUT_LONG_RESPONSE_LEN context=$ROLLOUT_MAX_CONTEXT_LEN" >&2
    exit 2
  fi
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

CKPT_ARGS=(--hf-checkpoint "$ROLLOUT_HF_CHECKPOINT" --ref-load "$REF_LOAD")
if [ "${DISABLE_CHECKPOINT_SAVE:-0}" != "1" ]; then
  CKPT_ARGS+=(--save "$SAVE_DIR" --save-interval "$SAVE_INTERVAL" --save-retain-last "$CHECKPOINT_KEEP_LAST")
fi
if [ -n "$LOAD" ]; then
  CKPT_ARGS+=(--load "$LOAD")
  if [ "$TOOLRL_RESUME" != "1" ]; then
    CKPT_ARGS+=(--finetune --no-load-optim --no-load-rng --start-rollout-id 0)
  fi
fi
if [ "${NO_SAVE_OPTIM:-0}" = "1" ]; then
  CKPT_ARGS+=(--no-save-optim --no-save-rng)
fi

TOOLRL_ARGS=(
  --distributed-timeout-minutes "$DISTRIBUTED_TIMEOUT_MINUTES"
  --rollout-function-path "$ROLLOUT_FUNCTION_PATH"
  --custom-rm-path "$CUSTOM_RM_PATH"
  --reward-key "$REWARD_KEY"

  --prompt-data "$PROMPT_DATA"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --apply-chat-template
  --apply-chat-template-kwargs "$APPLY_CHAT_TEMPLATE_KWARGS"
  --rollout-shuffle

  --advantage-estimator "$ADVANTAGE_ESTIMATOR"
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28

  --num-rollout "$NUM_ROLLOUT"
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
  --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
  --rollout-temperature "$ROLLOUT_TEMPERATURE"
  --rollout-num-gpus-per-engine "$ROLLOUT_NUM_GPUS_PER_ENGINE"
  --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --balance-data
)
if [ -n "$CUSTOM_GENERATE_FUNCTION_PATH" ]; then
  TOOLRL_ARGS+=(--custom-generate-function-path "$CUSTOM_GENERATE_FUNCTION_PATH")
fi
if [ -n "$CUSTOM_ROLLOUT_LOG_FUNCTION_PATH" ]; then
  TOOLRL_ARGS+=(--custom-rollout-log-function-path "$CUSTOM_ROLLOUT_LOG_FUNCTION_PATH")
fi
if [ -n "$ROLLOUT_ALL_SAMPLES_PROCESS_PATH" ]; then
  TOOLRL_ARGS+=(--rollout-all-samples-process-path "$ROLLOUT_ALL_SAMPLES_PROCESS_PATH")
fi
if [ -n "$ROLLOUT_LONG_RESPONSE_LEN" ]; then
  read -r -a ROLLOUT_LONG_TASK_TYPE_ARGS <<< "$ROLLOUT_LONG_TASK_TYPES"
  TOOLRL_ARGS+=(--rollout-long-response-len "$ROLLOUT_LONG_RESPONSE_LEN")
  TOOLRL_ARGS+=(--rollout-long-task-types "${ROLLOUT_LONG_TASK_TYPE_ARGS[@]}")
fi
if [ "$NORMALIZE_ADVANTAGES" = "1" ]; then
  TOOLRL_ARGS+=(--normalize-advantages)
fi
if [ -n "$DYNAMIC_SAMPLING_FILTER_PATH" ]; then
  TOOLRL_ARGS+=(--dynamic-sampling-filter-path "$DYNAMIC_SAMPLING_FILTER_PATH")
fi
if [ -n "${DYNAMIC_SAMPLING_MAX_DROPPED_GROUPS:-}" ]; then
  TOOLRL_ARGS+=(--dynamic-sampling-max-dropped-groups "$DYNAMIC_SAMPLING_MAX_DROPPED_GROUPS")
fi
if [ "${DYNAMIC_SAMPLING_STRICT_MAX_DROPS:-0}" = "1" ]; then
  TOOLRL_ARGS+=(--dynamic-sampling-strict-max-drops)
fi
if [ "$USE_ROLLOUT_LOGPROBS" = "1" ]; then
  # The generated tokens come from SGLang.  When its token log-probabilities
  # materially differ from Megatron's recomputation, use the actual behavior
  # policy as PPO's old-policy denominator instead of training against a
  # fictitious Megatron behavior distribution.
  TOOLRL_ARGS+=(--use-rollout-logprobs)
fi
if [ "$USE_KL_LOSS" = "1" ]; then
  TOOLRL_ARGS+=(--use-kl-loss --kl-loss-coef "$KL_LOSS_COEF" --kl-loss-type "$KL_LOSS_TYPE")
fi
if [ -n "${SGLANG_KV_CACHE_DTYPE:-}" ]; then
  TOOLRL_ARGS+=(--sglang-kv-cache-dtype "$SGLANG_KV_CACHE_DTYPE")
fi
if [ "${SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-0}" = "1" ]; then
  TOOLRL_ARGS+=(--sglang-disable-custom-all-reduce)
fi
# SGLang's CUDA graphs capture parameter/storage addresses.  During colocated
# RL the actor weights are updated in place between rollout batches; hybrid
# GDN/Mamba models have shown valid module tensors but corrupt graph execution
# after that update.  Keep this opt-in so unaffected profiles retain graph
# throughput, while large Qwen3.5 profiles can force eager execution.
if [ "${SGLANG_DISABLE_CUDA_GRAPH:-0}" = "1" ]; then
  TOOLRL_ARGS+=(--sglang-disable-cuda-graph)
fi
if [ "${SGLANG_DISABLE_OVERLAP_SCHEDULE:-0}" = "1" ]; then
  TOOLRL_ARGS+=(--sglang-disable-overlap-schedule)
fi
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
  --log-probs-chunk-size "$LOG_PROBS_CHUNK_SIZE"
)
if [ -n "${PIPELINE_MODEL_PARALLEL_LAYOUT:-}" ]; then
  PERF_ARGS+=(--pipeline-model-parallel-layout "$PIPELINE_MODEL_PARALLEL_LAYOUT")
else
  if [ -n "${NUM_LAYERS_IN_FIRST_PIPELINE_STAGE:-}" ]; then
    PERF_ARGS+=(--decoder-first-pipeline-num-layers "$NUM_LAYERS_IN_FIRST_PIPELINE_STAGE")
  fi
  if [ -n "${NUM_LAYERS_IN_LAST_PIPELINE_STAGE:-}" ]; then
    PERF_ARGS+=(--decoder-last-pipeline-num-layers "$NUM_LAYERS_IN_LAST_PIPELINE_STAGE")
  fi
fi
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
if [ "$RECOMPUTE_LOSS_FUNCTION" = "1" ]; then
  PERF_ARGS+=(--recompute-loss-function)
fi
if [ "$RECOMPUTE_VOCAB_LOG_PROBS" = "1" ]; then
  PERF_ARGS+=(--recompute-vocab-log-probs)
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "$LR"
  --min-lr "$MIN_LR"
  --lr-decay-style "${LR_DECAY_STYLE:-constant}"
  --lr-decay-iters "$LR_DECAY_ITERS"
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --adam-beta1 "${ADAM_BETA1:-0.9}"
  --adam-beta2 "${ADAM_BETA2:-0.95}"
)
if [ -n "$LR_WARMUP_FRACTION" ]; then
  OPTIMIZER_ARGS+=(--lr-warmup-fraction "$LR_WARMUP_FRACTION")
fi
if [ -n "$LR_WARMUP_INIT" ]; then
  OPTIMIZER_ARGS+=(--lr-warmup-init "$LR_WARMUP_INIT")
fi
if [ "${OPTIMIZER_CPU_OFFLOAD:-0}" = "1" ]; then
  OPTIMIZER_ARGS+=(--optimizer-cpu-offload)
  if [ -n "${OPTIMIZER_OFFLOAD_FRACTION:-}" ]; then
    OPTIMIZER_ARGS+=(--optimizer-offload-fraction "$OPTIMIZER_OFFLOAD_FRACTION")
  fi
  if [ "${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-1}" = "1" ]; then
    OPTIMIZER_ARGS+=(--overlap-cpu-optimizer-d2h-h2d)
  fi
fi
if [ "${USE_PRECISION_AWARE_OPTIMIZER:-0}" = "1" ]; then
  OPTIMIZER_ARGS+=(--use-precision-aware-optimizer)
fi
if [ "${OFFLOAD_OPTIMIZER_STATES:-0}" = "1" ]; then
  OPTIMIZER_ARGS+=(--offload-optimizer-states)
fi
if [ -n "${MAIN_GRADS_DTYPE:-}" ]; then
  OPTIMIZER_ARGS+=(--main-grads-dtype "$MAIN_GRADS_DTYPE")
fi
if [ -n "${MAIN_PARAMS_DTYPE:-}" ]; then
  OPTIMIZER_ARGS+=(--main-params-dtype "$MAIN_PARAMS_DTYPE")
fi
if [ -n "${EXP_AVG_DTYPE:-}" ]; then
  OPTIMIZER_ARGS+=(--exp-avg-dtype "$EXP_AVG_DTYPE")
fi
if [ -n "${EXP_AVG_SQ_DTYPE:-}" ]; then
  OPTIMIZER_ARGS+=(--exp-avg-sq-dtype "$EXP_AVG_SQ_DTYPE")
fi
if [ "${FP8_PARAM_GATHER:-0}" = "1" ]; then
  OPTIMIZER_ARGS+=(
    --fp8-format "${FP8_FORMAT:-e4m3}"
    --fp8-recipe "${FP8_RECIPE:-blockwise}"
    --fp8-param-gather
  )
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --attention-softmax-in-fp32
  --attention-backend flash
)
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
if [ "${ACCUMULATE_ALLREDUCE_GRADS_IN_FP32:-1}" = "1" ]; then
  MISC_ARGS+=(--accumulate-allreduce-grads-in-fp32)
fi
if [ "${MOE_ENABLE_DEEPEP:-0}" = "1" ]; then
  MISC_ARGS+=(--moe-token-dispatcher-type flex --moe-enable-deepep)
fi
if [ "${OVERLAP_GRAD_REDUCE:-0}" = "1" ]; then
  MISC_ARGS+=(--overlap-grad-reduce)
fi
if [ "${OVERLAP_PARAM_GATHER:-0}" = "1" ]; then
  MISC_ARGS+=(--overlap-param-gather)
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
  ray start --head \
    --node-ip-address "$MASTER_ADDR" \
    --num-gpus "$NUM_GPUS" \
    --num-cpus "$REAL_CPU" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHON_CPU_FIX_DIR}:${MEGATRON_LM_PATH}:${SLIME}:${PYTHONPATH:-}\",
    \"PYTHON_CPU_COUNT\": \"${REAL_CPU}\",
    \"PATH\": \"${PATH}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"CUDA_HOME\": \"${CUDA_HOME:-/usr/local/cuda}\",
    \"NVIDIA_VISIBLE_DEVICES\": \"${NVIDIA_VISIBLE_DEVICES:-all}\",
    \"NVIDIA_DRIVER_CAPABILITIES\": \"${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NVSHMEM_DISABLE_NCCL\": \"1\",
    \"NCCL_IB_DISABLE\": \"${NCCL_IB_DISABLE:-1}\"
    ,\"DRUG_AGENT_TRAINING_OFFLINE\": \"1\"
    ,\"DRUG_AGENT_ALLOW_TOOL_ENV\": \"0\"
    ,\"TOOLRL_REWARD_MODE\": \"${TOOLRL_REWARD_MODE}\"
    ,\"SLIME_VERIFY_FIRST_STEP_PARAMS\": \"${SLIME_VERIFY_FIRST_STEP_PARAMS:-0}\"
    ,\"SLIME_VERIFY_FIRST_UPDATE_EQUAL\": \"${SLIME_VERIFY_FIRST_UPDATE_EQUAL:-0}\"
    ,\"SLIME_LORA_SKIP_BASE_SYNC\": \"${SLIME_LORA_SKIP_BASE_SYNC:-0}\"
  }
}"

ray job submit --address="$RAY_DASHBOARD_ADDRESS" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$NUM_GPUS" \
  --num-gpus-per-node "$NUM_GPUS" \
  "${PLACEMENT_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${TOOLRL_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${LORA_ARGS[@]}"
