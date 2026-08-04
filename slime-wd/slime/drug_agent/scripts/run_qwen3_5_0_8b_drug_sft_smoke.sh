#!/bin/bash
set -ex
set -o pipefail

if [ -f /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh ]; then
  source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
else
  source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi

cd "$SLIME"
export OFFLOAD_OPTIMIZER_MASTER_WEIGHTS=${OFFLOAD_OPTIMIZER_MASTER_WEIGHTS:-${SFT_OFFLOAD_OPTIMIZER_MASTER_WEIGHTS:-1}}
source drug_agent/scripts/offline_training_env.sh
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-/root/Megatron-LM}

# Colocated SGLang uses TorchMemorySaver to release and restore GPU memory.
# TorchMemorySaver currently rejects PyTorch expandable allocator segments.
if [[ "${SFT_DEBUG_TRAIN_ONLY:-0}" != "1" && "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
  echo "[drug_agent] Unsetting incompatible PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" >&2
  unset PYTORCH_CUDA_ALLOC_CONF
fi
if [[ "${SFT_DEBUG_TRAIN_ONLY:-0}" != "1" && "${PYTORCH_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
  echo "[drug_agent] Unsetting incompatible PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF}" >&2
  unset PYTORCH_ALLOC_CONF
fi

unset RAY_ADDRESS || true
bash drug_agent/scripts/guard_ray_restart.sh
pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
pkill -9 -x raylet 2>/dev/null || true
pkill -9 -x gcs_server 2>/dev/null || true
sleep 2

export PYTHONBUFFERED=16
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

NUM_GPUS=${NUM_GPUS:-2}
REAL_CPU=${REAL_CPU:-$(nproc)}
SCRIPT_DIR="$SLIME/drug_agent"
OUTPUTS_ROOT=${OUTPUTS_ROOT:-${WD:-$GROUP_SPACE/slime-wd}/outputs}
DRUG_AGENT_DATA_ROOT=${DRUG_AGENT_DATA_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_data}
DRUG_AGENT_RUNS_ROOT=${DRUG_AGENT_RUNS_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_runs}
mkdir -p "$DRUG_AGENT_DATA_ROOT" "$DRUG_AGENT_RUNS_ROOT"

DEFAULT_REACT_DATA=${DEFAULT_REACT_DATA:-$DRUG_AGENT_DATA_ROOT/react_trajectories.jsonl}
REACT_DATA_SOURCE=${PROMPT_DATA:-${REACT_DATA_SOURCE:-$DEFAULT_REACT_DATA}}

if [ -d "$REACT_DATA_SOURCE" ]; then
  MATERIALIZED_REACT_PATH=${MATERIALIZED_REACT_PATH:-${REACT_DATA_SOURCE%/}.jsonl}
  MATERIALIZED_REACT_MANIFEST=${MATERIALIZED_REACT_MANIFEST:-${MATERIALIZED_REACT_PATH%.jsonl}.manifest.json}
  mkdir -p "$(dirname "$MATERIALIZED_REACT_PATH")"
  PROMPT_DATA="$MATERIALIZED_REACT_PATH"
  python drug_agent/data/materialize_sft_jsonl.py \
    --input "$REACT_DATA_SOURCE" \
    --output "$PROMPT_DATA" \
    --manifest "$MATERIALIZED_REACT_MANIFEST"
else
  PROMPT_DATA="$REACT_DATA_SOURCE"
fi

if [ ! -f "$PROMPT_DATA" ]; then
  echo "PROMPT_DATA not found: $PROMPT_DATA"
  exit 2
fi

python drug_agent/data/validate_sft_messages.py \
  --input "$PROMPT_DATA" \
  --protocol react_json

NUM_ROLLOUT=${NUM_ROLLOUT:-2}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
NUM_EPOCH=${NUM_EPOCH:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-2048}
RECOMPUTE_VOCAB_LOG_PROBS=${RECOMPUTE_VOCAB_LOG_PROBS:-0}
LR=${LR:-1e-5}
SFT_EPOCH_ONLY=${SFT_EPOCH_ONLY:-0}

TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}

MODEL_PARALLEL_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
if [ "$MODEL_PARALLEL_SIZE" -le 0 ] || [ $((NUM_GPUS % MODEL_PARALLEL_SIZE)) -ne 0 ]; then
  echo "NUM_GPUS must be divisible by TP*PP*CP: NUM_GPUS=$NUM_GPUS TP=$TENSOR_MODEL_PARALLEL_SIZE PP=$PIPELINE_MODEL_PARALLEL_SIZE CP=$CONTEXT_PARALLEL_SIZE" >&2
  exit 2
fi
DATA_PARALLEL_SIZE=$((NUM_GPUS / MODEL_PARALLEL_SIZE))
# Megatron builds the expert grid independently from the dense TP/CP grid:
# world_size must be divisible by ETP*EP*PP.  In particular CP is not part of
# this product (e.g. TP2/CP4/EP8 is valid on eight ranks when ETP=PP=1).
EXPERT_MODEL_SIZE=$((EXPERT_TENSOR_PARALLEL_SIZE * EXPERT_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE))
if [ "$EXPERT_MODEL_SIZE" -le 0 ] || [ $((NUM_GPUS % EXPERT_MODEL_SIZE)) -ne 0 ]; then
  echo "NUM_GPUS must be divisible by ETP*EP*PP: NUM_GPUS=$NUM_GPUS ETP=$EXPERT_TENSOR_PARALLEL_SIZE EP=$EXPERT_MODEL_PARALLEL_SIZE PP=$PIPELINE_MODEL_PARALLEL_SIZE" >&2
  exit 2
fi

if [ "$ROLLOUT_BATCH_SIZE" -lt "$GLOBAL_BATCH_SIZE" ]; then
  echo "ROLLOUT_BATCH_SIZE must be >= GLOBAL_BATCH_SIZE: ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE" >&2
  exit 2
fi
if [ $((ROLLOUT_BATCH_SIZE % GLOBAL_BATCH_SIZE)) -ne 0 ]; then
  echo "ROLLOUT_BATCH_SIZE must be an integer multiple of GLOBAL_BATCH_SIZE for the smoke path: ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE" >&2
  exit 2
fi
if [ $((GLOBAL_BATCH_SIZE % DATA_PARALLEL_SIZE)) -ne 0 ]; then
  echo "GLOBAL_BATCH_SIZE must be divisible by DATA_PARALLEL_SIZE: GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE DATA_PARALLEL_SIZE=$DATA_PARALLEL_SIZE" >&2
  exit 2
fi

if [ "$SFT_EPOCH_ONLY" = "1" ]; then
  echo "[drug_agent] SFT epoch-only mode: slime will derive num_rollout from dataset_size / ROLLOUT_BATCH_SIZE."
else
  DERIVED_TRAIN_ITERS=$((NUM_ROLLOUT * ROLLOUT_BATCH_SIZE / GLOBAL_BATCH_SIZE))
  if [ "$DERIVED_TRAIN_ITERS" -lt 1 ]; then
    echo "[drug_agent] Derived train_iters=$DERIVED_TRAIN_ITERS from NUM_ROLLOUT=$NUM_ROLLOUT, ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE, GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE; bumping LR_DECAY_ITERS to 1 so Megatron scheduler stays valid." >&2
    LR_DECAY_ITERS=${LR_DECAY_ITERS:-1}
  else
    LR_DECAY_ITERS=${LR_DECAY_ITERS:-$DERIVED_TRAIN_ITERS}
  fi
fi

echo "[drug_agent] SFT parallel/batch config: NUM_GPUS=$NUM_GPUS TP=$TENSOR_MODEL_PARALLEL_SIZE PP=$PIPELINE_MODEL_PARALLEL_SIZE CP=$CONTEXT_PARALLEL_SIZE EP=$EXPERT_MODEL_PARALLEL_SIZE DP=$DATA_PARALLEL_SIZE RBS=$ROLLOUT_BATCH_SIZE GBS=$GLOBAL_BATCH_SIZE NUM_ROLLOUT=$NUM_ROLLOUT NUM_EPOCH=$NUM_EPOCH EPOCH_ONLY=$SFT_EPOCH_ONLY MAX_TOKENS_PER_GPU=$MAX_TOKENS_PER_GPU"

SAVE_DIR=${SAVE_DIR:-$DRUG_AGENT_RUNS_ROOT/Qwen3.5-0.8B_drug_sft_smoke}
SAVE_INTERVAL=${SAVE_INTERVAL:-1}
CHECKPOINT_KEEP_LAST=${CHECKPOINT_KEEP_LAST:-2}
HF_CHECKPOINT=${HF_CHECKPOINT:-$DATA/Qwen3.5-0.8B}
REF_LOAD=${REF_LOAD:-$DATA/Qwen3.5-0.8B_torch_dist}
LOAD=${LOAD:-}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-0.8B.sh}
if [ ! -f "$MODEL_ARGS_FILE" ]; then
  echo "MODEL_ARGS_FILE not found: $MODEL_ARGS_FILE" >&2
  exit 2
fi
source "$MODEL_ARGS_FILE"

CKPT_ARGS=(--hf-checkpoint "$HF_CHECKPOINT" --ref-load "$REF_LOAD")
if [ "${DISABLE_CHECKPOINT_SAVE:-0}" != "1" ]; then
  CKPT_ARGS+=(
    --save "$SAVE_DIR"
    --save-interval "$SAVE_INTERVAL"
    --save-retain-last "$CHECKPOINT_KEEP_LAST"
  )
fi
if [ -n "$LOAD" ]; then
  CKPT_ARGS+=(--load "$LOAD")
fi
if [ "${NO_SAVE_OPTIM:-0}" = "1" ]; then
  CKPT_ARGS+=(--no-save-optim --no-save-rng)
fi

SFT_ARGS=(
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data "$PROMPT_DATA"
  --input-key messages
  --metadata-key metadata
  --rollout-shuffle

  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --global-batch-size "$GLOBAL_BATCH_SIZE"

  --loss-type sft_loss
  --loss-mask-type qwen3_5
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --log-probs-chunk-size "$LOG_PROBS_CHUNK_SIZE"
)
if [ -n "${SFT_MAX_SEQUENCE_LEN:-}" ]; then
  SFT_ARGS+=(--sft-max-sequence-len "$SFT_MAX_SEQUENCE_LEN")
  SFT_ARGS+=(--sft-truncation-head-tokens "${SFT_TRUNCATION_HEAD_TOKENS:-4096}")
fi
if [[ "$RECOMPUTE_VOCAB_LOG_PROBS" == 1 ]]; then
  SFT_ARGS+=(--recompute-vocab-log-probs)
fi
if [ "${RECOMPUTE_LOSS_FUNCTION:-0}" = "1" ]; then
  SFT_ARGS+=(--recompute-loss-function)
fi
if [ "$SFT_EPOCH_ONLY" = "1" ]; then
  SFT_ARGS+=(--num-epoch "$NUM_EPOCH")
else
  SFT_ARGS+=(--num-rollout "$NUM_ROLLOUT")
fi

if [ "${SFT_DEBUG_TRAIN_ONLY:-0}" = "1" ]; then
  SFT_ARGS+=(--debug-train-only)
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
if [ "${BALANCE_DATA:-0}" = "1" ]; then
  PERF_ARGS+=(--balance-data)
fi
if [ "${RECOMPUTE_FULL:-0}" = "1" ]; then
  PERF_ARGS+=(
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
  )
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "$LR"
  --lr-decay-style "${LR_DECAY_STYLE:-cosine}"
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --adam-beta1 "${ADAM_BETA1:-0.9}"
  --adam-beta2 "${ADAM_BETA2:-0.95}"
)
# Method-specific precision is useful on 8-GPU SFT: it can use precision-aware
# moments without making the colocated CPUAdam RL paths claim lower host state.
MAIN_GRADS_DTYPE=${SFT_MAIN_GRADS_DTYPE:-${MAIN_GRADS_DTYPE:-}}
MAIN_PARAMS_DTYPE=${SFT_MAIN_PARAMS_DTYPE:-${MAIN_PARAMS_DTYPE:-}}
EXP_AVG_DTYPE=${SFT_EXP_AVG_DTYPE:-${EXP_AVG_DTYPE:-}}
EXP_AVG_SQ_DTYPE=${SFT_EXP_AVG_SQ_DTYPE:-${EXP_AVG_SQ_DTYPE:-}}
if [ -n "${LR_DECAY_ITERS:-}" ]; then
  OPTIMIZER_ARGS+=(--lr-decay-iters "$LR_DECAY_ITERS")
fi
if [ -n "${MIN_LR:-}" ]; then
  OPTIMIZER_ARGS+=(--min-lr "$MIN_LR")
fi
if [ -n "${LR_WARMUP_FRACTION:-}" ]; then
  OPTIMIZER_ARGS+=(--lr-warmup-fraction "$LR_WARMUP_FRACTION")
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

PLACEMENT_ARGS=()
if [ "${SFT_DEBUG_TRAIN_ONLY:-0}" = "1" ] && [ "${SFT_DISABLE_OFFLOAD:-0}" = "1" ]; then
  # debug-train-only does not instantiate an SGLang rollout model. Avoid
  # --colocate here: colocate forces actor offload even though there is no
  # competing rollout model, adding a sleep/wake cycle around every rollout.
  PLACEMENT_ARGS+=(--no-offload-train --no-offload-rollout)
  echo "[drug_agent] Train-only SFT: actor remains resident on GPU (colocate/offload disabled)."
else
  PLACEMENT_ARGS+=(--colocate)
fi

ray start --head \
  --node-ip-address "$MASTER_ADDR" \
  --num-gpus "$NUM_GPUS" \
  --num-cpus "$REAL_CPU" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

collect_ray_job_logs_on_failure() {
  local submit_log="$1"
  local status="$2"
  local job_id
  job_id=$(grep -Eo 'raysubmit_[A-Za-z0-9]+' "$submit_log" | tail -1 || true)

  echo "[drug_agent] ray job submit failed with exit code ${status}" >&2
  echo "[drug_agent] ray submit log: ${submit_log}" >&2

  if [ -z "$job_id" ]; then
    echo "[drug_agent] could not find raysubmit_* job id in ${submit_log}" >&2
    return
  fi

  local job_log="${submit_log%.log}.${job_id}.full.log"
  local error_log="${submit_log%.log}.${job_id}.first_error.log"
  echo "[drug_agent] collecting full Ray job log for ${job_id}: ${job_log}" >&2
  ray job logs "$job_id" --address=http://127.0.0.1:8265 > "$job_log" 2>&1 || true

  local line
  line=$(grep -nEi 'traceback|runtimeerror|assertionerror|outofmemoryerror|raytaskerror|exception|sigkill|sigterm|killed|nccl' "$job_log" | head -1 | cut -d: -f1 || true)
  if [ -n "$line" ]; then
    local start=$((line > 40 ? line - 40 : 1))
    local end=$((line + 160))
    sed -n "${start},${end}p" "$job_log" > "$error_log" || true
    echo "[drug_agent] first error context: ${error_log}" >&2
    cat "$error_log" >&2
  else
    echo "[drug_agent] no traceback-like line found in ${job_log}; searching Ray worker logs" >&2
    grep -RniE 'traceback|runtimeerror|assertionerror|outofmemoryerror|raytaskerror|nccl|sigkill|sigterm|killed' \
      /tmp/ray/session_latest/logs 2>/dev/null | head -100 > "$error_log" || true
    echo "[drug_agent] Ray worker error grep: ${error_log}" >&2
    cat "$error_log" >&2 || true
  fi
}

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHON_CPU_FIX_DIR}:${MEGATRON_LM_PATH}:${SLIME}:${SCRIPT_DIR:-}:${PYTHONPATH:-}\",
    \"PYTHON_CPU_COUNT\": \"${REAL_CPU}\",
    \"PATH\": \"${PATH}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"CUDA_HOME\": \"${CUDA_HOME:-/usr/local/cuda}\",
    \"NVIDIA_VISIBLE_DEVICES\": \"${NVIDIA_VISIBLE_DEVICES:-all}\",
    \"NVIDIA_DRIVER_CAPABILITIES\": \"${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF:-}\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NVSHMEM_DISABLE_NCCL\": \"1\",
    \"NCCL_IB_DISABLE\": \"${NCCL_IB_DISABLE:-1}\",
    \"OUTPUTS_ROOT\": \"${OUTPUTS_ROOT}\",
    \"DRUG_AGENT_DATA_ROOT\": \"${DRUG_AGENT_DATA_ROOT}\",
    \"DRUG_AGENT_RUNS_ROOT\": \"${DRUG_AGENT_RUNS_ROOT}\"
    ,\"DRUG_AGENT_TRAINING_OFFLINE\": \"1\"
    ,\"DRUG_AGENT_ALLOW_TOOL_ENV\": \"0\"
  }
}"

RAY_SUBMIT_LOG=${RAY_SUBMIT_LOG:-$DRUG_AGENT_RUNS_ROOT/qwen3_5_sft_ray_submit_$(date +%Y%m%d_%H%M%S).log}
set +e
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$NUM_GPUS" \
  --num-gpus-per-node "$NUM_GPUS" \
  "${PLACEMENT_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${SFT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  2>&1 | tee "$RAY_SUBMIT_LOG"
RAY_JOB_STATUS=${PIPESTATUS[0]}
set -e

if [ "$RAY_JOB_STATUS" -ne 0 ]; then
  collect_ray_job_logs_on_failure "$RAY_SUBMIT_LOG" "$RAY_JOB_STATUS"
  exit "$RAY_JOB_STATUS"
fi
