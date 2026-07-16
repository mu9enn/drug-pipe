#!/bin/bash
set -ex
set -o pipefail

if [ -f /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh ]; then
  source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
else
  source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi

cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh

# Colocated SGLang uses TorchMemorySaver to release and restore GPU memory.
# TorchMemorySaver currently rejects PyTorch expandable allocator segments.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
  echo "[drug_agent] Unsetting incompatible PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" >&2
  unset PYTORCH_CUDA_ALLOC_CONF
fi
if [[ "${PYTORCH_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
  echo "[drug_agent] Unsetting incompatible PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF}" >&2
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
SCRIPT_DIR="$SLIME/drug_agent"
OUTPUTS_ROOT=${OUTPUTS_ROOT:-${WD:-$GROUP_SPACE/slime-wd}/outputs}
DRUG_AGENT_DATA_ROOT=${DRUG_AGENT_DATA_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_data}
DRUG_AGENT_RUNS_ROOT=${DRUG_AGENT_RUNS_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_runs}
mkdir -p "$DRUG_AGENT_DATA_ROOT" "$DRUG_AGENT_RUNS_ROOT"

DEFAULT_SFT_DATA=${DEFAULT_SFT_DATA:-$DATA/mcp_sft_all.train.jsonl}
PROMPT_DATA_SOURCE=${PROMPT_DATA:-${PROMPT_DATA_SOURCE:-$DEFAULT_SFT_DATA}}

if [ -d "$PROMPT_DATA_SOURCE" ]; then
  MATERIALIZED_SFT_PATH=${MATERIALIZED_SFT_PATH:-${PROMPT_DATA_SOURCE%/}.train.jsonl}
  MATERIALIZED_SFT_MANIFEST=${MATERIALIZED_SFT_MANIFEST:-${MATERIALIZED_SFT_PATH%.jsonl}.manifest.json}
  mkdir -p "$(dirname "$MATERIALIZED_SFT_PATH")"
  PROMPT_DATA="$MATERIALIZED_SFT_PATH"
  python drug_agent/data/materialize_sft_jsonl.py \
    --input "$PROMPT_DATA_SOURCE" \
    --output "$PROMPT_DATA" \
    --manifest "$MATERIALIZED_SFT_MANIFEST"
else
  PROMPT_DATA="$PROMPT_DATA_SOURCE"
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
LR=${LR:-1e-5}
SFT_EPOCH_ONLY=${SFT_EPOCH_ONLY:-0}

TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}

MODEL_PARALLEL_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE * EXPERT_MODEL_PARALLEL_SIZE))
if [ "$MODEL_PARALLEL_SIZE" -le 0 ] || [ $((NUM_GPUS % MODEL_PARALLEL_SIZE)) -ne 0 ]; then
  echo "NUM_GPUS must be divisible by TP*PP*CP*EP: NUM_GPUS=$NUM_GPUS TP=$TENSOR_MODEL_PARALLEL_SIZE PP=$PIPELINE_MODEL_PARALLEL_SIZE CP=$CONTEXT_PARALLEL_SIZE EP=$EXPERT_MODEL_PARALLEL_SIZE" >&2
  exit 2
fi
DATA_PARALLEL_SIZE=$((NUM_GPUS / MODEL_PARALLEL_SIZE))

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

CKPT_ARGS=(
  --hf-checkpoint "$HF_CHECKPOINT"
  --ref-load "$REF_LOAD"
  --save "$SAVE_DIR"
  --save-interval "$SAVE_INTERVAL"
)
if [ -n "$LOAD" ]; then
  CKPT_ARGS+=(--load "$LOAD")
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
)
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
if [ "$TENSOR_MODEL_PARALLEL_SIZE" -gt 1 ]; then
  PERF_ARGS+=(--sequence-parallel)
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
if [ -n "${LR_DECAY_ITERS:-}" ]; then
  OPTIMIZER_ARGS+=(--lr-decay-iters "$LR_DECAY_ITERS")
fi
if [ -n "${MIN_LR:-}" ]; then
  OPTIMIZER_ARGS+=(--min-lr "$MIN_LR")
fi
if [ -n "${LR_WARMUP_FRACTION:-}" ]; then
  OPTIMIZER_ARGS+=(--lr-warmup-fraction "$LR_WARMUP_FRACTION")
fi

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
    \"PYTHONPATH\": \"${PYTHON_CPU_FIX_DIR}:/root/Megatron-LM/:${SLIME}:${SCRIPT_DIR:-}:${PYTHONPATH:-}\",
    \"PYTHON_CPU_COUNT\": \"${REAL_CPU}\",
    \"PATH\": \"${PATH}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"CUDA_HOME\": \"${CUDA_HOME:-/usr/local/cuda}\",
    \"NVIDIA_VISIBLE_DEVICES\": \"${NVIDIA_VISIBLE_DEVICES:-all}\",
    \"NVIDIA_DRIVER_CAPABILITIES\": \"${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
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
  --colocate \
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
