#!/usr/bin/env bash
# Size-independent canonical-data production pipeline for Qwen3.5-9B on 8 x H200.
# The branches are SFT -> ToolRL and SFT -> GAD; GAD does not load ToolRL.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/resolve_slime_env.sh"
source "$SLIME_ENV"
cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONUNBUFFERED=1

SLIME_WD_ROOT="${WD:-$(cd "$SLIME/.." && pwd)}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-$SLIME_WD_ROOT/outputs}"
MODEL_DATA_ROOT="${DATA:-$SLIME_WD_ROOT/data}"
DATA_ROOT="${DRUG_AGENT_DATA_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_data}"
RUNS_ROOT="${DRUG_AGENT_RUNS_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_runs}"
LIVE_DATA_ROOT="${LIVE_DATA_ROOT:-$DATA_ROOT/live_tool_catalog_v3}"
CANONICAL_DATA="${CANONICAL_DATA:-$LIVE_DATA_ROOT/react_trajectories.jsonl}"
TOOLRL_DATA="${TOOLRL_DATA:-$LIVE_DATA_ROOT/toolrl/toolrl_steps.jsonl}"
GAD_DATA="${GAD_DATA:-$LIVE_DATA_ROOT/gad/gad_steps.jsonl}"
DRUG_AGENT_TOOL_CATALOG="${DRUG_AGENT_TOOL_CATALOG:-$LIVE_DATA_ROOT/tool_catalog.json}"
export DRUG_AGENT_TOOL_CATALOG
MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-scripts/models/qwen3.5-9B.sh}"
HF_CHECKPOINT="${HF_CHECKPOINT:-$MODEL_DATA_ROOT/Qwen3.5-9B}"
REF_LOAD="${REF_LOAD:-$MODEL_DATA_ROOT/Qwen3.5-9B_torch_dist}"
DISCRIMINATOR_MODEL_PATH="${DISCRIMINATOR_MODEL_PATH:-$MODEL_DATA_ROOT/Qwen3.5-0.8B}"
SERIAL_RUN_ID="${SERIAL_RUN_ID:-Qwen3.5-9B_serial_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$RUNS_ROOT/$SERIAL_RUN_ID}"
LOG_ROOT="$RUN_ROOT/logs"
ALIGNED_DATA_ROOT="$RUN_ROOT/aligned_data"
SFT_TRAIN_DATA="$ALIGNED_DATA_ROOT/react_trajectories.gbs2.jsonl"
TOOLRL_LENGTH_DATA="$ALIGNED_DATA_ROOT/toolrl_steps.max_prompt.jsonl"
GAD_LENGTH_DATA="$ALIGNED_DATA_ROOT/gad_steps.max_prompt.jsonl"
RL_LENGTH_MANIFEST="$ALIGNED_DATA_ROOT/rl_prompt_length.manifest.json"
TOOLRL_TRAIN_DATA="$ALIGNED_DATA_ROOT/toolrl_steps.rbs8.jsonl"
GAD_TRAIN_DATA="$ALIGNED_DATA_ROOT/gad_steps.rbs8.jsonl"

SFT_DIR="${SFT_DIR:-$RUN_ROOT/sft}"
TOOLRL_DIR="$RUN_ROOT/toolrl"
NEGATIVE_CACHE="$RUN_ROOT/gad_stage2_negatives.jsonl"
DISCRIMINATOR_DIR="$RUN_ROOT/gad_discriminator_warmup"
DISCRIMINATOR_LATEST="$DISCRIMINATOR_DIR/latest"
GAD_DIR="$RUN_ROOT/gad"
GAD_SERVICE_DIR="$RUN_ROOT/gad_discriminator_online"
GAD_SERVICE_PORT="${GAD_SERVICE_PORT:-8100}"
GAD_SERVICE_URL="http://127.0.0.1:${GAD_SERVICE_PORT}"
SFT_MAX_SEQUENCE_LEN="${SFT_MAX_SEQUENCE_LEN:-131072}"
SFT_TRUNCATION_HEAD_TOKENS="${SFT_TRUNCATION_HEAD_TOKENS:-8192}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-114688}"
SKIP_SFT="${SKIP_SFT:-0}"
RESTART_AFTER_MATERIALIZE="${RESTART_AFTER_MATERIALIZE:-0}"
export CHECKPOINT_KEEP_LAST="${CHECKPOINT_KEEP_LAST:-2}"
export CHECKPOINT_FINAL_KEEP="${CHECKPOINT_FINAL_KEEP:-1}"
export DISCRIMINATOR_KEEP_LAST="${DISCRIMINATOR_KEEP_LAST:-2}"

for value in "$CHECKPOINT_KEEP_LAST" "$CHECKPOINT_FINAL_KEEP" "$DISCRIMINATOR_KEEP_LAST"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Checkpoint retention values must be positive integers; got $value" >&2
    exit 2
  fi
done
if [[ ! "$SFT_MAX_SEQUENCE_LEN" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$SFT_TRUNCATION_HEAD_TOKENS" =~ ^[1-9][0-9]*$ ]] || \
   (( SFT_TRUNCATION_HEAD_TOKENS >= SFT_MAX_SEQUENCE_LEN )); then
  echo "Invalid SFT truncation contract: max=$SFT_MAX_SEQUENCE_LEN head=$SFT_TRUNCATION_HEAD_TOKENS" >&2
  exit 2
fi
for flag in "$SKIP_SFT" "$RESTART_AFTER_MATERIALIZE"; do
  if [[ "$flag" != 0 && "$flag" != 1 ]]; then
    echo "Boolean training flags must be 0 or 1; got $flag" >&2
    exit 2
  fi
done
if [[ ! "$ROLLOUT_MAX_PROMPT_LEN" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROLLOUT_MAX_PROMPT_LEN must be a positive integer; got $ROLLOUT_MAX_PROMPT_LEN" >&2
  exit 2
fi

require_path() {
  [[ -e "$1" ]] || { echo "Required path does not exist: $1" >&2; exit 2; }
}

finalize_checkpoint_stage() {
  local save_dir="$1"
  local complete_marker="$2"
  require_path "$save_dir/latest_checkpointed_iteration.txt"
  if ! python -m slime.utils.checkpoint_retention \
    --save-dir "$save_dir" --keep-last "$CHECKPOINT_FINAL_KEEP"; then
    rm -f "$complete_marker"
    echo "Checkpoint finalization failed; removed completion marker: $complete_marker" >&2
    return 1
  fi
}

require_path "$CANONICAL_DATA"
require_path "$TOOLRL_DATA"
require_path "$GAD_DATA"
require_path "$DRUG_AGENT_TOOL_CATALOG"
require_path "$HF_CHECKPOINT"
require_path "$DISCRIMINATOR_MODEL_PATH"
require_path "$MODEL_ARGS_FILE"
command -v nvidia-smi >/dev/null
command -v ray >/dev/null
command -v curl >/dev/null

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( GPU_COUNT != 8 )); then
  echo "This profile requires exactly 8 visible GPUs; found $GPU_COUNT" >&2
  exit 2
fi
NON_H200=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -vc 'H200' || true)
if (( NON_H200 != 0 )); then
  echo "This profile was sized for H200 GPUs:" >&2
  nvidia-smi --query-gpu=index,name,memory.total --format=csv >&2
  exit 2
fi
if [[ "${ALLOW_BUSY_GPUS:-0}" != 1 ]]; then
  BUSY_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  if [[ -n "$BUSY_PIDS" ]]; then
    echo "GPU compute processes already exist: $BUSY_PIDS" >&2
    exit 2
  fi
fi

SOURCE_COUNT=$(wc -l < "$CANONICAL_DATA")
if (( SOURCE_COUNT < 1 )); then
  echo "Canonical dataset is empty: $CANONICAL_DATA" >&2
  exit 2
fi

# Prepare the release checkpoint once. This is model conversion, not training.
if [[ ! -f "$REF_LOAD/latest_checkpointed_iteration.txt" ]]; then
  echo "[serial] preparing missing Qwen3.5-9B torch_dist checkpoint: $REF_LOAD"
  CUDA_VISIBLE_DEVICES=0 \
    MODEL_ARGS_FILE="$MODEL_ARGS_FILE" HF_CHECKPOINT="$HF_CHECKPOINT" \
    SAVE_DIR="$REF_LOAD" NUM_GPUS=1 \
    bash drug_agent/scripts/prepare_qwen3_5_9B_torch_dist.sh
fi
require_path "$REF_LOAD/latest_checkpointed_iteration.txt"

if [[ -e "$RUN_ROOT" && "${RESUME_SERIAL_RUN:-0}" != 1 && "$RESTART_AFTER_MATERIALIZE" != 1 ]]; then
  echo "RUN_ROOT already exists: $RUN_ROOT" >&2
  echo "Set RESUME_SERIAL_RUN=1 only to continue this exact serial run." >&2
  exit 2
fi
mkdir -p "$LOG_ROOT"
if [[ "${RESUME_SERIAL_RUN:-0}" == 1 ]]; then
  require_path "$RUN_ROOT/serial_config.env"
  echo "[$(date --iso-8601=seconds)] RESUME" | tee -a "$RUN_ROOT/serial_status.log"
elif [[ "$RESTART_AFTER_MATERIALIZE" == 1 ]]; then
  require_path "$RUN_ROOT/materialize_training_data.complete"
  if [[ -e "$RUN_ROOT/serial_config.env" ]]; then
    echo "RESTART_AFTER_MATERIALIZE is only for a failure before serial_config.env; use RESUME_SERIAL_RUN=1 instead" >&2
    exit 2
  fi
  echo "[$(date --iso-8601=seconds)] RESTART_AFTER_MATERIALIZE" | tee -a "$RUN_ROOT/serial_status.log"
fi

run_logged() {
  local stage="$1"
  shift
  echo "[$(date --iso-8601=seconds)] START $stage" | tee -a "$RUN_ROOT/serial_status.log"
  "$@" 2>&1 | tee -a "$LOG_ROOT/${stage}.log"
  touch "$RUN_ROOT/${stage}.complete"
  echo "[$(date --iso-8601=seconds)] COMPLETE $stage" | tee -a "$RUN_ROOT/serial_status.log"
}

DISCRIMINATOR_PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$DISCRIMINATOR_PID" ]] && kill -0 "$DISCRIMINATOR_PID" 2>/dev/null; then
    kill "$DISCRIMINATOR_PID" 2>/dev/null || true
    wait "$DISCRIMINATOR_PID" 2>/dev/null || true
  fi
  if (( status != 0 )); then
    echo "[$(date --iso-8601=seconds)] FAILED exit=$status" | tee -a "$RUN_ROOT/serial_status.log" >&2
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$ALIGNED_DATA_ROOT"
if [[ ! -f "$RUN_ROOT/materialize_training_data.complete" ]]; then
  run_logged materialize_training_data \
    bash -c '
      set -euo pipefail
      python drug_agent/scripts/materialize_batch_aligned_sft.py \
        --input "$1" --output "$2" --manifest "$3" --model "$4" --multiple 2
      python drug_agent/scripts/materialize_length_filtered_rl.py \
        --toolrl-input "$5" --gad-input "$8" \
        --toolrl-output "${11}" --gad-output "${12}" --manifest "${13}" \
        --model "$4" --max-prompt-tokens "${14}"
      python drug_agent/scripts/materialize_batch_aligned_jsonl.py \
        --input "${11}" --output "$6" --manifest "$7" --multiple 8
      python drug_agent/scripts/materialize_batch_aligned_jsonl.py \
        --input "${12}" --output "$9" --manifest "${10}" --multiple 8
    ' _ \
      "$CANONICAL_DATA" "$SFT_TRAIN_DATA" "$ALIGNED_DATA_ROOT/sft.manifest.json" "$HF_CHECKPOINT" \
      "$TOOLRL_DATA" "$TOOLRL_TRAIN_DATA" "$ALIGNED_DATA_ROOT/toolrl.manifest.json" \
      "$GAD_DATA" "$GAD_TRAIN_DATA" "$ALIGNED_DATA_ROOT/gad.manifest.json" \
      "$TOOLRL_LENGTH_DATA" "$GAD_LENGTH_DATA" "$RL_LENGTH_MANIFEST" "$ROLLOUT_MAX_PROMPT_LEN"
else
  echo "[serial] SKIP completed materialize_training_data"
fi
for path in "$SFT_TRAIN_DATA" "$TOOLRL_LENGTH_DATA" "$GAD_LENGTH_DATA" \
  "$RL_LENGTH_MANIFEST" "$TOOLRL_TRAIN_DATA" "$GAD_TRAIN_DATA" \
  "$ALIGNED_DATA_ROOT/sft.manifest.json" "$ALIGNED_DATA_ROOT/toolrl.manifest.json" \
  "$ALIGNED_DATA_ROOT/gad.manifest.json"; do
  require_path "$path"
done

SFT_COUNT=$(wc -l < "$SFT_TRAIN_DATA")
TOOLRL_COUNT=$(wc -l < "$TOOLRL_TRAIN_DATA")
GAD_COUNT=$(wc -l < "$GAD_TRAIN_DATA")
if (( SFT_COUNT % 2 != 0 || TOOLRL_COUNT % 8 != 0 || GAD_COUNT % 8 != 0 )); then
  echo "Aligned data contract failed: SFT=$SFT_COUNT ToolRL=$TOOLRL_COUNT GAD=$GAD_COUNT" >&2
  exit 2
fi

DATA_SHA256=$(sha256sum "$CANONICAL_DATA" | awk '{print $1}')
TOOLRL_SHA256=$(sha256sum "$TOOLRL_DATA" | awk '{print $1}')
GAD_SHA256=$(sha256sum "$GAD_DATA" | awk '{print $1}')
CODE_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
if [[ "${RESUME_SERIAL_RUN:-0}" != 1 ]]; then
cat > "$RUN_ROOT/serial_config.env" <<EOF
SERIAL_RUN_ID=$SERIAL_RUN_ID
CODE_COMMIT=$CODE_COMMIT
CANONICAL_DATA=$CANONICAL_DATA
CANONICAL_RECORDS=$SOURCE_COUNT
CANONICAL_SHA256=$DATA_SHA256
SFT_TRAIN_DATA=$SFT_TRAIN_DATA
SFT_TRAIN_RECORDS=$SFT_COUNT
TOOLRL_DATA=$TOOLRL_DATA
TOOLRL_RECORDS=$(wc -l < "$TOOLRL_DATA")
TOOLRL_SHA256=$TOOLRL_SHA256
TOOLRL_LENGTH_DATA=$TOOLRL_LENGTH_DATA
TOOLRL_LENGTH_RECORDS=$(wc -l < "$TOOLRL_LENGTH_DATA")
TOOLRL_TRAIN_DATA=$TOOLRL_TRAIN_DATA
TOOLRL_TRAIN_RECORDS=$TOOLRL_COUNT
GAD_DATA=$GAD_DATA
GAD_RECORDS=$(wc -l < "$GAD_DATA")
GAD_SHA256=$GAD_SHA256
GAD_LENGTH_DATA=$GAD_LENGTH_DATA
GAD_LENGTH_RECORDS=$(wc -l < "$GAD_LENGTH_DATA")
GAD_TRAIN_DATA=$GAD_TRAIN_DATA
GAD_TRAIN_RECORDS=$GAD_COUNT
MODEL_ARGS_FILE=$MODEL_ARGS_FILE
HF_CHECKPOINT=$HF_CHECKPOINT
REF_LOAD=$REF_LOAD
SFT_SAVE_DIR=$SFT_DIR
SKIP_SFT=$SKIP_SFT
RESTART_AFTER_MATERIALIZE=$RESTART_AFTER_MATERIALIZE
SFT_MAX_SEQUENCE_LEN=$SFT_MAX_SEQUENCE_LEN
SFT_TRUNCATION_HEAD_TOKENS=$SFT_TRUNCATION_HEAD_TOKENS
TOOLRL_SAVE_DIR=$TOOLRL_DIR
GAD_NEGATIVE_CACHE=$NEGATIVE_CACHE
GAD_DISCRIMINATOR_DIR=$DISCRIMINATOR_DIR
GAD_SAVE_DIR=$GAD_DIR
ROLLOUT_MAX_PROMPT_LEN=$ROLLOUT_MAX_PROMPT_LEN
ROLLOUT_MAX_RESPONSE_LEN=8192
ROLLOUT_LONG_RESPONSE_LEN=16384
ROLLOUT_LONG_TASK_TYPES=vs,pf
ROLLOUT_MAX_CONTEXT_LEN=131072
SFT_EPOCHS=${SFT_EPOCHS:-1}
TOOLRL_EPOCHS=${TOOLRL_EPOCHS:-1}
GAD_EPOCHS=${GAD_EPOCHS:-1}
SAVE_OPTIMIZER=1
CHECKPOINT_KEEP_LAST=$CHECKPOINT_KEEP_LAST
CHECKPOINT_FINAL_KEEP=$CHECKPOINT_FINAL_KEEP
DISCRIMINATOR_KEEP_LAST=$DISCRIMINATOR_KEEP_LAST
EOF
else
  RECORDED_SHA256=$(sed -n 's/^CANONICAL_SHA256=//p' "$RUN_ROOT/serial_config.env")
  if [[ "$RECORDED_SHA256" != "$DATA_SHA256" ]]; then
    echo "Canonical data changed since the original run: $RECORDED_SHA256 != $DATA_SHA256" >&2
    exit 2
  fi
  RECORDED_TOOLRL_SHA256=$(sed -n 's/^TOOLRL_SHA256=//p' "$RUN_ROOT/serial_config.env")
  RECORDED_GAD_SHA256=$(sed -n 's/^GAD_SHA256=//p' "$RUN_ROOT/serial_config.env")
  if [[ "$RECORDED_TOOLRL_SHA256" != "$TOOLRL_SHA256" || "$RECORDED_GAD_SHA256" != "$GAD_SHA256" ]]; then
    echo "Derived data changed since the original run; start a new RUN_ROOT" >&2
    exit 2
  fi
fi

echo "[serial] run_root=$RUN_ROOT"
echo "[serial] optimizer state is enabled for SFT, ToolRL, GAD generator, and discriminator checkpoints"
echo "[serial] branch topology: Qwen3.5-9B base -> SFT -> {ToolRL, GAD}"

# Dataset size controls only duration. Topology, batch shape and optimizer
# hyperparameters remain fixed; audited derivatives absorb non-divisible tails.
SFT_EPOCHS=${SFT_EPOCHS:-1}
TOOLRL_EPOCHS=${TOOLRL_EPOCHS:-1}
GAD_EPOCHS=${GAD_EPOCHS:-1}
for value in "$SFT_EPOCHS" "$TOOLRL_EPOCHS" "$GAD_EPOCHS"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Stage epoch counts must be positive integers; got $value" >&2
    exit 2
  fi
done
TOOLRL_NUM_ROLLOUT=$((TOOLRL_EPOCHS * TOOLRL_COUNT / 8))
GAD_STAGE2_NUM_ROLLOUT=$((GAD_COUNT / 8))
GAD_NUM_ROLLOUT=$((GAD_EPOCHS * GAD_COUNT))

# SFT: the measured throughput winner is TP4/PP1/DP2 with GBS2. A new
# derived-data release may reuse an explicitly supplied, completed SFT branch.
if [[ "$SKIP_SFT" == 1 ]]; then
  require_path "$SFT_DIR/latest_checkpointed_iteration.txt"
  touch "$RUN_ROOT/sft.complete"
  echo "[serial] REUSE completed external SFT checkpoint: $SFT_DIR"
elif [[ ! -f "$RUN_ROOT/sft.complete" ]]; then
run_logged sft \
  env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    PROMPT_DATA="$SFT_TRAIN_DATA" SAVE_DIR="$SFT_DIR" RUN_NAME="${SERIAL_RUN_ID}_sft" \
    MODEL_ARGS_FILE="$MODEL_ARGS_FILE" HF_CHECKPOINT="$HF_CHECKPOINT" REF_LOAD="$REF_LOAD" \
    NUM_GPUS=8 TENSOR_MODEL_PARALLEL_SIZE=4 PIPELINE_MODEL_PARALLEL_SIZE=1 \
    CONTEXT_PARALLEL_SIZE=1 EXPERT_MODEL_PARALLEL_SIZE=1 EXPERT_TENSOR_PARALLEL_SIZE=1 \
    NUM_EPOCH="$SFT_EPOCHS" ROLLOUT_BATCH_SIZE="$SFT_COUNT" GLOBAL_BATCH_SIZE=2 MAX_TOKENS_PER_GPU=16384 \
    SFT_MAX_SEQUENCE_LEN="$SFT_MAX_SEQUENCE_LEN" SFT_TRUNCATION_HEAD_TOKENS="$SFT_TRUNCATION_HEAD_TOKENS" \
    LR=5e-6 MIN_LR=5e-7 LR_WARMUP_FRACTION=0.05 SAVE_INTERVAL=100 \
    RECOMPUTE_FULL=1 RECOMPUTE_NUM_LAYERS=1 RECOMPUTE_LOSS_FUNCTION=1 \
    RECOMPUTE_VOCAB_LOG_PROBS=1 LOG_PROBS_CHUNK_SIZE=64 BALANCE_DATA=1 \
    SFT_DEBUG_TRAIN_ONLY=1 SFT_DISABLE_OFFLOAD=1 \
    bash drug_agent/scripts/run_qwen3_5_9b_drug_sft_full.sh
else
  echo "[serial] SKIP completed sft"
fi
finalize_checkpoint_stage "$SFT_DIR" "$RUN_ROOT/sft.complete"

# ToolRL: distinct prompts with REINFORCE++ avoid zero-variance grouped rewards.
if [[ ! -f "$RUN_ROOT/toolrl.complete" ]]; then
run_logged toolrl \
  env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    PROMPT_DATA="$TOOLRL_TRAIN_DATA" SAVE_DIR="$TOOLRL_DIR" LOAD="$SFT_DIR" TOOLRL_RESUME=0 \
    MODEL_ARGS_FILE="$MODEL_ARGS_FILE" HF_CHECKPOINT="$HF_CHECKPOINT" REF_LOAD="$REF_LOAD" \
    NUM_GPUS=8 TENSOR_MODEL_PARALLEL_SIZE=4 PIPELINE_MODEL_PARALLEL_SIZE=1 \
    CONTEXT_PARALLEL_SIZE=1 EXPERT_MODEL_PARALLEL_SIZE=1 EXPERT_TENSOR_PARALLEL_SIZE=1 \
    ROLLOUT_NUM_GPUS_PER_ENGINE=1 ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=1 \
    GLOBAL_BATCH_SIZE=8 NUM_ROLLOUT="$TOOLRL_NUM_ROLLOUT" \
    ADVANTAGE_ESTIMATOR=reinforce_plus_plus NORMALIZE_ADVANTAGES=1 \
    ROLLOUT_MAX_PROMPT_LEN="$ROLLOUT_MAX_PROMPT_LEN" ROLLOUT_MAX_RESPONSE_LEN=8192 ROLLOUT_LONG_RESPONSE_LEN=16384 \
    ROLLOUT_LONG_TASK_TYPES='vs pf' ROLLOUT_MAX_CONTEXT_LEN=131072 \
    CUSTOM_GENERATE_FUNCTION_PATH=drug_agent.rollout.length_aware_generate.generate \
    ROLLOUT_TEMPERATURE=0.8 SGLANG_MEM_FRACTION_STATIC=0.25 MAX_TOKENS_PER_GPU=16384 \
    LOG_PROBS_CHUNK_SIZE=64 RECOMPUTE_LOSS_FUNCTION=1 RECOMPUTE_VOCAB_LOG_PROBS=1 \
    TOOLRL_REWARD_MODE=molclaw LR=2e-7 MIN_LR=2e-8 LR_DECAY_STYLE=cosine \
    LR_WARMUP_FRACTION=0.03 WEIGHT_DECAY=0.1 SAVE_INTERVAL=200 \
    RECOMPUTE_FULL=1 RECOMPUTE_NUM_LAYERS=1 COLOCATE_OFFLOAD_TRAIN=0 COLOCATE_OFFLOAD_ROLLOUT=0 \
    SGLANG_DISABLE_CUDA_GRAPH=1 SGLANG_DISABLE_CUSTOM_ALL_REDUCE=1 SGLANG_DISABLE_OVERLAP_SCHEDULE=1 \
    USE_ROLLOUT_LOGPROBS=0 USE_KL_LOSS=0 \
    bash drug_agent/toolrl/scripts/run_toolrl_grpo.sh
else
  echo "[serial] SKIP completed toolrl"
fi
finalize_checkpoint_stage "$TOOLRL_DIR" "$RUN_ROOT/toolrl.complete"

# GAD Stage 2 starts from SFT (not ToolRL) and covers the aligned corpus once.
if [[ ! -f "$RUN_ROOT/gad_negative_generation.complete" ]]; then
run_logged gad_negative_generation \
  env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    PROMPT_DATA="$GAD_TRAIN_DATA" GAD_NEGATIVE_CACHE="$NEGATIVE_CACHE" STUDENT_LOAD="$SFT_DIR" \
    MODEL_ARGS_FILE="$MODEL_ARGS_FILE" HF_CHECKPOINT="$HF_CHECKPOINT" REF_LOAD="$REF_LOAD" \
    NUM_GPUS=8 TENSOR_MODEL_PARALLEL_SIZE=4 PIPELINE_MODEL_PARALLEL_SIZE=1 \
    ROLLOUT_NUM_GPUS_PER_ENGINE=1 ROLLOUT_BATCH_SIZE=8 NUM_ROLLOUT="$GAD_STAGE2_NUM_ROLLOUT" \
    ROLLOUT_MAX_PROMPT_LEN="$ROLLOUT_MAX_PROMPT_LEN" ROLLOUT_MAX_RESPONSE_LEN=8192 ROLLOUT_LONG_RESPONSE_LEN=16384 \
    ROLLOUT_LONG_TASK_TYPES='vs pf' ROLLOUT_MAX_CONTEXT_LEN=131072 \
    CUSTOM_GENERATE_FUNCTION_PATH=drug_agent.rollout.length_aware_generate.generate \
    ROLLOUT_TEMPERATURE=0.8 SGLANG_MEM_FRACTION_STATIC=0.25 MAX_TOKENS_PER_GPU=16384 \
    LOG_PROBS_CHUNK_SIZE=64 RECOMPUTE_VOCAB_LOG_PROBS=1 \
    COLOCATE_OFFLOAD_TRAIN=0 COLOCATE_OFFLOAD_ROLLOUT=0 \
    SGLANG_DISABLE_CUDA_GRAPH=1 SGLANG_DISABLE_CUSTOM_ALL_REDUCE=1 SGLANG_DISABLE_OVERLAP_SCHEDULE=1 \
    bash drug_agent/gad/scripts/generate_stage2_negatives.sh
else
  echo "[serial] SKIP completed gad_negative_generation"
fi
require_path "$NEGATIVE_CACHE"
[[ -s "$NEGATIVE_CACHE" ]] || { echo "Empty GAD negative cache: $NEGATIVE_CACHE" >&2; exit 2; }

ray stop --force >/dev/null 2>&1 || true
pkill -9 sglang 2>/dev/null || true

# A 0.8B discriminator leaves enough shared-HBM headroom for the resident 9B
# actor. Its independently validated default remains 8K; test 16K/32K before
# overriding this because the rollout response gate does not prove RM memory.
if [[ ! -f "$RUN_ROOT/gad_discriminator_warmup.complete" ]]; then
run_logged gad_discriminator_warmup \
  env CUDA_VISIBLE_DEVICES=7 \
    PAIRS="$NEGATIVE_CACHE" GENERATOR_WARMUP_LOAD="$SFT_DIR" \
    DISCRIMINATOR_MODEL_PATH="$DISCRIMINATOR_MODEL_PATH" DISCRIMINATOR_OUTPUT_DIR="$DISCRIMINATOR_DIR" \
    DISCRIMINATOR_EPOCHS="${DISCRIMINATOR_EPOCHS:-1}" DISCRIMINATOR_BATCH_SIZE=2 DISCRIMINATOR_LR=1e-6 \
    DISCRIMINATOR_MAX_LENGTH="${DISCRIMINATOR_MAX_LENGTH:-8192}" \
    DISCRIMINATOR_CLIP_GRAD=1.0 DISCRIMINATOR_SAVE_INTERVAL=400 \
    DISCRIMINATOR_KEEP_LAST="$DISCRIMINATOR_KEEP_LAST" \
    bash drug_agent/gad/scripts/run_stage2_discriminator_warmup.sh
else
  echo "[serial] SKIP completed gad_discriminator_warmup"
fi
require_path "$DISCRIMINATOR_LATEST"
require_path "$DISCRIMINATOR_DIR/warmup_manifest.json"

# Pin the online discriminator to physical GPU 7. CUDA visibility remaps it to
# logical `cuda`; the TP4/DP2 actor still uses all eight physical GPUs.
if [[ ! -f "$RUN_ROOT/gad.complete" ]]; then
CUDA_VISIBLE_DEVICES=7 \
  DISCRIMINATOR_MODEL_PATH="$DISCRIMINATOR_MODEL_PATH" \
  DISCRIMINATOR_RESUME="$DISCRIMINATOR_LATEST" \
  DISCRIMINATOR_OUTPUT_DIR="$GAD_SERVICE_DIR" \
  GAD_DISCRIMINATOR_HOST=127.0.0.1 GAD_DISCRIMINATOR_PORT="$GAD_SERVICE_PORT" \
  GAD_DISCRIMINATOR_DEVICE=cuda DISCRIMINATOR_LR=1e-6 \
  DISCRIMINATOR_MAX_LENGTH="${DISCRIMINATOR_MAX_LENGTH:-8192}" \
  DISCRIMINATOR_UPDATE_STEPS=1 DISCRIMINATOR_CLIP_GRAD=1.0 \
  DISCRIMINATOR_REWARD_CLIP=2.0 DISCRIMINATOR_SAVE_INTERVAL=400 \
  DISCRIMINATOR_KEEP_LAST="$DISCRIMINATOR_KEEP_LAST" \
  DISCRIMINATOR_OFFLOAD_AFTER_REQUEST=1 \
  bash drug_agent/gad/scripts/serve_discriminator.sh > "$LOG_ROOT/gad_discriminator_service.log" 2>&1 &
DISCRIMINATOR_PID=$!

SERVICE_READY=0
for _ in $(seq 1 180); do
  if ! kill -0 "$DISCRIMINATOR_PID" 2>/dev/null; then
    echo "GAD discriminator exited during startup; see $LOG_ROOT/gad_discriminator_service.log" >&2
    exit 2
  fi
  if curl -fsS "$GAD_SERVICE_URL/health" >/dev/null 2>&1; then
    SERVICE_READY=1
    break
  fi
  sleep 2
done
[[ "$SERVICE_READY" -eq 1 ]] || { echo "GAD discriminator did not become ready" >&2; exit 2; }

run_logged gad \
  env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    PROMPT_DATA="$GAD_TRAIN_DATA" SAVE_DIR="$GAD_DIR" \
    STUDENT_WARMUP_LOAD="$SFT_DIR" DISCRIMINATOR_WARMUP_LOAD="$DISCRIMINATOR_LATEST" \
    GAD_WARMUP_MANIFEST="$DISCRIMINATOR_DIR/warmup_manifest.json" \
    GAD_REWARD_MODE=hybrid GAD_DISCRIMINATOR_URL="$GAD_SERVICE_URL" \
    GAD_REWARD_COEF=0.8 GAD_FORMAT_REWARD_COEF=0.1 GAD_TOOL_REWARD_COEF=0.1 \
    MODEL_ARGS_FILE="$MODEL_ARGS_FILE" HF_CHECKPOINT="$HF_CHECKPOINT" REF_LOAD="$REF_LOAD" \
    NUM_GPUS=8 TENSOR_MODEL_PARALLEL_SIZE=4 PIPELINE_MODEL_PARALLEL_SIZE=1 \
    ROLLOUT_NUM_GPUS_PER_ENGINE=1 ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=8 \
    GLOBAL_BATCH_SIZE=8 NUM_ROLLOUT="$GAD_NUM_ROLLOUT" ADVANTAGE_ESTIMATOR=gspo \
    ROLLOUT_MAX_PROMPT_LEN="$ROLLOUT_MAX_PROMPT_LEN" ROLLOUT_MAX_RESPONSE_LEN=8192 ROLLOUT_LONG_RESPONSE_LEN=16384 \
    ROLLOUT_LONG_TASK_TYPES='vs pf' ROLLOUT_MAX_CONTEXT_LEN=131072 \
    CUSTOM_GENERATE_FUNCTION_PATH=drug_agent.rollout.length_aware_generate.generate \
    ROLLOUT_TEMPERATURE=0.8 GAD_SGLANG_MEM_FRACTION_STATIC=0.25 MAX_TOKENS_PER_GPU=16384 \
    LOG_PROBS_CHUNK_SIZE=64 RECOMPUTE_VOCAB_LOG_PROBS=1 ROUTER_POLICY=round_robin \
    DYNAMIC_SAMPLING_FILTER_PATH=slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std \
    DYNAMIC_SAMPLING_MAX_DROPPED_GROUPS=8 \
    COLOCATE_OFFLOAD_TRAIN=0 COLOCATE_OFFLOAD_ROLLOUT=0 \
    SGLANG_DISABLE_CUDA_GRAPH=1 SGLANG_DISABLE_CUSTOM_ALL_REDUCE=1 SGLANG_DISABLE_OVERLAP_SCHEDULE=1 \
    USE_ROLLOUT_LOGPROBS=0 USE_KL_LOSS=1 \
    STUDENT_LR=1e-7 STUDENT_MIN_LR=1e-8 LR_DECAY_STYLE=cosine LR_WARMUP_FRACTION=0.03 \
    WEIGHT_DECAY=0.1 KL_LOSS_COEF=0.001 SAVE_INTERVAL=200 \
    bash drug_agent/gad/scripts/run_stage3_gad_grpo_full.sh
require_path "$GAD_DIR/latest_checkpointed_iteration.txt"
finalize_checkpoint_stage "$GAD_DIR" "$RUN_ROOT/gad.complete"

curl -fsS -X POST "$GAD_SERVICE_URL/checkpoint" \
  -H 'Content-Type: application/json' \
  -d "{\"path\":\"$GAD_SERVICE_DIR/final\"}" \
  > "$RUN_ROOT/gad_discriminator_final_checkpoint.json"
kill "$DISCRIMINATOR_PID" 2>/dev/null || true
wait "$DISCRIMINATOR_PID" 2>/dev/null || true
DISCRIMINATOR_PID=""
else
  echo "[serial] SKIP completed gad"
fi

touch "$RUN_ROOT/ALL_COMPLETE"
echo "[$(date --iso-8601=seconds)] ALL COMPLETE" | tee -a "$RUN_ROOT/serial_status.log"
echo "RUN_ROOT=$RUN_ROOT"
