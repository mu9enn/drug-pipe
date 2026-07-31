#!/usr/bin/env bash
# One-shot formal 4-GPU pipeline for the current 373 canonical ReAct records.
# Algorithm branches are intentional: ToolRL and GAD both start from SFT.
set -euo pipefail

SLIME_ENV="${SLIME_ENV:-/root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh}"
if [[ ! -f "$SLIME_ENV" ]]; then
  SLIME_ENV=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi
if [[ ! -f "$SLIME_ENV" ]]; then
  echo "SLIME environment file not found: $SLIME_ENV" >&2
  exit 2
fi
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
CANONICAL_DATA="${CANONICAL_DATA:-$DATA_ROOT/react_trajectories.jsonl}"
TOOLRL_DATA="${TOOLRL_DATA:-$DATA_ROOT/toolrl/react_trajectories.toolrl_steps.jsonl}"
GAD_DATA="${GAD_DATA:-$DATA_ROOT/gad/gad_steps.jsonl}"
HF_CHECKPOINT="${HF_CHECKPOINT:-$MODEL_DATA_ROOT/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-$MODEL_DATA_ROOT/Qwen3.5-4B_torch_dist}"
SERIAL_RUN_ID="${SERIAL_RUN_ID:-Qwen3.5-4B_current373_serial_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$RUNS_ROOT/$SERIAL_RUN_ID}"
LOG_ROOT="$RUN_ROOT/logs"

SFT_DIR="$RUN_ROOT/sft"
TOOLRL_DIR="$RUN_ROOT/toolrl"
NEGATIVE_CACHE="$RUN_ROOT/gad_stage2_negatives.jsonl"
DISCRIMINATOR_DIR="$RUN_ROOT/gad_discriminator_warmup"
DISCRIMINATOR_LATEST="$DISCRIMINATOR_DIR/latest"
GAD_DIR="$RUN_ROOT/gad"
GAD_SERVICE_DIR="$RUN_ROOT/gad_discriminator_online"
GAD_SERVICE_PORT="${GAD_SERVICE_PORT:-8100}"
GAD_SERVICE_URL="http://127.0.0.1:${GAD_SERVICE_PORT}"
export CHECKPOINT_KEEP_LAST="${CHECKPOINT_KEEP_LAST:-2}"
export CHECKPOINT_FINAL_KEEP="${CHECKPOINT_FINAL_KEEP:-1}"
export DISCRIMINATOR_KEEP_LAST="${DISCRIMINATOR_KEEP_LAST:-2}"

for value in "$CHECKPOINT_KEEP_LAST" "$CHECKPOINT_FINAL_KEEP" "$DISCRIMINATOR_KEEP_LAST"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Checkpoint retention values must be positive integers; got $value" >&2
    exit 2
  fi
done

if [[ -e "$RUN_ROOT" ]]; then
  echo "RUN_ROOT already exists; choose a new SERIAL_RUN_ID or RUN_ROOT: $RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$LOG_ROOT"

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "Required path does not exist: $1" >&2
    exit 2
  fi
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
require_path "$HF_CHECKPOINT"
require_path "$REF_LOAD"
command -v nvidia-smi >/dev/null
command -v ray >/dev/null
command -v curl >/dev/null

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( GPU_COUNT < 4 )); then
  echo "This profile requires at least 4 visible GPUs; found $GPU_COUNT" >&2
  exit 2
fi
if [[ "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
  BUSY_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  if [[ -n "$BUSY_PIDS" ]]; then
    echo "GPU compute processes already exist: $BUSY_PIDS" >&2
    echo "Use a clean worker, or set ALLOW_BUSY_GPUS=1 only after checking them." >&2
    exit 2
  fi
fi

SOURCE_COUNT=$(wc -l < "$CANONICAL_DATA")
if [[ "$SOURCE_COUNT" -ne 373 ]]; then
  echo "Expected exactly 373 canonical records, found $SOURCE_COUNT in $CANONICAL_DATA" >&2
  exit 2
fi

run_logged() {
  local stage="$1"
  shift
  echo "[$(date --iso-8601=seconds)] START $stage" | tee -a "$RUN_ROOT/serial_status.log"
  "$@" 2>&1 | tee "$LOG_ROOT/${stage}.log"
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

DATA_SHA256=$(sha256sum "$CANONICAL_DATA" | awk '{print $1}')
CODE_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
cat > "$RUN_ROOT/serial_config.env" <<EOF
SERIAL_RUN_ID=$SERIAL_RUN_ID
CODE_COMMIT=$CODE_COMMIT
CANONICAL_DATA=$CANONICAL_DATA
CANONICAL_RECORDS=$SOURCE_COUNT
CANONICAL_SHA256=$DATA_SHA256
HF_CHECKPOINT=$HF_CHECKPOINT
REF_LOAD=$REF_LOAD
SFT_SAVE_DIR=$SFT_DIR
TOOLRL_SAVE_DIR=$TOOLRL_DIR
GAD_NEGATIVE_CACHE=$NEGATIVE_CACHE
GAD_DISCRIMINATOR_DIR=$DISCRIMINATOR_DIR
GAD_SAVE_DIR=$GAD_DIR
SAVE_OPTIMIZER=1
CHECKPOINT_KEEP_LAST=$CHECKPOINT_KEEP_LAST
CHECKPOINT_FINAL_KEEP=$CHECKPOINT_FINAL_KEEP
DISCRIMINATOR_KEEP_LAST=$DISCRIMINATOR_KEEP_LAST
EOF

echo "[serial] run_root=$RUN_ROOT"
echo "[serial] canonical_records=$SOURCE_COUNT sha256=$DATA_SHA256"
echo "[serial] branch topology: SFT -> {ToolRL, GAD}; ToolRL is run first but is not GAD initialization"

# Deterministically regenerate step-level data from this exact canonical source.
run_logged prepare_toolrl_data \
  python -m drug_agent.toolrl.convert_react_to_toolrl_steps \
    --input "$CANONICAL_DATA" \
    --output "$TOOLRL_DATA" \
    --skipped-report "$DATA_ROOT/toolrl/react_trajectories.toolrl_steps.skipped.jsonl" \
    --report "$DATA_ROOT/toolrl/react_trajectories.toolrl_steps.report.json"

run_logged prepare_gad_data \
  env INPUT="$CANONICAL_DATA" OUTPUT_ROOT="$DATA_ROOT/gad" \
    bash drug_agent/gad/scripts/prepare_gad_step_data.sh

TOOLRL_COUNT=$(wc -l < "$TOOLRL_DATA")
GAD_COUNT=$(wc -l < "$GAD_DATA")
if [[ "$TOOLRL_COUNT" -ne 3028 || "$GAD_COUNT" -ne 3234 ]]; then
  echo "Unexpected derived counts: ToolRL=$TOOLRL_COUNT (expected 3028), GAD=$GAD_COUNT (expected 3234)" >&2
  exit 2
fi

# SFT: the configuration already validated on the previous 4-GPU 4B run.
run_logged sft \
  env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    PROMPT_DATA="$CANONICAL_DATA" SAVE_DIR="$SFT_DIR" RUN_NAME="${SERIAL_RUN_ID}_sft" \
    HF_CHECKPOINT="$HF_CHECKPOINT" REF_LOAD="$REF_LOAD" \
    NUM_GPUS=4 TENSOR_MODEL_PARALLEL_SIZE=4 PIPELINE_MODEL_PARALLEL_SIZE=1 \
    CONTEXT_PARALLEL_SIZE=1 EXPERT_MODEL_PARALLEL_SIZE=1 EXPERT_TENSOR_PARALLEL_SIZE=1 \
    NUM_EPOCH=1 ROLLOUT_BATCH_SIZE=373 GLOBAL_BATCH_SIZE=1 MAX_TOKENS_PER_GPU=8192 \
    LR=1e-5 MIN_LR=1e-6 LR_WARMUP_FRACTION=0.1 SAVE_INTERVAL=100 \
    RECOMPUTE_FULL=1 RECOMPUTE_NUM_LAYERS=1 SFT_DEBUG_TRAIN_ONLY=1 SFT_DISABLE_OFFLOAD=1 \
    bash drug_agent/scripts/run_qwen3_5_4b_drug_sft_full.sh
finalize_checkpoint_stage "$SFT_DIR" "$RUN_ROOT/sft.complete"

# ToolRL official baseline: fixed history-only states, two samples per prompt,
# full long-context limits, and weights initialized from this run's SFT.
run_logged toolrl \
  env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    PROMPT_DATA="$TOOLRL_DATA" SAVE_DIR="$TOOLRL_DIR" LOAD="$SFT_DIR" TOOLRL_RESUME=0 \
    HF_CHECKPOINT="$HF_CHECKPOINT" REF_LOAD="$REF_LOAD" \
    NUM_GPUS=4 TENSOR_MODEL_PARALLEL_SIZE=4 PIPELINE_MODEL_PARALLEL_SIZE=1 \
    CONTEXT_PARALLEL_SIZE=1 EXPERT_MODEL_PARALLEL_SIZE=1 EXPERT_TENSOR_PARALLEL_SIZE=1 \
    ROLLOUT_NUM_GPUS_PER_ENGINE=1 ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=2 \
    GLOBAL_BATCH_SIZE=8 NUM_ROLLOUT=757 NUM_EPOCH=1 \
    ROLLOUT_MAX_PROMPT_LEN=98304 ROLLOUT_MAX_RESPONSE_LEN=4096 ROLLOUT_MAX_CONTEXT_LEN=102400 \
    ROLLOUT_TEMPERATURE=1.0 SGLANG_MEM_FRACTION_STATIC=0.75 MAX_TOKENS_PER_GPU=8192 \
    TOOLRL_REWARD_MODE=official LR=5e-7 SAVE_INTERVAL=200 RECOMPUTE_FULL=1 RECOMPUTE_NUM_LAYERS=1 \
    bash drug_agent/toolrl/scripts/run_qwen3_5_4b_toolrl_full.sh
finalize_checkpoint_stage "$TOOLRL_DIR" "$RUN_ROOT/toolrl.complete"

# GAD Stage 2a: generate one student negative per fixed expert state. This is
# a zero-learning-rate rollout and starts from SFT, not ToolRL.
run_logged gad_negative_generation \
  env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    PROMPT_DATA="$GAD_DATA" GAD_NEGATIVE_CACHE="$NEGATIVE_CACHE" STUDENT_LOAD="$SFT_DIR" \
    HF_CHECKPOINT="$HF_CHECKPOINT" REF_LOAD="$REF_LOAD" MODEL_ARGS_FILE=scripts/models/qwen3.5-4B.sh \
    NUM_GPUS=4 TENSOR_MODEL_PARALLEL_SIZE=4 ROLLOUT_NUM_GPUS_PER_ENGINE=1 \
    ROLLOUT_BATCH_SIZE=2 NUM_ROLLOUT=1617 \
    ROLLOUT_MAX_PROMPT_LEN=98304 ROLLOUT_MAX_RESPONSE_LEN=4096 ROLLOUT_MAX_CONTEXT_LEN=102400 \
    ROLLOUT_TEMPERATURE=0.8 SGLANG_MEM_FRACTION_STATIC=0.75 MAX_TOKENS_PER_GPU=8192 \
    bash drug_agent/gad/scripts/generate_stage2_negatives.sh
require_path "$NEGATIVE_CACHE"
if [[ ! -s "$NEGATIVE_CACHE" ]]; then
  echo "GAD negative cache is empty: $NEGATIVE_CACHE" >&2
  exit 2
fi

# Release Ray/SGLang before the direct single-GPU discriminator warmup.
ray stop --force >/dev/null 2>&1 || true
pkill -9 sglang 2>/dev/null || true

run_logged gad_discriminator_warmup \
  env CUDA_VISIBLE_DEVICES=3 \
    PAIRS="$NEGATIVE_CACHE" GENERATOR_WARMUP_LOAD="$SFT_DIR" \
    DISCRIMINATOR_MODEL_PATH="$HF_CHECKPOINT" DISCRIMINATOR_OUTPUT_DIR="$DISCRIMINATOR_DIR" \
    DISCRIMINATOR_EPOCHS=1 DISCRIMINATOR_BATCH_SIZE=2 DISCRIMINATOR_LR=1e-6 \
    DISCRIMINATOR_MAX_LENGTH=4096 DISCRIMINATOR_CLIP_GRAD=0.2 DISCRIMINATOR_SAVE_INTERVAL=200 \
    DISCRIMINATOR_KEEP_LAST="$DISCRIMINATOR_KEEP_LAST" \
    bash drug_agent/gad/scripts/run_stage2_discriminator_warmup.sh
require_path "$DISCRIMINATOR_LATEST"
require_path "$DISCRIMINATOR_DIR/warmup_manifest.json"

# Pure GAD needs the discriminator alive while the generator trains. Reserve
# physical GPU 3 for it and expose GPUs 0-2 to the TP1/DP3 generator.
CUDA_VISIBLE_DEVICES=3 \
  DISCRIMINATOR_MODEL_PATH="$HF_CHECKPOINT" \
  DISCRIMINATOR_RESUME="$DISCRIMINATOR_LATEST" \
  DISCRIMINATOR_OUTPUT_DIR="$GAD_SERVICE_DIR" \
  GAD_DISCRIMINATOR_HOST=127.0.0.1 GAD_DISCRIMINATOR_PORT="$GAD_SERVICE_PORT" \
  DISCRIMINATOR_LR=1e-6 DISCRIMINATOR_MAX_LENGTH=4096 DISCRIMINATOR_UPDATE_STEPS=1 \
  DISCRIMINATOR_CLIP_GRAD=0.2 DISCRIMINATOR_REWARD_CLIP=2.0 DISCRIMINATOR_SAVE_INTERVAL=200 \
  DISCRIMINATOR_KEEP_LAST="$DISCRIMINATOR_KEEP_LAST" \
  bash drug_agent/gad/scripts/serve_discriminator.sh > "$LOG_ROOT/gad_discriminator_service.log" 2>&1 &
DISCRIMINATOR_PID=$!

SERVICE_READY=0
for _ in $(seq 1 90); do
  if ! kill -0 "$DISCRIMINATOR_PID" 2>/dev/null; then
    echo "GAD discriminator service exited during startup; see $LOG_ROOT/gad_discriminator_service.log" >&2
    exit 2
  fi
  if curl -fsS "$GAD_SERVICE_URL/health" >/dev/null 2>&1; then
    SERVICE_READY=1
    break
  fi
  sleep 2
done
if [[ "$SERVICE_READY" -ne 1 ]]; then
  echo "GAD discriminator did not become ready: $GAD_SERVICE_URL" >&2
  exit 2
fi

run_logged gad \
  env CUDA_VISIBLE_DEVICES=0,1,2 \
    PROMPT_DATA="$GAD_DATA" SAVE_DIR="$GAD_DIR" \
    STUDENT_WARMUP_LOAD="$SFT_DIR" DISCRIMINATOR_WARMUP_LOAD="$DISCRIMINATOR_LATEST" \
    GAD_WARMUP_MANIFEST="$DISCRIMINATOR_DIR/warmup_manifest.json" \
    GAD_REWARD_MODE=pure GAD_DISCRIMINATOR_URL="$GAD_SERVICE_URL" \
    HF_CHECKPOINT="$HF_CHECKPOINT" REF_LOAD="$REF_LOAD" MODEL_ARGS_FILE=scripts/models/qwen3.5-4B.sh \
    NUM_GPUS=3 TENSOR_MODEL_PARALLEL_SIZE=1 ROLLOUT_NUM_GPUS_PER_ENGINE=1 \
    ROLLOUT_BATCH_SIZE=2 N_SAMPLES_PER_PROMPT=3 GLOBAL_BATCH_SIZE=6 NUM_ROLLOUT=1617 \
    ROLLOUT_MAX_PROMPT_LEN=98304 ROLLOUT_MAX_RESPONSE_LEN=4096 ROLLOUT_MAX_CONTEXT_LEN=102400 \
    ROLLOUT_TEMPERATURE=0.8 SGLANG_MEM_FRACTION_STATIC=0.75 MAX_TOKENS_PER_GPU=8192 \
    STUDENT_LR=2e-7 KL_LOSS_COEF=0.001 SAVE_INTERVAL=200 \
    bash drug_agent/gad/scripts/run_stage3_gad_grpo_full.sh
finalize_checkpoint_stage "$GAD_DIR" "$RUN_ROOT/gad.complete"

# Persist the final online discriminator state before terminating its service.
curl -fsS -X POST "$GAD_SERVICE_URL/checkpoint" \
  -H 'Content-Type: application/json' \
  -d "{\"path\":\"$GAD_SERVICE_DIR/final\"}" \
  > "$RUN_ROOT/gad_discriminator_final_checkpoint.json"
kill "$DISCRIMINATOR_PID" 2>/dev/null || true
wait "$DISCRIMINATOR_PID" 2>/dev/null || true
DISCRIMINATOR_PID=""

touch "$RUN_ROOT/ALL_COMPLETE"
echo "[$(date --iso-8601=seconds)] ALL COMPLETE" | tee -a "$RUN_ROOT/serial_status.log"
echo "RUN_ROOT=$RUN_ROOT"
