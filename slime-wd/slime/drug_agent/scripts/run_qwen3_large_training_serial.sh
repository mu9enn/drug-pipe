#!/usr/bin/env bash
# Formal serial training for the measured 8xH200 large-model profiles.
# Algorithm branches are intentional: SFT -> ToolRL and SFT -> GAD.
set -euo pipefail

PROFILE=${MODEL_PROFILE:?Set MODEL_PROFILE to qwen35-27b-8xh200 or qwen35-122b-8xh200}
case "$PROFILE" in
  qwen35-27b-8xh200|qwen35-122b-8xh200) ;;
  *) echo "Unsupported formal serial profile: $PROFILE" >&2; exit 2 ;;
esac

SLIME_ENV=${SLIME_ENV:-/root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh}
if [[ ! -f "$SLIME_ENV" ]]; then
  SLIME_ENV=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi
source "$SLIME_ENV"
cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh
source drug_agent/scripts/qwen3_large_profile.sh

# The formal 27B corpus contains trajectories up to ~94K tokens.  The faster
# TP2/PP2/DP2 bucket is safe for bounded examples, but the final PP stage OOMs
# when it owns 30 transformer layers plus the full-vocabulary output tensor.
# Keep every 27B actor stage on the measured long-sequence topology so SFT and
# both RL branches have identical checkpoint-compatible parallelism.
FORMAL_PP=$PIPELINE_MODEL_PARALLEL_SIZE
FORMAL_FIRST=${NUM_LAYERS_IN_FIRST_PIPELINE_STAGE:-}
FORMAL_LAST=${NUM_LAYERS_IN_LAST_PIPELINE_STAGE:-}
FORMAL_LAYOUT=${PIPELINE_MODEL_PARALLEL_LAYOUT:-}
if [[ "$PROFILE" == qwen35-27b-8xh200 ]]; then
  FORMAL_PP=4
  # The 18/16/16/14 layout completed eight updates, then the loss rank reached
  # 140.03 GiB and could not allocate the fused CE's 122 MiB tile.  Move two
  # more layers to the embedding rank and use the single-node recompute path,
  # whose TP2 forward/backward was checked numerically before this retry.
  FORMAL_FIRST=20
  FORMAL_LAST=12
fi
FORMAL_PARALLEL_ENV=(
  "PIPELINE_MODEL_PARALLEL_SIZE=$FORMAL_PP"
  "NUM_LAYERS_IN_FIRST_PIPELINE_STAGE=$FORMAL_FIRST"
  "NUM_LAYERS_IN_LAST_PIPELINE_STAGE=$FORMAL_LAST"
  "PIPELINE_MODEL_PARALLEL_LAYOUT=$FORMAL_LAYOUT"
)
if [[ "$PROFILE" == qwen35-27b-8xh200 ]]; then
  FORMAL_PARALLEL_ENV+=(
    "RECOMPUTE_VOCAB_LOG_PROBS=1"
    "LOG_PROBS_CHUNK_SIZE=64"
    # The pinned HybridDeviceOptimizer produced corrupted post-step rollout
    # weights in both overlapped and synchronous modes.  The GPU precision-
    # aware Adam path already passed the formal SFT topology; BF16 moments
    # approximately offset the HBM formerly saved by 35% CPU offload.
    "OVERLAP_CPU_OPTIMIZER_D2H_H2D=0"
    "TOOLRL_OPTIMIZER_CPU_OFFLOAD=0"
    "TOOLRL_OPTIMIZER_OFFLOAD_FRACTION=0"
    "GAD_OPTIMIZER_CPU_OFFLOAD=0"
    "GAD_OPTIMIZER_OFFLOAD_FRACTION=0"
    "EXP_AVG_DTYPE=bf16"
    "EXP_AVG_SQ_DTYPE=bf16"
    "DYNAMIC_SAMPLING_MAX_DROPPED_GROUPS=8"
  )
fi
if [[ "$PROFILE" == qwen35-122b-8xh200 ]]; then
  # Keep the profile's uniform 12/12/12/12 layout.  Moving a layer away from
  # the output stage created a 13-layer third stage, which independently
  # reproduced the known 20--24 MiB FP8 Adam expansion failure.  Uniform is
  # the only measured topology with no 13-layer stage.
  # Uniform 12/12/12/12 passed a complete optimizer update from the official
  # FP8-derived actor.  The 7168-token retry still exhausted rank 0 on a later
  # long GDN microbatch (only 210 MiB free for a 632 MiB allocation).  Use
  # 6144 so variable-length batches retain durable activation headroom.
  FORMAL_PARALLEL_ENV+=(
    "MAX_TOKENS_PER_GPU=6144"
    # Dynamic batching cannot split a single ~94K-token trajectory.  Preserve
    # its system/task prefix and most recent supervised decisions while
    # bounding the GatedDelta activation workspace for the 8xH200 actor.
    # 32K and 16K completed forward/backward but did not retain durable room
    # for the next FP8 optimizer-state reload.  Twelve thousand tokens keeps
    # the task prefix plus twice as much recent supervision as an 8K fallback
    # while reclaiming 25% of the sequence-proportional activation workspace.
    "SFT_MAX_SEQUENCE_LEN=12288"
    "SFT_TRUNCATION_HEAD_TOKENS=4096"
  )
  # The capacity-safe streamed FP8 Adam path completed two updates at the
  # original schedule, but the second warmup update reached 3.66e-7 and the
  # following backward produced a local NaN on the output rank.  Use the
  # conservative full-model FP8 schedule until a longer numerical gate proves
  # a higher rate safe on these trajectories.
  SFT_LR=${QWEN122_FORMAL_SFT_LR:-1e-7}
  SFT_MIN_LR=${QWEN122_FORMAL_SFT_MIN_LR:-1e-8}
fi

export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export PYTHONUNBUFFERED=1
SERIAL_RUN_ID=${SERIAL_RUN_ID:-${PROFILE}_serial_$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-$DRUG_AGENT_RUNS_ROOT/$SERIAL_RUN_ID}
LOG_ROOT=$RUN_ROOT/logs
SFT_DIR=$RUN_ROOT/sft
TOOLRL_DIR=$RUN_ROOT/toolrl
NEGATIVE_CACHE=$RUN_ROOT/gad_stage2_negatives.jsonl
DISCRIMINATOR_DIR=$RUN_ROOT/gad_discriminator_warmup
DISCRIMINATOR_LATEST=$DISCRIMINATOR_DIR/latest
GAD_DIR=$RUN_ROOT/gad
GAD_SERVICE_DIR=$RUN_ROOT/gad_discriminator_online
GAD_SERVICE_PORT=${GAD_SERVICE_PORT:-8100}
GAD_SERVICE_URL=http://127.0.0.1:$GAD_SERVICE_PORT
CHECKPOINT_KEEP_LAST=${CHECKPOINT_KEEP_LAST:-2}
DISCRIMINATOR_KEEP_LAST=${DISCRIMINATOR_KEEP_LAST:-2}

require_path() {
  [[ -e "$1" ]] || { echo "Required path does not exist: $1" >&2; exit 2; }
}

run_stage() {
  local stage=$1
  shift
  if [[ -f "$RUN_ROOT/$stage.complete" ]]; then
    echo "[$(date --iso-8601=seconds)] SKIP completed $stage" | tee -a "$RUN_ROOT/serial_status.log"
    return
  fi
  echo "[$(date --iso-8601=seconds)] START $stage" | tee -a "$RUN_ROOT/serial_status.log"
  "$@" 2>&1 | tee -a "$LOG_ROOT/$stage.log"
  # `ray job stop` is reported as a successful CLI exit by Ray.  Do not turn
  # an operator-stopped or otherwise checkpoint-less training job into a false
  # completed stage merely because the submission command returned zero.
  case "$stage" in
    sft) require_path "$SFT_DIR/latest_checkpointed_iteration.txt" ;;
    toolrl) require_path "$TOOLRL_DIR/latest_checkpointed_iteration.txt" ;;
    gad_negative_generation)
      [[ -s "$NEGATIVE_CACHE" ]] || { echo "Empty GAD negative cache: $NEGATIVE_CACHE" >&2; return 2; }
      ;;
    gad_discriminator_warmup)
      require_path "$DISCRIMINATOR_LATEST"
      require_path "$DISCRIMINATOR_DIR/warmup_manifest.json"
      ;;
    gad) require_path "$GAD_DIR/latest_checkpointed_iteration.txt" ;;
  esac
  touch "$RUN_ROOT/$stage.complete"
  echo "[$(date --iso-8601=seconds)] COMPLETE $stage" | tee -a "$RUN_ROOT/serial_status.log"
}

DISCRIMINATOR_PID=
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

require_path "$CANONICAL_DATA"
require_path "$TOOLRL_DATA"
require_path "$GAD_DATA"
require_path "$HF_CHECKPOINT"
require_path "$REF_LOAD/latest_checkpointed_iteration.txt"
require_path "$MODEL_ARGS_FILE"
require_path "$DISCRIMINATOR_MODEL_PATH"

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
H200_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c H200 || true)
if (( GPU_COUNT != 8 || H200_COUNT != 8 )); then
  echo "This serial profile requires exactly 8 visible H200 GPUs; found GPUs=$GPU_COUNT H200=$H200_COUNT" >&2
  exit 2
fi
if [[ ${ALLOW_BUSY_GPUS:-0} != 1 ]]; then
  BUSY_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  [[ -z "$BUSY_PIDS" ]] || { echo "GPU compute processes already exist: $BUSY_PIDS" >&2; exit 2; }
fi

SFT_COUNT=$(wc -l < "$CANONICAL_DATA")
TOOLRL_COUNT=$(wc -l < "$TOOLRL_DATA")
GAD_COUNT=$(wc -l < "$GAD_DATA")
if (( SFT_COUNT != 364 || TOOLRL_COUNT != 3182 || GAD_COUNT != 3147 )); then
  echo "Unexpected dataset counts: SFT=$SFT_COUNT ToolRL=$TOOLRL_COUNT GAD=$GAD_COUNT" >&2
  exit 2
fi

if [[ -e "$RUN_ROOT" && ${RESUME_SERIAL_RUN:-0} != 1 ]]; then
  echo "RUN_ROOT already exists: $RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$LOG_ROOT"
if [[ ! -f "$RUN_ROOT/serial_config.env" ]]; then
  cat > "$RUN_ROOT/serial_config.env" <<EOF
MODEL_PROFILE=$PROFILE
SERIAL_RUN_ID=$SERIAL_RUN_ID
RUN_ROOT=$RUN_ROOT
CANONICAL_DATA=$CANONICAL_DATA
TOOLRL_DATA=$TOOLRL_DATA
GAD_DATA=$GAD_DATA
HF_CHECKPOINT=$HF_CHECKPOINT
ROLLOUT_HF_CHECKPOINT=${ROLLOUT_HF_CHECKPOINT:-$HF_CHECKPOINT}
REF_LOAD=$REF_LOAD
SFT_SAVE_DIR=$SFT_DIR
TOOLRL_SAVE_DIR=$TOOLRL_DIR
GAD_SAVE_DIR=$GAD_DIR
SFT_RECORDS=$SFT_COUNT
TOOLRL_RECORDS=$TOOLRL_COUNT
GAD_RECORDS=$GAD_COUNT
EOF
fi
echo "[$(date --iso-8601=seconds)] PIPELINE profile=$PROFILE root=$RUN_ROOT topology=tp${TENSOR_MODEL_PARALLEL_SIZE}-pp${FORMAL_PP}-first${FORMAL_FIRST}-last${FORMAL_LAST} layout=${FORMAL_LAYOUT:-equal-middle}" | tee -a "$RUN_ROOT/serial_status.log"

# One full canonical SFT epoch. The measured 8-GPU profiles supply topology,
# uneven PP layout, precision-aware optimizer dtypes, recompute and balancing.
SFT_SAVE_INTERVAL=30
SFT_NO_SAVE_OPTIM=0
if [[ "$PROFILE" == qwen35-122b-8xh200 ]]; then
  # A 122B optimizer checkpoint/save transient has not passed the 1 TiB gate.
  # Save trainable weights only, at the final planned update.
  SFT_SAVE_INTERVAL=91
  SFT_NO_SAVE_OPTIM=1
fi
run_stage sft \
  env "${FORMAL_PARALLEL_ENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    PROMPT_DATA="$CANONICAL_DATA" SAVE_DIR="$SFT_DIR" \
    NUM_GPUS=8 SFT_EPOCH_ONLY=1 NUM_EPOCH=1 ROLLOUT_BATCH_SIZE=364 GLOBAL_BATCH_SIZE=4 \
    LR="$SFT_LR" MIN_LR="$SFT_MIN_LR" LR_WARMUP_FRACTION=0.03 \
    SAVE_INTERVAL="$SFT_SAVE_INTERVAL" CHECKPOINT_KEEP_LAST="$CHECKPOINT_KEEP_LAST" \
    NO_SAVE_OPTIM="$SFT_NO_SAVE_OPTIM" SFT_DEBUG_TRAIN_ONLY=1 SFT_DISABLE_OFFLOAD=1 \
    SLIME_DROP_OPTIMIZER_STATE_BEFORE_WEIGHTS_ONLY_SAVE="$SFT_NO_SAVE_OPTIM" \
    bash drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh
require_path "$SFT_DIR/latest_checkpointed_iteration.txt"

if [[ "$PROFILE" == qwen35-122b-8xh200 ]]; then
  cat > "$RUN_ROOT/WAITING_FOR_EXTERNAL_RL" <<EOF
SFT completed. The measured colocated FP8 ToolRL computation fits, but actor
pause exceeds this worker's 1 TiB host cgroup. Start ToolRL/GAD only after
external rollout engines and an external GAD discriminator are assigned.
EOF
  echo "[$(date --iso-8601=seconds)] WAITING external rollout/discriminator for 122B ToolRL/GAD" | tee -a "$RUN_ROOT/serial_status.log"
  exit 0
fi

# ToolRL is a branch from SFT, not a continuation of the SFT optimizer.  The
# strict four-sample GRPO groups repeatedly had identical rewards and therefore
# exactly zero advantage.  Use the existing dense MolClaw reward with
# REINFORCE++ normalization across an eight-prompt batch; this preserves the
# same tool-call objective while providing a learning signal across prompts.
TOOLRL_ROLLOUTS=$(((TOOLRL_COUNT + 7) / 8))
# A 5e-8 production attempt destroyed ReAct validity after its first update:
# batch 0 had two exact +1 outcomes, while batches 1--64 contained only invalid
# -0.3/-0.5 outcomes.  Use a five-times smaller peak rate, twenty-step warmup,
# and lower sampling temperature.  Keep the optional reference-KL path off by
# default: an isolation run with a mathematically zero first-step LR still
# collapsed only when the ref/actor switching path was active.
FORMAL_TOOLRL_LR=${FORMAL_TOOLRL_LR:-1e-8}
FORMAL_USE_KL_LOSS=${FORMAL_USE_KL_LOSS:-0}
run_stage toolrl \
  env "${FORMAL_PARALLEL_ENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    PROMPT_DATA="$TOOLRL_DATA" SAVE_DIR="$TOOLRL_DIR" LOAD="$SFT_DIR" TOOLRL_RESUME=0 \
    TOOLRL_REWARD_MODE="${TOOLRL_REWARD_MODE:-molclaw}" \
    ADVANTAGE_ESTIMATOR=reinforce_plus_plus NORMALIZE_ADVANTAGES=1 \
    USE_ROLLOUT_LOGPROBS=1 USE_KL_LOSS="$FORMAL_USE_KL_LOSS" KL_LOSS_COEF=0.001 KL_LOSS_TYPE=low_var_kl \
    USE_PRECISION_AWARE_OPTIMIZER=0 EXP_AVG_DTYPE=fp32 EXP_AVG_SQ_DTYPE=fp32 \
    SGLANG_MEM_FRACTION_STATIC=0.12 SGLANG_DISABLE_CUDA_GRAPH=1 COLOCATE_OFFLOAD_ROLLOUT=0 \
    NUM_GPUS=8 NUM_ROLLOUT="$TOOLRL_ROLLOUTS" ROLLOUT_BATCH_SIZE=8 \
    N_SAMPLES_PER_PROMPT=1 GLOBAL_BATCH_SIZE=8 LR="$FORMAL_TOOLRL_LR" LR_DECAY_STYLE=constant \
    LR_WARMUP_FRACTION=0.05 LR_WARMUP_INIT=0 ROLLOUT_TEMPERATURE=0.7 \
    ROLLOUT_MAX_PROMPT_LEN=65536 ROLLOUT_MAX_CONTEXT_LEN=69632 \
    SAVE_INTERVAL=200 CHECKPOINT_KEEP_LAST="$CHECKPOINT_KEEP_LAST" \
    bash drug_agent/toolrl/scripts/run_toolrl_grpo.sh
require_path "$TOOLRL_DIR/latest_checkpointed_iteration.txt"

# GAD is a separate branch from the same SFT checkpoint.
GAD_ROLLOUTS=$(((GAD_COUNT + 1) / 2))
run_stage gad_negative_generation \
  env "${FORMAL_PARALLEL_ENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    PROMPT_DATA="$GAD_DATA" GAD_NEGATIVE_CACHE="$NEGATIVE_CACHE" STUDENT_LOAD="$SFT_DIR" \
    NUM_GPUS=8 NUM_ROLLOUT="$GAD_ROLLOUTS" ROLLOUT_BATCH_SIZE=2 COLOCATE_OFFLOAD_ROLLOUT=0 \
    ROLLOUT_MAX_PROMPT_LEN=65536 ROLLOUT_MAX_CONTEXT_LEN=69632 \
    bash drug_agent/gad/scripts/generate_stage2_negatives.sh
[[ -s "$NEGATIVE_CACHE" ]] || { echo "Empty GAD negative cache: $NEGATIVE_CACHE" >&2; exit 2; }

ray stop --force >/dev/null 2>&1 || true
pkill -9 sglang 2>/dev/null || true
run_stage gad_discriminator_warmup \
  env CUDA_VISIBLE_DEVICES=7 PAIRS="$NEGATIVE_CACHE" GENERATOR_WARMUP_LOAD="$SFT_DIR" \
    DISCRIMINATOR_OUTPUT_DIR="$DISCRIMINATOR_DIR" DISCRIMINATOR_EPOCHS=1 \
    DISCRIMINATOR_BATCH_SIZE=1 DISCRIMINATOR_LR=1e-6 DISCRIMINATOR_MAX_LENGTH=8192 \
    DISCRIMINATOR_CLIP_GRAD=1.0 DISCRIMINATOR_SAVE_INTERVAL=400 \
    DISCRIMINATOR_KEEP_LAST="$DISCRIMINATOR_KEEP_LAST" \
    bash drug_agent/gad/scripts/run_stage2_discriminator_warmup.sh
require_path "$DISCRIMINATOR_LATEST"
require_path "$DISCRIMINATOR_DIR/warmup_manifest.json"

# All eight GPUs are needed by the actor topology. Run the 0.8B discriminator
# on CPU; the 1 TiB worker passed the corresponding host-memory preflight.
GAD_DISCRIMINATOR_DEVICE=cpu DISCRIMINATOR_RESUME="$DISCRIMINATOR_LATEST" \
  DISCRIMINATOR_OUTPUT_DIR="$GAD_SERVICE_DIR" GAD_DISCRIMINATOR_HOST=127.0.0.1 \
  GAD_DISCRIMINATOR_PORT="$GAD_SERVICE_PORT" DISCRIMINATOR_LR=1e-6 \
  DISCRIMINATOR_MAX_LENGTH=8192 DISCRIMINATOR_UPDATE_STEPS=1 \
  DISCRIMINATOR_SAVE_INTERVAL=400 DISCRIMINATOR_KEEP_LAST="$DISCRIMINATOR_KEEP_LAST" \
  bash drug_agent/gad/scripts/serve_discriminator.sh > "$LOG_ROOT/gad_discriminator_service.log" 2>&1 &
DISCRIMINATOR_PID=$!
SERVICE_READY=0
for _ in $(seq 1 180); do
  kill -0 "$DISCRIMINATOR_PID" 2>/dev/null || break
  if curl -fsS "$GAD_SERVICE_URL/health" >/dev/null 2>&1; then
    SERVICE_READY=1
    break
  fi
  sleep 2
done
[[ "$SERVICE_READY" == 1 ]] || { echo "GAD discriminator service failed to start" >&2; exit 2; }

run_stage gad \
  env "${FORMAL_PARALLEL_ENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    PROMPT_DATA="$GAD_DATA" SAVE_DIR="$GAD_DIR" STUDENT_WARMUP_LOAD="$SFT_DIR" \
    DISCRIMINATOR_WARMUP_LOAD="$DISCRIMINATOR_LATEST" \
    GAD_WARMUP_MANIFEST="$DISCRIMINATOR_DIR/warmup_manifest.json" \
    GAD_REWARD_MODE=pure GAD_DISCRIMINATOR_URL="$GAD_SERVICE_URL" \
    DYNAMIC_SAMPLING_FILTER_PATH=slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std \
    USE_ROLLOUT_LOGPROBS=1 \
    USE_PRECISION_AWARE_OPTIMIZER=0 EXP_AVG_DTYPE=fp32 EXP_AVG_SQ_DTYPE=fp32 \
    GAD_SGLANG_MEM_FRACTION_STATIC=0.12 SGLANG_DISABLE_CUDA_GRAPH=1 COLOCATE_OFFLOAD_ROLLOUT=0 \
    NUM_GPUS=8 NUM_ROLLOUT="$GAD_ROLLOUTS" ROLLOUT_BATCH_SIZE=2 \
    N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=8 STUDENT_LR="$GAD_LR" \
    ROLLOUT_MAX_PROMPT_LEN=65536 ROLLOUT_MAX_CONTEXT_LEN=69632 \
    KL_LOSS_COEF=0.001 SAVE_INTERVAL=200 CHECKPOINT_KEEP_LAST="$CHECKPOINT_KEEP_LAST" \
    bash drug_agent/gad/scripts/run_stage3_gad_grpo.sh
require_path "$GAD_DIR/latest_checkpointed_iteration.txt"

curl -fsS -X POST "$GAD_SERVICE_URL/checkpoint" -H 'Content-Type: application/json' \
  -d "{\"path\":\"$GAD_SERVICE_DIR/final\"}" > "$RUN_ROOT/gad_discriminator_final_checkpoint.json"
kill "$DISCRIMINATOR_PID" 2>/dev/null || true
wait "$DISCRIMINATOR_PID" 2>/dev/null || true
DISCRIMINATOR_PID=
touch "$RUN_ROOT/ALL_COMPLETE"
echo "[$(date --iso-8601=seconds)] ALL COMPLETE" | tee -a "$RUN_ROOT/serial_status.log"
