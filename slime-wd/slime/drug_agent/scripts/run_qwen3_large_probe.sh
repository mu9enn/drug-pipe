#!/usr/bin/env bash
# Reproducible, deliberately short gates for the large Qwen profiles.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  MODEL_PROFILE=<profile> bash drug_agent/scripts/run_qwen3_large_probe.sh <action>

Actions:
  validate                Static model/data/topology/memory audit (no GPU)
  preflight               Read-only worker/CUDA/path audit
  convert                 HF -> torch_dist conversion into REF_LOAD
  sft-one-step            One train-only SFT optimizer step
  toolrl-one-group        One ToolRL GRPO group (cold start; optional SFT_LOAD)
  gad-negatives-one       Generate one GAD negative (requires SFT_LOAD)
  gad-discriminator-one   Warm up discriminator on probe negatives
  gad-serve               Serve warmed discriminator (foreground)
  gad-one-group           One GAD generator group against the service

The script has no full-epoch/full-dataset action. Use RUN_TAG to choose fresh
output names. One-step checkpoints omit optimizer/RNG state by design.
EOF
}

ACTION=${1:-}
if [[ -z "$ACTION" || "$ACTION" == "-h" || "$ACTION" == "--help" ]]; then
  usage
  [[ -n "$ACTION" ]] || exit 2
  exit 0
fi

SLIME_ENV=${SLIME_ENV:-/root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh}
if [[ ! -f "$SLIME_ENV" ]]; then
  SLIME_ENV=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi
source "$SLIME_ENV"
cd "$SLIME"

: "${MODEL_PROFILE:?Set MODEL_PROFILE to a qwen3_large_profile.sh profile}"
source drug_agent/scripts/qwen3_large_profile.sh
RUN_TAG=${RUN_TAG:-$MODEL_PROFILE}
PROBE_ROOT=${PROBE_ROOT:-$DRUG_AGENT_RUNS_ROOT/large_model_probes/$RUN_TAG}

validate_profile() {
  MODEL_PROFILE="$MODEL_PROFILE" bash drug_agent/scripts/validate_qwen3_large_profile.sh
}

worker_preflight() {
  local required_host_gib=${1:-$MIN_HOST_MEMORY_GIB}
  EXPECTED_GPUS="$NUM_GPUS" MIN_HOST_MEMORY_GIB="$required_host_gib" \
    HF_CHECKPOINT="$HF_CHECKPOINT" bash drug_agent/scripts/preflight_large_model_worker.sh
}

require_path() {
  local name=$1
  local path=$2
  if [[ ! -e "$path" ]]; then
    echo "Required $name does not exist: $path" >&2
    exit 2
  fi
}

require_fresh_dir() {
  local path=$1
  if [[ -e "$path" ]] && find "$path" -mindepth 1 -print -quit | grep -q .; then
    echo "Probe output is not empty: $path" >&2
    echo "Set a new RUN_TAG or explicitly inspect/move the old probe first." >&2
    exit 2
  fi
}

require_colocated_online_support() {
  local method=$1
  local supported=$2
  if [[ "$supported" != 1 && "${ALLOW_UNSUPPORTED_COLOCATED_RL:-0}" != 1 ]]; then
    echo "$MODEL_PROFILE does not fit colocated $method on its current $NUM_GPUS actor GPUs." >&2
    echo "Use separate rollout/discriminator resources, then set ALLOW_UNSUPPORTED_COLOCATED_RL=1 only after adapting the launcher." >&2
    exit 2
  fi
}

case "$ACTION" in
  validate)
    validate_profile
    ;;
  preflight)
    validate_profile
    worker_preflight
    ;;
  convert)
    validate_profile
    worker_preflight
    NUM_GPUS="$NUM_GPUS" MODEL_ARGS_FILE="$MODEL_ARGS_FILE" \
      HF_CHECKPOINT="$HF_CHECKPOINT" SAVE_DIR="$REF_LOAD" \
      bash drug_agent/scripts/prepare_qwen3_torch_dist.sh
    ;;
  sft-one-step)
    validate_profile
    worker_preflight
    require_path torch_dist "$REF_LOAD"
    SAVE_DIR=${SAVE_DIR:-$PROBE_ROOT/sft_one_step}
    require_fresh_dir "$SAVE_DIR"
    PROMPT_DATA=${PROMPT_DATA:-$CANONICAL_DATA} \
    SAVE_DIR="$SAVE_DIR" NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=1 \
    SFT_DEBUG_TRAIN_ONLY=1 SFT_DISABLE_OFFLOAD=1 NO_SAVE_OPTIM=1 SAVE_INTERVAL=100000 \
    DISABLE_CHECKPOINT_SAVE="${PROBE_DISABLE_CHECKPOINT_SAVE:-1}" \
    LR="$SFT_LR" MIN_LR="$SFT_MIN_LR" LR_DECAY_ITERS=1 LR_WARMUP_FRACTION=0 \
      bash drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh
    ;;
  toolrl-one-group)
    require_colocated_online_support ToolRL "${COLOCATED_TOOLRL_SUPPORTED:-1}"
    validate_profile
    worker_preflight
    require_path torch_dist "$REF_LOAD"
    require_path rollout_HF "${ROLLOUT_HF_CHECKPOINT:-$HF_CHECKPOINT}"
    TOOLRL_LOAD_ENV=()
    if [[ -n "${SFT_LOAD:-}" ]]; then
      require_path SFT_LOAD "$SFT_LOAD"
      TOOLRL_LOAD_ENV+=("LOAD=$SFT_LOAD")
    fi
    SAVE_DIR=${SAVE_DIR:-$PROBE_ROOT/toolrl_one_group}
    require_fresh_dir "$SAVE_DIR"
    env "${TOOLRL_LOAD_ENV[@]}" PROMPT_DATA="${PROMPT_DATA:-$TOOLRL_DATA}" SAVE_DIR="$SAVE_DIR" \
    NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=4 \
    NO_SAVE_OPTIM="${PROBE_NO_SAVE_OPTIM:-1}" \
    DISABLE_CHECKPOINT_SAVE="${PROBE_DISABLE_CHECKPOINT_SAVE:-1}" \
    LR="$TOOLRL_LR" LR_DECAY_STYLE=constant \
      bash drug_agent/toolrl/scripts/run_toolrl_grpo.sh
    ;;
  gad-negatives-one)
    require_colocated_online_support GAD-negative-generation "${COLOCATED_GAD_SUPPORTED:-1}"
    validate_profile
    worker_preflight
    : "${SFT_LOAD:?Set SFT_LOAD to a completed SFT checkpoint}"
    require_path SFT_LOAD "$SFT_LOAD"
    require_path torch_dist "$REF_LOAD"
    require_path rollout_HF "${ROLLOUT_HF_CHECKPOINT:-$HF_CHECKPOINT}"
    GAD_NEGATIVE_CACHE=${GAD_NEGATIVE_CACHE:-$PROBE_ROOT/gad_negative_one.jsonl}
    if [[ -e "$GAD_NEGATIVE_CACHE" ]]; then
      echo "Probe negative cache already exists: $GAD_NEGATIVE_CACHE" >&2
      exit 2
    fi
    mkdir -p "$(dirname "$GAD_NEGATIVE_CACHE")"
    PROMPT_DATA=${PROMPT_DATA:-$GAD_DATA} STUDENT_LOAD="$SFT_LOAD" \
    GAD_NEGATIVE_CACHE="$GAD_NEGATIVE_CACHE" NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=1 \
      bash drug_agent/gad/scripts/generate_stage2_negatives.sh
    ;;
  gad-discriminator-one)
    : "${SFT_LOAD:?Set SFT_LOAD to the generator SFT checkpoint}"
    GAD_NEGATIVE_CACHE=${GAD_NEGATIVE_CACHE:-$PROBE_ROOT/gad_negative_one.jsonl}
    require_path GAD_NEGATIVE_CACHE "$GAD_NEGATIVE_CACHE"
    require_path SFT_LOAD "$SFT_LOAD"
    DISCRIMINATOR_OUTPUT_DIR=${DISCRIMINATOR_OUTPUT_DIR:-$PROBE_ROOT/gad_discriminator_one}
    require_fresh_dir "$DISCRIMINATOR_OUTPUT_DIR"
    PAIRS="$GAD_NEGATIVE_CACHE" GENERATOR_WARMUP_LOAD="$SFT_LOAD" \
    DISCRIMINATOR_OUTPUT_DIR="$DISCRIMINATOR_OUTPUT_DIR" DISCRIMINATOR_BATCH_SIZE=1 \
    DISCRIMINATOR_EPOCHS=1 DISCRIMINATOR_MAX_LENGTH=${DISCRIMINATOR_MAX_LENGTH:-8192} \
      bash drug_agent/gad/scripts/run_stage2_discriminator_warmup.sh
    ;;
  gad-serve)
    : "${DISCRIMINATOR_RESUME:?Set DISCRIMINATOR_RESUME to the probe discriminator checkpoint}"
    require_path DISCRIMINATOR_RESUME "$DISCRIMINATOR_RESUME"
    CUDA_VISIBLE_DEVICES=${GAD_DISCRIMINATOR_GPU:-$((NUM_GPUS - 1))} \
      bash drug_agent/gad/scripts/serve_discriminator.sh
    ;;
  gad-one-group)
    require_colocated_online_support GAD "${COLOCATED_GAD_SUPPORTED:-1}"
    validate_profile
    # Online GAD adds a colocated discriminator to the actor, rollout engine,
    # and CPUAdam transient. This has a stricter measured memory gate than
    # SFT, ToolRL, negative generation, or discriminator-only warmup.
    worker_preflight "$GAD_MIN_HOST_MEMORY_GIB"
    : "${SFT_LOAD:?Set SFT_LOAD to the generator SFT checkpoint}"
    : "${DISCRIMINATOR_RESUME:?Set DISCRIMINATOR_RESUME to the discriminator checkpoint}"
    : "${GAD_WARMUP_MANIFEST:?Set GAD_WARMUP_MANIFEST to warmup_manifest.json}"
    : "${GAD_DISCRIMINATOR_URL:?Set GAD_DISCRIMINATOR_URL to the running service}"
    require_path SFT_LOAD "$SFT_LOAD"
    require_path DISCRIMINATOR_RESUME "$DISCRIMINATOR_RESUME"
    require_path GAD_WARMUP_MANIFEST "$GAD_WARMUP_MANIFEST"
    require_path rollout_HF "${ROLLOUT_HF_CHECKPOINT:-$HF_CHECKPOINT}"
    SAVE_DIR=${SAVE_DIR:-$PROBE_ROOT/gad_one_group}
    require_fresh_dir "$SAVE_DIR"
    PROMPT_DATA=${PROMPT_DATA:-$GAD_DATA} STUDENT_WARMUP_LOAD="$SFT_LOAD" \
    DISCRIMINATOR_WARMUP_LOAD="$DISCRIMINATOR_RESUME" SAVE_DIR="$SAVE_DIR" \
    NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=4 \
    STUDENT_LR="$GAD_LR" NO_SAVE_OPTIM=1 DISABLE_CHECKPOINT_SAVE=1 SAVE_INTERVAL=100000 \
      bash drug_agent/gad/scripts/run_stage3_gad_grpo.sh
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    usage >&2
    exit 2
    ;;
esac
