#!/usr/bin/env bash
# Resumable 8xH200 Qwen3.5-9B SFT -> hierarchical policy-boundary ToolRL
# using an audited prematerialized SFT+ToolRL release. Historical v4-mol
# defaults remain for compatibility; v5 wrappers override the contract.
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
DATA_ROOT="${DRUG_AGENT_DATA_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_data}"
RUNS_ROOT="${DRUG_AGENT_RUNS_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_runs}"
MODEL_ROOT="${DATA:-$SLIME_WD_ROOT/data}"
LIVE_DATA_ROOT="${LIVE_DATA_ROOT:-$DATA_ROOT/live_tool_catalog_v4_mol}"
CANONICAL_DATA="$LIVE_DATA_ROOT/react_trajectories.jsonl"
TOOL_CATALOG="$LIVE_DATA_ROOT/tool_catalog.json"
EXPECTED_CANONICAL_SHA256="${EXPECTED_CANONICAL_SHA256:-d3c5fa5954fbf1c85f859f0789d75d9b349e47b5735bc165e36eab0952f59d98}"
EXPECTED_CANONICAL_RECORDS="${EXPECTED_CANONICAL_RECORDS:-365}"
EXPECTED_DATASET_VERSION="${EXPECTED_DATASET_VERSION:-live_tool_catalog_v4_mol}"
EXPECTED_EXCLUDED_RECORDS="${EXPECTED_EXCLUDED_RECORDS:-240}"
PREMATERIALIZED_RL_VIEW_ROOT="${PREMATERIALIZED_RL_VIEW_ROOT:-$LIVE_DATA_ROOT}"

HF_CHECKPOINT="${HF_CHECKPOINT:-$MODEL_ROOT/Qwen3.5-9B}"
REF_LOAD="${REF_LOAD:-$MODEL_ROOT/Qwen3.5-9B_torch_dist}"
MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-scripts/models/qwen3.5-9B.sh}"
RUN_ID="${RUN_ID:-Qwen3.5-9B_v4_mol_sft_toolrl_v2_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$RUNS_ROOT/$RUN_ID}"
LOG_ROOT="$RUN_ROOT/logs"
VIEW_ROOT="$RUN_ROOT/training_data"
GATE_ROOT="$RUN_ROOT/gates"
SFT_DIR="$RUN_ROOT/sft"
SFT_DATA="$VIEW_ROOT/react_trajectories.gbs2.jsonl"
SFT_MANIFEST="$VIEW_ROOT/sft.manifest.json"
SFT_PROBES="$VIEW_ROOT/sft_probes"
SFT_MAX_SEQUENCE_LEN=131072
SFT_TRUNCATION_HEAD_TOKENS=8192
RESUME_MOL_RUN="${RESUME_MOL_RUN:-0}"
PIPELINE_TOOLRL_REWARD_MODE="${TOOLRL_REWARD_MODE:-hierarchical}"
PIPELINE_TOOLRL_USE_KL_LOSS="${TOOLRL_USE_KL_LOSS:-1}"
PIPELINE_TOOLRL_KL_COEF="${TOOLRL_KL_COEF:-0.0}"
PIPELINE_TOOLRL_KL_LOSS_COEF="${TOOLRL_KL_LOSS_COEF:-0.001}"
PIPELINE_TOOLRL_KL_LOSS_TYPE="${TOOLRL_KL_LOSS_TYPE:-low_var_kl}"

require_path() { [[ -e "$1" ]] || { echo "Required path does not exist: $1" >&2; exit 2; }; }
mark_complete() { touch "$RUN_ROOT/$1.complete"; }
run_logged() {
  local stage=$1; shift
  local log="$LOG_ROOT/$stage.log"
  if [[ -s "$log" ]]; then
    mv "$log" "$log.attempt_$(date +%Y%m%d_%H%M%S)"
  fi
  echo "[$(date --iso-8601=seconds)] START $stage" | tee -a "$RUN_ROOT/status.log"
  "$@" 2>&1 | tee "$log"
  echo "[$(date --iso-8601=seconds)] COMMAND_COMPLETE $stage" | tee -a "$RUN_ROOT/status.log"
}

drain_previous_stage_runtime() {
  local stage_log="$LOG_ROOT/sft_to_toolrl_transition.log"
  local deadline=$((SECONDS + 180))

  echo "[$(date --iso-8601=seconds)] START sft_to_toolrl_transition" | tee -a "$RUN_ROOT/status.log" "$stage_log"

  # The SFT submit command is blocking, so reaching this point means its Ray
  # job has finished.  Keep the serial launcher alive while explicitly
  # dismantling the SFT-owned Ray runtime; otherwise the ToolRL preflight can
  # observe stale CUDA actors, and a PID-based fallback queue can mistake the
  # stage boundary for the end of the whole pipeline.
  bash drug_agent/scripts/guard_ray_restart.sh 2>&1 | tee -a "$stage_log"
  ray stop --force 2>&1 | tee -a "$stage_log" || true

  while (( SECONDS < deadline )); do
    local busy
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
    if [[ -z "$busy" ]]; then
      echo "[$(date --iso-8601=seconds)] COMMAND_COMPLETE sft_to_toolrl_transition" \
        | tee -a "$RUN_ROOT/status.log" "$stage_log"
      return 0
    fi
    echo "[$(date --iso-8601=seconds)] waiting_for_sft_gpu_teardown pids=$(tr '\n' ',' <<<"$busy")" \
      | tee -a "$stage_log"
    sleep 2
  done

  echo "SFT GPU processes did not exit within 180 seconds" | tee -a "$stage_log" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    2>&1 | tee -a "$stage_log" >&2 || true
  return 2
}

for path in "$CANONICAL_DATA" "$TOOL_CATALOG" "$LIVE_DATA_ROOT/dataset_manifest.json" \
  "$PREMATERIALIZED_RL_VIEW_ROOT/materialize.complete" "$PREMATERIALIZED_RL_VIEW_ROOT/manifest.json" \
  "$PREMATERIALIZED_RL_VIEW_ROOT/toolrl/toolrl_steps.jsonl" \
  "$HF_CHECKPOINT/config.json" "$HF_CHECKPOINT/tokenizer_config.json" \
  "$REF_LOAD/latest_checkpointed_iteration.txt" "$MODEL_ARGS_FILE"; do
  require_path "$path"
done
[[ "$(sha256sum "$CANONICAL_DATA" | awk '{print $1}')" == "$EXPECTED_CANONICAL_SHA256" ]] || {
  echo "canonical dataset hash mismatch" >&2; exit 2;
}
[[ "$(wc -l < "$CANONICAL_DATA")" == "$EXPECTED_CANONICAL_RECORDS" ]] || {
  echo "Expected $EXPECTED_CANONICAL_RECORDS trajectories" >&2; exit 2;
}
python - "$LIVE_DATA_ROOT/dataset_manifest.json" "$PREMATERIALIZED_RL_VIEW_ROOT/manifest.json" \
  "$EXPECTED_DATASET_VERSION" "$EXPECTED_CANONICAL_RECORDS" "$EXPECTED_EXCLUDED_RECORDS" <<'PY'
import json, sys
data, rl = (json.load(open(path)) for path in sys.argv[1:3])
version, records, excluded = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
assert data["dataset_version"] == version
sft = data.get("canonical_react", data.get("sft", {}))
assert int(sft["records"]) == records
assert int(data.get("excluded_parent_records", data.get("excluded_parent_trajectories", 0))) == excluded
assert rl["source_sha256"] == sft["sha256"]
assert rl["limits"] == {"context": 262144, "prompt": 245760, "response": 16384}
PY

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[[ "$GPU_COUNT" == 8 ]] || { echo "Expected 8 GPUs, found $GPU_COUNT" >&2; exit 2; }
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -vc H200 || true)" == 0 ]] || {
  echo "This launcher requires 8 H200 GPUs" >&2; exit 2;
}
if [[ "${ALLOW_BUSY_GPUS:-0}" != 1 ]]; then
  BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  [[ -z "$BUSY" ]] || { echo "GPU processes already exist: $BUSY" >&2; exit 2; }
fi
if [[ -e "$RUN_ROOT" && "$RESUME_MOL_RUN" != 1 ]]; then
  echo "RUN_ROOT already exists: $RUN_ROOT" >&2; exit 2
fi
mkdir -p "$LOG_ROOT" "$VIEW_ROOT" "$GATE_ROOT"

if [[ ! -f "$RUN_ROOT/sft_data.complete" ]]; then
  run_logged sft_materialize python drug_agent/scripts/materialize_batch_aligned_sft.py \
    --input "$CANONICAL_DATA" --output "$SFT_DATA" --manifest "$SFT_MANIFEST" \
    --model "$HF_CHECKPOINT" --multiple 2
  python drug_agent/scripts/build_qwen35_9b_sft_probe_sets.py \
    --input "$SFT_DATA" --model "$HF_CHECKPOINT" --output-dir "$SFT_PROBES" --count 2 \
    > "$LOG_ROOT/sft_probes.log"
  EXPECTED_ALIGNED_RECORDS=$(( (EXPECTED_CANONICAL_RECORDS + 1) / 2 * 2 ))
  [[ "$(wc -l < "$SFT_DATA")" == "$EXPECTED_ALIGNED_RECORDS" ]] || {
    echo "Expected $EXPECTED_ALIGNED_RECORDS batch-aligned SFT rows" >&2; exit 2;
  }
  mark_complete sft_data
fi

cat > "$RUN_ROOT/serial_config.env" <<EOF
RUN_ID=$RUN_ID
TRAINING_PIPELINE=SFT_TO_TOOLRL
DATASET_VERSION=$EXPECTED_DATASET_VERSION
CANONICAL_DATA=$CANONICAL_DATA
CANONICAL_SHA256=$EXPECTED_CANONICAL_SHA256
CANONICAL_RECORDS=$EXPECTED_CANONICAL_RECORDS
EXCLUDED_PARENT_RECORDS=$EXPECTED_EXCLUDED_RECORDS
SFT_DATA=$SFT_DATA
SFT_RECORDS=$(( (EXPECTED_CANONICAL_RECORDS + 1) / 2 * 2 ))
SFT_TP=4
SFT_PP=1
SFT_DP=2
SFT_GBS=2
SFT_EPOCHS=1
SFT_LR=5e-6
SFT_MIN_LR=5e-7
SFT_MAX_SEQUENCE_LEN=$SFT_MAX_SEQUENCE_LEN
TOOLRL_REWARD_MODE=$PIPELINE_TOOLRL_REWARD_MODE
TOOLRL_SELECTOR=deterministic_static_coverage
TOOLRL_USE_KL_LOSS=$PIPELINE_TOOLRL_USE_KL_LOSS
TOOLRL_KL_COEF=$PIPELINE_TOOLRL_KL_COEF
TOOLRL_KL_LOSS_COEF=$PIPELINE_TOOLRL_KL_LOSS_COEF
TOOLRL_KL_LOSS_TYPE=$PIPELINE_TOOLRL_KL_LOSS_TYPE
TOOLRL_RBS=4
TOOLRL_N=4
TOOLRL_GBS=16
ROLLOUT_MAX_CONTEXT_LEN=262144
ROLLOUT_MAX_PROMPT_LEN=245760
ROLLOUT_MAX_RESPONSE_LEN=16384
HF_CHECKPOINT=$HF_CHECKPOINT
REF_LOAD=$REF_LOAD
CODE_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
CODE_DIRTY=$(git status --porcelain | wc -l)
EOF

SFT_COMMON=(
  env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  MODEL_ARGS_FILE="$MODEL_ARGS_FILE" HF_CHECKPOINT="$HF_CHECKPOINT" REF_LOAD="$REF_LOAD"
  NUM_GPUS=8 TENSOR_MODEL_PARALLEL_SIZE=4 PIPELINE_MODEL_PARALLEL_SIZE=1
  CONTEXT_PARALLEL_SIZE=1 EXPERT_MODEL_PARALLEL_SIZE=1 EXPERT_TENSOR_PARALLEL_SIZE=1
  GLOBAL_BATCH_SIZE=2 MAX_TOKENS_PER_GPU=16384
  SFT_MAX_SEQUENCE_LEN="$SFT_MAX_SEQUENCE_LEN" SFT_TRUNCATION_HEAD_TOKENS="$SFT_TRUNCATION_HEAD_TOKENS"
  LR=5e-6 MIN_LR=5e-7 LR_WARMUP_FRACTION=0.05 LR_DECAY_STYLE=cosine
  RECOMPUTE_FULL=1 RECOMPUTE_NUM_LAYERS=1 RECOMPUTE_LOSS_FUNCTION=1
  RECOMPUTE_VOCAB_LOG_PROBS=1 LOG_PROBS_CHUNK_SIZE=64 BALANCE_DATA=1
  SFT_DEBUG_TRAIN_ONLY=1 SFT_DISABLE_OFFLOAD=1
)

if [[ ! -f "$RUN_ROOT/sft_smoke.complete" ]]; then
  run_logged sft_smoke "${SFT_COMMON[@]}" \
    PROMPT_DATA="$SFT_PROBES/sft_short_2.jsonl" SAVE_DIR="$GATE_ROOT/sft_smoke" \
    RUN_NAME="${RUN_ID}_sft_smoke" NUM_EPOCH=1 ROLLOUT_BATCH_SIZE=2 DISABLE_CHECKPOINT_SAVE=1 \
    bash drug_agent/scripts/run_qwen3_5_9b_drug_sft_full.sh
  grep -Eq "train/(loss|grad_norm)" "$LOG_ROOT/sft_smoke.log" || {
    echo "SFT smoke did not emit training metrics" >&2; exit 2;
  }
  mark_complete sft_smoke
fi

if [[ ! -f "$RUN_ROOT/sft.complete" ]]; then
  SFT_RESUME=()
  [[ -f "$SFT_DIR/latest_checkpointed_iteration.txt" ]] && SFT_RESUME+=(RESUME_DIR="$SFT_DIR")
  run_logged sft "${SFT_COMMON[@]}" "${SFT_RESUME[@]}" \
    PROMPT_DATA="$SFT_DATA" SAVE_DIR="$SFT_DIR" RUN_NAME="${RUN_ID}_sft" \
    NUM_EPOCH=1 ROLLOUT_BATCH_SIZE=$(( (EXPECTED_CANONICAL_RECORDS + 1) / 2 * 2 )) SAVE_INTERVAL=100 CHECKPOINT_KEEP_LAST=4 \
    bash drug_agent/scripts/run_qwen3_5_9b_drug_sft_full.sh
  require_path "$SFT_DIR/latest_checkpointed_iteration.txt"
  mark_complete sft
fi

TOOLRL_CANDIDATES=$(wc -l < "$PREMATERIALIZED_RL_VIEW_ROOT/toolrl/toolrl_steps.jsonl")
TOOLRL_NUM_ROLLOUT="${TOOLRL_NUM_ROLLOUT:-$(( (TOOLRL_CANDIDATES + 3) / 4 ))}"
cat >> "$RUN_ROOT/serial_config.env" <<EOF
TOOLRL_DATA=$PREMATERIALIZED_RL_VIEW_ROOT/toolrl/toolrl_steps.jsonl
TOOLRL_CANDIDATES=$TOOLRL_CANDIDATES
TOOLRL_NUM_ROLLOUT=$TOOLRL_NUM_ROLLOUT
EOF

drain_previous_stage_runtime
echo "[$(date --iso-8601=seconds)] START toolrl_v2_pipeline" | tee -a "$RUN_ROOT/status.log"
exec env \
  LIVE_DATA_ROOT="$LIVE_DATA_ROOT" \
  CANONICAL_DATA="$CANONICAL_DATA" \
  DRUG_AGENT_TOOL_CATALOG="$TOOL_CATALOG" \
  EXPECTED_CANONICAL_SHA256="$EXPECTED_CANONICAL_SHA256" \
  EXPECTED_CANONICAL_RECORDS="$EXPECTED_CANONICAL_RECORDS" \
  PREMATERIALIZED_RL_VIEW_ROOT="$PREMATERIALIZED_RL_VIEW_ROOT" \
  BASE_SFT_DIR="$SFT_DIR" \
  RUN_ID="$RUN_ID" RUN_ROOT="$RUN_ROOT" RESUME_V2_RUN=1 \
  TOOLRL_NUM_ROLLOUT="$TOOLRL_NUM_ROLLOUT" TOOLRL_REWARD_MODE="$PIPELINE_TOOLRL_REWARD_MODE" \
  TOOLRL_USE_KL_LOSS="$PIPELINE_TOOLRL_USE_KL_LOSS" TOOLRL_KL_COEF="$PIPELINE_TOOLRL_KL_COEF" \
  TOOLRL_KL_LOSS_COEF="$PIPELINE_TOOLRL_KL_LOSS_COEF" TOOLRL_KL_LOSS_TYPE="$PIPELINE_TOOLRL_KL_LOSS_TYPE" \
  bash drug_agent/scripts/run_qwen3_5_9b_v4_sft_toolrl_v2.sh
