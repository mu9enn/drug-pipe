#!/usr/bin/env bash
# Backward-compatible entrypoint for the Qwen3.5-9B SFT -> ToolRL v2 path.
# Planning supervision is intentionally deferred until the 605 trajectories
# have LLM-authored, reviewed plan targets; this launcher does not build or
# train a heuristic Plan-SFT view.
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
LIVE_DATA_ROOT="${LIVE_DATA_ROOT:-$DATA_ROOT/live_tool_catalog_v4}"
CANONICAL_DATA="${CANONICAL_DATA:-$LIVE_DATA_ROOT/react_trajectories.jsonl}"
TOOL_CATALOG="${DRUG_AGENT_TOOL_CATALOG:-$LIVE_DATA_ROOT/tool_catalog.json}"
export DRUG_AGENT_TOOL_CATALOG="$TOOL_CATALOG"
EXPECTED_CANONICAL_SHA256="${EXPECTED_CANONICAL_SHA256:-be4ed789b45b280b338a3344558736cc43847b19478df7d71d53853a2de91e1e}"
EXPECTED_CANONICAL_RECORDS="${EXPECTED_CANONICAL_RECORDS:-605}"

HF_CHECKPOINT="${HF_CHECKPOINT:-$MODEL_ROOT/Qwen3.5-9B}"
REF_LOAD="${REF_LOAD:-$MODEL_ROOT/Qwen3.5-9B_torch_dist}"
MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-scripts/models/qwen3.5-9B.sh}"
BASE_SFT_DIR="${BASE_SFT_DIR:-$RUNS_ROOT/Qwen3.5-9B_v4_sft_toolrl_decision_aware_20260812_173253/sft}"
TOOLRL_REF_LOAD="${TOOLRL_REF_LOAD:-$BASE_SFT_DIR}"
RUN_ID="${RUN_ID:-Qwen3.5-9B_v4_sft_toolrl_v2_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$RUNS_ROOT/$RUN_ID}"
LOG_ROOT="$RUN_ROOT/logs"
VIEW_ROOT="$RUN_ROOT/training_data"
GATE_ROOT="$RUN_ROOT/gates"
TOOLRL_DIR="$RUN_ROOT/toolrl"

DERIVATIVE="$VIEW_ROOT/toolrl_steps_v3.jsonl"
CONVERT_REPORT="$VIEW_ROOT/toolrl_conversion_v3.json"
SKIPPED="$VIEW_ROOT/toolrl_skipped_v3.jsonl"
TOOLRL_DATA="$VIEW_ROOT/toolrl_policy_boundary_candidates.jsonl"
TOOLRL_MANIFEST="$VIEW_ROOT/toolrl_policy_boundary.manifest.json"
TOOLRL_PROBES="$VIEW_ROOT/toolrl_probes"
LEARNABILITY_LOG="$RUN_ROOT/learnability.jsonl"
PREMATERIALIZED_RL_VIEW_ROOT="${PREMATERIALIZED_RL_VIEW_ROOT:-}"

if [[ -n "$PREMATERIALIZED_RL_VIEW_ROOT" ]]; then
  TOOLRL_DATA="${TOOLRL_DATA_OVERRIDE:-$PREMATERIALIZED_RL_VIEW_ROOT/toolrl/toolrl_steps.jsonl}"
  TOOLRL_MANIFEST="${TOOLRL_MANIFEST_OVERRIDE:-$PREMATERIALIZED_RL_VIEW_ROOT/toolrl/context_manifest.json}"
fi

ROLLOUT_MAX_CONTEXT_LEN=262144
ROLLOUT_MAX_PROMPT_LEN=245760
ROLLOUT_MAX_RESPONSE_LEN=16384
TOOLRL_NUM_ROLLOUT="${TOOLRL_NUM_ROLLOUT:-}"
TOOLRL_SAVE_INTERVAL="${TOOLRL_SAVE_INTERVAL:-25}"
TOOLRL_CHECKPOINT_KEEP_LAST="${TOOLRL_CHECKPOINT_KEEP_LAST:-4}"
TOOLRL_LONG_BATCH_GATE_UPDATES="${TOOLRL_LONG_BATCH_GATE_UPDATES:-2}"
TOOLRL_RETAIN_GATE_CHECKPOINTS="${TOOLRL_RETAIN_GATE_CHECKPOINTS:-0}"
TOOLRL_REWARD_MODE="${TOOLRL_REWARD_MODE:-hierarchical}"
TOOLRL_USE_KL_LOSS="${TOOLRL_USE_KL_LOSS:-1}"
TOOLRL_KL_COEF="${TOOLRL_KL_COEF:-0.0}"
TOOLRL_KL_LOSS_COEF="${TOOLRL_KL_LOSS_COEF:-0.001}"
TOOLRL_KL_LOSS_TYPE="${TOOLRL_KL_LOSS_TYPE:-low_var_kl}"
TOOLRL_NORMALIZE_ADVANTAGES="${TOOLRL_NORMALIZE_ADVANTAGES:-0}"
TOOLRL_DISABLE_REWARDS_NORMALIZATION="${TOOLRL_DISABLE_REWARDS_NORMALIZATION:-0}"
TOOLRL_CUSTOM_ADVANTAGE_FUNCTION_PATH="${TOOLRL_CUSTOM_ADVANTAGE_FUNCTION_PATH:-}"
TOOLRL_ENTROPY_COEF="${TOOLRL_ENTROPY_COEF:-0.0}"
TOOLRL_CALCULATE_PER_TOKEN_LOSS="${TOOLRL_CALCULATE_PER_TOKEN_LOSS:-0}"
TOOLRL_TRAINING_PIPELINE="${TOOLRL_TRAINING_PIPELINE:-SFT_TO_TOOLRL}"
DYNAMIC_MAX_DROPPED="${DYNAMIC_MAX_DROPPED:-128}"
TOOLRL_ENABLE_DYNAMIC_FILTER="${TOOLRL_ENABLE_DYNAMIC_FILTER:-1}"
TOOLRL_REQUIRE_EXACT_EPOCH="${TOOLRL_REQUIRE_EXACT_EPOCH:-0}"
TOOLRL_MIN_NONZERO_GROUP_RATIO="${TOOLRL_MIN_NONZERO_GROUP_RATIO:-1.0}"
GATE_CANDIDATES="${GATE_CANDIDATES:-32}"
RESUME_V2_RUN="${RESUME_V2_RUN:-0}"

require_path() { [[ -e "$1" ]] || { echo "Required path does not exist: $1" >&2; exit 2; }; }
mark_complete() { touch "$RUN_ROOT/$1.complete"; }
run_logged() {
  local stage=$1; shift
  local stage_log="$LOG_ROOT/$stage.log"
  # A resumed, incomplete stage must have an attempt-local log. Otherwise a
  # successful retry is rejected by the gate checker because the same file
  # still contains the earlier attempt's traceback/OOM signature.
  if [[ -s "$stage_log" ]]; then
    local archive="$stage_log.attempt_$(date +%Y%m%d_%H%M%S)"
    mv "$stage_log" "$archive"
    echo "[$(date --iso-8601=seconds)] ARCHIVED_INCOMPLETE_LOG $stage $archive" | tee -a "$RUN_ROOT/status.log"
  fi
  echo "[$(date --iso-8601=seconds)] START $stage" | tee -a "$RUN_ROOT/status.log"
  "$@" 2>&1 | tee "$stage_log"
  echo "[$(date --iso-8601=seconds)] COMMAND_COMPLETE $stage" | tee -a "$RUN_ROOT/status.log"
}

for path in "$CANONICAL_DATA" "$TOOL_CATALOG" "$HF_CHECKPOINT" "$REF_LOAD/latest_checkpointed_iteration.txt" "$BASE_SFT_DIR/latest_checkpointed_iteration.txt" "$TOOLRL_REF_LOAD/latest_checkpointed_iteration.txt" "$MODEL_ARGS_FILE"; do
  require_path "$path"
done
[[ "$(realpath "$TOOLRL_REF_LOAD")" == "$(realpath "$BASE_SFT_DIR")" ]] || {
  echo "TOOLRL_REF_LOAD must be the frozen SFT checkpoint: ref=$TOOLRL_REF_LOAD sft=$BASE_SFT_DIR" >&2
  exit 2
}
[[ "$(sha256sum "$CANONICAL_DATA" | awk '{print $1}')" == "$EXPECTED_CANONICAL_SHA256" ]] || { echo "Canonical v4 hash mismatch" >&2; exit 2; }
[[ "$(wc -l < "$CANONICAL_DATA")" == "$EXPECTED_CANONICAL_RECORDS" ]] || {
  echo "Expected $EXPECTED_CANONICAL_RECORDS canonical trajectories" >&2
  exit 2
}
python - "$HF_CHECKPOINT/config.json" "$HF_CHECKPOINT/tokenizer_config.json" <<'PY'
import json, sys
model = json.load(open(sys.argv[1])); tok = json.load(open(sys.argv[2]))
assert int(model.get("text_config", model).get("max_position_embeddings", 0)) >= 262144
assert int(tok.get("model_max_length", 0)) >= 262144
PY
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" == 8 ]] || { echo "Expected 8 GPUs" >&2; exit 2; }
if [[ "${ALLOW_BUSY_GPUS:-0}" != 1 ]]; then
  BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  [[ -z "$BUSY" ]] || { echo "GPU processes already exist: $BUSY" >&2; exit 2; }
fi
if [[ -e "$RUN_ROOT" && "$RESUME_V2_RUN" != 1 ]]; then
  echo "RUN_ROOT already exists: $RUN_ROOT" >&2; exit 2
fi
mkdir -p "$LOG_ROOT" "$VIEW_ROOT" "$GATE_ROOT"

if [[ ! -f "$RUN_ROOT/materialize.complete" ]]; then
if [[ -n "$PREMATERIALIZED_RL_VIEW_ROOT" ]]; then
require_path "$PREMATERIALIZED_RL_VIEW_ROOT/materialize.complete"
require_path "$PREMATERIALIZED_RL_VIEW_ROOT/manifest.json"
require_path "$TOOLRL_DATA"
require_path "$TOOLRL_MANIFEST"
python - "$PREMATERIALIZED_RL_VIEW_ROOT/manifest.json" "$EXPECTED_CANONICAL_SHA256" <<'PY'
import json,sys
manifest=json.load(open(sys.argv[1]))
assert manifest["source_sha256"] == sys.argv[2], (manifest["source_sha256"], sys.argv[2])
assert manifest["limits"] == {"context":262144,"prompt":245760,"response":16384}
PY
run_logged probes python drug_agent/scripts/build_toolrl_length_probes.py \
  --input "$TOOLRL_DATA" --output-dir "$TOOLRL_PROBES" --candidates-per-tier "$GATE_CANDIDATES"
else
run_logged convert python -m drug_agent.toolrl.convert_react_to_toolrl_steps \
  --input "$CANONICAL_DATA" --output "$DERIVATIVE" --skipped-report "$SKIPPED" --report "$CONVERT_REPORT"
python - "$CONVERT_REPORT" <<'PY'
import json, sys
r=json.load(open(sys.argv[1])); c=r["counts"]
expected={"kept":11909,"kept_tool_call":11304,"kept_final_answer":605,"target_tool_call_total":19232}
bad={k:(c.get(k),v) for k,v in expected.items() if c.get(k)!=v}
if bad or r.get("skipped_rows") != 0: raise SystemExit(f"v4 derivative mismatch: {bad}, skipped={r.get('skipped_rows')}")
if c.get("kept_role_planning",0) or c.get("kept_role_initial_tool_step",0): raise SystemExit("planning leaked into ToolRL roles")
PY
run_logged select_toolrl python drug_agent/scripts/select_toolrl_decisions.py \
  --input "$DERIVATIVE" --output "$TOOLRL_DATA" --manifest "$TOOLRL_MANIFEST" --model "$HF_CHECKPOINT" \
  --max-prompt-tokens "$ROLLOUT_MAX_PROMPT_LEN" --max-response-tokens "$ROLLOUT_MAX_RESPONSE_LEN" --summary-max-tokens 32768
python - "$TOOLRL_MANIFEST" <<'PY'
import json,sys
view=json.load(open(sys.argv[1]))
assert view["copies_added"] == 0
assert view["role_counts"].get("final") == 605
assert set(view["role_counts"]) <= {"tool_step","final"}
assert view["coverage"]["task_type_count"] == 5 and view["coverage"]["tool_count"] == 83
assert view["context"]["observed_max_prompt_tokens"] <= 245760
assert view["context"]["observed_max_target_tokens"] <= 16384
PY
run_logged probes python drug_agent/scripts/build_toolrl_length_probes.py \
  --input "$TOOLRL_DATA" --output-dir "$TOOLRL_PROBES" --candidates-per-tier "$GATE_CANDIDATES"
fi
mark_complete materialize
fi

TOOLRL_CANDIDATES=$(wc -l < "$TOOLRL_DATA")
if [[ "$TOOLRL_REQUIRE_EXACT_EPOCH" == 1 ]]; then
  (( TOOLRL_CANDIDATES % 4 == 0 )) || {
    echo "Exact ToolRL epoch requires decisions divisible by RBS=4: $TOOLRL_CANDIDATES" >&2
    exit 2
  }
  EXPECTED_TOOLRL_NUM_ROLLOUT=$(( TOOLRL_CANDIDATES / 4 ))
else
  EXPECTED_TOOLRL_NUM_ROLLOUT=$(( (TOOLRL_CANDIDATES + 3) / 4 ))
fi
if [[ -z "$TOOLRL_NUM_ROLLOUT" ]]; then
  TOOLRL_NUM_ROLLOUT="$EXPECTED_TOOLRL_NUM_ROLLOUT"
elif [[ "$TOOLRL_NUM_ROLLOUT" != "$EXPECTED_TOOLRL_NUM_ROLLOUT" && "${ALLOW_TOOLRL_ROLLOUT_OVERRIDE:-0}" != 1 ]]; then
  echo "TOOLRL_NUM_ROLLOUT must equal ceil(decisions/4): got=$TOOLRL_NUM_ROLLOUT expected=$EXPECTED_TOOLRL_NUM_ROLLOUT decisions=$TOOLRL_CANDIDATES" >&2
  exit 2
fi

cat > "$RUN_ROOT/resolved_config.env" <<EOF
RUN_ID=$RUN_ID
BASE_SFT_DIR=$BASE_SFT_DIR
TOOLRL_DIR=$TOOLRL_DIR
CANONICAL_SHA256=$EXPECTED_CANONICAL_SHA256
CANONICAL_RECORDS=$EXPECTED_CANONICAL_RECORDS
TRAINING_PIPELINE=$TOOLRL_TRAINING_PIPELINE
PLANNING_IMPROVEMENT=LLM_CLEAN_FIRST_THOUGHT_NO_DEDICATED_RL_WHEN_PRESENT
PREMATERIALIZED_RL_VIEW_ROOT=$PREMATERIALIZED_RL_VIEW_ROOT
TOOLRL_CANDIDATES=$(wc -l < "$TOOLRL_DATA")
TOOLRL_REWARD_MODE=$TOOLRL_REWARD_MODE
TOOLRL_REF_LOAD=$TOOLRL_REF_LOAD
TOOLRL_USE_KL_LOSS=$TOOLRL_USE_KL_LOSS
TOOLRL_KL_COEF=$TOOLRL_KL_COEF
TOOLRL_KL_LOSS_COEF=$TOOLRL_KL_LOSS_COEF
TOOLRL_KL_LOSS_TYPE=$TOOLRL_KL_LOSS_TYPE
TOOLRL_CUSTOM_ADVANTAGE_FUNCTION_PATH=$TOOLRL_CUSTOM_ADVANTAGE_FUNCTION_PATH
TOOLRL_CANDIDATES=$TOOLRL_CANDIDATES
TOOLRL_NUM_ROLLOUT=$TOOLRL_NUM_ROLLOUT
TOOLRL_SELECTOR=$([[ "$TOOLRL_ENABLE_DYNAMIC_FILTER" == 1 ]] && echo current_policy_reward_variance_n4 || echo deterministic_static_coverage)
TOOLRL_DYNAMIC_MAX_DROPPED=$DYNAMIC_MAX_DROPPED
TOOLRL_MEMORY_STRATEGY=resident_actor_offloaded_rollout
TOOLRL_LONG_BATCH_GATE_UPDATES=$TOOLRL_LONG_BATCH_GATE_UPDATES
TOOLRL_SAVE_INTERVAL=$TOOLRL_SAVE_INTERVAL
TOOLRL_CHECKPOINT_KEEP_LAST=$TOOLRL_CHECKPOINT_KEEP_LAST
TOOLRL_RETAIN_GATE_CHECKPOINTS=$TOOLRL_RETAIN_GATE_CHECKPOINTS
ROLLOUT_MAX_PROMPT_LEN=$ROLLOUT_MAX_PROMPT_LEN
ROLLOUT_MAX_RESPONSE_LEN=$ROLLOUT_MAX_RESPONSE_LEN
ROLLOUT_MAX_CONTEXT_LEN=$ROLLOUT_MAX_CONTEXT_LEN
EOF

require_path "$BASE_SFT_DIR/latest_checkpointed_iteration.txt"

TOOLRL_COMMON=(
  env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MODEL_ARGS_FILE="$MODEL_ARGS_FILE"
  HF_CHECKPOINT="$HF_CHECKPOINT" REF_LOAD="$TOOLRL_REF_LOAD" NUM_GPUS=8
  TENSOR_MODEL_PARALLEL_SIZE=4 PIPELINE_MODEL_PARALLEL_SIZE=2 CONTEXT_PARALLEL_SIZE=1
  EXPERT_MODEL_PARALLEL_SIZE=1 EXPERT_TENSOR_PARALLEL_SIZE=1 ROLLOUT_NUM_GPUS_PER_ENGINE=1
  ADVANTAGE_ESTIMATOR=grpo NORMALIZE_ADVANTAGES="$TOOLRL_NORMALIZE_ADVANTAGES" ROLLOUT_MAX_PROMPT_LEN="$ROLLOUT_MAX_PROMPT_LEN"
  ROLLOUT_MAX_RESPONSE_LEN="$ROLLOUT_MAX_RESPONSE_LEN" ROLLOUT_MAX_CONTEXT_LEN="$ROLLOUT_MAX_CONTEXT_LEN"
  ROLLOUT_TEMPERATURE=1.0 SGLANG_MEM_FRACTION_STATIC=0.25 MAX_TOKENS_PER_GPU=16384 LOG_PROBS_CHUNK_SIZE=64
  RECOMPUTE_FULL=1 RECOMPUTE_NUM_LAYERS=1 RECOMPUTE_LOSS_FUNCTION=1 RECOMPUTE_VOCAB_LOG_PROBS=1
  TOOLRL_REWARD_MODE="$TOOLRL_REWARD_MODE" CUSTOM_ROLLOUT_LOG_FUNCTION_PATH=drug_agent.toolrl.metrics.augment_rollout_metrics
  ROLLOUT_ALL_SAMPLES_PROCESS_PATH=drug_agent.toolrl.policy_boundary.audit_all_groups
  TOOLRL_LEARNABILITY_LOG="$LEARNABILITY_LOG" LR=2e-7 MIN_LR=0 LR_DECAY_STYLE=constant WEIGHT_DECAY=0.1
  DISTRIBUTED_TIMEOUT_MINUTES=60 ADAM_BETA1=0.9 ADAM_BETA2=0.95 USE_ROLLOUT_LOGPROBS=0
  USE_KL_LOSS="$TOOLRL_USE_KL_LOSS" KL_COEF="$TOOLRL_KL_COEF"
  KL_LOSS_COEF="$TOOLRL_KL_LOSS_COEF" KL_LOSS_TYPE="$TOOLRL_KL_LOSS_TYPE"
  DISABLE_REWARDS_NORMALIZATION="$TOOLRL_DISABLE_REWARDS_NORMALIZATION"
  CUSTOM_ADVANTAGE_FUNCTION_PATH="$TOOLRL_CUSTOM_ADVANTAGE_FUNCTION_PATH"
  ENTROPY_COEF="$TOOLRL_ENTROPY_COEF"
  CALCULATE_PER_TOKEN_LOSS="$TOOLRL_CALCULATE_PER_TOKEN_LOSS"
  COLOCATE_OFFLOAD_TRAIN=0 COLOCATE_OFFLOAD_ROLLOUT=1 SGLANG_DISABLE_CUDA_GRAPH=1
  SGLANG_DISABLE_CUSTOM_ALL_REDUCE=1 SGLANG_DISABLE_OVERLAP_SCHEDULE=1
)
if [[ "$TOOLRL_ENABLE_DYNAMIC_FILTER" == 1 ]]; then
  TOOLRL_COMMON+=(
    DYNAMIC_SAMPLING_FILTER_PATH=drug_agent.toolrl.policy_boundary.policy_boundary_filter
    DYNAMIC_SAMPLING_MAX_DROPPED_GROUPS="$DYNAMIC_MAX_DROPPED"
    DYNAMIC_SAMPLING_STRICT_MAX_DROPS=1
  )
else
  TOOLRL_COMMON+=(DYNAMIC_SAMPLING_FILTER_PATH= DYNAMIC_SAMPLING_MAX_DROPPED_GROUPS= DYNAMIC_SAMPLING_STRICT_MAX_DROPS=0)
fi

for tier in shortest p50 p95 near_limit; do
  stage="toolrl_gate_${tier}"
  if [[ ! -f "$RUN_ROOT/$stage.complete" ]]; then
    run_logged "$stage" "${TOOLRL_COMMON[@]}" PROMPT_DATA="$TOOLRL_PROBES/toolrl_${tier}.jsonl" \
      TOOLRL_LEARNABILITY_LOG="$RUN_ROOT/traversal_${stage}.jsonl" \
      SAVE_DIR="$GATE_ROOT/$tier" LOAD="$BASE_SFT_DIR" TOOLRL_RESUME=0 NUM_ROLLOUT=2 ROLLOUT_BATCH_SIZE=1 \
      N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=4 DISABLE_CHECKPOINT_SAVE=1 \
      bash drug_agent/toolrl/scripts/run_toolrl_grpo.sh
    GATE_ZERO_ARGS=()
    if [[ "${TOOLRL_ALLOW_ZERO_VARIANCE_PROBES:-0}" == 1 ]]; then
      GATE_ZERO_ARGS+=(--allow-all-zero-gradients)
    fi
    python -m drug_agent.scripts.check_rl_training_gate "$LOG_ROOT/$stage.log" 2 \
      --minimum-nonzero-group-ratio "$TOOLRL_MIN_NONZERO_GROUP_RATIO" "${GATE_ZERO_ARGS[@]}"
    mark_complete "$stage"
  fi
done

# The single-prompt near-limit gate above proves that one maximum-length
# sample fits, but production uses RBS=4/n=4 and therefore trains 16 samples
# per update.  Keep a separate fail-closed stress gate with the exact
# production batch geometry.  Two updates are required so the second rollout
# also validates generation after a complete SGLang release/resume cycle.
if [[ ! -f "$RUN_ROOT/toolrl_gate_near_limit_rbs4_offload.complete" ]]; then
  run_logged toolrl_gate_near_limit_rbs4_offload "${TOOLRL_COMMON[@]}" \
    TOOLRL_LEARNABILITY_LOG="$RUN_ROOT/traversal_toolrl_gate_near_limit_rbs4_offload.jsonl" \
    PROMPT_DATA="$TOOLRL_PROBES/toolrl_near_limit.jsonl" \
    SAVE_DIR="$GATE_ROOT/near_limit_rbs4_offload" LOAD="$BASE_SFT_DIR" TOOLRL_RESUME=0 \
    NUM_ROLLOUT="$TOOLRL_LONG_BATCH_GATE_UPDATES" ROLLOUT_BATCH_SIZE=4 \
    N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=16 SAVE_INTERVAL=1 CHECKPOINT_KEEP_LAST=1 \
    bash drug_agent/toolrl/scripts/run_toolrl_grpo.sh
  GATE_ZERO_ARGS=()
  if [[ "${TOOLRL_ALLOW_ZERO_VARIANCE_PROBES:-0}" == 1 ]]; then
    GATE_ZERO_ARGS+=(--allow-all-zero-gradients)
  fi
  python -m drug_agent.scripts.check_rl_training_gate \
    "$LOG_ROOT/toolrl_gate_near_limit_rbs4_offload.log" "$TOOLRL_LONG_BATCH_GATE_UPDATES" \
    --minimum-nonzero-group-ratio "$TOOLRL_MIN_NONZERO_GROUP_RATIO" "${GATE_ZERO_ARGS[@]}"
  require_path "$GATE_ROOT/near_limit_rbs4_offload/latest_checkpointed_iteration.txt"
  mark_complete toolrl_gate_near_limit_rbs4_offload
  # This gate only proves that production-shaped state can be serialized.
  # Its checkpoint starts from the same frozen SFT model and is never a
  # production resume source.  Keep logs/markers, but reclaim the full model
  # copy after a successful save unless explicitly requested for debugging.
  if [[ "$TOOLRL_RETAIN_GATE_CHECKPOINTS" != 1 ]]; then
    rm -rf -- "$GATE_ROOT/near_limit_rbs4_offload"
    echo "[$(date --iso-8601=seconds)] CLEANED_GATE_CHECKPOINT toolrl_gate_near_limit_rbs4_offload" \
      | tee -a "$RUN_ROOT/status.log"
  fi
fi

if [[ ! -f "$RUN_ROOT/toolrl_multi_update.complete" ]]; then
  run_logged toolrl_multi_update "${TOOLRL_COMMON[@]}" PROMPT_DATA="$TOOLRL_DATA" \
    TOOLRL_LEARNABILITY_LOG="$RUN_ROOT/traversal_toolrl_multi_update.jsonl" \
    SAVE_DIR="$GATE_ROOT/multi_update" LOAD="$BASE_SFT_DIR" TOOLRL_RESUME=0 NUM_ROLLOUT=10 \
    ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=16 DISABLE_CHECKPOINT_SAVE=1 \
    bash drug_agent/toolrl/scripts/run_toolrl_grpo.sh
  python -m drug_agent.scripts.check_rl_training_gate "$LOG_ROOT/toolrl_multi_update.log" 10 --minimum-nonzero-group-ratio "$TOOLRL_MIN_NONZERO_GROUP_RATIO"
  mark_complete toolrl_multi_update
fi

if [[ ! -f "$RUN_ROOT/toolrl.complete" ]]; then
  TOOLRL_LOAD="$BASE_SFT_DIR"
  TOOLRL_RESUME_FLAG=0
  if [[ -f "$TOOLRL_DIR/latest_checkpointed_iteration.txt" ]]; then
    TOOLRL_LOAD="$TOOLRL_DIR"
    TOOLRL_RESUME_FLAG=1
    echo "[$(date --iso-8601=seconds)] RESUME toolrl from $TOOLRL_DIR" | tee -a "$RUN_ROOT/status.log"
  fi
  run_logged toolrl "${TOOLRL_COMMON[@]}" PROMPT_DATA="$TOOLRL_DATA" SAVE_DIR="$TOOLRL_DIR" \
    TOOLRL_LEARNABILITY_LOG="$RUN_ROOT/toolrl_fixed_traversal.jsonl" \
    LOAD="$TOOLRL_LOAD" TOOLRL_RESUME="$TOOLRL_RESUME_FLAG" NUM_ROLLOUT="$TOOLRL_NUM_ROLLOUT" ROLLOUT_BATCH_SIZE=4 \
    N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=16 SAVE_INTERVAL="$TOOLRL_SAVE_INTERVAL" \
    CHECKPOINT_KEEP_LAST="$TOOLRL_CHECKPOINT_KEEP_LAST" \
    bash drug_agent/toolrl/scripts/run_toolrl_grpo.sh
fi
require_path "$TOOLRL_DIR/latest_checkpointed_iteration.txt"
if [[ "$TOOLRL_REQUIRE_EXACT_EPOCH" == 1 ]]; then
  python -m drug_agent.scripts.validate_fixed_toolrl_traversal \
    --dataset "$TOOLRL_DATA" \
    --audit "$RUN_ROOT/toolrl_fixed_traversal.jsonl" \
    --output "$RUN_ROOT/toolrl_fixed_traversal.audit.json"
fi
mark_complete toolrl
echo "[$(date --iso-8601=seconds)] PIPELINE COMPLETE" | tee -a "$RUN_ROOT/status.log"
