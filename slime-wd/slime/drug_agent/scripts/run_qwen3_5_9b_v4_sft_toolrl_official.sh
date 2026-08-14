#!/usr/bin/env bash
# Resumable Qwen3.5-9B v4 SFT -> decision-aware ToolRL pipeline for one 8xH200 worker.
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

HF_CHECKPOINT="${HF_CHECKPOINT:-$MODEL_ROOT/Qwen3.5-9B}"
REF_LOAD="${REF_LOAD:-$MODEL_ROOT/Qwen3.5-9B_torch_dist}"
MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-scripts/models/qwen3.5-9B.sh}"
RUN_ID="${RUN_ID:-Qwen3.5-9B_v4_sft_toolrl_official_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$RUNS_ROOT/$RUN_ID}"
LOG_ROOT="$RUN_ROOT/logs"
DATA_VIEW_ROOT="$RUN_ROOT/training_data"
PROBE_ROOT="$RUN_ROOT/gates"
SFT_DIR="${SFT_DIR:-$RUN_ROOT/sft}"
TOOLRL_DIR="${TOOLRL_DIR:-$RUN_ROOT/toolrl}"

TOOLRL_DERIVATIVE="$DATA_VIEW_ROOT/toolrl_steps.jsonl"
TOOLRL_CONVERT_REPORT="$DATA_VIEW_ROOT/toolrl_conversion.json"
TOOLRL_SKIPPED="$DATA_VIEW_ROOT/toolrl_skipped.jsonl"
SFT_DATA="$DATA_VIEW_ROOT/react_trajectories.gbs2.jsonl"
SFT_MANIFEST="$DATA_VIEW_ROOT/sft.manifest.json"
SFT_PROBES="$DATA_VIEW_ROOT/sft_probes"
TOOLRL_DATA="$DATA_VIEW_ROOT/toolrl_steps.decision_aware_rbs4.jsonl"
TOOLRL_MANIFEST="$DATA_VIEW_ROOT/toolrl_decision_aware.manifest.json"
TOOLRL_PROBES="$DATA_VIEW_ROOT/toolrl_probes"

SFT_MAX_SEQUENCE_LEN=131072
SFT_TRUNCATION_HEAD_TOKENS=8192
ROLLOUT_MAX_PROMPT_LEN=245760
ROLLOUT_MAX_RESPONSE_LEN=16384
ROLLOUT_MAX_CONTEXT_LEN=262144
EXPECTED_SFT_RECORDS=606
EXPECTED_DERIVATIVE_RECORDS=11909
EXPECTED_TOOLRL_RECORDS=5036
EXPECTED_TOOLRL_UNIQUE=3710
EXPECTED_TOOLRL_TOOLS=83
EXPECTED_TOOLRL_TASK_TYPES=5
TOOLRL_NUM_ROLLOUT=1259
TOOLRL_DISTRIBUTED_TIMEOUT_MINUTES=60
TOOLRL_SAVE_INTERVAL=100
TOOLRL_GATE_CANDIDATES=4
TOOLRL_GATE_MIN_NONZERO_GROUP_RATIO=0.25

RESUME_SERIAL_RUN="${RESUME_SERIAL_RUN:-0}"
SKIP_SFT="${SKIP_SFT:-0}"
for flag in "$RESUME_SERIAL_RUN" "$SKIP_SFT"; do
  [[ "$flag" == 0 || "$flag" == 1 ]] || { echo "boolean flags must be 0 or 1: $flag" >&2; exit 2; }
done

require_path() { [[ -e "$1" ]] || { echo "Required path does not exist: $1" >&2; exit 2; }; }
mark_complete() { touch "$RUN_ROOT/$1.complete"; }
run_logged() {
  local stage=$1
  shift
  echo "[$(date --iso-8601=seconds)] START $stage" | tee -a "$RUN_ROOT/status.log"
  "$@" 2>&1 | tee -a "$LOG_ROOT/$stage.log"
  echo "[$(date --iso-8601=seconds)] COMMAND_COMPLETE $stage" | tee -a "$RUN_ROOT/status.log"
}

for path in "$CANONICAL_DATA" "$TOOL_CATALOG" "$HF_CHECKPOINT" "$REF_LOAD/latest_checkpointed_iteration.txt" "$MODEL_ARGS_FILE"; do
  require_path "$path"
done
python - "$HF_CHECKPOINT/config.json" "$HF_CHECKPOINT/tokenizer_config.json" "$ROLLOUT_MAX_CONTEXT_LEN" <<'PY'
import json, sys
config_path, tokenizer_path, required = sys.argv[1], sys.argv[2], int(sys.argv[3])
config = json.load(open(config_path))
tokenizer = json.load(open(tokenizer_path))
model_limit = int(config.get("text_config", config).get("max_position_embeddings", 0))
tokenizer_limit = int(tokenizer.get("model_max_length", 0))
if model_limit < required or tokenizer_limit < required:
    raise SystemExit(
        f"Qwen context contract failed: model={model_limit} tokenizer={tokenizer_limit} required={required}"
    )
PY
[[ "$(sha256sum "$CANONICAL_DATA" | awk '{print $1}')" == "$EXPECTED_CANONICAL_SHA256" ]] || {
  echo "Canonical v4 hash mismatch; refusing to train" >&2
  exit 2
}
[[ "$(wc -l < "$CANONICAL_DATA")" == 605 ]] || { echo "Expected 605 canonical trajectories" >&2; exit 2; }

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[[ "$GPU_COUNT" == 8 ]] || { echo "Expected 8 GPUs, found $GPU_COUNT" >&2; exit 2; }
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -vc H200 || true)" == 0 ]] || {
  echo "This launcher requires 8 H200 GPUs" >&2; exit 2;
}
if [[ "${ALLOW_BUSY_GPUS:-0}" != 1 ]]; then
  BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  [[ -z "$BUSY" ]] || { echo "GPU processes already exist: $BUSY" >&2; exit 2; }
fi

if [[ -e "$RUN_ROOT" && "$RESUME_SERIAL_RUN" != 1 ]]; then
  echo "RUN_ROOT already exists: $RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$LOG_ROOT" "$DATA_VIEW_ROOT" "$PROBE_ROOT"
if [[ "$RESUME_SERIAL_RUN" == 1 ]]; then
  require_path "$RUN_ROOT/resolved_config.env"
  RECORDED_HASH=$(sed -n 's/^CANONICAL_SHA256=//p' "$RUN_ROOT/resolved_config.env")
  [[ "$RECORDED_HASH" == "$EXPECTED_CANONICAL_SHA256" ]] || { echo "Resume data hash mismatch" >&2; exit 2; }
fi

# Rebuild old single-prompt gate views when resuming this run. Candidate rows
# execute sequentially (RBS=1), so near-limit capacity is not multiplied.
PROBE_MANIFEST_OK=0
if [[ -f "$TOOLRL_PROBES/manifest.json" ]]; then
  if python - "$TOOLRL_PROBES/manifest.json" "$TOOLRL_GATE_CANDIDATES" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
if manifest.get("schema_version") != "toolrl_length_probe_candidates_v2":
    raise SystemExit(1)
if manifest.get("candidates_per_tier") != expected:
    raise SystemExit(1)
if any(probe.get("candidate_count") != expected for probe in manifest.get("probes", {}).values()):
    raise SystemExit(1)
PY
  then
    PROBE_MANIFEST_OK=1
  fi
fi
if [[ -f "$RUN_ROOT/materialize.complete" && "$PROBE_MANIFEST_OK" != 1 ]]; then
  require_path "$TOOLRL_DATA"
  python drug_agent/scripts/build_toolrl_length_probes.py \
    --input "$TOOLRL_DATA" --output-dir "$TOOLRL_PROBES" \
    --candidates-per-tier "$TOOLRL_GATE_CANDIDATES"
fi

if [[ ! -f "$RUN_ROOT/materialize.complete" ]]; then
  run_logged toolrl_convert python -m drug_agent.toolrl.convert_react_to_toolrl_steps \
    --input "$CANONICAL_DATA" --output "$TOOLRL_DERIVATIVE" \
    --skipped-report "$TOOLRL_SKIPPED" --report "$TOOLRL_CONVERT_REPORT"
  python - "$TOOLRL_CONVERT_REPORT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
c = r["counts"]
expected = {
    "kept": 11909, "kept_tool_call": 11304, "kept_final_answer": 605,
    "target_tool_call_total": 19232, "kept_tool_call_with_local": 4280,
    "kept_tool_call_mixed_local_molclaw": 50,
}
bad = {k: (c.get(k), v) for k, v in expected.items() if c.get(k) != v}
if bad or r.get("skipped_rows") != 0:
    raise SystemExit(f"v4 ToolRL derivative contract failed: bad={bad} skipped={r.get('skipped_rows')}")
PY
  python -m drug_agent.toolrl.validate_toolrl_offline_data --input "$TOOLRL_DERIVATIVE"
  python drug_agent/scripts/materialize_batch_aligned_sft.py \
    --input "$CANONICAL_DATA" --output "$SFT_DATA" --manifest "$SFT_MANIFEST" \
    --model "$HF_CHECKPOINT" --multiple 2
  python drug_agent/scripts/build_qwen35_9b_sft_probe_sets.py \
    --input "$SFT_DATA" --model "$HF_CHECKPOINT" --output-dir "$SFT_PROBES" --count 2
  python drug_agent/scripts/materialize_decision_aware_toolrl_view.py \
    --input "$TOOLRL_DERIVATIVE" --output "$TOOLRL_DATA" --manifest "$TOOLRL_MANIFEST" \
    --model "$HF_CHECKPOINT" --max-prompt-tokens "$ROLLOUT_MAX_PROMPT_LEN" \
    --max-response-tokens "$ROLLOUT_MAX_RESPONSE_LEN" --summary-max-tokens 32768 \
    --intermediate-budget 2500 --min-per-tool 8 --max-per-trajectory 8 --multiple 4 --seed 42
  python - "$SFT_DATA" "$TOOLRL_DERIVATIVE" "$TOOLRL_DATA" "$TOOLRL_MANIFEST" <<'PY'
import json, sys
sft, derivative, view, manifest = sys.argv[1:]
line_count = lambda p: sum(1 for line in open(p) if line.strip())
m = json.load(open(manifest))
actual = (line_count(sft), line_count(derivative), line_count(view), m["unique_records"], m["intermediate_selected"])
expected = (606, 11909, 5036, 3710, 2500)
if actual != expected:
    raise SystemExit(f"materialized data contract failed: {actual} != {expected}")
coverage = m["coverage"]
if coverage["selected_tool_count"] != 83 or coverage["missing_tools"] or len(coverage["selected_task_types"]) != 5:
    raise SystemExit(f"curated ToolRL coverage contract failed: {coverage}")
if m["contract"]["max_context_tokens"] != 262144:
    raise SystemExit(f"context contract failed: {m['contract']}")
if m["rejected_records"] != 8 or any(row["decision_role"] != "tool_step" for row in m["rejected"]):
    raise SystemExit(f"unexpected oversized-target exclusions: {m['rejected']}")
max_prompt = max_target = 0
for line_number, line in enumerate(open(view), 1):
    row = json.loads(line)
    metadata = row.get("metadata") or {}
    prompt_tokens = int(metadata.get("prompt_tokens_final", -1))
    target_tokens = int(metadata.get("canonical_target_tokens", -1))
    if prompt_tokens < 0 or target_tokens < 0:
        raise SystemExit(f"missing exact token audit at {view}:{line_number}")
    max_prompt = max(max_prompt, prompt_tokens)
    max_target = max(max_target, target_tokens)
if max_prompt > 245760 or max_target > 16384:
    raise SystemExit(f"materialized token contract failed: prompt={max_prompt} target={max_target}")
PY
  python drug_agent/scripts/build_toolrl_length_probes.py \
    --input "$TOOLRL_DATA" --output-dir "$TOOLRL_PROBES" \
    --candidates-per-tier "$TOOLRL_GATE_CANDIDATES"
  mark_complete materialize
fi

cat > "$RUN_ROOT/resolved_config.env" <<EOF
RUN_ID=$RUN_ID
CANONICAL_DATA=$CANONICAL_DATA
CANONICAL_SHA256=$EXPECTED_CANONICAL_SHA256
CANONICAL_RECORDS=605
TOOL_CATALOG=$TOOL_CATALOG
TOOLRL_DERIVATIVE=$TOOLRL_DERIVATIVE
TOOLRL_DERIVATIVE_RECORDS=$EXPECTED_DERIVATIVE_RECORDS
SFT_DATA=$SFT_DATA
SFT_RECORDS=$EXPECTED_SFT_RECORDS
TOOLRL_DATA=$TOOLRL_DATA
TOOLRL_UNIQUE=$EXPECTED_TOOLRL_UNIQUE
TOOLRL_RECORDS=$EXPECTED_TOOLRL_RECORDS
HF_CHECKPOINT=$HF_CHECKPOINT
REF_LOAD=$REF_LOAD
SFT_DIR=$SFT_DIR
TOOLRL_DIR=$TOOLRL_DIR
SFT_TP=4
SFT_PP=1
SFT_DP=2
SFT_GBS=2
SFT_EPOCHS=1
SFT_LR=5e-6
SFT_MIN_LR=5e-7
SFT_MAX_SEQUENCE_LEN=$SFT_MAX_SEQUENCE_LEN
SFT_TRUNCATION_HEAD_TOKENS=$SFT_TRUNCATION_HEAD_TOKENS
TOOLRL_ADVANTAGE_ESTIMATOR=grpo
TOOLRL_RBS=4
TOOLRL_N=4
TOOLRL_GBS=16
TOOLRL_NUM_ROLLOUT=$TOOLRL_NUM_ROLLOUT
TOOLRL_LR=2e-7
TOOLRL_DISTRIBUTED_TIMEOUT_MINUTES=$TOOLRL_DISTRIBUTED_TIMEOUT_MINUTES
TOOLRL_SAVE_INTERVAL=$TOOLRL_SAVE_INTERVAL
TOOLRL_GATE_CANDIDATES=$TOOLRL_GATE_CANDIDATES
TOOLRL_GATE_MIN_NONZERO_GROUP_RATIO=$TOOLRL_GATE_MIN_NONZERO_GROUP_RATIO
TOOLRL_REWARD_MODE=decision_aware
ROLLOUT_MAX_PROMPT_LEN=$ROLLOUT_MAX_PROMPT_LEN
ROLLOUT_MAX_RESPONSE_LEN=$ROLLOUT_MAX_RESPONSE_LEN
ROLLOUT_MAX_CONTEXT_LEN=$ROLLOUT_MAX_CONTEXT_LEN
CODE_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
CODE_DIRTY=$(git status --porcelain | wc -l)
EOF

SFT_COMMON=(
  env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
  attempt="$PROBE_ROOT/sft_smoke_$(date +%Y%m%d_%H%M%S)"
  run_logged sft_smoke "${SFT_COMMON[@]}" \
    PROMPT_DATA="$SFT_PROBES/sft_short_2.jsonl" SAVE_DIR="$attempt" RUN_NAME="${RUN_ID}_sft_smoke" \
    NUM_EPOCH=1 ROLLOUT_BATCH_SIZE=2 DISABLE_CHECKPOINT_SAVE=1 \
    bash drug_agent/scripts/run_qwen3_5_9b_drug_sft_full.sh
  grep -Eq "train/(loss|grad_norm)" "$LOG_ROOT/sft_smoke.log" || {
    echo "SFT smoke did not emit training metrics" >&2; exit 2;
  }
  mark_complete sft_smoke
fi

if [[ "$SKIP_SFT" == 1 ]]; then
  require_path "$SFT_DIR/latest_checkpointed_iteration.txt"
  mark_complete sft
elif [[ ! -f "$RUN_ROOT/sft.complete" ]]; then
  SFT_RESUME=()
  [[ -f "$SFT_DIR/latest_checkpointed_iteration.txt" ]] && SFT_RESUME+=(RESUME_DIR="$SFT_DIR")
  run_logged sft "${SFT_COMMON[@]}" "${SFT_RESUME[@]}" \
    PROMPT_DATA="$SFT_DATA" SAVE_DIR="$SFT_DIR" RUN_NAME="${RUN_ID}_sft" \
    NUM_EPOCH=1 ROLLOUT_BATCH_SIZE="$EXPECTED_SFT_RECORDS" SAVE_INTERVAL=100 CHECKPOINT_KEEP_LAST=4 \
    bash drug_agent/scripts/run_qwen3_5_9b_drug_sft_full.sh
fi
require_path "$SFT_DIR/latest_checkpointed_iteration.txt"
mark_complete sft

TOOLRL_COMMON=(
  env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  MODEL_ARGS_FILE="$MODEL_ARGS_FILE" HF_CHECKPOINT="$HF_CHECKPOINT" REF_LOAD="$REF_LOAD"
  NUM_GPUS=8 TENSOR_MODEL_PARALLEL_SIZE=4 PIPELINE_MODEL_PARALLEL_SIZE=2
  CONTEXT_PARALLEL_SIZE=1 EXPERT_MODEL_PARALLEL_SIZE=1 EXPERT_TENSOR_PARALLEL_SIZE=1
  ROLLOUT_NUM_GPUS_PER_ENGINE=1 ADVANTAGE_ESTIMATOR=grpo NORMALIZE_ADVANTAGES=0
  ROLLOUT_MAX_PROMPT_LEN="$ROLLOUT_MAX_PROMPT_LEN" ROLLOUT_MAX_RESPONSE_LEN="$ROLLOUT_MAX_RESPONSE_LEN"
  ROLLOUT_MAX_CONTEXT_LEN="$ROLLOUT_MAX_CONTEXT_LEN" ROLLOUT_TEMPERATURE=1.0
  SGLANG_MEM_FRACTION_STATIC=0.25 MAX_TOKENS_PER_GPU=16384 LOG_PROBS_CHUNK_SIZE=64
  RECOMPUTE_FULL=1 RECOMPUTE_NUM_LAYERS=1 RECOMPUTE_LOSS_FUNCTION=1 RECOMPUTE_VOCAB_LOG_PROBS=1
  TOOLRL_REWARD_MODE=decision_aware CUSTOM_ROLLOUT_LOG_FUNCTION_PATH=drug_agent.toolrl.metrics.augment_rollout_metrics
  LR=2e-7 MIN_LR=0 LR_DECAY_STYLE=constant WEIGHT_DECAY=0.1
  DISTRIBUTED_TIMEOUT_MINUTES="$TOOLRL_DISTRIBUTED_TIMEOUT_MINUTES"
  ADAM_BETA1=0.9 ADAM_BETA2=0.95 USE_ROLLOUT_LOGPROBS=0 USE_KL_LOSS=0
  COLOCATE_OFFLOAD_TRAIN=0 COLOCATE_OFFLOAD_ROLLOUT=0
  SGLANG_DISABLE_CUDA_GRAPH=1 SGLANG_DISABLE_CUSTOM_ALL_REDUCE=1 SGLANG_DISABLE_OVERLAP_SCHEDULE=1
)

check_rl_gate() {
  python -m drug_agent.scripts.check_rl_training_gate "$1" "$2" \
    --minimum-nonzero-group-ratio "${3:-0}"
}

for length_gate in shortest p50 p95 near_limit; do
  gate_marker="toolrl_${length_gate}"
  if [[ ! -f "$RUN_ROOT/$gate_marker.complete" ]]; then
    gate_stamp=$(date +%Y%m%d_%H%M%S)
    gate_stage="${gate_marker}_candidates_${gate_stamp}"
    gate_dir="$PROBE_ROOT/$gate_stage"
    run_logged "$gate_stage" "${TOOLRL_COMMON[@]}" \
      PROMPT_DATA="$TOOLRL_PROBES/toolrl_${length_gate}.jsonl" SAVE_DIR="$gate_dir" \
      LOAD="$SFT_DIR" TOOLRL_RESUME=0 NUM_ROLLOUT="$TOOLRL_GATE_CANDIDATES" ROLLOUT_BATCH_SIZE=1 \
      N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=4 DISABLE_CHECKPOINT_SAVE=1 \
      bash drug_agent/toolrl/scripts/run_toolrl_grpo.sh
    check_rl_gate "$LOG_ROOT/$gate_stage.log" "$TOOLRL_GATE_CANDIDATES" \
      "$TOOLRL_GATE_MIN_NONZERO_GROUP_RATIO"
    mark_complete "$gate_marker"
  fi
done

if [[ ! -f "$RUN_ROOT/toolrl_multi_update.complete" ]]; then
  gate_dir="$PROBE_ROOT/toolrl_multi_update_$(date +%Y%m%d_%H%M%S)"
  run_logged toolrl_multi_update "${TOOLRL_COMMON[@]}" \
    PROMPT_DATA="$TOOLRL_DATA" SAVE_DIR="$gate_dir" LOAD="$SFT_DIR" TOOLRL_RESUME=0 \
    NUM_ROLLOUT=10 ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=16 \
    DISABLE_CHECKPOINT_SAVE=1 \
    bash drug_agent/toolrl/scripts/run_toolrl_grpo.sh
  check_rl_gate "$LOG_ROOT/toolrl_multi_update.log" 10
  mark_complete toolrl_multi_update
fi

if [[ ! -f "$RUN_ROOT/toolrl.complete" ]]; then
  TOOLRL_LOAD=(LOAD="$SFT_DIR" TOOLRL_RESUME=0)
  if [[ -f "$TOOLRL_DIR/latest_checkpointed_iteration.txt" ]]; then
    TOOLRL_LOAD=(LOAD="$TOOLRL_DIR" TOOLRL_RESUME=1)
  fi
  run_logged toolrl "${TOOLRL_COMMON[@]}" "${TOOLRL_LOAD[@]}" \
    PROMPT_DATA="$TOOLRL_DATA" SAVE_DIR="$TOOLRL_DIR" \
    NUM_ROLLOUT="$TOOLRL_NUM_ROLLOUT" ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=16 \
    SAVE_INTERVAL="$TOOLRL_SAVE_INTERVAL" CHECKPOINT_KEEP_LAST=16 \
    bash drug_agent/toolrl/scripts/run_toolrl_grpo.sh
fi
require_path "$TOOLRL_DIR/latest_checkpointed_iteration.txt"
mark_complete toolrl
echo "[$(date --iso-8601=seconds)] PIPELINE COMPLETE" | tee -a "$RUN_ROOT/status.log"
