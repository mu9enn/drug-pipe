#!/usr/bin/env bash
# Qwen3.5-9B SFT -> fixed-view ToolRL turn-semantics launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/resolve_slime_env.sh"
source "$SLIME_ENV"

SLIME_WD_ROOT="${WD:-$(cd "$SLIME/.." && pwd)}"
DATA_ROOT="${DRUG_AGENT_DATA_ROOT:-$SLIME_WD_ROOT/outputs/slime_drug_agent_data}"
V6_VARIANT="${V6_VARIANT:-full}"
V6_PROFILE="${V6_PROFILE:-official_baseline}"

case "$V6_VARIANT" in
  full)
    LIVE_DATA_ROOT="${LIVE_DATA_ROOT:-$DATA_ROOT/live_tool_catalog_v6-final-sftnrl}"
    EXPECTED_DATASET_VERSION="live_tool_catalog_v6-final-sftnrl"
    EXPECTED_CANONICAL_RECORDS=605
    EXPECTED_EXCLUDED_RECORDS=0
    RUN_PREFIX="Qwen3.5-9B_v6_turn_full_sft_toolrl"
    ;;
  mol)
    LIVE_DATA_ROOT="${LIVE_DATA_ROOT:-$DATA_ROOT/live_tool_catalog_v6-final-mol-sftnrl}"
    EXPECTED_DATASET_VERSION="live_tool_catalog_v6-final-mol-sftnrl"
    EXPECTED_CANONICAL_RECORDS=365
    EXPECTED_EXCLUDED_RECORDS=240
    RUN_PREFIX="Qwen3.5-9B_v6_turn_mol_sft_toolrl"
    ;;
  *) echo "V6_VARIANT must be full or mol, got: $V6_VARIANT" >&2; exit 2 ;;
esac

for path in "$LIVE_DATA_ROOT/RELEASE_COMPLETE" "$LIVE_DATA_ROOT/dataset_manifest.json" \
  "$LIVE_DATA_ROOT/react_trajectories.jsonl" "$LIVE_DATA_ROOT/toolrl/toolrl_steps.jsonl" \
  "$LIVE_DATA_ROOT/toolrl/toolrl_steps.official_baseline.jsonl" \
  "$LIVE_DATA_ROOT/toolrl/context_manifest.official_baseline.json" \
  "$LIVE_DATA_ROOT/materialize.complete" "$LIVE_DATA_ROOT/manifest.json" "$LIVE_DATA_ROOT/tool_catalog.json"; do
  [[ -f "$path" ]] || { echo "missing v6 release artifact: $path" >&2; exit 2; }
done

read -r SFT_RECORDS PRODUCTION_RECORDS BASELINE_RECORDS <<<"$(python - "$LIVE_DATA_ROOT/dataset_manifest.json" "$EXPECTED_DATASET_VERSION" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
assert manifest["dataset_version"] == sys.argv[2]
assert manifest["protocol"] == "toolrl_turn_v1"
assert manifest["training_flow"] == "SFT -> ToolRL"
assert manifest["training_flows"]["official_baseline"] == "base checkpoint -> ToolRL"
assert manifest["training_flows"]["drug_pipe_production"] == "SFT -> ToolRL"
assert manifest["tool_catalog_injected_into_prompts"] is False
production = int(manifest["toolrl"]["records"])
baseline = int(manifest["toolrl_official_baseline"]["records"])
sft = int(manifest["sft"]["records"])
assert production > 0 and production % 4 == 0
assert baseline > 0 and baseline % 4 == 0
assert manifest["toolrl"]["selection"]["runtime_filter"] is None
assert manifest["toolrl_official_baseline"]["selection"]["runtime_filter"] is None
assert manifest["toolrl"]["accounting"]["grpo_group_count"] == production
assert manifest["toolrl_official_baseline"]["accounting"]["grpo_group_count"] == baseline
print(sft, production, baseline)
PY
)"
[[ -z "${DYNAMIC_SAMPLING_FILTER_PATH:-}" ]] || {
  echo "v6 fixed-view launcher rejects DYNAMIC_SAMPLING_FILTER_PATH" >&2
  exit 2
}

EXPECTED_CANONICAL_SHA256="$(sha256sum "$LIVE_DATA_ROOT/react_trajectories.jsonl" | awk '{print $1}')"
RUN_ID="${RUN_ID:-${RUN_PREFIX}_${V6_PROFILE}_$(date +%Y%m%d_%H%M%S)}"

COMMON_ENV=(
  env
  LIVE_DATA_ROOT="$LIVE_DATA_ROOT" PREMATERIALIZED_RL_VIEW_ROOT="$LIVE_DATA_ROOT" \
  EXPECTED_DATASET_VERSION="$EXPECTED_DATASET_VERSION" \
  EXPECTED_CANONICAL_RECORDS="$SFT_RECORDS" \
  EXPECTED_EXCLUDED_RECORDS="$EXPECTED_EXCLUDED_RECORDS" \
  EXPECTED_CANONICAL_SHA256="$EXPECTED_CANONICAL_SHA256" RUN_ID="$RUN_ID" \
  DRUG_AGENT_TOOL_CATALOG="$LIVE_DATA_ROOT/tool_catalog.json"
  TOOLRL_ENABLE_DYNAMIC_FILTER=0 TOOLRL_REQUIRE_EXACT_EPOCH=1
  TOOLRL_MIN_NONZERO_GROUP_RATIO=0.0 TOOLRL_ALLOW_ZERO_VARIANCE_PROBES=1 EPS_CLIP=0.2 EPS_CLIP_HIGH=0.2
)

case "$V6_PROFILE" in
  official_baseline)
    BASE_MODEL_TORCH_DIST="${REF_LOAD:-${DATA:-$SLIME_WD_ROOT/data}/Qwen3.5-9B_torch_dist}"
    exec "${COMMON_ENV[@]}" \
      BASE_SFT_DIR="$BASE_MODEL_TORCH_DIST" TOOLRL_REF_LOAD="$BASE_MODEL_TORCH_DIST" \
      TOOLRL_DATA_OVERRIDE="$LIVE_DATA_ROOT/toolrl/toolrl_steps.official_baseline.jsonl" \
      TOOLRL_MANIFEST_OVERRIDE="$LIVE_DATA_ROOT/toolrl/context_manifest.official_baseline.json" \
      TOOLRL_NUM_ROLLOUT="$(( BASELINE_RECORDS / 4 ))" \
      TOOLRL_REWARD_MODE=toolrl_official_8cee13e TOOLRL_TRAINING_PIPELINE=TOOLRL_OFFICIAL_BASELINE \
      TOOLRL_STRUCTURED_FINAL_EXACT=0 \
      TOOLRL_USE_KL_LOSS=0 TOOLRL_KL_COEF=0.001 TOOLRL_KL_LOSS_COEF=0 \
      TOOLRL_KL_LOSS_TYPE=k1 TOOLRL_DISABLE_REWARDS_NORMALIZATION=1 \
      TOOLRL_CUSTOM_ADVANTAGE_FUNCTION_PATH=drug_agent.toolrl.official_grpo.compute_official_8cee13e_advantages \
      TOOLRL_ENTROPY_COEF=0.001 \
      TOOLRL_CALCULATE_PER_TOKEN_LOSS=1 \
      bash "$SCRIPT_DIR/run_qwen3_5_9b_v4_plan_sft_toolrl_v2.sh"
    ;;
  drug_pipe_production)
    exec "${COMMON_ENV[@]}" \
      TOOLRL_NUM_ROLLOUT="$(( PRODUCTION_RECORDS / 4 ))" \
      TOOLRL_REWARD_MODE=hierarchical TOOLRL_USE_KL_LOSS=1 TOOLRL_KL_COEF=0 \
      TOOLRL_STRUCTURED_FINAL_EXACT=1 \
      TOOLRL_KL_LOSS_COEF=0.001 TOOLRL_KL_LOSS_TYPE=low_var_kl \
      bash "$SCRIPT_DIR/run_qwen3_5_9b_v4_mol_sft_toolrl_v2.sh"
    ;;
  *) echo "V6_PROFILE must be official_baseline or drug_pipe_production" >&2; exit 2 ;;
esac
