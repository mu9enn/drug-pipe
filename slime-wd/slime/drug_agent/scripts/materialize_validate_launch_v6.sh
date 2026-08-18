#!/usr/bin/env bash
# Materialize one immutable v6-final release, validate both views, then launch
# the Drug-Pipe production SFT -> ToolRL profile on the current 8-GPU worker.
set -euo pipefail

VARIANT="${1:?usage: materialize_validate_launch_v6.sh full|mol}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/resolve_slime_env.sh"
source "$SLIME_ENV"
cd "$SLIME"

SLIME_WD_ROOT="${WD:-$(cd "$SLIME/.." && pwd)}"
DATA_ROOT="${DRUG_AGENT_DATA_ROOT:-$SLIME_WD_ROOT/outputs/slime_drug_agent_data}"
MODEL_ROOT="${DATA:-$SLIME_WD_ROOT/data}/Qwen3.5-9B"

case "$VARIANT" in
  full)
    INPUT="$DATA_ROOT/live_tool_catalog_v5-sftnrl/react_trajectories.jsonl"
    OUTPUT="$DATA_ROOT/live_tool_catalog_v6-final-sftnrl"
    DATASET_VERSION="live_tool_catalog_v6-final-sftnrl"
    CATALOG="$DATA_ROOT/live_tool_catalog_v6-turn-sftnrl/tool_catalog.json"
    PLANNING="$DATA_ROOT/live_tool_catalog_v6-turn-sftnrl/planning_annotations.jsonl"
    PARENT="live_tool_catalog_v5-sftnrl"
    EXCLUDED_PARENT=0
    ;;
  mol)
    INPUT="$DATA_ROOT/live_tool_catalog_v5-mol-sftnrl/react_trajectories.jsonl"
    OUTPUT="$DATA_ROOT/live_tool_catalog_v6-final-mol-sftnrl"
    DATASET_VERSION="live_tool_catalog_v6-final-mol-sftnrl"
    CATALOG="$DATA_ROOT/live_tool_catalog_v6-turn-mol-sftnrl/tool_catalog.json"
    PLANNING="$DATA_ROOT/live_tool_catalog_v6-turn-mol-sftnrl/planning_annotations.jsonl"
    PARENT="live_tool_catalog_v5-mol-sftnrl"
    EXCLUDED_PARENT=240
    ;;
  *)
    echo "variant must be full or mol: $VARIANT" >&2
    exit 2
    ;;
esac

for path in "$INPUT" "$MODEL_ROOT" "$CATALOG" "$PLANNING"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 2; }
done
if [[ -e "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing non-empty v6-final output: $OUTPUT" >&2
  exit 2
fi

python -m drug_agent.scripts.materialize_toolrl_turn_v6 \
  --input-react "$INPUT" \
  --output-root "$OUTPUT" \
  --model "$MODEL_ROOT" \
  --dataset-version "$DATASET_VERSION" \
  --tool-catalog "$CATALOG" \
  --planning-annotations "$PLANNING" \
  --parent-dataset-version "$PARENT" \
  --excluded-parent-trajectories "$EXCLUDED_PARENT"

for view in production official_baseline; do
  python -m drug_agent.scripts.validate_toolrl_turn_release \
    --root "$OUTPUT" \
    --model "$MODEL_ROOT" \
    --view "$view" \
    --output "$OUTPUT/audit/validation.${view}.json"
done

[[ "${V6_START_TRAINING:-1}" == 1 ]] || exit 0
RUN_ID="${RUN_ID:-Qwen3.5-9B_v6_final_${VARIANT}_production_20260817}"
export V6_VARIANT="$VARIANT" V6_PROFILE=drug_pipe_production RUN_ID
exec bash "$SCRIPT_DIR/run_qwen3_5_9b_v6_turn_sft_toolrl.sh"
