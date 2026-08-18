#!/usr/bin/env bash
# Launch a fresh Qwen3.5-9B SFT -> ToolRL run from a published v5 release.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/resolve_slime_env.sh"
source "$SLIME_ENV"

SLIME_WD_ROOT="${WD:-$(cd "$SLIME/.." && pwd)}"
DATA_ROOT="${DRUG_AGENT_DATA_ROOT:-$SLIME_WD_ROOT/outputs/slime_drug_agent_data}"
V5_VARIANT="${V5_VARIANT:-full}"

case "$V5_VARIANT" in
  full)
    LIVE_DATA_ROOT="${LIVE_DATA_ROOT:-$DATA_ROOT/live_tool_catalog_v5-sftnrl}"
    EXPECTED_DATASET_VERSION="live_tool_catalog_v5-sftnrl"
    EXPECTED_CANONICAL_RECORDS=605
    EXPECTED_EXCLUDED_RECORDS=0
    RUN_PREFIX="Qwen3.5-9B_v5_full_sft_toolrl"
    ;;
  mol)
    LIVE_DATA_ROOT="${LIVE_DATA_ROOT:-$DATA_ROOT/live_tool_catalog_v5-mol-sftnrl}"
    EXPECTED_DATASET_VERSION="live_tool_catalog_v5-mol-sftnrl"
    EXPECTED_CANONICAL_RECORDS=365
    EXPECTED_EXCLUDED_RECORDS=240
    RUN_PREFIX="Qwen3.5-9B_v5_mol_sft_toolrl"
    ;;
  *) echo "V5_VARIANT must be full or mol, got: $V5_VARIANT" >&2; exit 2 ;;
esac

for path in "$LIVE_DATA_ROOT/RELEASE_COMPLETE" "$LIVE_DATA_ROOT/dataset_manifest.json" \
  "$LIVE_DATA_ROOT/react_trajectories.jsonl" "$LIVE_DATA_ROOT/toolrl/toolrl_steps.jsonl" \
  "$LIVE_DATA_ROOT/materialize.complete" "$LIVE_DATA_ROOT/manifest.json"; do
  [[ -f "$path" ]] || { echo "missing v5 release artifact: $path" >&2; exit 2; }
done

EXPECTED_CANONICAL_SHA256="$(sha256sum "$LIVE_DATA_ROOT/react_trajectories.jsonl" | awk '{print $1}')"
RUN_ID="${RUN_ID:-${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)}"

exec env \
  LIVE_DATA_ROOT="$LIVE_DATA_ROOT" PREMATERIALIZED_RL_VIEW_ROOT="$LIVE_DATA_ROOT" \
  EXPECTED_DATASET_VERSION="$EXPECTED_DATASET_VERSION" \
  EXPECTED_CANONICAL_RECORDS="$EXPECTED_CANONICAL_RECORDS" \
  EXPECTED_EXCLUDED_RECORDS="$EXPECTED_EXCLUDED_RECORDS" \
  EXPECTED_CANONICAL_SHA256="$EXPECTED_CANONICAL_SHA256" RUN_ID="$RUN_ID" \
  bash "$SCRIPT_DIR/run_qwen3_5_9b_v4_mol_sft_toolrl_v2.sh"
