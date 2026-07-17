#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE_DIR="$REPO_DIR/pipeline"
PYTHON_BIN="${PYTHON_BIN:-python}"

RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RESULTS_ROOT/postprocess_candidates}"
ANSWER_HIT_ONLY=0
SKIP_EXPORT=0
SKIP_SFT=0
LLM_CLEAN=0
CLAUDE_BIN="${CLAUDE_BIN:-claude}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_postprocess.sh [options]

Options:
  --results-root PATH      Default: <repo>/results
  --output-root PATH       Default: <results-root>/postprocess_candidates
  --answer-hit-only        Only affects vs/ac/pf when building SFT/RL
  --llm-clean              Run protected Claude semantic cleaning before the final gate
  --claude-bin PATH        Claude executable used with --llm-clean
  --skip-export            Skip trajectory_exporter stage
  --skip-sft               Skip post_process_sft stage
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-root) RESULTS_ROOT="${2:-}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --answer-hit-only) ANSWER_HIT_ONLY=1; shift ;;
    --llm-clean) LLM_CLEAN=1; shift ;;
    --claude-bin) CLAUDE_BIN="${2:-}"; shift 2 ;;
    --skip-export) SKIP_EXPORT=1; shift ;;
    --skip-sft) SKIP_SFT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

RESULTS_ROOT="$(realpath -m "$RESULTS_ROOT")"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"

if [[ ! -d "$RESULTS_ROOT" ]]; then
  echo "[error] results root not found: $RESULTS_ROOT" >&2
  exit 1
fi

if [[ "$SKIP_EXPORT" -eq 0 ]]; then
  echo "[postprocess] stage1: trajectory_exporter (rebuild from raw complete_session)"
  mapfile -t RUN_CFGS < <(find "$RESULTS_ROOT" -type f -name run_config.json | sort)
  for cfg in "${RUN_CFGS[@]}"; do
    run_dir="$(dirname "$cfg")"
    echo "  - export: $run_dir"
    export_cmd=("$PYTHON_BIN" "$PIPELINE_DIR/postprocess/trajectory_exporter.py" "$run_dir")
    if [[ "$LLM_CLEAN" -eq 1 ]]; then
      export_cmd+=(--llm-clean --claude-bin "$CLAUDE_BIN")
    fi
    "${export_cmd[@]}" >/dev/null
  done
fi

if [[ "$SKIP_SFT" -eq 0 ]]; then
  echo "[postprocess] stage2: aggregate canonical ReAct trajectories"
  cmd=(
    "$PYTHON_BIN" "$PIPELINE_DIR/postprocess/aggregate_react.py"
    --results-root "$RESULTS_ROOT"
    --output-root "$OUTPUT_ROOT"
  )
  if [[ "$ANSWER_HIT_ONLY" -eq 1 ]]; then
    cmd+=(--answer-hit-only)
  fi
  "${cmd[@]}"
fi

echo "[done] postprocess pipeline finished"
echo "  results_root: $RESULTS_ROOT"
echo "  output_root:  $OUTPUT_ROOT"
if [[ "$SKIP_SFT" -eq 0 ]]; then
  echo "  canonical_react: $OUTPUT_ROOT/react_trajectories.jsonl"
  echo "  curation_audit:   $OUTPUT_ROOT/curation_audit.jsonl"
  echo "  rejected:        $OUTPUT_ROOT/rejected.jsonl"
fi
