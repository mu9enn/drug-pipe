#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE_DIR="$REPO_DIR/pipeline"
PYTHON_BIN="${PYTHON_BIN:-python}"

RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RESULTS_ROOT/postprocess_candidates}"
ANSWER_HIT_ONLY=0
SPLIT_MULTI_TOOL_CALLS=0
SKIP_EXPORT=0
SKIP_SCAN=0
SKIP_SFT=0
SKIP_PRECHECK=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_postprocess.sh [options]

Options:
  --results-root PATH      Default: <repo>/results
  --output-root PATH       Default: <results-root>/postprocess_candidates
  --answer-hit-only        Only affects vs/ac/pf when building SFT/RL
  --split-multi-tool-calls  Split multiple tool calls from one assistant event
  --skip-export            Skip trajectory_exporter stage
  --skip-scan              Skip scan_molclaw_usage stage
  --skip-sft               Skip post_process_sft stage
  --skip-precheck          Skip the non-blocking pre-LLM semantic flag report
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-root) RESULTS_ROOT="${2:-}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --answer-hit-only) ANSWER_HIT_ONLY=1; shift ;;
    --split-multi-tool-calls) SPLIT_MULTI_TOOL_CALLS=1; shift ;;
    --skip-export) SKIP_EXPORT=1; shift ;;
    --skip-scan) SKIP_SCAN=1; shift ;;
    --skip-sft) SKIP_SFT=1; shift ;;
    --skip-precheck) SKIP_PRECHECK=1; shift ;;
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
    "$PYTHON_BIN" "$PIPELINE_DIR/postprocess/trajectory_exporter.py" "$run_dir" >/dev/null
  done
fi

if [[ "$SKIP_SCAN" -eq 0 ]]; then
  echo "[postprocess] stage2: scan_molclaw_usage (accepted candidates)"
  "$PYTHON_BIN" "$PIPELINE_DIR/postprocess/scan_molclaw_usage.py" \
    --results-root "$RESULTS_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --use-accepted-only
fi

if [[ "$SKIP_SFT" -eq 0 ]]; then
  echo "[postprocess] stage3: post_process_sft (script-1 ReAct/RL formatting and hard-clean)"
  cmd=(
    "$PYTHON_BIN" "$PIPELINE_DIR/postprocess/post_process_sft.py"
    --input-root "$OUTPUT_ROOT"
  )
  if [[ "$ANSWER_HIT_ONLY" -eq 1 ]]; then
    cmd+=(--answer-hit-only)
  fi
  if [[ "$SPLIT_MULTI_TOOL_CALLS" -eq 1 ]]; then
    cmd+=(--split-multi-tool-calls)
  fi
  "${cmd[@]}"
fi

if [[ "$SKIP_SFT" -eq 0 && "$SKIP_PRECHECK" -eq 0 ]]; then
  echo "[postprocess] stage4: pre-LLM semantic detector (flags only; never rejects or rewrites)"
  "$PYTHON_BIN" "$SCRIPT_DIR/validate_llm_cleaned.py" \
    --mode pre-llm \
    --input-dir "$OUTPUT_ROOT/sft_outputs/mcp_sft_all" \
    --output-json "$OUTPUT_ROOT/sft_outputs/pre_llm_semantic_report.json" \
    --output-md "$OUTPUT_ROOT/sft_outputs/pre_llm_semantic_report.md"
fi

echo "[done] postprocess pipeline finished"
echo "  results_root: $RESULTS_ROOT"
echo "  output_root:  $OUTPUT_ROOT"
if [[ "$SKIP_SFT" -eq 0 ]]; then
  echo "  sft_all:      $OUTPUT_ROOT/sft_outputs/mcp_sft_all/"
  echo "  sft_all_compat_jsonl: $OUTPUT_ROOT/sft_outputs/mcp_sft_all.jsonl"
  echo "  rl_all:       $OUTPUT_ROOT/sft_outputs/mcp_rl_prompts_all.jsonl"
  if [[ "$SKIP_PRECHECK" -eq 0 ]]; then
    echo "  pre_llm_report: $OUTPUT_ROOT/sft_outputs/pre_llm_semantic_report.json"
  fi
  if [[ "$SPLIT_MULTI_TOOL_CALLS" -eq 1 ]]; then
    echo "  split_tools:  enabled"
  fi
fi
