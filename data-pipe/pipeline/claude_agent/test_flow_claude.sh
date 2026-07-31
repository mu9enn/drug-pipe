#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PIPELINE_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_DIR/.." && pwd)"
cd "$REPO_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash claude_agent/test_flow_claude.sh [provider] [claude_bin] [limit] [num_rollouts] [parallel_rollouts] [task] [dataset_csv] [skip_provider_switch] [max_workers]

Defaults:
  provider=manual
  claude_bin=claude
  limit=0
  num_rollouts=1
  parallel_rollouts=1
  task=vs
  dataset_csv=<repo>/molbench/molbench-<task>-900.csv (for e2e: <repo>/molbench/MolBench-E2E/e2e_dataset.csv; for kg: must be explicit)
  skip_provider_switch=0
  max_workers=0 (compatibility: use parallel_rollouts)
EOF
  exit 0
fi

PROVIDER="${1:-manual}"
CLAUDE_BIN="${2:-claude}"
LIMIT="${3:-0}"
NUM_ROLLOUTS="${4:-1}"
PARALLEL_ROLLOUTS="${5:-1}"
TASK="${6:-vs}"
DATASET_CSV="${7:-}"
SKIP_PROVIDER_SWITCH="${8:-0}"
MAX_WORKERS="${9:-0}"

TASK="$(echo "$TASK" | tr '[:upper:]' '[:lower:]')"
if [[ "$TASK" != "vs" && "$TASK" != "ac" && "$TASK" != "pf" && "$TASK" != "e2e" && "$TASK" != "kg" ]]; then
  echo "[error] unsupported task: $TASK" >&2
  exit 1
fi

if [[ -z "$DATASET_CSV" ]]; then
  if [[ "$TASK" == "e2e" ]]; then
    DATASET_CSV="$REPO_DIR/molbench/MolBench-E2E/e2e_dataset.csv"
  elif [[ "$TASK" == "kg" ]]; then
    echo "[error] task=kg requires explicit dataset_csv argument" >&2
    exit 1
  else
    DATASET_CSV="$REPO_DIR/molbench/molbench-${TASK}-900.csv"
  fi
fi

SKILLS_ROOT="$PROJECT_ROOT/molclaw-skills"
SYSTEM_PROMPT_FILE="$SKILLS_ROOT/system_prompt_FULL.md"

RESULTS_ROOT="$REPO_DIR/results"
LAUNCH_SCRIPT="$PIPELINE_DIR/claude_agent/launch_claude.sh"
EVAL_SCRIPT="$PIPELINE_DIR/evaluate/run_eval_bench.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "$DATASET_CSV" ]]; then
  echo "[error] dataset csv not found: $DATASET_CSV" >&2
  exit 1
fi
if [[ ! -f "$SYSTEM_PROMPT_FILE" ]]; then
  echo "[error] system prompt file not found: $SYSTEM_PROMPT_FILE" >&2
  exit 1
fi
if [[ ! -x "$LAUNCH_SCRIPT" ]]; then
  echo "[error] launch script not executable: $LAUNCH_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "[error] eval script not found: $EVAL_SCRIPT" >&2
  exit 1
fi

echo "[route] task=$TASK dataset_csv=$DATASET_CSV"
echo "[route] skills_root=$SKILLS_ROOT"
echo "[route] system_prompt=$SYSTEM_PROMPT_FILE"

TMP_LOG="$(mktemp -t test_flow_claude_log.XXXXXX)"

CMD=(
  bash "$LAUNCH_SCRIPT"
  --run-dataset
  --task "$TASK"
  --dataset-csv "$DATASET_CSV"
  --skills-root "$SKILLS_ROOT"
  --results-root "$RESULTS_ROOT"
  --provider "$PROVIDER"
  --claude-bin "$CLAUDE_BIN"
  --limit "$LIMIT"
  --num-rollouts "$NUM_ROLLOUTS"
  --parallel-rollouts "$PARALLEL_ROLLOUTS"
  --max-workers "$MAX_WORKERS"
)
if [[ "$SKIP_PROVIDER_SWITCH" == "1" ]]; then
  CMD+=(--skip-provider-switch)
fi

set +e
"${CMD[@]}" | tee "$TMP_LOG"
RC=${PIPESTATUS[0]}
set -e

if [[ "$RC" -ne 0 ]]; then
  echo "[error] dataset run failed with code $RC" >&2
  rm -f "$TMP_LOG"
  exit "$RC"
fi

RESULTS_DIR="$(grep '^RESULTS_DIR=' "$TMP_LOG" | tail -n 1 | cut -d= -f2-)"
rm -f "$TMP_LOG"

if [[ -z "$RESULTS_DIR" ]]; then
  echo "[error] test flow did not output RESULTS_DIR=..." >&2
  exit 1
fi

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "[error] RESULTS_DIR does not exist: $RESULTS_DIR" >&2
  exit 1
fi

if [[ "$TASK" == "e2e" || "$TASK" == "kg" ]]; then
  echo "[skip] evaluation skipped for $TASK task"
else
  "$PYTHON_BIN" "$EVAL_SCRIPT" "$RESULTS_DIR" --task "$TASK"
fi

echo "[done] full pipeline completed (task=$TASK)"
