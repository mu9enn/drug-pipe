#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_DIR="$SCRIPT_DIR"
PIPELINE_DIR="$(cd "$E2E_DIR/.." && pwd)"
REPO_DIR="$(cd "$PIPELINE_DIR/.." && pwd)"
BUILD_SCRIPT="$E2E_DIR/scripts/build_e2e_dataset.py"
TEST_FLOW="$PIPELINE_DIR/claude_agent/test_flow_claude.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"

QUESTIONS=""
PROVIDER="${PROVIDER:-manual}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
LIMIT="${LIMIT:-0}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
PARALLEL_ROLLOUTS="${PARALLEL_ROLLOUTS:-1}"
MAX_WORKERS="${MAX_WORKERS:-1}"
SKIP_PROVIDER_SWITCH="${SKIP_PROVIDER_SWITCH:-1}"

usage() {
  cat <<EOF
Usage:
  bash pipeline/e2e/run_e2e_pipeline.sh [options]

Options:
  --questions CSV_IDS          Comma-separated ids (e.g. E2E-Q03,E2E-Q05)
  --provider NAME              Default: manual (set model via external cc-switch)
  --claude-bin BIN             Default: claude
  --limit N                    Default: 0 (no limit)
  --num-rollouts N             Default: 1
  --parallel-rollouts N        Default: 1
  --max-workers N              Maximum concurrent Claude invocations. Default: 1
  --skip-provider-switch 0|1   Default: 1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --questions) QUESTIONS="${2:-}"; shift 2 ;;
    --provider) PROVIDER="${2:-}"; shift 2 ;;
    --claude-bin) CLAUDE_BIN="${2:-}"; shift 2 ;;
    --limit) LIMIT="${2:-}"; shift 2 ;;
    --num-rollouts) NUM_ROLLOUTS="${2:-}"; shift 2 ;;
    --parallel-rollouts) PARALLEL_ROLLOUTS="${2:-}"; shift 2 ;;
    --max-workers) MAX_WORKERS="${2:-}"; shift 2 ;;
    --skip-provider-switch) SKIP_PROVIDER_SWITCH="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if ! [[ "$LIMIT" =~ ^[0-9]+$ && "$NUM_ROLLOUTS" =~ ^[0-9]+$ && "$PARALLEL_ROLLOUTS" =~ ^[0-9]+$ && "$MAX_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "[error] limit/num-rollouts/parallel-rollouts/max-workers must be non-negative integers" >&2
  exit 1
fi
if (( MAX_WORKERS <= 0 )); then
  echo "[error] --max-workers must be > 0" >&2
  exit 1
fi
if [[ "$SKIP_PROVIDER_SWITCH" != "0" && "$SKIP_PROVIDER_SWITCH" != "1" ]]; then
  echo "[error] --skip-provider-switch must be 0 or 1" >&2
  exit 1
fi
if [[ ! -f "$BUILD_SCRIPT" ]]; then
  echo "[error] build script not found: $BUILD_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$TEST_FLOW" ]]; then
  echo "[error] test_flow script not executable: $TEST_FLOW" >&2
  exit 1
fi

ts="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$E2E_DIR/runs/$ts"
DATASET_CSV="$RUN_DIR/e2e_dataset.csv"
DATASET_META="$RUN_DIR/dataset_manifest.json"
PIPELINE_LOG="$RUN_DIR/pipeline.log"
MANIFEST_JSON="$RUN_DIR/manifest.json"
mkdir -p "$RUN_DIR"

build_cmd=(
  "$PYTHON_BIN" "$BUILD_SCRIPT"
  --out-csv "$DATASET_CSV"
  --manifest-out "$DATASET_META"
)
if [[ -n "$QUESTIONS" ]]; then
  build_cmd+=(--questions "$QUESTIONS")
fi

echo "[run] building E2E dataset..."
"${build_cmd[@]}"

echo "[run] running pipeline full flow (task=e2e)..."
set +e
bash "$TEST_FLOW" \
  "$PROVIDER" \
  "$CLAUDE_BIN" \
  "$LIMIT" \
  "$NUM_ROLLOUTS" \
  "$PARALLEL_ROLLOUTS" \
  "e2e" \
  "$DATASET_CSV" \
  "$SKIP_PROVIDER_SWITCH" \
  "$MAX_WORKERS" | tee "$PIPELINE_LOG"
rc=${PIPESTATUS[0]}
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "[error] pipeline failed with code $rc" >&2
  exit "$rc"
fi

results_dir="$(grep '^RESULTS_DIR=' "$PIPELINE_LOG" | tail -n 1 | cut -d= -f2-)"
if [[ -z "$results_dir" ]]; then
  echo "[error] cannot find RESULTS_DIR in $PIPELINE_LOG" >&2
  exit 1
fi

"$PYTHON_BIN" - "$DATASET_META" "$MANIFEST_JSON" "$results_dir" "$PIPELINE_LOG" "$PROVIDER" "$CLAUDE_BIN" "$LIMIT" "$NUM_ROLLOUTS" "$PARALLEL_ROLLOUTS" "$MAX_WORKERS" "$SKIP_PROVIDER_SWITCH" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

dataset_meta = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
results_dir = Path(sys.argv[3]).resolve()
pipeline_log = Path(sys.argv[4]).resolve()
provider = sys.argv[5]
claude_bin = sys.argv[6]
limit = int(sys.argv[7])
num_rollouts = int(sys.argv[8])
parallel_rollouts = int(sys.argv[9])
max_workers = int(sys.argv[10])
skip_provider_switch = int(sys.argv[11])

meta = json.loads(dataset_meta.read_text(encoding="utf-8"))
manifest = {
    "generated_at": datetime.now().isoformat(),
    "task": "e2e",
    "provider": provider,
    "claude_bin": claude_bin,
    "limit": limit,
    "num_rollouts": num_rollouts,
    "parallel_rollouts": parallel_rollouts,
    "max_workers": max_workers,
    "skip_provider_switch": bool(skip_provider_switch),
    "dataset_csv": meta.get("out_csv"),
    "selected_question_ids": meta.get("selected_question_ids", []),
    "results_dir": str(results_dir),
    "pipeline_log": str(pipeline_log),
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY

echo "[done] E2E pipeline completed"
echo "  run_dir: $RUN_DIR"
echo "  results_dir: $results_dir"
echo "  manifest: $MANIFEST_JSON"
