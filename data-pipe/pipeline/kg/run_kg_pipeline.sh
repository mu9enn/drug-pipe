#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$PIPELINE_DIR/.." && pwd)"
LAUNCH_SCRIPT="$PIPELINE_DIR/claude_agent/launch_claude.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"

KG_TASK_FILE=""
N_CASES=""
PROVIDER="${PROVIDER:-manual}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
PARALLEL_ROLLOUTS="${PARALLEL_ROLLOUTS:-1}"
MAX_WORKERS="${MAX_WORKERS:-1}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/results/kg_sampled}"
SKIP_PROVIDER_SWITCH="${SKIP_PROVIDER_SWITCH:-1}"

usage() {
  cat <<USAGE
Usage:
  bash pipeline/kg/run_kg_pipeline.sh --kg-task-file <kg_sampled_tasks.jsonl> --n-cases <N> [options]

Options:
  --kg-task-file PATH          Input KGTaskSpec JSONL
  --n-cases N                  Number of tasks to run from the head of JSONL
  --provider NAME              Default: manual (set model via external cc-switch)
  --claude-bin BIN             Default: claude
  --num-rollouts N             Default: 1
  --parallel-rollouts N        Default: 1
  --max-workers N              Maximum concurrent Claude invocations. Default: 1
  --results-root PATH          Default: results/kg_sampled
  --skip-provider-switch 0|1   Default: 1
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kg-task-file) KG_TASK_FILE="${2:-}"; shift 2 ;;
    --n-cases) N_CASES="${2:-}"; shift 2 ;;
    --provider) PROVIDER="${2:-}"; shift 2 ;;
    --claude-bin) CLAUDE_BIN="${2:-}"; shift 2 ;;
    --num-rollouts) NUM_ROLLOUTS="${2:-}"; shift 2 ;;
    --parallel-rollouts) PARALLEL_ROLLOUTS="${2:-}"; shift 2 ;;
    --max-workers) MAX_WORKERS="${2:-}"; shift 2 ;;
    --results-root) RESULTS_ROOT="${2:-}"; shift 2 ;;
    --skip-provider-switch) SKIP_PROVIDER_SWITCH="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$KG_TASK_FILE" || -z "$N_CASES" ]]; then
  echo "[error] --kg-task-file and --n-cases are required" >&2
  usage >&2
  exit 1
fi

if ! [[ "$N_CASES" =~ ^[0-9]+$ && "$NUM_ROLLOUTS" =~ ^[0-9]+$ && "$PARALLEL_ROLLOUTS" =~ ^[0-9]+$ && "$MAX_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "[error] --n-cases/--num-rollouts/--parallel-rollouts/--max-workers must be non-negative integers" >&2
  exit 1
fi
if (( N_CASES <= 0 )); then
  echo "[error] --n-cases must be > 0" >&2
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

KG_TASK_FILE="$(realpath "$KG_TASK_FILE")"
RESULTS_ROOT="$(realpath -m "$RESULTS_ROOT")"

if [[ ! -f "$KG_TASK_FILE" ]]; then
  echo "[error] kg task file not found: $KG_TASK_FILE" >&2
  exit 1
fi
if [[ ! -x "$LAUNCH_SCRIPT" ]]; then
  echo "[error] launch script not executable: $LAUNCH_SCRIPT" >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$PIPELINE_DIR/kg/runs/$TS"
EXEC_CSV="$RUN_DIR/kg_tasks_exec.csv"
SELECTED_JSONL="$RUN_DIR/selected_tasks.jsonl"
PIPELINE_LOG="$RUN_DIR/pipeline.log"
MANIFEST_JSON="$RUN_DIR/manifest.json"
mkdir -p "$RUN_DIR"

"$PYTHON_BIN" - "$KG_TASK_FILE" "$N_CASES" "$EXEC_CSV" "$SELECTED_JSONL" <<'PY'
import csv
import json
import sys
from pathlib import Path

kg_task_file = Path(sys.argv[1]).resolve()
max_n = int(sys.argv[2])
out_csv = Path(sys.argv[3]).resolve()
out_selected = Path(sys.argv[4]).resolve()

rows = []
selected = []
with kg_task_file.open("r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        rows.append(obj)

for obj in rows:
    question = str(obj.get("question") or "").strip()
    task_id = str(obj.get("task_id") or "").strip()
    if not question or not task_id:
        continue
    selected.append(obj)
    if len(selected) >= max_n:
        break

if not selected:
    raise RuntimeError("No valid KG tasks selected from input jsonl.")

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["index", "question_id", "question", "answer", "raw_question_json"])
    writer.writeheader()
    for i, task in enumerate(selected, start=1):
        writer.writerow(
            {
                "index": i,
                "question_id": task["task_id"],
                "question": task["question"],
                "answer": "[]",
                "raw_question_json": json.dumps(task, ensure_ascii=False),
            }
        )

with out_selected.open("w", encoding="utf-8") as f:
    for task in selected:
        f.write(json.dumps(task, ensure_ascii=False) + "\n")

print(json.dumps({
    "input": str(kg_task_file),
    "requested_n_cases": max_n,
    "selected": len(selected),
    "exec_csv": str(out_csv),
    "selected_jsonl": str(out_selected),
}, ensure_ascii=False, indent=2))
PY

CMD=(
  bash "$LAUNCH_SCRIPT"
  --run-dataset
  --task kg
  --dataset-csv "$EXEC_CSV"
  --results-root "$RESULTS_ROOT"
  --provider "$PROVIDER"
  --claude-bin "$CLAUDE_BIN"
  --limit 0
  --num-rollouts "$NUM_ROLLOUTS"
  --parallel-rollouts "$PARALLEL_ROLLOUTS"
  --max-workers "$MAX_WORKERS"
)
if [[ "$SKIP_PROVIDER_SWITCH" == "1" ]]; then
  CMD+=(--skip-provider-switch)
fi

echo "[run] launching KG pipeline"
set +e
"${CMD[@]}" | tee "$PIPELINE_LOG"
RC=${PIPESTATUS[0]}
set -e
if [[ "$RC" -ne 0 ]]; then
  echo "[error] launch failed with code $RC" >&2
  exit "$RC"
fi

RESULTS_DIR="$(grep '^RESULTS_DIR=' "$PIPELINE_LOG" | tail -n 1 | cut -d= -f2-)"
if [[ -z "$RESULTS_DIR" ]]; then
  echo "[error] cannot find RESULTS_DIR in pipeline log" >&2
  exit 1
fi
RESULTS_DIR="$(realpath "$RESULTS_DIR")"

"$PYTHON_BIN" - "$SELECTED_JSONL" "$RESULTS_DIR" "$MANIFEST_JSON" "$KG_TASK_FILE" "$PIPELINE_LOG" "$PROVIDER" "$CLAUDE_BIN" "$NUM_ROLLOUTS" "$PARALLEL_ROLLOUTS" "$MAX_WORKERS" "$SKIP_PROVIDER_SWITCH" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

selected_path = Path(sys.argv[1]).resolve()
results_dir = Path(sys.argv[2]).resolve()
manifest_path = Path(sys.argv[3]).resolve()
kg_task_file = Path(sys.argv[4]).resolve()
pipeline_log = Path(sys.argv[5]).resolve()
provider = sys.argv[6]
claude_bin = sys.argv[7]
num_rollouts = int(sys.argv[8])
parallel_rollouts = int(sys.argv[9])
max_workers = int(sys.argv[10])
skip_provider_switch = int(sys.argv[11])

selected = []
with selected_path.open("r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            selected.append(obj)

summary_path = results_dir / "run_summary.jsonl"
summary_rows = []
if summary_path.is_file():
    with summary_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                summary_rows.append(obj)

by_dataset = {}
for row in summary_rows:
    idx = str(row.get("dataset_index") or "")
    if not idx:
        continue
    by_dataset.setdefault(idx, []).append(
        {
            "rollout_index": row.get("rollout_index"),
            "sample_dir": row.get("sample_dir"),
            "return_code": row.get("return_code"),
            "timed_out": row.get("timed_out"),
        }
    )

mappings = []
for i, task in enumerate(selected, start=1):
    tid = str(task.get("task_id") or f"task_{i:06d}")
    mappings.append(
        {
            "task_id": tid,
            "row_number": i,
            "dataset_index": tid,
            "source": task.get("source", {}),
            "run_entries": by_dataset.get(tid, []),
        }
    )

manifest = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "task": "kg",
    "kg_task_file": str(kg_task_file),
    "provider": provider,
    "claude_bin": claude_bin,
    "num_rollouts": num_rollouts,
    "parallel_rollouts": parallel_rollouts,
    "max_workers": max_workers,
    "skip_provider_switch": bool(skip_provider_switch),
    "selected_count": len(selected),
    "results_dir": str(results_dir),
    "pipeline_log": str(pipeline_log),
    "task_mappings": mappings,
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY

echo "[done] KG pipeline completed"
echo "  run_dir: $RUN_DIR"
echo "  results_dir: $RESULTS_DIR"
echo "  manifest: $MANIFEST_JSON"
