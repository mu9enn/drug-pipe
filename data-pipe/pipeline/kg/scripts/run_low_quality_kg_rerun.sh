#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_PIPE_ROOT="$(cd "$KG_DIR/../.." && pwd)"

if [[ -f "$DATA_PIPE_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$DATA_PIPE_ROOT/.env"
  set +a
fi

OLD_TASKS="$KG_DIR/data/legacy_graph_run_20260601_123052/kg_sampled_tasks.jsonl"
NEW_TASKS="$KG_DIR/data/legacy_graph_20260601_resample_20260728_131422/kg_sampled_tasks.jsonl"
SELECTION="$KG_DIR/low_quality_rerun_selection_20260729.jsonl"
CLAUDE_BIN_ARG="${CLAUDE_BIN:-}"
MAX_WORKERS=2
NUM_ROLLOUTS=1
RESULTS_ROOT=""
PREPARE_ONLY=0
MCP_TOOL_TIMEOUT_MS="${MOLCLAW_MCP_TOOL_TIMEOUT_MS:-14400000}"

usage() {
  cat <<'EOF'
Usage:
  bash pipeline/kg/scripts/run_low_quality_kg_rerun.sh [options]

Options:
  --selection PATH       JSONL containing the task_id values to rerun
  --old-tasks PATH       Original Tool-KG sampled tasks JSONL
  --new-tasks PATH       New Tool-KG sampled tasks JSONL
  --claude-bin PATH      Claude executable (default: $CLAUDE_BIN or PATH lookup)
  --max-workers N        Concurrent top-level Claude tasks (default: 2)
  --num-rollouts N       Rollouts per selected task (default: 1)
  --mcp-tool-timeout-ms N
                         MolClaw per-tool hard timeout in milliseconds (default: 14400000)
  --results-root PATH    Result root (default: data-pipe/results/kg_low_quality_rerun_<timestamp>)
  --prepare-only         Validate and create the selected task file without Claude/MCP calls
  -h, --help             Show this help

Claude execution policy is applied by the shared runtimes:
  CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY=2
  CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1
The MolClaw server uses a per-server tool timeout; no global MCP_TOOL_TIMEOUT is set.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --selection) SELECTION="$2"; shift 2 ;;
    --old-tasks) OLD_TASKS="$2"; shift 2 ;;
    --new-tasks) NEW_TASKS="$2"; shift 2 ;;
    --claude-bin) CLAUDE_BIN_ARG="$2"; shift 2 ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --num-rollouts) NUM_ROLLOUTS="$2"; shift 2 ;;
    --mcp-tool-timeout-ms) MCP_TOOL_TIMEOUT_MS="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in "$MAX_WORKERS" "$NUM_ROLLOUTS"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[error] worker and rollout values must be positive integers: $value" >&2
    exit 2
  fi
done
if [[ ! "$MCP_TOOL_TIMEOUT_MS" =~ ^[1-9][0-9]*$ ]] || (( MCP_TOOL_TIMEOUT_MS < 1000 )); then
  echo "[error] --mcp-tool-timeout-ms must be an integer >= 1000: $MCP_TOOL_TIMEOUT_MS" >&2
  exit 2
fi

for path in "$OLD_TASKS" "$NEW_TASKS" "$SELECTION"; do
  if [[ ! -f "$path" ]]; then
    echo "[error] required input not found: $path" >&2
    exit 1
  fi
done

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
PREP_DIR="$KG_DIR/runs/low_quality_rerun_${RUN_TAG}"
SELECTED_TASKS="$PREP_DIR/selected_tasks.jsonl"
SELECTION_MANIFEST="$PREP_DIR/selection_manifest.json"
mkdir -p "$PREP_DIR"

python - "$OLD_TASKS" "$NEW_TASKS" "$SELECTION" "$SELECTED_TASKS" "$SELECTION_MANIFEST" "$MCP_TOOL_TIMEOUT_MS" <<'PY'
import json
import sys
from pathlib import Path

old_path, new_path, selection_path, output_path, manifest_path = map(Path, sys.argv[1:6])
mcp_tool_timeout_ms = int(sys.argv[6])

def read_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records

selection = read_jsonl(selection_path)
selected_ids = [str(item.get("task_id") or "").strip() for item in selection]
if any(not task_id for task_id in selected_ids):
    raise ValueError("every selection record must contain a non-empty task_id")
if len(selected_ids) != len(set(selected_ids)):
    raise ValueError("selection contains duplicate task_id values")

task_by_id = {}
source_by_id = {}
for source, path in (("original", old_path), ("new_sampling", new_path)):
    for task in read_jsonl(path):
        task_id = str(task.get("task_id") or "").strip()
        if not task_id:
            raise ValueError(f"task in {path} is missing task_id")
        if task_id in task_by_id:
            raise ValueError(f"duplicate task_id across task inputs: {task_id}")
        task_by_id[task_id] = task
        source_by_id[task_id] = source

missing = [task_id for task_id in selected_ids if task_id not in task_by_id]
if missing:
    raise ValueError(f"selected task IDs are absent from task inputs: {missing}")

selected = [task_by_id[task_id] for task_id in selected_ids]
output_path.write_text(
    "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in selected),
    encoding="utf-8",
)
manifest = {
    "schema_version": "kg_quality_rerun_selection_v1",
    "selection_file": str(selection_path.resolve()),
    "selected_tasks_file": str(output_path.resolve()),
    "selected_count": len(selected),
    "selected_task_ids": selected_ids,
    "sources": {task_id: source_by_id[task_id] for task_id in selected_ids},
    "execution_policy": {
        "claude_code_max_tool_use_concurrency": 2,
        "claude_code_disable_background_tasks": True,
        "molclaw_mcp_tool_timeout_ms": mcp_tool_timeout_ms,
    },
}
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY

SELECTED_COUNT="$(python - "$SELECTION_MANIFEST" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_count"])
PY
)"

echo "[prepared] selected_tasks=$SELECTED_TASKS"
echo "[prepared] selection_manifest=$SELECTION_MANIFEST"
if [[ "$PREPARE_ONLY" -eq 1 ]]; then
  echo "[done] prepare-only mode; no Claude or MCP calls were made"
  exit 0
fi

if [[ -z "$CLAUDE_BIN_ARG" ]]; then
  CLAUDE_BIN_ARG="${CLAUDE_BIN:-$(command -v claude || true)}"
fi
if [[ -z "$CLAUDE_BIN_ARG" || ! -x "$CLAUDE_BIN_ARG" ]]; then
  echo "[error] Claude executable is missing or not executable: ${CLAUDE_BIN_ARG:-<empty>}" >&2
  echo "        pass --claude-bin /absolute/path/to/claude" >&2
  exit 1
fi
if [[ -z "${MOLCLAW_SCP_MCP_URL:-}" || -z "${MOLCLAW_SCP_MCP_AUTH:-}" ]]; then
  echo "[error] data-pipe/.env must provide MOLCLAW_SCP_MCP_URL and MOLCLAW_SCP_MCP_AUTH" >&2
  exit 1
fi
if [[ -z "$RESULTS_ROOT" ]]; then
  RESULTS_ROOT="$DATA_PIPE_ROOT/results/kg_low_quality_rerun_${RUN_TAG}"
fi

echo "[run] selected=$SELECTED_COUNT max_workers=$MAX_WORKERS num_rollouts=$NUM_ROLLOUTS mcp_tool_timeout_ms=$MCP_TOOL_TIMEOUT_MS"
echo "[run] results_root=$RESULTS_ROOT"
export MOLCLAW_MCP_TOOL_TIMEOUT_MS="$MCP_TOOL_TIMEOUT_MS"
PROVIDER="${CC_SWITCH_PROVIDER:-manual}" \
bash "$KG_DIR/run_kg_pipeline.sh" \
  --kg-task-file "$SELECTED_TASKS" \
  --n-cases "$SELECTED_COUNT" \
  --claude-bin "$CLAUDE_BIN_ARG" \
  --num-rollouts "$NUM_ROLLOUTS" \
  --parallel-rollouts 1 \
  --max-workers "$MAX_WORKERS" \
  --results-root "$RESULTS_ROOT" \
  --skip-provider-switch 1

echo "[done] preparation_dir=$PREP_DIR"
echo "[done] results_root=$RESULTS_ROOT"
