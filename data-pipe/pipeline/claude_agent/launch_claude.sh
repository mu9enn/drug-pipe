#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PIPELINE_DIR/.." && pwd)"
cd "$REPO_DIR"

# Auto-load shared env file from the merged workspace root if present.
ROOT_ENV_FILE="${ROOT_ENV_FILE:-$REPO_DIR/.env}"
if [[ -f "$ROOT_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT_ENV_FILE"
  set +a
fi

# Shared options
TASK="${TASK:-vs}"
SKILLS_ROOT="${SKILLS_ROOT:-}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-}"
PROVIDER="${CC_SWITCH_PROVIDER:-manual}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_PROVIDER_SWITCH=0
SKIP_MCP_VERIFY=0

# Every task uses the same canonical MCP server.
MCP_SERVER_NAME="molclaw-scp"
MCP_SERVER_URL="${MOLCLAW_SCP_MCP_URL:-}"
MCP_SERVER_AUTH="${MOLCLAW_SCP_MCP_AUTH:-}"
MCP_SERVER_AUTH_HEADER="${MOLCLAW_SCP_MCP_AUTH_HEADER:-SCP-HUB-API-KEY}"
MCP_SERVER_SCOPE="${MCP_SERVER_SCOPE:-project}"

# Single-sample mode options
WORKDIR=""
PROMPT_FILE=""
PROMPT_TEXT=""

# Dataset mode options
RUN_DATASET=0
DATASET_CSV=""
RESULTS_ROOT="results"
START_ROW=1
END_ROW=0
LIMIT=0
NUM_ROLLOUTS=1
ROLLOUT_SEED_BASE=0
PARALLEL_ROLLOUTS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"; shift 2 ;;
    --run-dataset)
      RUN_DATASET=1; shift ;;
    --workdir)
      WORKDIR="$2"; shift 2 ;;
    --prompt-file)
      PROMPT_FILE="$2"; shift 2 ;;
    --prompt)
      PROMPT_TEXT="$2"; shift 2 ;;
    --skills-root)
      SKILLS_ROOT="$2"; shift 2 ;;
    --system-prompt-file)
      SYSTEM_PROMPT_FILE="$2"; shift 2 ;;
    --dataset-csv)
      DATASET_CSV="$2"; shift 2 ;;
    --results-root)
      RESULTS_ROOT="$2"; shift 2 ;;
    --provider)
      PROVIDER="$2"; shift 2 ;;
    --claude-bin)
      CLAUDE_BIN="$2"; shift 2 ;;
    --start-row)
      START_ROW="$2"; shift 2 ;;
    --end-row)
      END_ROW="$2"; shift 2 ;;
    --limit)
      LIMIT="$2"; shift 2 ;;
    --num-rollouts)
      NUM_ROLLOUTS="$2"; shift 2 ;;
    --rollout-seed-base)
      ROLLOUT_SEED_BASE="$2"; shift 2 ;;
    --parallel-rollouts)
      PARALLEL_ROLLOUTS="$2"; shift 2 ;;
    --skip-provider-switch)
      SKIP_PROVIDER_SWITCH=1; shift ;;
    --skip-mcp-verify)
      SKIP_MCP_VERIFY=1; shift ;;
    -h|--help)
      cat <<EOF
Usage: bash claude_agent/launch_claude.sh [options]

Modes:
  1) Single-sample mode (default):
     --workdir PATH
     --prompt-file PATH | --prompt TEXT

  2) Dataset mode:
     --run-dataset
     [--task vs|ac|pf|e2e|kg]
     [--dataset-csv PATH]
     [--results-root PATH]
     [--start-row N] [--end-row N] [--limit N]
     [--num-rollouts N] [--parallel-rollouts N] [--rollout-seed-base N]

Shared options:
  --task TASK
  --skills-root PATH
  --system-prompt-file NAME
  --provider ID                 (default: manual; set model via external cc-switch)
  --claude-bin PATH_OR_NAME     (default: claude)
  --skip-provider-switch
  --skip-mcp-verify
EOF
      exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1 ;;
  esac
done

TASK="$(echo "$TASK" | tr '[:upper:]' '[:lower:]')"
if [[ "$TASK" != "vs" && "$TASK" != "ac" && "$TASK" != "pf" && "$TASK" != "e2e" && "$TASK" != "kg" ]]; then
  echo "[error] unsupported --task: $TASK" >&2
  exit 1
fi

# Task-aware defaults
if [[ "$TASK" == "vs" ]]; then
  : "${SKILLS_ROOT:=skills/skills_vs}"
  : "${SYSTEM_PROMPT_FILE:=system_prompt_result.md}"
  : "${DATASET_CSV:=$REPO_DIR/molbench/molbench-vs-900.csv}"
elif [[ "$TASK" == "e2e" ]]; then
  : "${SKILLS_ROOT:=skills/skills_full}"
  : "${SYSTEM_PROMPT_FILE:=system_prompt_FULL.md}"
  : "${DATASET_CSV:=$REPO_DIR/molbench/MolBench-E2E/e2e_dataset.csv}"
elif [[ "$TASK" == "kg" ]]; then
  : "${SKILLS_ROOT:=skills/skills_full}"
  : "${SYSTEM_PROMPT_FILE:=system_prompt_FULL.md}"
else
  : "${SKILLS_ROOT:=skills/skills_full}"
  : "${SYSTEM_PROMPT_FILE:=system_prompt_FULL.md}"
  : "${DATASET_CSV:=$REPO_DIR/molbench/molbench-${TASK}-900.csv}"
fi

if [[ "$TASK" == "kg" && -z "${DATASET_CSV:-}" ]]; then
  echo "[error] task=kg requires explicit --dataset-csv path" >&2
  exit 1
fi

if [[ -z "${MCP_SERVER_URL:-}" ]]; then
  cat >&2 <<'EOF'
[error] MCP server URL is empty.
Please configure the canonical MolClaw MCP endpoint:
  export MOLCLAW_SCP_MCP_URL='http://.../mcp'
  export MOLCLAW_SCP_MCP_AUTH='...'
EOF
  exit 1
fi
if [[ -z "${MCP_SERVER_AUTH:-}" ]]; then
  echo "[error] MCP auth token is empty. Please set MOLCLAW_SCP_MCP_AUTH." >&2
  exit 1
fi

# if [[ "$SKIP_PROVIDER_SWITCH" -eq 0 ]]; then
#   if ! command -v cc-switch >/dev/null 2>&1; then
#     echo "[error] cc-switch not found in PATH, but is required for provider switch. Either install cc-switch or use --skip-provider-switch flag." >&2
#     exit 1
#   fi
# fi

MCP_CONFIG_FILE="$(mktemp -t "claude_mcp_${TASK}.XXXXXX.json")"
cleanup_mcp_config() {
  if [[ -n "${MCP_CONFIG_FILE:-}" && -f "$MCP_CONFIG_FILE" ]]; then
    rm -f "$MCP_CONFIG_FILE"
  fi
}
trap cleanup_mcp_config EXIT

write_task_mcp_config() {
  "$PYTHON_BIN" - \
    "$MCP_CONFIG_FILE" \
    "$MCP_SERVER_NAME" "$MCP_SERVER_URL" "$MCP_SERVER_AUTH_HEADER" "$MCP_SERVER_AUTH" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
name, url, header, token = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

if not name or not url:
    raise SystemExit("no valid MCP servers to write")

server = {"type": "http", "url": url}
if header and token:
    server["headers"] = {header: token}
cfg = {"mcpServers": {name: server}}

out_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

if [[ "$SKIP_PROVIDER_SWITCH" -eq 0 ]]; then
  # cc-switch provider switch "$PROVIDER"
  # echo "[run] provider switched via cc-switch: ${PROVIDER}"
  echo "[run] provider switch step disabled in script (expect external cc-switch before run)"
fi

write_task_mcp_config
if [[ "$SKIP_MCP_VERIFY" -eq 0 ]]; then
  echo "[run] using strict MCP config: ${MCP_CONFIG_FILE} (server: ${MCP_SERVER_NAME})"
fi

if [[ ! -d "$SKILLS_ROOT" ]]; then
  echo "[error] skills root not found: $SKILLS_ROOT" >&2
  exit 1
fi
echo "[route] task=${TASK} skills_root=${SKILLS_ROOT} system_prompt=${SYSTEM_PROMPT_FILE} mcp_server=${MCP_SERVER_NAME} mcp_scope=${MCP_SERVER_SCOPE}"

if [[ "$RUN_DATASET" -eq 1 ]]; then
  RUNNER="$PIPELINE_DIR/claude_agent/run_claude.py"
  if [[ ! -f "$RUNNER" ]]; then
    echo "[error] run_claude.py not found: $RUNNER" >&2
    exit 1
  fi

  cmd=(
    "$PYTHON_BIN" "$RUNNER"
    --task "$TASK"
    --dataset-csv "$DATASET_CSV"
    --skills-root "$SKILLS_ROOT"
    --system-prompt-file "$SYSTEM_PROMPT_FILE"
    --results-root "$RESULTS_ROOT"
    --provider "$PROVIDER"
    --claude-bin "$CLAUDE_BIN"
    --start-row "$START_ROW"
    --end-row "$END_ROW"
    --limit "$LIMIT"
    --num-rollouts "$NUM_ROLLOUTS"
    --rollout-seed-base "$ROLLOUT_SEED_BASE"
    --parallel-rollouts "$PARALLEL_ROLLOUTS"
    --mcp-config-file "$MCP_CONFIG_FILE"
    --strict-mcp-config
    --skip-provider-switch
  )
  "${cmd[@]}"
  exit $?
fi

if [[ -z "$WORKDIR" ]]; then
  echo "[error] --workdir is required in single-sample mode" >&2
  exit 1
fi
if [[ -z "$PROMPT_FILE" && -z "$PROMPT_TEXT" ]]; then
  echo "[error] --prompt-file or --prompt is required in single-sample mode" >&2
  exit 1
fi
if [[ -n "$PROMPT_FILE" && -n "$PROMPT_TEXT" ]]; then
  echo "[error] only one of --prompt-file / --prompt can be provided" >&2
  exit 1
fi

mkdir -p "$WORKDIR"
cp -a "$SKILLS_ROOT"/. "$WORKDIR"/

if [[ -n "$PROMPT_FILE" ]]; then
  if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "[error] prompt file not found: $PROMPT_FILE" >&2
    exit 1
  fi
  PROMPT_TEXT="$(cat "$PROMPT_FILE")"
fi

printf '%s\n' "$PROMPT_TEXT" > "$WORKDIR/prompt.txt"

set +e
(
  cd "$WORKDIR" || exit 1
  "$CLAUDE_BIN" \
    --dangerously-skip-permissions \
    --verbose \
    --output-format stream-json \
    --mcp-config "$MCP_CONFIG_FILE" \
    --strict-mcp-config \
    -p "$PROMPT_TEXT"
) > "$WORKDIR/complete_session.jsonl" 2>&1
RC=$?
set -e

"$PYTHON_BIN" - "$WORKDIR" "$PROVIDER" "$CLAUDE_BIN" "$RC" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

workdir = Path(sys.argv[1]).resolve()
provider = sys.argv[2]
claude_bin = sys.argv[3]
rc = int(sys.argv[4])
meta = {
    "timestamp": datetime.now().isoformat(),
    "provider": provider,
    "claude_bin": claude_bin,
    "workdir": str(workdir),
    "return_code": rc,
    "timed_out": False,
    "timeout_sec": None,
}
(workdir / "run_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY

echo "WORKDIR=$WORKDIR"
exit "$RC"
