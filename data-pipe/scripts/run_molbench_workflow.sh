#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GET_DIR="$ROOT_DIR/get-molbench"
PIPELINE_DIR="$ROOT_DIR/pipeline"
ENV_FILE="$ROOT_DIR/.env"

SEED=""
N_CASES=""

usage() {
  cat <<USAGE
Usage:
  bash scripts/run_molbench_workflow.sh --seed 609 --n-cases 1

Description:
  1) Generate AC/VS/PF datasets under get-molbench/outputs/auto/{ac,vs,pf}
  2) Merge PF v0/v1 to molbench-pf-<N>-<SEED>.csv
  3) Send three pipeline jobs via tmux send-keys:
     - pipe-vs-1:0 (task=vs)
     - pipe-ac-2:0 (task=ac)
     - pipe-pf-3:0 (task=pf)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="$2"
      shift 2
      ;;
    --n-cases)
      N_CASES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SEED" || -z "$N_CASES" ]]; then
  echo "[error] --seed and --n-cases are required" >&2
  usage >&2
  exit 1
fi

if ! [[ "$SEED" =~ ^[0-9]+$ && "$N_CASES" =~ ^[0-9]+$ ]]; then
  echo "[error] --seed and --n-cases must be non-negative integers" >&2
  exit 1
fi

if (( N_CASES <= 0 )); then
  echo "[error] --n-cases must be > 0" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "[error] tmux not found in PATH" >&2
  exit 1
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

MOLCLAW_SCP_MCP_URL="${MOLCLAW_SCP_MCP_URL:-}"
MOLCLAW_SCP_MCP_AUTH_HEADER="${MOLCLAW_SCP_MCP_AUTH_HEADER:-SCP-HUB-API-KEY}"
MOLCLAW_SCP_MCP_AUTH="${MOLCLAW_SCP_MCP_AUTH:-}"

for required_var in MOLCLAW_SCP_MCP_URL MOLCLAW_SCP_MCP_AUTH; do
  if [[ -z "${!required_var}" ]]; then
    echo "[error] $required_var is required; configure it once in $ENV_FILE" >&2
    exit 1
  fi
done

PYTHON_BIN="${PYTHON_BIN:-python}"
PROVIDER="${PROVIDER:-${CC_SWITCH_PROVIDER:-manual}}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"

AC_OUT_DIR="$GET_DIR/outputs/auto/ac"
VS_OUT_DIR="$GET_DIR/outputs/auto/vs"
PF_OUT_DIR="$GET_DIR/outputs/auto/pf"
mkdir -p "$AC_OUT_DIR" "$VS_OUT_DIR" "$PF_OUT_DIR"

AC_NAME="molbench-ac-${N_CASES}-${SEED}.csv"
VS_NAME="molbench-vs-${N_CASES}-${SEED}.csv"

PF_V0_CASES=$(( N_CASES / 2 ))
PF_V1_CASES=$(( N_CASES - PF_V0_CASES ))
PF_V0_SEED="$SEED"
PF_V1_SEED=$(( SEED + 1 ))

PF_V0_NAME="molbench-pf-v0-${PF_V0_CASES}-${PF_V0_SEED}.csv"
PF_V1_NAME="molbench-pf-v1-${PF_V1_CASES}-${PF_V1_SEED}.csv"
PF_MERGED_NAME="molbench-pf-${N_CASES}-${SEED}.csv"

if (( PF_V0_CASES <= 0 || PF_V1_CASES <= 0 )); then
  echo "[error] PF split requires n-cases >= 2 (current: ${N_CASES})" >&2
  exit 1
fi

echo "[run] Generate AC"
"$PYTHON_BIN" "$GET_DIR/pipelines/generate_molbench_ac.py" \
  --n-cases "$N_CASES" \
  --seed "$SEED" \
  --out-dir "outputs/auto/ac" \
  --out-name "$AC_NAME"

echo "[run] Generate VS"
"$PYTHON_BIN" "$GET_DIR/pipelines/generate_molbench_vs.py" \
  --n-cases "$N_CASES" \
  --seed "$SEED" \
  --out-dir "outputs/auto/vs" \
  --out-name "$VS_NAME" \
  --no-remote-target-name

echo "[run] Generate PF v0"
"$PYTHON_BIN" "$GET_DIR/pipelines/generate_molbench_pf.py" \
  --variant v0 \
  --n-cases "$PF_V0_CASES" \
  --seed "$PF_V0_SEED" \
  --out-dir "outputs/auto/pf" \
  --out-name "$PF_V0_NAME"

echo "[run] Generate PF v1"
"$PYTHON_BIN" "$GET_DIR/pipelines/generate_molbench_pf.py" \
  --variant v1 \
  --n-cases "$PF_V1_CASES" \
  --seed "$PF_V1_SEED" \
  --out-dir "outputs/auto/pf" \
  --out-name "$PF_V1_NAME"

echo "[run] Merge PF"
"$PYTHON_BIN" "$GET_DIR/scripts/merge_molbench_pf.py" \
  --v0-csv "$PF_OUT_DIR/$PF_V0_NAME" \
  --v1-csv "$PF_OUT_DIR/$PF_V1_NAME" \
  --out "$PF_OUT_DIR/$PF_MERGED_NAME"

AC_CSV="$(realpath "$AC_OUT_DIR/$AC_NAME")"
VS_CSV="$(realpath "$VS_OUT_DIR/$VS_NAME")"
PF_CSV="$(realpath "$PF_OUT_DIR/$PF_MERGED_NAME")"

ensure_tmux_target() {
  local target="$1"
  if ! tmux list-panes -t "$target" >/dev/null 2>&1; then
    echo "[error] tmux target not found: $target" >&2
    exit 1
  fi
}

ensure_tmux_target "pipe-vs-1:0"
ensure_tmux_target "pipe-ac-2:0"
ensure_tmux_target "pipe-pf-3:0"

if [[ -f "$ENV_FILE" ]]; then
  printf -v ENV_BOOTSTRAP 'set -a; source "%s"; set +a; export PYTHON_BIN=%q; ' "$ENV_FILE" "$PYTHON_BIN"
else
  printf -v ENV_BOOTSTRAP \
    'export MOLCLAW_SCP_MCP_URL=%q MOLCLAW_SCP_MCP_AUTH_HEADER=%q MOLCLAW_SCP_MCP_AUTH=%q PYTHON_BIN=%q; ' \
    "$MOLCLAW_SCP_MCP_URL" "$MOLCLAW_SCP_MCP_AUTH_HEADER" "$MOLCLAW_SCP_MCP_AUTH" "$PYTHON_BIN"
fi
VS_CMD="$ENV_BOOTSTRAP bash $PIPELINE_DIR/claude_agent/test_flow_claude.sh $PROVIDER $CLAUDE_BIN 0 1 1 vs $VS_CSV 1"
AC_CMD="$ENV_BOOTSTRAP bash $PIPELINE_DIR/claude_agent/test_flow_claude.sh $PROVIDER $CLAUDE_BIN 0 1 1 ac $AC_CSV 1"
PF_CMD="$ENV_BOOTSTRAP bash $PIPELINE_DIR/claude_agent/test_flow_claude.sh $PROVIDER $CLAUDE_BIN 0 1 1 pf $PF_CSV 1"

tmux send-keys -t pipe-vs-1:0 "$VS_CMD" C-m
tmux send-keys -t pipe-ac-2:0 "$AC_CMD" C-m
tmux send-keys -t pipe-pf-3:0 "$PF_CMD" C-m

echo "[done] tmux commands sent"
echo "  pipe-vs-1:0 -> $VS_CMD"
echo "  pipe-ac-2:0 -> $AC_CMD"
echo "  pipe-pf-3:0 -> $PF_CMD"
