#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DRUG_PIPE_DIR="$(cd "$DATA_PIPE_DIR/.." && pwd)"
SLIME_DIR="$DRUG_PIPE_DIR/slime-wd/slime"
PYTHON_BIN="${PYTHON_BIN:-python}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
AGENT_HARNESS="${AGENT_HARNESS:-claude}"
DSH_BIN="${DSH_BIN:-dsh}"
DSH_NODE_BIN="${DSH_NODE_BIN:-node}"
DSH_MODEL="${DSH_MODEL:-deepseek-v4-flash}"
DSH_PROVIDER="${DSH_PROVIDER:-${CC_SWITCH_PROVIDER:-dsv4flash}}"
RESULTS_ROOT="${RESULTS_ROOT:-$DATA_PIPE_DIR/results}"
PRECLEAN_ROOT="${PRECLEAN_ROOT:-}"
WORK_ROOT="${WORK_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
L1_AUGMENTED_ROOT="${L1_AUGMENTED_ROOT:-}"
L1_SKILLS_ROOT="${L1_SKILLS_ROOT:-$DRUG_PIPE_DIR/workdir-skills/molclaw-l1-workspace/.agents/skills}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-$DATA_PIPE_DIR/pipeline/cleaning/prompts/qwen35_system.md}"
USER_PROMPT_PREFIX_FILE="${USER_PROMPT_PREFIX_FILE:-$DRUG_PIPE_DIR/workdir-skills/molclaw-l1-workspace/prompt_prefix.md}"
DEPLOYMENT_TOOL_SET="${DEPLOYMENT_TOOL_SET:-}"
TOOL_VISIBILITY="${TOOL_VISIBILITY:-all}"
TOKENIZER="${TOKENIZER:-}"
RELEASE_ROOT="${RELEASE_ROOT:-}"
TIMEOUT_SEC="${LLM_CLEAN_TIMEOUT_SEC:-300}"
LIMIT=0
MAX_WORKERS="${MAX_WORKERS:-1}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_cleaning.sh --deployment-tool-set PATH --tokenizer PATH --release-root NEW_PATH [options]

Builds the structured data path:
  Claude raw -> uncleaned Qwen-native audit projection
             -> semantic mother dataset + answer recovery
             -> native first-use skill augmentation -> reasoning clean
             -> Qwen3.5 structured SFT view

Options:
  --tokenizer PATH             Local checkpoint tokenizer; required for final release
  --release-root PATH          New, non-existing release directory
  --results-root PATH
  --preclean-root PATH          Uncleaned raw-event native-message audit view
  --work-root PATH
  --output-root PATH
  --l1-augmented-root PATH
  --l1-skills-root PATH
  --deployment-tool-set PATH   Exact tools visible to the student at deployment
  --tool-visibility MODE       all (required for formal releases)
  --system-prompt-file PATH    Qwen adapter system prompt
  --user-prompt-prefix-file PATH
  --claude-bin PATH
  --harness claude|deepseek
  --dsh-bin PATH
  --dsh-node-bin PATH
  --dsh-model MODEL
  --dsh-provider CC_SWITCH_ID
  --timeout-sec SECONDS
  --limit N
  --max-workers N
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tokenizer) TOKENIZER="${2:-}"; shift 2 ;;
    --release-root) RELEASE_ROOT="${2:-}"; shift 2 ;;
    --results-root) RESULTS_ROOT="${2:-}"; shift 2 ;;
    --preclean-root) PRECLEAN_ROOT="${2:-}"; shift 2 ;;
    --work-root) WORK_ROOT="${2:-}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --l1-augmented-root) L1_AUGMENTED_ROOT="${2:-}"; shift 2 ;;
    --l1-skills-root) L1_SKILLS_ROOT="${2:-}"; shift 2 ;;
    --deployment-tool-set) DEPLOYMENT_TOOL_SET="${2:-}"; shift 2 ;;
    --tool-visibility) TOOL_VISIBILITY="${2:-}"; shift 2 ;;
    --system-prompt-file) SYSTEM_PROMPT_FILE="${2:-}"; shift 2 ;;
    --user-prompt-prefix-file) USER_PROMPT_PREFIX_FILE="${2:-}"; shift 2 ;;
    --claude-bin) CLAUDE_BIN="${2:-}"; shift 2 ;;
    --harness) AGENT_HARNESS="${2:-}"; shift 2 ;;
    --dsh-bin) DSH_BIN="${2:-}"; shift 2 ;;
    --dsh-node-bin) DSH_NODE_BIN="${2:-}"; shift 2 ;;
    --dsh-model) DSH_MODEL="${2:-}"; shift 2 ;;
    --dsh-provider) DSH_PROVIDER="${2:-}"; shift 2 ;;
    --timeout-sec) TIMEOUT_SEC="${2:-}"; shift 2 ;;
    --limit) LIMIT="${2:-}"; shift 2 ;;
    --max-workers) MAX_WORKERS="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

PRECLEAN_ROOT="${PRECLEAN_ROOT:-$RESULTS_ROOT/qwen35_native_raw}"
WORK_ROOT="${WORK_ROOT:-$RESULTS_ROOT/semantic_work}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RESULTS_ROOT/cleaned}"
L1_AUGMENTED_ROOT="${L1_AUGMENTED_ROOT:-$RESULTS_ROOT/l1_augmented}"

if [[ -z "$DEPLOYMENT_TOOL_SET" || ! -f "$DEPLOYMENT_TOOL_SET" ]]; then
  echo "[error] --deployment-tool-set must name the exact deployment-visible tool manifest" >&2
  exit 1
fi
if [[ ! -f "$SYSTEM_PROMPT_FILE" ]]; then
  echo "[error] Qwen adapter system prompt not found: $SYSTEM_PROMPT_FILE" >&2
  exit 1
fi
if [[ "$TOOL_VISIBILITY" != "all" ]]; then
  echo "[error] unsupported --tool-visibility: $TOOL_VISIBILITY" >&2
  exit 1
fi
if [[ "$AGENT_HARNESS" != "claude" && "$AGENT_HARNESS" != "deepseek" ]]; then
  echo "[error] unsupported --harness: $AGENT_HARNESS" >&2
  exit 1
fi

if [[ ! -d "$TOKENIZER" || -z "$RELEASE_ROOT" || -e "$RELEASE_ROOT" ]]; then
  echo "[error] provide --tokenizer and a new --release-root before cleaning" >&2
  exit 1
fi

RESULTS_ROOT="$(realpath -m "$RESULTS_ROOT")"
PRECLEAN_ROOT="$(realpath -m "$PRECLEAN_ROOT")"
WORK_ROOT="$(realpath -m "$WORK_ROOT")"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
DEPLOYMENT_TOOL_SET="$(realpath "$DEPLOYMENT_TOOL_SET")"
SYSTEM_PROMPT_FILE="$(realpath "$SYSTEM_PROMPT_FILE")"
USER_PROMPT_PREFIX_FILE="$(realpath "$USER_PROMPT_PREFIX_FILE")"
L1_SKILLS_ROOT="$(realpath "$L1_SKILLS_ROOT")"

export PYTHONPATH="$DATA_PIPE_DIR:$SLIME_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$PRECLEAN_ROOT" "$WORK_ROOT" "$OUTPUT_ROOT"
cd "$DATA_PIPE_DIR"

echo "[cleaning] raw events -> uncleaned Qwen3.5 native-message audit view"
"$PYTHON_BIN" -m pipeline.cleaning.preclean_native \
  --results-root "$RESULTS_ROOT" \
  --output-root "$PRECLEAN_ROOT"

echo "[cleaning] raw events -> semantic mother dataset"
"$PYTHON_BIN" -m pipeline.cleaning.python_clean \
  --results-root "$RESULTS_ROOT" \
  --output-root "$WORK_ROOT"

VIEW_INPUT="$WORK_ROOT/semantic_trajectories.jsonl"
echo "[cleaning] semantic -> native first-use skill augmentation"
"$PYTHON_BIN" -m pipeline.cleaning.skill_native_augmentation \
  --input "$VIEW_INPUT" \
  --output-root "$L1_AUGMENTED_ROOT" \
  --skills-root "$L1_SKILLS_ROOT"
VIEW_INPUT="$L1_AUGMENTED_ROOT/semantic_trajectories.jsonl"


echo "[cleaning] semantic reasoning clean + pending answer recovery"
"$PYTHON_BIN" -m pipeline.cleaning.llm_clean \
  --input "$VIEW_INPUT" \
  --output-root "$OUTPUT_ROOT" \
  --claude-bin "$CLAUDE_BIN" \
  --harness "$AGENT_HARNESS" \
  --dsh-bin "$DSH_BIN" \
  --dsh-node-bin "$DSH_NODE_BIN" \
  --dsh-model "$DSH_MODEL" \
  --dsh-provider "$DSH_PROVIDER" \
  --timeout-sec "$TIMEOUT_SEC" \
  --limit "$LIMIT" \
  --max-workers "$MAX_WORKERS"
VIEW_INPUT="$OUTPUT_ROOT/semantic_trajectories.jsonl"


echo "[cleaning] publish complete trajectories with actual tokenizer and loss mask"
"$PYTHON_BIN" -m pipeline.cleaning.publish_dataset \
  --input "$VIEW_INPUT" --output-root "$RELEASE_ROOT" --tokenizer "$TOKENIZER" \
  --deployment-tool-set "$DEPLOYMENT_TOOL_SET" \
  --system-prompt-file "$SYSTEM_PROMPT_FILE" \
  --user-prompt-prefix-file "$USER_PROMPT_PREFIX_FILE" \
  --audit "$WORK_ROOT/python_audit.jsonl" \
  --audit "$OUTPUT_ROOT/llm_clean_audit.jsonl"
echo "[done] release: $RELEASE_ROOT/release_manifest.json"
