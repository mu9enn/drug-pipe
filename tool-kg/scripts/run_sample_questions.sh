#!/usr/bin/env bash
#
# Run tool-kg Stage3 question sampling for an existing completed KG run.
#
# Preconditions:
#   1. Stage1 + Stage2 must already be complete for the target run_id.
#   2. The canonical results directory must contain graph.jsonl,
#      tool_catalog.jsonl, edge_decisions.jsonl, and run_manifest.json.
#   3. The fixed local Science-KB must already be built with:
#        python scripts/build_science_kb.py --replace
#   4. The project .env must define MOLCLAW_SCP_API_KEY.
#
# What this script does:
#   - Samples tool workflows from the generated ToolKG.
#   - Calls the Claude Code agent to generate non-leaking public questions.
#   - Writes canonical tasks to runs/<run_id>/results/tasks.jsonl.
#   - Writes retry/workdir/audit state under runs/<run_id>/intermediate/stage3/.
#
# Main usage (simple_default is the default named profile):
#   bash scripts/run_sample_questions.sh <run_id>
#
# Examples:
#   # Simple success-first mode: stop at 20 usable questions or 200 attempts.
#   bash scripts/run_sample_questions.sh run_20260601_123052 \
#     --sampling-profile simple_default \
#     --target-successes 20 --max-attempts 200 \
#     --grounding-selection random_seeded \
#     --max-repeat-target 2 --max-repeat-compound 2
#
#   # Simple-mode smoke test: generate one usable question.
#   bash scripts/run_sample_questions.sh run_20260601_123052 \
#     --sampling-profile simple_default --target-successes 1 --max-attempts 10
#
#   # Override walk/anchor hop range. Defaults are 2 to 4.
#   bash scripts/run_sample_questions.sh run_20260601_123052 \
#     --target-successes 20 --max-attempts 200 --min-hops 2 --max-hops 5
#
# Important notes:
#   - Public questions should not expose tool names or explicit tool order.
#   - Internal expected trajectories may contain tool IDs for evaluation/training.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="${1:-}"
SAMPLING_PROFILE="simple_default"
MIN_HOPS=""
MAX_HOPS=""
SEED=""
TARGET_SUCCESSES=""
MAX_ATTEMPTS=""
JSON_REPAIR_ROUNDS=""
SEMANTIC_REPAIR_ROUNDS=""
SCIENCE_KB_TOPK=""
GROUNDING_SELECTION=""
MAX_REPEAT_TARGET=""
MAX_REPEAT_COMPOUND=""
MAX_WORKERS=1

usage() {
  echo "Usage:"
  echo "  Default simple profile: $0 <run_id> [--target-successes <N>] [--max-attempts <N>] [--max-workers <1-4>] [...]"
}

if [[ -z "$RUN_ID" || "$RUN_ID" == "--help" || "$RUN_ID" == "-h" ]]; then
  usage
  exit 0
fi
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sampling-profile)
      SAMPLING_PROFILE="${2:-}"
      shift 2
      ;;
    --target-successes)
      TARGET_SUCCESSES="${2:-}"
      shift 2
      ;;
    --max-attempts)
      MAX_ATTEMPTS="${2:-}"
      shift 2
      ;;
    --json-repair-rounds)
      JSON_REPAIR_ROUNDS="${2:-}"
      shift 2
      ;;
    --semantic-repair-rounds)
      SEMANTIC_REPAIR_ROUNDS="${2:-}"
      shift 2
      ;;
    --science-kb-topk)
      SCIENCE_KB_TOPK="${2:-}"
      shift 2
      ;;
    --grounding-selection)
      GROUNDING_SELECTION="${2:-}"
      shift 2
      ;;
    --max-repeat-target)
      MAX_REPEAT_TARGET="${2:-}"
      shift 2
      ;;
    --max-repeat-compound)
      MAX_REPEAT_COMPOUND="${2:-}"
      shift 2
      ;;
    --min-hops)
      MIN_HOPS="${2:-}"
      shift 2
      ;;
    --max-hops)
      MAX_HOPS="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! "$MAX_WORKERS" =~ ^[1-4]$ ]]; then
  echo "ERROR: --max-workers must be an integer from 1 to 4" >&2
  exit 2
fi

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

API_KEY="${MOLCLAW_SCP_API_KEY:-}"
if [[ -z "$API_KEY" ]]; then
  echo "ERROR: MOLCLAW_SCP_API_KEY is required" >&2
  exit 1
fi

RUN_DIR="$PROJECT_ROOT/runs/$RUN_ID"
for f in graph.jsonl tool_catalog.jsonl edge_decisions.jsonl run_manifest.json; do
  if [[ ! -f "$RUN_DIR/results/$f" ]]; then
    echo "ERROR: missing required file: $RUN_DIR/results/$f" >&2
    exit 2
  fi
done

for f in science_kb/processed/science_kb.sqlite science_kb/manifests/science_kb_manifest.json; do
  if [[ ! -f "$PROJECT_ROOT/$f" ]]; then
    echo "ERROR: missing fixed Science-KB artifact: $PROJECT_ROOT/$f" >&2
    echo "Build it once with: python scripts/build_science_kb.py --replace" >&2
    exit 2
  fi
done

cmd=(
  python3 -m molclaw_kg.cli
  --project-root "$PROJECT_ROOT"
  --run-id "$RUN_ID"
  --mode claude_cc
  --max-workers "$MAX_WORKERS"
  sample-questions
  --sampling-profile "$SAMPLING_PROFILE"
)

append_override() {
  local value="$1"
  local flag="$2"
  if [[ -n "$value" ]]; then
    cmd+=("$flag" "$value")
  fi
}

append_override "$TARGET_SUCCESSES" --target-successes
append_override "$MAX_ATTEMPTS" --max-attempts
append_override "$JSON_REPAIR_ROUNDS" --json-repair-rounds
append_override "$SEMANTIC_REPAIR_ROUNDS" --semantic-repair-rounds
append_override "$SCIENCE_KB_TOPK" --science-kb-topk
append_override "$GROUNDING_SELECTION" --grounding-selection
append_override "$MAX_REPEAT_TARGET" --max-repeat-target
append_override "$MAX_REPEAT_COMPOUND" --max-repeat-compound
append_override "$MIN_HOPS" --min-hops
append_override "$MAX_HOPS" --max-hops
append_override "$SEED" --seed

echo "[sample-questions] run_id=$RUN_ID profile=$SAMPLING_PROFILE max_workers=$MAX_WORKERS (only explicit flags override the profile)"
PYTHONPATH="$PROJECT_ROOT/src" "${cmd[@]}"
echo "[sample-questions] complete: $RUN_DIR/results/tasks.jsonl"
