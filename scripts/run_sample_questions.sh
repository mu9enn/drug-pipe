#!/usr/bin/env bash
#
# Run MolClaw-KG Stage3 question sampling for an existing completed KG run.
#
# Preconditions:
#   1. Stage1 + Stage2 must already be complete for the target run_id.
#   2. The run directory must contain:
#        runs/<run_id>/graph_all.jsonl
#        runs/<run_id>/tool_cards.jsonl
#        runs/<run_id>/tool_snapshot.jsonl
#        runs/<run_id>/edge_debug_sidecar.jsonl
#   3. The fixed local Science-KB must already be built with:
#        python scripts/build_science_kb.py --replace
#   4. The project .env must define MOLCLAW_SCP_API_KEY.
#
# What this script does:
#   - Samples tool workflows from the generated ToolKG.
#   - Calls the Claude Code agent to generate non-leaking public questions.
#   - Writes all Stage3 outputs under:
#        runs/<run_id>/sample_workdir/   # per-sample Claude Code workdirs/traces
#        runs/<run_id>/sample_results/   # JSONL/CSV/results/reports
#
# Main usage (simple_toolchain_question is the default mode):
#   bash scripts/run_sample_questions.sh <run_id> \
#     --target-successes <N> --max-attempts <N>
#
# Examples:
#   # Simple success-first mode: stop at 20 usable questions or 200 attempts.
#   bash scripts/run_sample_questions.sh run_20260601_123052 \
#     --target-successes 20 --max-attempts 200 \
#     --grounding-selection random_seeded \
#     --max-repeat-target 2 --max-repeat-compound 2
#
#   # Simple-mode smoke test: generate one usable question.
#   bash scripts/run_sample_questions.sh run_20260601_123052 \
#     --target-successes 1 --max-attempts 10
#
#   # Legacy closure mode must now be selected explicitly.
#   bash scripts/run_sample_questions.sh run_20260601_123052 \
#     --sampling-mode dag_closure --sample-size 20
#
#   # Override walk/anchor hop range. Defaults are 2 to 4.
#   bash scripts/run_sample_questions.sh run_20260601_123052 \
#     --target-successes 20 --max-attempts 200 --min-hops 2 --max-hops 5
#
# Sampling modes:
#   - simple_toolchain_question: default success-first mode. Samples valid hidden
#     toolchains and asks the Agent only for a compact natural-language question.
#   - dag_closure: legacy explicit mode. Uses dependency-closure logic and
#     trajectory_v2 graph outputs where available.
#   - linear_debug: debug-only linear sampling mode, not recommended for final
#     KG-sampled task generation.
#
# Quality controls:
#   - --edge-profile core_strict|core_expanded
#       Default core_strict only samples high-confidence core edges.
#   - --partial-policy closure_required|exclude
#       Default closure_required allows mapped partial edges, but a sample only
#       succeeds after the final Agent-proposed workflow is dependency-closed.
#   - --max-repair-rounds N
#       Number of Agent repair attempts after Python validation feedback.
#
# Important notes:
#   - --sample-size is the number of sampling attempts, not guaranteed successes.
#   - Public questions should not expose tool names or explicit tool order.
#   - Internal expected trajectories may contain tool IDs for evaluation/training.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="${1:-}"
SAMPLE_SIZE=""
MIN_HOPS=2
MAX_HOPS=4
SEED=""
SAMPLING_MODE="simple_toolchain_question"
PARTIAL_POLICY="closure_required"
EDGE_PROFILE="core_strict"
MAX_REPAIR_ROUNDS=2
TARGET_SUCCESSES=""
MAX_ATTEMPTS=""
JSON_REPAIR_ROUNDS=1
SCIENCE_KB_TOPK=3
GROUNDING_SELECTION="random_seeded"
MAX_REPEAT_TARGET=2
MAX_REPEAT_COMPOUND=2

usage() {
  echo "Usage:"
  echo "  Default simple mode: $0 <run_id> --target-successes <N> --max-attempts <N> [--grounding-selection random_seeded] [...]"
  echo "  Legacy strict mode: $0 <run_id> --sampling-mode dag_closure|linear_debug --sample-size <N> [...]"
}

if [[ -z "$RUN_ID" || "$RUN_ID" == "--help" || "$RUN_ID" == "-h" ]]; then
  usage
  exit 0
fi
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample-size)
      SAMPLE_SIZE="${2:-}"
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
      JSON_REPAIR_ROUNDS="${2:-1}"
      shift 2
      ;;
    --science-kb-topk)
      SCIENCE_KB_TOPK="${2:-3}"
      shift 2
      ;;
    --grounding-selection)
      GROUNDING_SELECTION="${2:-random_seeded}"
      shift 2
      ;;
    --max-repeat-target)
      MAX_REPEAT_TARGET="${2:-2}"
      shift 2
      ;;
    --max-repeat-compound)
      MAX_REPEAT_COMPOUND="${2:-2}"
      shift 2
      ;;
    --min-hops)
      MIN_HOPS="${2:-2}"
      shift 2
      ;;
    --max-hops)
      MAX_HOPS="${2:-4}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --sampling-mode)
      SAMPLING_MODE="${2:-simple_toolchain_question}"
      shift 2
      ;;
    --partial-policy)
      PARTIAL_POLICY="${2:-closure_required}"
      shift 2
      ;;
    --edge-profile)
      EDGE_PROFILE="${2:-core_strict}"
      shift 2
      ;;
    --max-repair-rounds)
      MAX_REPAIR_ROUNDS="${2:-2}"
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

if [[ "$SAMPLING_MODE" == "simple_toolchain_question" ]]; then
  if [[ -z "$TARGET_SUCCESSES" || -z "$MAX_ATTEMPTS" ]]; then
    echo "ERROR: simple mode requires --target-successes and --max-attempts" >&2
    usage
    exit 2
  fi
elif [[ -z "$SAMPLE_SIZE" ]]; then
    echo "ERROR: strict modes require --sample-size" >&2
    usage
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
for f in graph_all.jsonl tool_cards.jsonl tool_snapshot.jsonl edge_debug_sidecar.jsonl; do
  if [[ ! -f "$RUN_DIR/$f" ]]; then
    echo "ERROR: missing required file: $RUN_DIR/$f" >&2
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
  --api-key "$API_KEY"
  --mode claude_cc
  sample-questions
  --min-hops "$MIN_HOPS"
  --max-hops "$MAX_HOPS"
  --sampling-mode "$SAMPLING_MODE"
  --partial-policy "$PARTIAL_POLICY"
  --edge-profile "$EDGE_PROFILE"
  --max-repair-rounds "$MAX_REPAIR_ROUNDS"
)

if [[ "$SAMPLING_MODE" == "simple_toolchain_question" ]]; then
  cmd+=(
    --target-successes "$TARGET_SUCCESSES"
    --max-attempts "$MAX_ATTEMPTS"
    --json-repair-rounds "$JSON_REPAIR_ROUNDS"
    --science-kb-topk "$SCIENCE_KB_TOPK"
    --grounding-selection "$GROUNDING_SELECTION"
    --max-repeat-target "$MAX_REPEAT_TARGET"
    --max-repeat-compound "$MAX_REPEAT_COMPOUND"
  )
else
  cmd+=(--sample-size "$SAMPLE_SIZE")
fi

if [[ -n "$SEED" ]]; then
  cmd+=(--seed "$SEED")
fi

echo "[sample-questions] run_id=$RUN_ID mode=$SAMPLING_MODE sample_size=${SAMPLE_SIZE:-n/a} target_successes=${TARGET_SUCCESSES:-n/a} max_attempts=${MAX_ATTEMPTS:-n/a} hops=[$MIN_HOPS,$MAX_HOPS] partial=$PARTIAL_POLICY edges=$EDGE_PROFILE repairs=$MAX_REPAIR_ROUNDS json_repairs=$JSON_REPAIR_ROUNDS kb_topk=$SCIENCE_KB_TOPK grounding=$GROUNDING_SELECTION max_repeat_target=$MAX_REPEAT_TARGET max_repeat_compound=$MAX_REPEAT_COMPOUND seed=${SEED:-none}"
PYTHONPATH="$PROJECT_ROOT/src" "${cmd[@]}"
echo "[sample-questions] complete: $RUN_DIR/sample_results"
