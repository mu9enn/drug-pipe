#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
RUNNER="$REPO_DIR/llm_clean/run_llm_clean.py"
FINALIZER="$REPO_DIR/pipeline/postprocess/final_hard_clean.py"
VALIDATOR="$SCRIPT_DIR/validate_llm_cleaned.py"
SKIP_LLM=0
FAIL_ON_INVALID=1
INPUT_DIR=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_llm_clean.sh INPUT_DIR [options]

Runs the complete LLM + script-2 stage:
  LLM semantic repair -> cleaned/
  final hard-clean/gate -> cleaned_final/ (unresolved samples are quarantined)
  post-LLM validator -> cleaned_final_validation.{json,md}

Options:
  --skip-llm             Reuse an existing INPUT_DIR/cleaned directory
  --allow-invalid        Do not return non-zero if final validation finds invalid files
  --claude-bin PATH      Default: claude
  --python-bin PATH      Default: python
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-llm) SKIP_LLM=1; shift ;;
    --allow-invalid) FAIL_ON_INVALID=0; shift ;;
    --claude-bin) CLAUDE_BIN="${2:-}"; shift 2 ;;
    --python-bin) PYTHON_BIN="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "[error] unknown option: $1" >&2; usage >&2; exit 1 ;;
    *)
      if [[ -n "$INPUT_DIR" ]]; then
        echo "[error] only one INPUT_DIR is allowed" >&2
        exit 1
      fi
      INPUT_DIR="$1"
      shift
      ;;
  esac
done

if [[ -z "$INPUT_DIR" ]]; then
  echo "[error] INPUT_DIR is required" >&2
  usage >&2
  exit 1
fi
INPUT_DIR="$(realpath -m "$INPUT_DIR")"
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "[error] input directory not found: $INPUT_DIR" >&2
  exit 1
fi
for required in "$RUNNER" "$FINALIZER" "$VALIDATOR"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] required script not found: $required" >&2
    exit 1
  fi
done

LLM_RC=0
if [[ "$SKIP_LLM" -eq 0 ]]; then
  echo "[llm-clean] semantic repair: $INPUT_DIR -> $INPUT_DIR/cleaned"
  set +e
  "$PYTHON_BIN" "$RUNNER" "$INPUT_DIR" --python-bin "$PYTHON_BIN" --claude-bin "$CLAUDE_BIN"
  LLM_RC=$?
  set -e
  if [[ "$LLM_RC" -ne 0 ]]; then
    echo "[warn] LLM clean reported skipped/failed samples (rc=$LLM_RC); continuing with collected cleaned files" >&2
  fi
else
  echo "[llm-clean] reusing existing cleaned directory (--skip-llm)"
fi

CLEANED_DIR="$INPUT_DIR/cleaned"
FINAL_DIR="$INPUT_DIR/cleaned_final"
REPORT_DIR="$INPUT_DIR/cleaned_final_reports"
if [[ ! -d "$CLEANED_DIR" ]]; then
  echo "[error] cleaned directory not found: $CLEANED_DIR" >&2
  exit 1
fi
if ! find "$CLEANED_DIR" -maxdepth 1 -type f -name '*.json' -print -quit | grep -q .; then
  echo "[error] no cleaned JSON files were produced: $CLEANED_DIR" >&2
  exit 1
fi

echo "[llm-clean] script-2 final hard-clean and deterministic gate"
"$PYTHON_BIN" "$FINALIZER" \
  --input-dir "$CLEANED_DIR" \
  --output-dir "$FINAL_DIR" \
  --report-dir "$REPORT_DIR"

echo "[llm-clean] post-LLM final validator: $FINAL_DIR"
validator_cmd=(
  "$PYTHON_BIN" "$VALIDATOR"
  --mode post-llm
  --input-dir "$FINAL_DIR"
  --output-json "$INPUT_DIR/cleaned_final_validation.json"
  --output-md "$INPUT_DIR/cleaned_final_validation.md"
  --quarantine-dir "$REPORT_DIR/quarantine_validator"
)
if [[ "$FAIL_ON_INVALID" -eq 1 ]]; then
  validator_cmd+=(--fail-on-invalid)
fi
VALIDATOR_RC=0
set +e
"${validator_cmd[@]}"
VALIDATOR_RC=$?
set -e

echo "[done] LLM + script-2 pipeline finished"
echo "  llm_cleaned:   $CLEANED_DIR"
echo "  cleaned_final: $FINAL_DIR"
echo "  quarantine:    $REPORT_DIR/quarantine"
echo "  validator_quarantine: $REPORT_DIR/quarantine_validator"
echo "  validation:    $INPUT_DIR/cleaned_final_validation.json"

if [[ "$VALIDATOR_RC" -ne 0 ]]; then
  exit "$VALIDATOR_RC"
fi
exit 0
