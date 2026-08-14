#!/usr/bin/env bash
# CPU-side ReAct -> ToolRL/GAD materialization with a shared LLM summary cache.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/resolve_slime_env.sh"
source "$SLIME_ENV"
cd "$SLIME"

INPUT=${INPUT:?Set INPUT to cleaned react_trajectories.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new versioned data-view directory}
HF_CHECKPOINT=${HF_CHECKPOINT:?Set HF_CHECKPOINT to the Qwen3.5-9B tokenizer/model directory}
CLAUDE_BIN=${CLAUDE_BIN:-claude}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-245760}
MAX_RESPONSE_TOKENS=${MAX_RESPONSE_TOKENS:-16384}
MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS:-262144}
SUMMARY_MAX_TOKENS=${SUMMARY_MAX_TOKENS:-32768}
LLM_MAX_ATTEMPTS=${LLM_MAX_ATTEMPTS:-3}
LLM_TIMEOUT_SEC=${LLM_TIMEOUT_SEC:-600}

[[ -f "$INPUT" ]] || { echo "Missing cleaned ReAct input: $INPUT" >&2; exit 2; }
[[ -f "$HF_CHECKPOINT/tokenizer_config.json" ]] || { echo "Missing tokenizer: $HF_CHECKPOINT" >&2; exit 2; }
command -v "$CLAUDE_BIN" >/dev/null || { echo "Claude/LLM CLI is unavailable: $CLAUDE_BIN" >&2; exit 2; }
[[ ! -e "$OUTPUT_ROOT/materialize.complete" ]] || { echo "Already materialized: $OUTPUT_ROOT" >&2; exit 2; }

python - "$HF_CHECKPOINT" "$MAX_CONTEXT_TOKENS" <<'PY'
import json, pathlib, sys
from transformers import AutoTokenizer
model = pathlib.Path(sys.argv[1]); required = int(sys.argv[2])
config = json.loads((model / "config.json").read_text())
tokenizer_config = json.loads((model / "tokenizer_config.json").read_text())
model_limit = int(config.get("text_config", config).get("max_position_embeddings", 0))
tokenizer_limit = int(tokenizer_config.get("model_max_length", 0))
assert model_limit >= required, (model_limit, required)
assert tokenizer_limit >= required, (tokenizer_limit, required)
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.apply_chat_template(
    [{"role": "system", "content": "preflight"}, {"role": "user", "content": "preflight"}],
    tokenize=True, add_generation_prompt=True, enable_thinking=False,
)
PY

mkdir -p "$OUTPUT_ROOT/toolrl" "$OUTPUT_ROOT/gad" "$OUTPUT_ROOT/summary_cache"

SOURCE_SHA256=$(sha256sum "$INPUT" | awk '{print $1}')

python -m drug_agent.toolrl.convert_react_to_toolrl_steps \
  --input "$INPUT" \
  --output "$OUTPUT_ROOT/toolrl/toolrl_steps.raw.jsonl" \
  --skipped-report "$OUTPUT_ROOT/toolrl/toolrl_steps.skipped.jsonl" \
  --report "$OUTPUT_ROOT/toolrl/toolrl_steps.report.json"

python -m drug_agent.gad.data \
  --input "$INPUT" \
  --output "$OUTPUT_ROOT/gad/gad_steps.raw.jsonl" \
  --skipped-report "$OUTPUT_ROOT/gad/gad_steps.skipped.jsonl" \
  --report "$OUTPUT_ROOT/gad/gad_steps.report.json"

python drug_agent/scripts/select_toolrl_decisions.py \
  --input "$OUTPUT_ROOT/toolrl/toolrl_steps.raw.jsonl" \
  --output "$OUTPUT_ROOT/toolrl/toolrl_steps.jsonl" \
  --manifest "$OUTPUT_ROOT/toolrl/context_manifest.json" \
  --model "$HF_CHECKPOINT" \
  --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
  --max-response-tokens "$MAX_RESPONSE_TOKENS" \
  --max-context-tokens "$MAX_CONTEXT_TOKENS" \
  --summary-max-tokens "$SUMMARY_MAX_TOKENS" \
  --semantic-summarizer claude \
  --summary-cache-root "$OUTPUT_ROOT/summary_cache" \
  --claude-bin "$CLAUDE_BIN" \
  --llm-timeout-sec "$LLM_TIMEOUT_SEC" \
  --llm-max-attempts "$LLM_MAX_ATTEMPTS"

python drug_agent/scripts/compact_rl_context.py \
  --input "$OUTPUT_ROOT/gad/gad_steps.raw.jsonl" \
  --output "$OUTPUT_ROOT/gad/gad_steps.jsonl" \
  --audit "$OUTPUT_ROOT/gad/context_manifest.json" \
  --excluded "$OUTPUT_ROOT/gad/context_excluded.jsonl" \
  --tokenizer "$HF_CHECKPOINT" \
  --max-tokens "$MAX_PROMPT_TOKENS" \
  --max-response-tokens "$MAX_RESPONSE_TOKENS" \
  --max-context-tokens "$MAX_CONTEXT_TOKENS" \
  --summary-max-tokens "$SUMMARY_MAX_TOKENS" \
  --semantic-summarizer claude \
  --summary-cache-root "$OUTPUT_ROOT/summary_cache" \
  --claude-bin "$CLAUDE_BIN" \
  --llm-timeout-sec "$LLM_TIMEOUT_SEC" \
  --llm-max-attempts "$LLM_MAX_ATTEMPTS"

python - "$OUTPUT_ROOT" "$INPUT" "$SOURCE_SHA256" "$MAX_PROMPT_TOKENS" "$MAX_RESPONSE_TOKENS" "$MAX_CONTEXT_TOKENS" <<'PY'
import json, pathlib, sys
root, source, source_hash = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
max_prompt, max_response, max_context = map(int, sys.argv[4:7])
tool = json.loads((root / "toolrl/context_manifest.json").read_text())
gad = json.loads((root / "gad/context_manifest.json").read_text())
assert tool["context"]["observed_max_prompt_tokens"] <= max_prompt
assert tool["context"]["observed_max_target_tokens"] <= max_response
assert gad["output_max_tokens"] <= max_prompt
manifest = {
    "schema_version": "react_rl_views_v1",
    "source": str(pathlib.Path(source).resolve()),
    "source_sha256": source_hash,
    "limits": {"context": max_context, "prompt": max_prompt, "response": max_response},
    "toolrl_manifest": str((root / "toolrl/context_manifest.json").resolve()),
    "gad_manifest": str((root / "gad/context_manifest.json").resolve()),
    "shared_summary_cache": str((root / "summary_cache").resolve()),
}
(root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
PY

touch "$OUTPUT_ROOT/materialize.complete"
echo "Materialized ToolRL/GAD views under $OUTPUT_ROOT (source=$SOURCE_SHA256)"
