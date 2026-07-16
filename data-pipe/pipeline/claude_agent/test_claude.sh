#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

PROVIDER="${CC_SWITCH_PROVIDER:-manual}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
SKILLS_ROOT="$REPO_DIR/skills/skills_vs"
SYSTEM_PROMPT_FILE="$SKILLS_ROOT/system_prompt_result.md"
LAUNCH_SCRIPT="$REPO_DIR/claude_agent/launch_claude.sh"
WORKDIR="$REPO_DIR/results/test_workdir"

if [[ ! -x "$LAUNCH_SCRIPT" ]]; then
  echo "[error] launch script not executable: $LAUNCH_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$SYSTEM_PROMPT_FILE" ]]; then
  echo "[error] system prompt file not found: $SYSTEM_PROMPT_FILE" >&2
  exit 1
fi

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

SYSTEM_PROMPT="$(cat "$SYSTEM_PROMPT_FILE")"
QUESTION_PAYLOAD="请简短回答你当前模型与可见的mcp server名称，然后列出你可用的所有工具"
PROMPT_FILE="$WORKDIR/prompt.txt"
printf '%s\n\nQuestion payload (MolBench-VS):\n%s\n' "$SYSTEM_PROMPT" "$QUESTION_PAYLOAD" > "$PROMPT_FILE"

echo "[test] running single question via launch_claude.sh"
bash "$LAUNCH_SCRIPT" \
  --workdir "$WORKDIR" \
  --prompt-file "$PROMPT_FILE" \
  --skills-root "$SKILLS_ROOT" \
  --provider "$PROVIDER" \
  --claude-bin "$CLAUDE_BIN"

if [[ ! -f "$WORKDIR/complete_session.jsonl" ]]; then
  echo "[error] missing session file: $WORKDIR/complete_session.jsonl" >&2
  exit 1
fi
if [[ ! -f "$WORKDIR/run_meta.json" ]]; then
  echo "[error] missing run meta file: $WORKDIR/run_meta.json" >&2
  exit 1
fi

echo "[test] assistant reply:"
python - "$WORKDIR/complete_session.jsonl" <<'PY'
import json
import sys
from pathlib import Path

session_path = Path(sys.argv[1])
final_text = ""
assistant_text = ""
raw_lines = []
for line in session_path.read_text(encoding="utf-8", errors="ignore").splitlines():
    raw_lines.append(line)
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(obj, dict) and obj.get("type") == "result":
        text = obj.get("result")
        if isinstance(text, str):
            final_text = text
    if isinstance(obj, dict) and obj.get("type") == "assistant":
        msg = obj.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list):
                buf = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text"), str):
                        buf.append(c["text"])
                if buf:
                    assistant_text = "".join(buf)
if final_text:
    print(final_text)
elif assistant_text:
    print(assistant_text)
else:
    print("\n".join(raw_lines))
PY

echo "[test] ok: session and run meta generated in $WORKDIR"
