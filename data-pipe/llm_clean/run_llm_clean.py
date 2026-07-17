#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DATA_PIPE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPE_DIR))

from pipeline.cleaning.llm_cleaner import clean_with_llm  # noqa: E402


def _extract_json(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1])
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("llm_output_not_object")
    return parsed


def _claude_rewriter(claude_bin: str, prompt: str):
    def rewrite(sample: dict[str, Any]) -> dict[str, Any]:
        request = f"{prompt}\n\nINPUT RECORD:\n{json.dumps(sample, ensure_ascii=False)}"
        process = subprocess.run(
            [claude_bin, "-p", request],
            check=False,
            text=True,
            capture_output=True,
        )
        if process.returncode != 0:
            raise RuntimeError(f"claude_exit_code:{process.returncode}")
        return _extract_json(process.stdout)

    return rewrite


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantically clean canonical ReAct JSONL with protected facts.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    audit = Path(args.audit).expanduser().resolve()
    prompt = (Path(__file__).resolve().parent / "prompt.md").read_text(encoding="utf-8")
    rewrite = _claude_rewriter(args.claude_bin, prompt)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with source.open("r", encoding="utf-8") as input_handle, output.open(
        "w", encoding="utf-8"
    ) as output_handle, audit.open("w", encoding="utf-8") as audit_handle:
        for line in input_handle:
            if not line.strip():
                continue
            if args.limit > 0 and processed >= args.limit:
                break
            record = json.loads(line)
            cleaned, report = clean_with_llm(record, rewrite)
            output_handle.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            audit_handle.write(json.dumps({"id": record.get("id"), **report}, ensure_ascii=False) + "\n")
            processed += 1
    print(json.dumps({"processed": processed, "output": str(output), "audit": str(audit)}, indent=2))


if __name__ == "__main__":
    main()
