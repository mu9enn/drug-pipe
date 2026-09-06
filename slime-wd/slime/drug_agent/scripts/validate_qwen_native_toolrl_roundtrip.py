#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_agent.toolrl.qwen_native_parser import parse_qwen_native_completion
from transformers import AutoTokenizer


def _completion(tokenizer, row: dict) -> str:
    prompt = tokenizer.apply_chat_template(
        row["prompt"], tools=row["tools"], tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        row["prompt"] + [row["label"]["target_assistant"]],
        tools=row["tools"],
        tokenize=False,
    )
    if not full.startswith(prompt):
        raise ValueError("target render is not a continuation of the generation prompt")
    suffix = full[len(prompt):]
    end = suffix.rfind("<|im_end|>")
    return suffix[:end] if end >= 0 else suffix


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate configured native parsers against tokenizer-rendered targets.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-parser", required=True)
    parser.add_argument("--tool-parser", required=True)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    checked = 0
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            parsed = parse_qwen_native_completion(
                _completion(tokenizer, row),
                tools_schema=row["tools"],
                reasoning_parser=args.reasoning_parser,
                tool_parser=args.tool_parser,
            )
            label = row["label"]
            expected_calls = label["target_tool_calls"]
            actual_calls = [
                {"name": call["name"], "arguments": call["arguments"]}
                for call in parsed["tool_calls"]
            ]
            if not parsed["ok"] or actual_calls != expected_calls:
                if label["decision_type"] != "final_answer" or parsed["final_answer"] != label["target_final_answer"]:
                    raise ValueError(f"native parser round-trip failed for {label['decision_id']}: {parsed}")
            checked += 1
            if checked >= args.limit:
                break
    if checked == 0:
        raise ValueError("no ToolRL rows checked")
    print(json.dumps({"ok": True, "checked": checked}, ensure_ascii=False))


if __name__ == "__main__":
    main()
