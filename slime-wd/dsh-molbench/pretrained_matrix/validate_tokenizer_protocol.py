#!/usr/bin/env python3
"""Validate the structured-v8 messages against one checkpoint chat template."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    tools = [{
        "type": "function",
        "function": {
            "name": "probe",
            "description": "Protocol probe",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }]
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "use tools"},
        {
            "role": "assistant",
            "reasoning_content": "single call",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "id": "call-1",
                "function": {"name": "probe", "arguments": {"value": "one"}},
            }],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "probe", "content": "one-result"},
        {
            "role": "assistant",
            "reasoning_content": "multiple calls",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call-2",
                    "function": {"name": "probe", "arguments": {"value": "two"}},
                },
                {
                    "type": "function",
                    "id": "call-3",
                    "function": {"name": "probe", "arguments": {"value": "three"}},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call-2", "name": "probe", "content": "two-result"},
        {"role": "tool", "tool_call_id": "call-3", "name": "probe", "content": "three-result"},
        {
            "role": "assistant",
            "reasoning_content": "final reasoning",
            "content": '{"answer_smiles":"CC","evidence":[]}',
        },
    ]
    rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    required = [
        "<think>", "single call", "multiple calls", "final reasoning",
        "<tool_call>", "<tool_response>", "one-result", "two-result", "three-result",
        '{"answer_smiles":"CC","evidence":[]}',
    ]
    missing = [marker for marker in required if marker not in rendered]
    if missing:
        raise RuntimeError(f"chat template protocol markers missing: {missing}")
    if "<final_answer>" in rendered:
        raise RuntimeError("checkpoint chat template injected the legacy final_answer envelope")
    encoded = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True)
    token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if not token_ids:
        raise RuntimeError("chat template produced no tokens")
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    decoded_missing = [marker for marker in required if marker not in decoded]
    if decoded_missing:
        raise RuntimeError(f"tokenize/decode round-trip lost protocol markers: {decoded_missing}")
    result = {
        "schema_version": "qwen35_structured_v8_tokenizer_gate_v1",
        "model_dir": str(args.model_dir),
        "tokenizer_config_sha256": hashlib.sha256(
            (args.model_dir / "tokenizer_config.json").read_bytes()
        ).hexdigest(),
        "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "token_count": len(token_ids),
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
