#!/usr/bin/env python3
"""Exercise the exact strict parser used by production runtime and reward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from drug_agent.protocol.react_protocol import parse_runtime_decision
from drug_agent.toolrl.parse_tool_calls import parse_tool_calls


def _call(name: str) -> str:
    return json.dumps({"tool_name": name, "arguments": {"value": name}}, separators=(",", ":"))


def audit(*, native_qwen_thinking: bool = False) -> dict[str, Any]:
    a, b, c = (_call(name) for name in ("A", "B", "C"))
    cases = {
        "whitespace_separated_objects": f"<tool_call>\n{a}\n{b}\n{c}\n</tool_call>",
        "comma_separated_objects": f"<tool_call>\n{a},{b},{c}\n</tool_call>",
        "json_array": f"<tool_call>\n[{a},{b},{c}]\n</tool_call>",
        "multiple_containers": f"<tool_call>{a}</tool_call><tool_call>{b}</tool_call>",
    }
    expected = {
        "whitespace_separated_objects": (True, 3),
        "comma_separated_objects": (False, 0),
        "json_array": (False, 0),
        "multiple_containers": (False, 0),
    }
    results = {}
    for name, text in cases.items():
        response = f"audited reasoning\n</think>\n\n{text}" if native_qwen_thinking else text
        runtime = parse_runtime_decision(
            response,
            strict_toolrl_turn=True,
            native_qwen_thinking=native_qwen_thinking,
        )
        reward = parse_tool_calls(
            response,
            keep_non_molclaw=True,
            strict_toolrl_turn=True,
            native_qwen_thinking=native_qwen_thinking,
        )
        runtime_count = len(runtime.get("tool_calls") or []) if runtime.get("ok") else 0
        reward_count = len(reward.get("tool_calls") or []) if reward.get("ok") else 0
        want_valid, want_count = expected[name]
        if bool(runtime.get("ok")) != want_valid or runtime_count != want_count:
            raise ValueError(f"runtime parser mismatch for {name}: {runtime}")
        if bool(reward.get("ok")) != want_valid or reward_count != want_count:
            raise ValueError(f"reward parser mismatch for {name}: {reward}")
        results[name] = {
            "expected_valid": want_valid,
            "runtime_valid": bool(runtime.get("ok")),
            "runtime_invocation_count": runtime_count,
            "reward_valid": bool(reward.get("ok")),
            "reward_invocation_count": reward_count,
        }
    return {
        "schema_version": "toolrl_runtime_parser_compatibility_v2",
        "ok": True,
        "native_qwen_thinking": native_qwen_thinking,
        "production_runtime_call": (
            "parse_runtime_decision(..., strict_toolrl_turn=True, native_qwen_thinking=True)"
            if native_qwen_thinking
            else "parse_runtime_decision(..., strict_toolrl_turn=True)"
        ),
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native-qwen-thinking", action="store_true")
    args = parser.parse_args()
    report = audit(native_qwen_thinking=args.native_qwen_thinking)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
