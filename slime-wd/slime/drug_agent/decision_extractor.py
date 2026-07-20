"""Pure, shared extraction of history-only ReAct decision states."""
from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Iterable


TOOL_CALL_RE = re.compile(r"<tool_call>([\s\S]*?)</tool_call>")
FINAL_ANSWER_RE = re.compile(r"<final_answer>([\s\S]*?)</final_answer>")
MOLCLAW_PREFIX_RE = re.compile(r"^mcp__molclaw-scp__")


def normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    raw_name = str(call.get("tool_name") or call.get("name") or "").strip()
    arguments = call.get("arguments")
    if arguments is None:
        arguments = call.get("input")
    return {
        "tool_name_raw": raw_name,
        "tool_name": MOLCLAW_PREFIX_RE.sub("", raw_name),
        "arguments": deepcopy(arguments) if isinstance(arguments, dict) else {},
        "id": str(call.get("id") or ""),
        "raw_payload": deepcopy(call),
    }


def parse_assistant_decision(content: Any) -> dict[str, Any]:
    if not isinstance(content, str):
        return {
            "ok": False,
            "decision_type": None,
            "tool_calls": [],
            "final_answer": None,
            "error": "assistant_content_not_string",
        }
    tool_blocks = TOOL_CALL_RE.findall(content)
    final_blocks = FINAL_ANSWER_RE.findall(content)
    tool_calls: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, block in enumerate(tool_blocks):
        try:
            payload = json.loads(block.strip())
        except json.JSONDecodeError as exc:
            errors.append(f"tool_call_{index}_json: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"tool_call_{index}_not_object")
            continue
        normalized = normalize_tool_call(payload)
        if not normalized["tool_name"]:
            errors.append(f"tool_call_{index}_missing_name")
            continue
        normalized["index"] = index
        tool_calls.append(normalized)

    final_answer: Any = None
    if final_blocks:
        try:
            final_answer = json.loads(final_blocks[-1].strip())
        except json.JSONDecodeError:
            final_answer = final_blocks[-1].strip()
    if tool_blocks and final_blocks:
        errors.append("mixed_tool_call_and_final_answer")
    if tool_blocks:
        decision_type = "tool_call"
    elif final_blocks:
        decision_type = "final_answer"
    else:
        decision_type = None
    return {
        "ok": not errors,
        "decision_type": decision_type,
        "tool_calls": tool_calls,
        "final_answer": final_answer,
        "error": "; ".join(errors) if errors else None,
    }


def iter_react_decisions(messages: Iterable[dict[str, Any]]):
    materialized = list(messages)
    for assistant_index, message in enumerate(materialized):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        parsed = parse_assistant_decision(message.get("content"))
        if parsed["decision_type"] is None:
            continue
        state_messages = []
        for item in materialized[:assistant_index]:
            if not isinstance(item, dict):
                continue
            state_message = {
                key: deepcopy(item[key])
                for key in ("role", "content", "name")
                if key in item
            }
            state_messages.append(state_message)
        yield {
            "assistant_index": assistant_index,
            "state_messages": state_messages,
            "target_assistant": {
                "role": "assistant",
                "content": message.get("content"),
            },
            "decision_type": parsed["decision_type"],
            "tool_calls": parsed["tool_calls"],
            "final_answer": parsed["final_answer"],
            "parse": parsed,
        }
