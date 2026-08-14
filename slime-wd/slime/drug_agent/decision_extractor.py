"""Pure, shared extraction of history-only ReAct decision states."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from drug_agent.protocol.react_protocol import parse_runtime_decision


def normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    raw_name = str(call.get("tool_name") or call.get("name") or "").strip()
    arguments = call.get("arguments")
    if arguments is None:
        arguments = call.get("input")
    return {
        "tool_name_raw": raw_name,
        "tool_name": raw_name.removeprefix("mcp__molclaw-scp__"),
        "arguments": deepcopy(arguments) if isinstance(arguments, dict) else {},
        "id": str(call.get("id") or ""),
        "raw_payload": deepcopy(call),
    }


def parse_assistant_decision(content: Any) -> dict[str, Any]:
    if not isinstance(content, str):
        return {
            "ok": False,
            "decision_type": None,
            "thoughts": [],
            "tool_calls": [],
            "final_answer": None,
            "error": "assistant_content_not_string",
        }
    runtime = parse_runtime_decision(content)
    decision_type = runtime.get("decision_type")
    tool_calls = []
    for index, call in enumerate(runtime.get("tool_calls") or []):
        payload = call.get("raw_payload") if isinstance(call.get("raw_payload"), dict) else {
            "tool_name": call.get("tool_name"),
            "arguments": call.get("arguments"),
        }
        normalized = normalize_tool_call(payload)
        normalized["index"] = index
        tool_calls.append(normalized)
    final_answer = runtime.get("final_answer")
    error = runtime.get("error_message")
    if decision_type is None and error == "assistant generation must contain a tool_call or final_answer":
        error = None
    return {
        "ok": error is None,
        "decision_type": decision_type,
        "thoughts": [str(item) for item in (runtime.get("thoughts") or []) if str(item).strip()],
        "tool_calls": tool_calls,
        "final_answer": final_answer,
        "error": error,
    }


def iter_react_decisions(messages: Iterable[dict[str, Any]]):
    materialized = list(messages)
    for assistant_index, message in enumerate(materialized):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        parsed = parse_assistant_decision(message.get("content"))
        if parsed["decision_type"] is None:
            continue
        if (
            assistant_index > 0
            and isinstance(materialized[assistant_index - 1], dict)
            and materialized[assistant_index - 1].get("role") == "assistant"
        ):
            parsed = {
                **parsed,
                "ok": False,
                "error": "consecutive_assistant_state_boundary",
            }
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
