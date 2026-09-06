from __future__ import annotations

import copy
import json
from typing import Any

from pipeline.cleaning.deployment_tools import DeploymentToolSet
from pipeline.cleaning.invariants import validate_semantic_record
from pipeline.cleaning.models import QWEN35_SFT_SCHEMA_VERSION, qwen35_sft_schema_findings


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def semantic_to_qwen35_sft(
    record: dict[str, Any],
    *,
    deployment_tools: DeploymentToolSet,
    system_prompt: str,
    tool_visibility: str = "all",
) -> dict[str, Any]:
    if not system_prompt.strip():
        raise ValueError("Qwen3.5 adapter system prompt must be non-empty")
    semantic_validation = validate_semantic_record(record)
    if not semantic_validation["ok"]:
        raise ValueError(f"invalid semantic trajectory: {semantic_validation['errors']}")
    visible_names = deployment_tools.visible_tool_names(record, tool_visibility)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt.strip(), "step_loss_mask": 0},
        {"role": "user", "content": record["user_task"], "step_loss_mask": 0},
    ]
    pending_calls: list[str] = []
    for event in record["events"]:
        if event["type"] == "assistant_decision":
            if pending_calls:
                raise ValueError(f"assistant decision before observations for {pending_calls}")
            calls = event["tool_calls"]
            message: dict[str, Any] = {
                "role": "assistant",
                "reasoning_content": event["reasoning"],
                "content": event["final_answer"] or "",
                "step_loss_mask": 1,
            }
            if calls:
                message["tool_calls"] = [
                    {
                        "id": call["source_tool_use_id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": copy.deepcopy(call["arguments"]),
                        },
                    }
                    for call in calls
                ]
                pending_calls = [call["source_tool_use_id"] for call in calls]
            messages.append(message)
            continue
        if not pending_calls or event["source_tool_use_id"] != pending_calls[0]:
            raise ValueError(f"observation order mismatch at {event['source_tool_use_id']}")
        messages.append(
            {
                "role": "tool",
                "name": event["name"],
                "tool_call_id": event["source_tool_use_id"],
                "content": _content(event["content"]),
                "step_loss_mask": 0,
            }
        )
        pending_calls.pop(0)
    if pending_calls:
        raise ValueError(f"missing terminal observations for {pending_calls}")
    output = {
        "schema_version": QWEN35_SFT_SCHEMA_VERSION,
        "id": record["id"],
        "messages": messages,
        "tools": deployment_tools.qwen_tools(visible_names),
    }
    findings = qwen35_sft_schema_findings(output)
    if findings:
        raise ValueError(f"Qwen3.5 SFT schema validation failed: {findings}")
    return output
