"""ToolRL-only projections retained outside the formal SFT materializer.

The legacy ToolRL runtime is intentionally not changed in the SFT refactor.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from pipeline.cleaning.deployment_tools import DeploymentToolSet
from pipeline.cleaning.models import TOOLRL_SCHEMA_VERSION, toolrl_schema_findings
from pipeline.cleaning.sft_views import semantic_to_qwen35_sft


def semantic_to_toolrl(
    record: dict[str, Any],
    *,
    deployment_tools: DeploymentToolSet,
    system_prompt: str,
) -> list[dict[str, Any]]:
    sft = semantic_to_qwen35_sft(
        record,
        deployment_tools=deployment_tools,
        system_prompt=system_prompt,
    )
    prompt = copy.deepcopy(sft["messages"][:2])
    rows: list[dict[str, Any]] = []
    messages = sft["messages"]
    assistant_positions = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    semantic_decisions = [event for event in record["events"] if event["type"] == "assistant_decision"]
    for ordinal, position in enumerate(assistant_positions):
        target = copy.deepcopy(messages[position])
        target.pop("step_loss_mask", None)
        calls = target.get("tool_calls") or []
        decision_type = "tool_call" if calls else "final_answer"
        source_event = semantic_decisions[ordinal]
        label_calls = [
            {
                "name": call["function"]["name"],
                "arguments": copy.deepcopy(call["function"]["arguments"]),
            }
            for call in calls
        ]
        row = {
            "schema_version": TOOLRL_SCHEMA_VERSION,
            "prompt": copy.deepcopy(prompt),
            "tools": copy.deepcopy(sft["tools"]),
            "label": {
                "source_id": record["id"],
                "decision_id": source_event["source_message_id"],
                "decision_ordinal": ordinal,
                "decision_type": decision_type,
                "target_tool_calls": label_calls,
                "target_final_answer": target["content"] if decision_type == "final_answer" else None,
                "target_assistant": target,
            },
            "metadata": {
                "source_id": record["id"],
                "decision_id": source_event["source_message_id"],
                "decision_ordinal": ordinal,
                "decision_type": decision_type,
                "decision_role": "final" if decision_type == "final_answer" else "tool_step",
                "task_type": (record.get("metadata") or {}).get("task_type"),
                "tool_names": [call["name"] for call in label_calls],
                "is_initial_step": ordinal == 0,
                "is_final_step": ordinal == len(assistant_positions) - 1,
                "trajectory_decision_count": len(assistant_positions),
                "target_tool_calls": copy.deepcopy(label_calls),
                "deployment_tool_set_sha256": deployment_tools.sha256,
            },
        }
        findings = toolrl_schema_findings(row)
        if findings:
            raise ValueError(f"ToolRL schema validation failed: {findings}")
        rows.append(row)

        context_target = copy.deepcopy(messages[position])
        context_target["step_loss_mask"] = 0
        prompt.append(context_target)
        cursor = position + 1
        while cursor < len(messages) and messages[cursor]["role"] == "tool":
            prompt.append(copy.deepcopy(messages[cursor]))
            cursor += 1

    observations_by_decision: dict[str, list[dict[str, Any]]] = {}
    current_id: str | None = None
    for event in record["events"]:
        if event["type"] == "assistant_decision":
            current_id = event["source_message_id"]
            observations_by_decision.setdefault(current_id, [])
        elif current_id is not None:
            observations_by_decision[current_id].append(event)
    seen: dict[str, tuple[int, bool]] = {}
    successful_ordinals: set[int] = set()
    for row in rows:
        metadata = row["metadata"]
        ordinal = metadata["decision_ordinal"]
        calls = row["label"]["target_tool_calls"]
        signature = json.dumps(calls, ensure_ascii=False, sort_keys=True, separators=(",", ":")) if calls else None
        previous = seen.get(signature) if signature is not None else None
        no_progress = bool(
            previous and previous[1]
            and not any(previous[0] < item < ordinal for item in successful_ordinals)
        )
        observations = observations_by_decision.get(row["label"]["decision_id"], [])
        usable_success = bool(observations) and all(not item["is_error"] for item in observations)
        metadata.update(
            {
                "is_no_progress_repeat": no_progress,
                "observation_status": "success" if usable_success else ("failure" if observations else "not_applicable"),
                "observation_usable_success": usable_success,
            }
        )
        if usable_success:
            successful_ordinals.add(ordinal)
        if signature is not None:
            seen[signature] = (ordinal, usable_success)
    return rows
