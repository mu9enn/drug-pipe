from __future__ import annotations

import copy
import json
from typing import Any

from pipeline.output_contracts import normalize_final_answer, task_constraints
from pipeline.benchmark_release import require_training_task
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
    user_prompt_prefix: str = "",
    tool_visibility: str = "all",
) -> dict[str, Any]:
    if not system_prompt.strip():
        raise ValueError("Qwen3.5 adapter system prompt must be non-empty")
    from pipeline.cleaning.answer_recovery import strip_review_text, answer_key_findings
    if answer_key_findings(record):
        raise ValueError("explicit answer-key consultation")
    record = strip_review_text(record)
    semantic_validation = validate_semantic_record(record)
    if not semantic_validation["ok"]:
        raise ValueError(f"invalid semantic trajectory: {semantic_validation['errors']}")
    task_type = str((record.get("metadata") or {}).get("task_type") or "")
    require_training_task(record["user_task"], task_type, source_task_ids=tuple(record.get("metadata", {}).get("source_task_ids", [])))
    finals = [event["final_answer"] for event in record["events"] if event.get("final_answer") is not None]
    if len(finals) != 1:
        raise ValueError("expected exactly one terminal answer")
    normalize_final_answer(finals[0], task_type, constraints=task_constraints(record["user_task"], task_type))
    visible_names = deployment_tools.visible_tool_names(record, tool_visibility)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt.strip(), "step_loss_mask": 0},
        {
            "role": "user",
            "content": (
                f"{user_prompt_prefix.strip()}\n\n# Task\n\n{record['user_task']}"
                if user_prompt_prefix.strip()
                else record["user_task"]
            ),
            "step_loss_mask": 0,
        },
    ]
    skill_metadata = (record.get("metadata") or {}).get("skill_native_augmentation") or {}
    if skill_metadata.get("catalog_message"):
        messages.append({"role": "user", "content": skill_metadata["catalog_message"], "step_loss_mask": 0})
    import jsonschema
    validators = {t["name"]: jsonschema.Draft202012Validator(t["input_schema"]) for t in deployment_tools.tools}
    observations = {event["source_tool_use_id"]: event for event in record["events"] if event["type"] == "tool_observation"}
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
                for call in calls:
                    name = deployment_tools.require_public_name(str(call["name"]))
                    errors = list(validators[name].iter_errors(call["arguments"]))
                    observation = observations[call["source_tool_use_id"]]
                    if errors and not observation.get("is_error") and observation.get("status") != "error":
                        raise ValueError(f"successful tool call incompatible with deployment schema: {name}: {errors[0].message}")
                message["tool_calls"] = [
                    {
                        "id": call["source_tool_use_id"],
                        "type": "function",
                        "function": {
                            "name": deployment_tools.require_public_name(str(call["name"])),
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
                "name": deployment_tools.require_public_name(str(event["name"])),
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
