from __future__ import annotations

import copy
import json
from typing import Any

from pipeline.cleaning.models import semantic_schema_findings


LEGACY_PROTOCOL_MARKERS = ("<thought>", "<tool_call>", "<observation>", "<final_answer>")


def validate_semantic_record(record: dict[str, Any]) -> dict[str, Any]:
    errors = semantic_schema_findings(record)
    if "deployment_tool_set_sha256" in (record.get("metadata") or {}):
        errors.append("semantic_contains_deployment_tool_set_sha256")
    pending: list[str] = []
    decision_ids: set[str] = set()
    final_count = 0
    call_count = 0
    for index, event in enumerate(record.get("events") or []):
        if event.get("type") == "assistant_decision":
            if pending:
                errors.append(f"decision_before_observations:{index}:{pending}")
            decision_id = str(event.get("source_message_id") or "")
            if decision_id in decision_ids:
                errors.append(f"duplicate_decision_id:{decision_id}")
            decision_ids.add(decision_id)
            calls = event.get("tool_calls") or []
            final = event.get("final_answer")
            if bool(calls) == bool(final):
                errors.append(f"decision_must_have_calls_xor_final:{index}")
            pending = [str(call.get("source_tool_use_id") or "") for call in calls]
            call_count += len(calls)
            final_count += int(bool(final))
        elif event.get("type") == "tool_observation":
            call_id = str(event.get("source_tool_use_id") or "")
            if not pending or pending[0] != call_id:
                errors.append(f"observation_order_mismatch:{index}:{call_id}:{pending[:1]}")
            else:
                pending.pop(0)
    if pending:
        errors.append(f"missing_observations:{pending}")
    if final_count != 1:
        errors.append(f"terminal_final_count:{final_count}")
    if call_count == 0:
        errors.append("no_tool_calls")
    serialized = json.dumps(record, ensure_ascii=False)
    for marker in LEGACY_PROTOCOL_MARKERS:
        if marker in serialized:
            errors.append(f"legacy_protocol_marker:{marker}")
    return {"ok": not errors, "errors": errors, "decision_count": len(decision_ids), "tool_call_count": call_count}


def immutable_semantic_facts(record: dict[str, Any]) -> dict[str, Any]:
    facts = copy.deepcopy(record)
    for event in facts.get("events") or []:
        if event.get("type") == "assistant_decision":
            event.pop("reasoning", None)
    return facts


def compare_immutable_facts(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    return [] if immutable_semantic_facts(before) == immutable_semantic_facts(after) else ["immutable_semantic_facts_changed"]
