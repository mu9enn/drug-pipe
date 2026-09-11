"""Comparison boundaries derived only from native decisions, once at intake."""
import json
import re

from drug_agent.toolrl.v8_dataset import stable_json

VERSION = "native_calls_cross_task_v3"


def prior_outcome(prompt):
    observations = [m for m in prompt if m.get("role") == "tool"]
    if not observations:
        return "unknown"
    latest = observations[-1]
    content = latest.get("content") or ""
    if latest.get("is_error") is True:
        return "error"
    if isinstance(content, str):
        content = content.strip()
        if content == "The operation timed out." or re.match(r"^\d+ validation errors? for call\[", content):
            return "error"
        if content.startswith("<skill_content"):
            return "success"
        try:
            content = json.loads(content)
        except ValueError:
            return "unknown"
    if isinstance(content, dict):
        if content.get("is_error") is True or content.get("ok") is False or content.get("status") in ("error", "failed", "timeout"):
            return "error"
        if content.get("is_error") is False or content.get("ok") is True or content.get("status") in ("success", "completed"):
            return "success"
    return "unknown"


def comparison_scope(row):
    calls = []
    for call in row["label"]["target_assistant"].get("tool_calls", []):
        function = call["function"]
        arguments = function["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        item = {"name": function["name"]}
        if function["name"] == "skill":
            item["skill"] = arguments["name"]
        item.update({key: arguments[key] for key in ("dry_run", "mode", "operation", "action", "method") if key in arguments})
        calls.append(item)
    scope = {"decision_type": "tool" if calls else "final_answer", "prior_outcome": prior_outcome(row["prompt"])}
    if calls:
        scope["calls"] = calls
    else:
        scope["task_type"] = row["metadata"]["task_type"]
    return stable_json(scope)
