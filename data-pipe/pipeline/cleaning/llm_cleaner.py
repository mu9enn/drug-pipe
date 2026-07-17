from __future__ import annotations

import copy
import json
import re
from typing import Any, Callable


TOOL_CALL_RE = re.compile(r"<tool_call>([\s\S]*?)</tool_call>")
OBSERVATION_RE = re.compile(r'<observation\s+tool_name="([^"]+)">([\s\S]*?)</observation>')
FINAL_RE = re.compile(r"<final_answer>([\s\S]*?)</final_answer>")
PREDICTION_KEYS = {
    "answer",
    "result",
    "answer_smiles",
    "ranked_smiles",
    "selected_smiles",
    "selected_smiles_list",
}
RewriteFunction = Callable[[dict[str, Any]], dict[str, Any]]


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except (TypeError, ValueError):
        return None


def _protocol_signature(sample: dict[str, Any]) -> dict[str, Any]:
    calls: list[Any] = []
    observations: list[dict[str, Any]] = []
    prediction: dict[str, Any] = {}
    for message in sample.get("messages", []):
        if not isinstance(message, dict):
            continue
        content = str(message.get("content") or "")
        for match in TOOL_CALL_RE.finditer(content):
            calls.append(_parse_json(match.group(1)))
        for match in OBSERVATION_RE.finditer(content):
            observations.append({"tool_name": match.group(1), "payload": _parse_json(match.group(2))})
        for match in FINAL_RE.finditer(content):
            payload = _parse_json(match.group(1))
            if isinstance(payload, dict):
                prediction = {key: payload[key] for key in PREDICTION_KEYS if key in payload}
    return {
        "calls": calls,
        "observations": observations,
        "prediction": prediction,
    }


def validate_llm_rewrite(source: dict[str, Any], cleaned: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    if not isinstance(cleaned, dict):
        return ["llm_output_not_object"]
    if cleaned.get("id") != source.get("id"):
        findings.append("llm_changed_sample_id")
    if cleaned.get("schema_version") != source.get("schema_version"):
        findings.append("llm_changed_schema_version")
    if not isinstance(cleaned.get("messages"), list):
        findings.append("llm_messages_not_list")
        return findings
    before = _protocol_signature(source)
    after = _protocol_signature(cleaned)
    if before["calls"] != after["calls"]:
        findings.append("llm_changed_tool_calls_or_order")
    if before["observations"] != after["observations"]:
        findings.append("llm_changed_observations_or_order")
    if before["prediction"] != after["prediction"]:
        findings.append("llm_changed_task_prediction")
    before_roles = [message.get("role") for message in source.get("messages", []) if isinstance(message, dict)]
    after_roles = [message.get("role") for message in cleaned.get("messages", []) if isinstance(message, dict)]
    if before_roles != after_roles:
        findings.append("llm_changed_message_roles_or_order")
    return findings


def clean_with_llm(
    sample: dict[str, Any],
    rewrite: RewriteFunction | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply a semantic text rewrite while protecting executable/scientific facts.

    The caller owns the actual model invocation. A rejected rewrite returns the
    untouched source record and findings for the final acceptance gate.
    """
    source = copy.deepcopy(sample)
    report: dict[str, Any] = {
        "status": "not_run" if rewrite is None else "cleaned",
        "findings": [],
        "actions": [],
    }
    if rewrite is None:
        return source, report
    try:
        candidate = rewrite(copy.deepcopy(source))
    except Exception as exc:
        report["status"] = "failed"
        report["findings"].append(f"llm_clean_exception:{type(exc).__name__}")
        return source, report
    findings = validate_llm_rewrite(source, candidate)
    if findings:
        report["status"] = "unsafe_rewrite"
        report["findings"] = findings
        return source, report
    report["actions"].append("semantic_text_rewrite")
    return candidate, report
