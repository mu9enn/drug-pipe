from __future__ import annotations

import json
import re
from collections import deque
from typing import Any

from pipeline.cleaning.artifacts import (
    ABSOLUTE_PATH_RE,
    ARTIFACT_RE,
    RELATIVE_PATH_RE,
    inspect_observation_status,
)
from pipeline.cleaning.models import react_schema_findings


TOOL_CALL_RE = re.compile(r"<tool_call>([\s\S]*?)</tool_call>")
OBSERVATION_RE = re.compile(r'<observation\s+tool_name="([^"]+)">([\s\S]*?)</observation>')
THOUGHT_RE = re.compile(r"<thought>([\s\S]*?)</thought>")
FINAL_RE = re.compile(r"<final_answer>([\s\S]*?)</final_answer>")
SUCCESS_CLAIM_RE = re.compile(r"(?i)\b(success(?:ful(?:ly)?)?|completed|produced|saved|written)\b")
TEACHER_SCAFFOLD_RE = re.compile(
    r"(?ix)(?:"
    r"claude\s+code|claude\.md|\.claude(?:/|\\)skills|"
    r"\bskills?\b|\bphase\s+\d+(?:\.\d+)?(?:\s*:\s*execution)?\b|\btask\s+type\s+triage\b|"
    r"\b(?:run_log|results?)\.md\b|\bfile\s+inventory\b|"
    r"\b(?:execution|decision|run)\s+log\b|\b(?:result|scientific)\s+report\b|"
    r"\b(?:using|use)\s+`?(?:Read|Write|Edit)`?\b|"
    r"`(?:Read|Write|Edit)`|\bls\s+-la\b|\blocal\s+workspace\b|"
    r"\ball\s+(?:output\s+)?files?\s+(?:are\s+)?in\s+place\b|"
    r"\b(?:compile|compiling|prepare|preparing|confirm|confirming)\b[^\n.]{0,60}"
    r"\bfinal\s+(?:results?|answer|summary)\b|"
    r"\breviewing\s+the\s+current\s+state\s+of\s+the\s+workflow\b|"
    r"\beverything\s+is\s+done\b|"
    r"\b(?:read|write|edit|update|record|compile)\b[^\n.]{0,80}"
    r"\b(?:file|log|report|inventory)\b"
    r")"
)
MOLECULE_KEYS = ("smiles", "ligand", "molecule", "candidate")


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except (TypeError, ValueError):
        return None


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _upcoming_tool_name(content: str, after: int) -> str | None:
    match = TOOL_CALL_RE.search(content, after)
    if not match:
        return None
    payload = _parse_json(match.group(1))
    return str(payload.get("tool_name") or "") if isinstance(payload, dict) else None


def _premature_completion_claim(text: str, upcoming_tool: str | None) -> bool:
    tool = str(upcoming_tool or "").lower()
    if "server_file_to_base64" in tool or "base64_to_server_file" in tool:
        operation = r"(?:file\s+)?(?:transfer|conversion)|file\s+transferr?ed"
    elif "retrieve" in tool:
        operation = r"retriev(?:al|e|ed)"
    elif "fix" in tool or "repair" in tool:
        operation = r"repair(?:ed)?|fix(?:ed)?"
    elif "dock" in tool or "quickvina" in tool:
        operation = r"dock(?:ing|ed)?"
    else:
        return False
    return bool(
        re.search(
            rf"(?i)\b(?:{operation})\b[^\n.]{{0,30}}\b(?:complete|completed|successful|succeeded)\b|"
            rf"\b(?:successfully|already)\b[^\n.]{{0,20}}\b(?:{operation})\b|"
            rf"\bfile\s+(?:was\s+)?transferred\b",
            text,
        )
    )


def _assistant_prose_findings(sample: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for message_index, message in enumerate(sample.get("messages") or []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "")
        for segment_index, match in enumerate(THOUGHT_RE.finditer(content)):
            reasons: list[str] = []
            text = match.group(1)
            if TEACHER_SCAFFOLD_RE.search(text):
                reasons.append("teacher_engineering_scaffolding")
            upcoming_tool = _upcoming_tool_name(content, match.end())
            if _premature_completion_claim(text, upcoming_tool):
                reasons.append(f"premature_completion_before_tool:{upcoming_tool}")
            if reasons:
                findings.append(
                    {
                        "message_index": message_index,
                        "segment_type": "thought",
                        "segment_index": segment_index,
                        "reasons": reasons,
                    }
                )
        for match in FINAL_RE.finditer(content):
            payload = _parse_json(match.group(1))
            summary = payload.get("summary") if isinstance(payload, dict) else None
            if isinstance(summary, str) and TEACHER_SCAFFOLD_RE.search(summary):
                findings.append(
                    {
                        "message_index": message_index,
                        "segment_type": "final_summary",
                        "segment_index": 0,
                        "reasons": ["teacher_engineering_scaffolding"],
                    }
                )
    return findings


def protocol_parts(sample: dict[str, Any]) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    malformed: list[str] = []
    for message_index, message in enumerate(sample.get("messages") or []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "assistant":
            for match in TOOL_CALL_RE.finditer(content):
                payload = _parse_json(match.group(1))
                if not isinstance(payload, dict):
                    malformed.append(f"message_{message_index}_tool_call_json_invalid")
                else:
                    calls.append(payload)
            for match in FINAL_RE.finditer(content):
                payload = _parse_json(match.group(1))
                if not isinstance(payload, dict):
                    malformed.append(f"message_{message_index}_final_json_invalid")
                else:
                    finals.append(payload)
        if role == "user":
            for match in OBSERVATION_RE.finditer(content):
                payload = _parse_json(match.group(2))
                if not isinstance(payload, dict):
                    malformed.append(f"message_{message_index}_observation_json_invalid")
                else:
                    observations.append({"tool_name": match.group(1), "payload": payload})
    return {"calls": calls, "observations": observations, "finals": finals, "malformed": malformed}


def immutable_facts(sample: dict[str, Any]) -> dict[str, Any]:
    parts = protocol_parts(sample)
    predictions = []
    for payload in parts["finals"]:
        predictions.append({key: value for key, value in payload.items() if key != "summary"})
    messages = sample.get("messages") if isinstance(sample.get("messages"), list) else []
    return {
        "schema_version": sample.get("schema_version"),
        "id": sample.get("id"),
        "roles": [message.get("role") for message in messages if isinstance(message, dict)],
        "loss_masks": [message.get("step_loss_mask") for message in messages if isinstance(message, dict)],
        "fixed_messages": [
            {"index": index, "role": message.get("role"), "content": message.get("content")}
            for index, message in enumerate(messages)
            if isinstance(message, dict) and message.get("role") in {"system", "user"}
            and "<observation " not in str(message.get("content") or "")
        ],
        "calls": parts["calls"],
        "observations": parts["observations"],
        "predictions": predictions,
    }


def compare_immutable_facts(source: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    before = immutable_facts(source)
    after = immutable_facts(candidate)
    findings: list[str] = []
    for key in (
        "schema_version", "id", "roles", "loss_masks", "fixed_messages",
        "calls", "observations", "predictions",
    ):
        if before[key] != after[key]:
            findings.append(f"immutable_changed:{key}")
    return findings


def _molecule_from_arguments(arguments: Any) -> str | None:
    if not isinstance(arguments, dict):
        return None
    for key in MOLECULE_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip() and "<artifact:" not in value:
            return value.strip()
    for value in arguments.values():
        nested = _molecule_from_arguments(value) if isinstance(value, dict) else None
        if nested:
            return nested
    return None


def _find_numeric(value: Any, preferred_keys: set[str] | None = None) -> list[float]:
    numbers: list[float] = []
    if isinstance(value, dict):
        for key, item in value.items():
            relevant = preferred_keys is None or any(token in str(key).lower() for token in preferred_keys)
            if relevant and isinstance(item, (int, float)) and not isinstance(item, bool):
                numbers.append(float(item))
            if isinstance(item, (dict, list)):
                numbers.extend(_find_numeric(item, preferred_keys))
    elif isinstance(value, list):
        for item in value:
            numbers.extend(_find_numeric(item, preferred_keys))
    return numbers


def _check_final_consistency(
    calls: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    final_payload: dict[str, Any] | None,
    report: dict[str, Any],
) -> None:
    if final_payload is None:
        report["errors"].append("missing_structured_final_answer")
        return
    final_text = _serialize(final_payload)
    observation_text = _serialize(observations)
    missing_artifacts = sorted(set(ARTIFACT_RE.findall(final_text)) - set(ARTIFACT_RE.findall(observation_text)))
    if missing_artifacts:
        report["errors"].append("final_references_unknown_artifact")
        report["missing_artifacts"] = missing_artifacts

    last_failed = bool(
        observations and inspect_observation_status(observations[-1]["payload"])["is_error"]
    )
    summary = str(final_payload.get("summary") or "")
    if last_failed and SUCCESS_CLAIM_RE.search(summary):
        report["errors"].append("final_claims_success_after_failed_critical_tool")

    final_numbers = _find_numeric(final_payload.get("evidence"), {"score", "affinity", "value", "count"})
    observed_numbers = _find_numeric([item["payload"] for item in observations])
    unsupported = [
        number for number in final_numbers
        if not any(abs(number - observed) <= 1e-9 for observed in observed_numbers)
    ]
    if unsupported:
        report["errors"].append("final_numeric_evidence_not_in_observations")
        report["unsupported_final_numbers"] = unsupported

    if str(final_payload.get("task_type") or "").lower() != "vs":
        return
    ranking = final_payload.get("ranked_smiles")
    if not isinstance(ranking, list):
        report["errors"].append("vs_final_missing_ranking")
        return
    scores: dict[str, float] = {}
    for call, observation in zip(calls, observations):
        tool_name = str(call.get("tool_name") or "").lower()
        if "dock" not in tool_name and "quickvina" not in tool_name:
            continue
        molecule = _molecule_from_arguments(call.get("arguments"))
        values = _find_numeric(observation["payload"], {"score", "affinity"})
        if molecule and values and not inspect_observation_status(observation["payload"])["is_error"]:
            scores[molecule] = values[0]
    scored = [(str(molecule), scores[str(molecule)]) for molecule in ranking if str(molecule) in scores]
    if any(current[1] < previous[1] for previous, current in zip(scored, scored[1:])):
        report["errors"].append("vs_ranking_inconsistent_with_tool_scores")
        report["vs_scored_ranking"] = scored


def validate_final_record(sample: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "errors": react_schema_findings(sample),
        "warnings": [],
        "status_conflicts": [],
        "sequence_mismatches": [],
        "prose_findings": _assistant_prose_findings(sample),
    }
    parts = protocol_parts(sample)
    report["errors"].extend(parts["malformed"])
    messages = sample.get("messages") if isinstance(sample.get("messages"), list) else []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = str(message.get("content") or "")
        if ABSOLUTE_PATH_RE.search(content) or RELATIVE_PATH_RE.search(content):
            report["errors"].append(f"message_{message_index}_unsanitized_path")

    pending: deque[str] = deque(str(call.get("tool_name") or "") for call in parts["calls"])
    for observation in parts["observations"]:
        inspection = inspect_observation_status(observation["payload"])
        if inspection["conflict"]:
            report["errors"].append("observation_status_conflict")
            report["status_conflicts"].append({"tool_name": observation["tool_name"], **inspection})
        if not pending:
            report["errors"].append("orphan_observation")
            continue
        expected = pending.popleft()
        if expected != observation["tool_name"]:
            report["errors"].append("tool_observation_order_mismatch")
            report["sequence_mismatches"].append(
                {"expected": expected, "observed": observation["tool_name"]}
            )
    if pending:
        report["errors"].append(f"missing_observations:{len(pending)}")
    if len(parts["finals"]) != 1:
        report["errors"].append(f"structured_final_answer_count:{len(parts['finals'])}")
    _check_final_consistency(
        parts["calls"], parts["observations"],
        parts["finals"][-1] if parts["finals"] else None, report,
    )
    report["errors"] = list(dict.fromkeys(report["errors"]))
    report["counts"] = {
        "tool_calls": len(parts["calls"]),
        "observations": len(parts["observations"]),
        "final_answers": len(parts["finals"]),
        "errors": len(report["errors"]),
        "prose_findings": len(report["prose_findings"]),
    }
    return report


def collect_repair_hints(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample.get("id"),
        "editable_findings": _assistant_prose_findings(sample),
    }
