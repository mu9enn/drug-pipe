from __future__ import annotations

import json
import re
from collections import deque
from typing import Any

from pipeline.cleaning.artifacts import (
    ABSOLUTE_PATH_RE,
    MALFORMED_ARTIFACT_RE,
    RELATIVE_PATH_RE,
    artifact_references,
    inspect_observation_status,
)
from pipeline.cleaning.models import react_schema_findings
from pipeline.cleaning.vs_ranking import (
    rank_by_best_quickvina_score,
    successful_quickvina_result,
)


TOOL_CALL_RE = re.compile(r"<tool_call>([\s\S]*?)</tool_call>")
OBSERVATION_RE = re.compile(r'<observation\s+tool_name="([^"]+)">([\s\S]*?)</observation>')
THOUGHT_RE = re.compile(r"<thought>([\s\S]*?)</thought>")
FINAL_RE = re.compile(r"<final_answer>([\s\S]*?)</final_answer>")
L2_L3_ORCHESTRATION_RE = re.compile(
    r"(?ix)(?:"
    r"(?:\.claude(?:/|\\)skills|skills?)[^\n]{0,100}\bL[23](?:[-_/ ][A-Za-z0-9.-]+)?\b|"
    r"\bL[23](?:[-_/ ][A-Za-z0-9.-]+)?\b[^\n]{0,100}\b(?:skill|workflow|methodolog)|"
    r"\b(?:workflow|methodology)[- ]level\s+skills?\b|"
    r"\b(?:read|load|inspect|invoke|follow|consult)(?:ing)?\b[^\n.]{0,100}"
    r"\b(?:L[23]|workflow|methodology)\b[^\n.]{0,60}\b(?:skill|document|workflow)\b"
    r")"
)
LOCAL_TOOL_NARRATION_RE = re.compile(
    r"(?ix)(?:"
    r"\b(?:Read|Write|Edit|Bash|Grep|Glob|Skill)\s+(?:tool|call)\b|"
    r"\b(?:use|using|invoke|run|call)(?:\s+the)?\s+`?(?:Read|Write|Edit|Bash|Grep|Glob|Skill)`?\b|"
    r"\b(?:read|load|inspect|consult)(?:ing)?\b[^\n.]{0,100}\bskills?\b|"
    r"\b(?:read|load|inspect|consult)(?:ing)?\b[^\n.]{0,80}"
    r"\b(?:L1|tool[- ]level)\b[^\n.]{0,40}\bskills?\b|"
    r"\b(?:read|write|edit|update|append|create|inspect|list)\b[^\n.]{0,80}"
    r"\b(?:file|directory|run_log\.md|results?\.md|report|log|file\s+inventory|local\s+workspace)\b|"
    r"\b(?:run_log\.md|results?\.md|file\s+inventory|local\s+workspace)\b"
    r")"
)
TEACHER_SIDECAR_NARRATION_RE = re.compile(
    r"(?ix)(?:"
    r"\b(?:question\.json|parsed_answer\.json|run_meta\.json|complete_session\.jsonl|"
    r"prompt\.txt|CLAUDE\.md)\b|"
    r"\b(?:read|inspect|list|check|locate|open)(?:ing)?\b[^\n.]{0,80}"
    r"\b(?:skills?\s+(?:directory|catalog|hierarchy)|\.claude/skills)\b"
    r")"
)
HIGHER_ORDER_RANKING_MARKERS = ("equiscore", "consensus")


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except (TypeError, ValueError):
        return None


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _assistant_prose_findings(
    sample: dict[str, Any],
    *,
    only_molclaw_tool: bool = False,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for message_index, message in enumerate(sample.get("messages") or []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "")
        for segment_index, match in enumerate(THOUGHT_RE.finditer(content)):
            reasons: list[str] = []
            text = match.group(1)
            if L2_L3_ORCHESTRATION_RE.search(text):
                reasons.append("l2_l3_teacher_orchestration")
            if TEACHER_SIDECAR_NARRATION_RE.search(text):
                reasons.append("teacher_sidecar_or_skill_catalog_narration")
            if only_molclaw_tool and LOCAL_TOOL_NARRATION_RE.search(text):
                reasons.append("local_tool_narration_removed_in_only_molclaw_mode")
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
            reasons = []
            if isinstance(summary, str) and L2_L3_ORCHESTRATION_RE.search(summary):
                reasons.append("l2_l3_teacher_orchestration")
            if (
                isinstance(summary, str)
                and TEACHER_SIDECAR_NARRATION_RE.search(summary)
            ):
                reasons.append("teacher_sidecar_or_skill_catalog_narration")
            if (
                only_molclaw_tool
                and isinstance(summary, str)
                and LOCAL_TOOL_NARRATION_RE.search(summary)
            ):
                reasons.append("local_tool_narration_removed_in_only_molclaw_mode")
            if reasons:
                findings.append(
                    {
                        "message_index": message_index,
                        "segment_type": "final_summary",
                        "segment_index": 0,
                        "reasons": reasons,
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
    known_artifacts = artifact_references(calls) | artifact_references(observations)
    missing_artifacts = sorted(artifact_references(final_text) - known_artifacts)
    if missing_artifacts:
        report["errors"].append("final_references_unknown_artifact")
        report["missing_artifacts"] = missing_artifacts

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
    if any(
        any(marker in str(call.get("tool_name") or "").lower() for marker in HIGHER_ORDER_RANKING_MARKERS)
        and not inspect_observation_status(observation["payload"])["is_error"]
        for call, observation in zip(calls, observations)
    ):
        return
    quickvina_results: list[dict[str, Any]] = []
    for call, observation in zip(calls, observations):
        result = successful_quickvina_result(
            str(call.get("tool_name") or ""),
            call.get("arguments"),
            observation["payload"],
        )
        if result is not None:
            quickvina_results.append(result)
    expected, score_audit = rank_by_best_quickvina_score(
        [str(molecule) for molecule in ranking],
        quickvina_results,
    )
    if expected != [str(molecule) for molecule in ranking]:
        report["errors"].append("vs_ranking_inconsistent_with_tool_scores")
        report["vs_expected_ranking"] = expected
        report["vs_quickvina_score_audit"] = score_audit


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
        if MALFORMED_ARTIFACT_RE.search(content):
            report["errors"].append(f"message_{message_index}_malformed_artifact_reference")

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
