from __future__ import annotations

import copy
import json
import re
from collections import Counter, deque
from typing import Any

try:
    from pipeline.postprocess.react_constructor import sanitize_artifacts
except ImportError:
    from postprocess.react_constructor import sanitize_artifacts


TOOL_CALL_RE = re.compile(r"<tool_call>([\s\S]*?)</tool_call>")
OBSERVATION_RE = re.compile(r'<observation\s+tool_name="([^"]+)">([\s\S]*?)</observation>')
FINAL_RE = re.compile(r"<final_answer>([\s\S]*?)</final_answer>")
ARTIFACT_RE = re.compile(r"<artifact:[^>]+>")
RELATIVE_PATH_RE = re.compile(
    r"(?<![:/A-Za-z0-9])(?P<path>(?:\.\.?/)+(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+"
    r"\.(?:json|pdb|cif|mmcif|sdf|mol2|pdbqt|csv|tsv|txt|npy|npz|pt|pkl))"
)
DEBUG_KEYS = {
    "metadata",
    "pointers",
    "raw_pointer",
    "tool_use_id",
    "raw_tool_name",
    "raw_status",
    "raw_is_error",
    "raw_event_index",
    "fence_wrapper_stripped",
}
ERROR_STATUSES = {"error", "failed", "failure", "timeout", "timed_out", "invalid"}
SUCCESS_STATUSES = {"ok", "success", "succeeded", "complete", "completed", "partial_success"}
SUCCESS_CLAIM_RE = re.compile(r"(?i)\b(success(?:ful(?:ly)?)?|completed|produced|saved|written)\b")
MOLECULE_KEYS = ("smiles", "ligand", "molecule", "candidate")


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except (TypeError, ValueError):
        return None


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sanitize_relative_paths(text: str, report: dict[str, Any]) -> str:
    protected: list[str] = []

    def protect(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"__HARD_CLEAN_ARTIFACT_{len(protected) - 1}__"

    value = ARTIFACT_RE.sub(protect, text)

    def replace(match: re.Match[str]) -> str:
        raw = match.group("path")
        name = raw.replace("\\", "/").split("/")[-1]
        artifact = f"<artifact:local/{name}>"
        report["actions"].append({"type": "sanitize_relative_path", "before": raw, "after": artifact})
        return artifact

    value = RELATIVE_PATH_RE.sub(replace, value)
    for index, artifact in enumerate(protected):
        value = value.replace(f"__HARD_CLEAN_ARTIFACT_{index}__", artifact)
    return value


def _strip_debug_keys(value: Any, removed: Counter[str]) -> None:
    if isinstance(value, dict):
        for key in list(value):
            if key in DEBUG_KEYS:
                removed[key] += 1
                value.pop(key)
            else:
                _strip_debug_keys(value[key], removed)
    elif isinstance(value, list):
        for item in value:
            _strip_debug_keys(item, removed)


def _payload_error(payload: dict[str, Any]) -> tuple[bool, str]:
    status = str(payload.get("status") or "").strip().lower()
    content = payload.get("content")
    content_status = (
        str(content.get("status") or "").strip().lower() if isinstance(content, dict) else ""
    )
    error_value = content.get("error") if isinstance(content, dict) else None
    is_error = (
        payload.get("is_error") is True
        or status in ERROR_STATUSES
        or content_status in ERROR_STATUSES
        or error_value not in (None, "", False, [], {})
    )
    return is_error, content_status or status


def _clean_protocol_messages(
    messages: list[dict[str, Any]], report: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    calls: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    final_payload: dict[str, Any] | None = None
    path_map: dict[str, str] = {}
    for message_index, message in enumerate(messages):
        content = str(message.get("content") or "")
        content = str(sanitize_artifacts(content, path_map))
        content = _sanitize_relative_paths(content, report)

        def clean_call(match: re.Match[str]) -> str:
            payload = _parse_json(match.group(1))
            if not isinstance(payload, dict):
                report["errors"].append(f"message_{message_index}_tool_call_json_invalid")
                return match.group(0)
            calls.append(payload)
            return f"<tool_call>{_serialize(payload)}</tool_call>"

        def clean_observation(match: re.Match[str]) -> str:
            tool_name = match.group(1)
            payload = _parse_json(match.group(2))
            if not isinstance(payload, dict):
                report["errors"].append(f"message_{message_index}_observation_json_invalid")
                return match.group(0)
            payload = sanitize_artifacts(payload, path_map)
            removed: Counter[str] = Counter()
            _strip_debug_keys(payload, removed)
            if removed:
                report["actions"].append(
                    {
                        "type": "remove_debug_metadata",
                        "message_index": message_index,
                        "tool_name": tool_name,
                        "keys": dict(removed),
                    }
                )
            is_error, semantic_status = _payload_error(payload)
            outer_status = str(payload.get("status") or "").strip().lower()
            if (
                payload.get("is_error") is False
                and semantic_status in ERROR_STATUSES
                or payload.get("is_error") is True
                and outer_status in SUCCESS_STATUSES
            ):
                report["errors"].append("observation_status_conflict")
                report["status_conflicts"].append(
                    {"message_index": message_index, "tool_name": tool_name, "payload": payload}
                )
            payload["is_error"] = is_error
            payload["status"] = "error" if is_error else outer_status or semantic_status or "success"
            observations.append({"tool_name": tool_name, "payload": payload})
            return f'<observation tool_name="{tool_name}">{_serialize(payload)}</observation>'

        role = str(message.get("role") or "")
        if role == "assistant":
            content = TOOL_CALL_RE.sub(clean_call, content)
        if role in {"user", "tool"}:
            content = OBSERVATION_RE.sub(clean_observation, content)
        matches = list(FINAL_RE.finditer(content)) if role == "assistant" else []
        if matches:
            parsed = _parse_json(matches[-1].group(1))
            if not isinstance(parsed, dict):
                report["errors"].append(f"message_{message_index}_final_json_invalid")
            else:
                final_payload = parsed
                content = FINAL_RE.sub(
                    lambda match: f"<final_answer>{_serialize(_parse_json(match.group(1)))}</final_answer>"
                    if isinstance(_parse_json(match.group(1)), dict)
                    else match.group(0),
                    content,
                )
        message["content"] = content
    if path_map:
        report["artifact_mappings"].update(path_map)
    return messages, calls, observations, final_payload


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
            key_is_relevant = preferred_keys is None or any(token in str(key).lower() for token in preferred_keys)
            if key_is_relevant and isinstance(item, (int, float)) and not isinstance(item, bool):
                numbers.append(float(item))
            numbers.extend(_find_numeric(item, preferred_keys))
    elif isinstance(value, list):
        for item in value:
            numbers.extend(_find_numeric(item, preferred_keys))
    return numbers


def _docking_score(payload: dict[str, Any]) -> float | None:
    values = _find_numeric(payload, {"score", "affinity"})
    return values[0] if values else None


def _check_call_observation_sequence(
    calls: list[dict[str, Any]], observations: list[dict[str, Any]], report: dict[str, Any]
) -> None:
    pending: deque[str] = deque(str(call.get("tool_name") or "") for call in calls)
    for observation in observations:
        observed_name = observation["tool_name"]
        if not pending:
            report["errors"].append("orphan_observation_after_clean")
            continue
        expected_name = pending.popleft()
        if expected_name != observed_name:
            report["errors"].append("tool_observation_order_mismatch")
            report["sequence_mismatches"].append({"expected": expected_name, "observed": observed_name})
    if pending:
        report["errors"].append(f"missing_observations_after_clean:{len(pending)}")


def _check_final_consistency(
    calls: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    final_payload: dict[str, Any] | None,
    report: dict[str, Any],
) -> None:
    if not final_payload:
        report["errors"].append("missing_structured_final_answer")
        return
    final_text = _serialize(final_payload)
    observation_text = _serialize(observations)
    final_artifacts = set(ARTIFACT_RE.findall(final_text))
    observation_artifacts = set(ARTIFACT_RE.findall(observation_text))
    missing_artifacts = sorted(final_artifacts - observation_artifacts)
    if missing_artifacts:
        report["errors"].append("final_references_unknown_artifact")
        report["missing_artifacts"] = missing_artifacts

    last_failed = bool(observations and _payload_error(observations[-1]["payload"])[0])
    summary = str(final_payload.get("summary") or final_payload.get("steps_summary") or "")
    if last_failed and SUCCESS_CLAIM_RE.search(summary):
        report["errors"].append("final_claims_success_after_failed_critical_tool")

    final_evidence_numbers = _find_numeric(final_payload.get("evidence"), {"score", "affinity", "value", "count"})
    observation_numbers = _find_numeric([item["payload"] for item in observations])
    unsupported_numbers = [
        number
        for number in final_evidence_numbers
        if not any(abs(number - observed) <= 1e-9 for observed in observation_numbers)
    ]
    if unsupported_numbers:
        report["errors"].append("final_numeric_evidence_not_in_observations")
        report["unsupported_final_numbers"] = unsupported_numbers

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
        score = _docking_score(observation["payload"])
        if molecule and score is not None and not _payload_error(observation["payload"])[0]:
            scores[molecule] = score
    scored_ranking = [(str(molecule), scores[str(molecule)]) for molecule in ranking if str(molecule) in scores]
    if any(current[1] < previous[1] for previous, current in zip(scored_ranking, scored_ranking[1:])):
        report["errors"].append("vs_ranking_inconsistent_with_tool_scores")
        report["vs_scored_ranking"] = scored_ranking


def hard_clean(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply deterministic protection and report findings without assigning status."""
    cleaned = copy.deepcopy(sample)
    report: dict[str, Any] = {
        "errors": [],
        "warnings": [],
        "actions": [],
        "status_conflicts": [],
        "sequence_mismatches": [],
        "artifact_mappings": {},
    }
    if set(cleaned) != {"schema_version", "id", "messages"} or not isinstance(cleaned.get("messages"), list):
        report["errors"].append("invalid_top_level_shape")
        return cleaned, report
    messages, calls, observations, final_payload = _clean_protocol_messages(cleaned["messages"], report)
    _check_call_observation_sequence(calls, observations, report)
    _check_final_consistency(calls, observations, final_payload, report)
    report["errors"] = list(dict.fromkeys(report["errors"]))
    report["warnings"] = list(dict.fromkeys(report["warnings"]))
    report["counts"] = {
        "tool_calls": len(calls),
        "observations": len(observations),
        "actions": len(report["actions"]),
        "errors": len(report["errors"]),
        "warnings": len(report["warnings"]),
    }
    return cleaned, report
