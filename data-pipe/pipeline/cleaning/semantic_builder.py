from __future__ import annotations

import base64
import hashlib
import json
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.cleaning.models import SEMANTIC_SCHEMA_VERSION, semantic_schema_findings
from pipeline.cleaning.path_sanitizer import TrajectoryPathNormalizer


SKILL_RUNTIME_RE = re.compile(
    r"(?:^|[\s/\"'])(?:\.claude/skills/)?(?:L[23]_[^/\s\"']+|LR_research|auto-generated-skills|execute-molclaw-trajectory)(?:/|$|[\s\"'])",
    re.IGNORECASE,
)
L1_RE = re.compile(r"(?:^|[\s/\"'])(?:\.claude/skills/)?L1_tools(?:/|$|[\s\"'])", re.IGNORECASE)
L1_SKILL_PATH_RE = re.compile(
    r"(?P<path>(?:[^\s;&|\"']*/)?L1_tools/[A-Za-z0-9_.-]+/SKILL\.md)",
    re.IGNORECASE,
)
CLAUDE_RUNTIME_RE = re.compile(r"(?:^|[\s/\"'])CLAUDE\.md(?:$|[\s\"'])", re.IGNORECASE)
TEACHER_SIDECARS = frozenset(
    {
        "prompt.txt", "system_prompt.md", "run_meta.json",
        "run_config.json", "complete_session.jsonl", "parsed_answer.json",
        "selected_attempt_artifacts.json", "completion_report.json",
    }
)
BASE64_RE = re.compile(r"^[A-Za-z0-9+/\r\n]+={0,2}$")


class SemanticConstructionError(ValueError):
    pass


@dataclass
class AssistantResponse:
    message_id: str
    first_line: int
    content: list[dict[str, Any]] = field(default_factory=list)
    seen_tool_ids: set[str] = field(default_factory=set)
    seen_prose: set[tuple[str, str]] = field(default_factory=set)


def _content_items(event: dict[str, Any]) -> list[dict[str, Any]]:
    message = event.get("message") if isinstance(event.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    if isinstance(content, str) and content.strip():
        return [{"type": "text", "text": content}]
    return []


def _walk_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _walk_strings(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _walk_strings(item)]
    return []


def _canonical_tool_name(raw_name: str) -> str:
    name = str(raw_name or "").strip()
    if name.startswith("mcp__") and "__" in name[5:]:
        return name.rsplit("__", 1)[-1]
    return name


def _drop_reason(raw_name: str, arguments: dict[str, Any]) -> tuple[str | None, str | None]:
    canonical = _canonical_tool_name(raw_name)
    if not canonical:
        return None, "missing_tool_name"
    if canonical == "Skill":
        return None, "teacher_skill_runtime_access"
    strings = [text.replace("\\", "/") for text in _walk_strings(arguments)]
    if any(SKILL_RUNTIME_RE.search(text) or CLAUDE_RUNTIME_RE.search(text) for text in strings):
        return None, "teacher_skill_runtime_access"
    for text in strings:
        lowered = text.lower()
        tokens = re.split(r"[\s|;&<>\"'(),:{}\[\]]+", lowered)
        if any(token and token.rsplit("/", 1)[-1] in TEACHER_SIDECARS for token in tokens):
            return None, "teacher_sidecar_access"
        if ".claude/projects" in lowered:
            return None, "teacher_transcript_access"
        claude_tokens = [token for token in tokens if ".claude" in token]
        if any(not L1_RE.search(token) for token in claude_tokens):
            return None, "teacher_skill_runtime_access"
        if "/.claude/skills/" in lowered and not L1_RE.search(text):
            return None, "teacher_skill_runtime_access"
    return canonical, None


def _parse_content(value: Any) -> Any:
    if isinstance(value, list):
        text_parts = [str(item.get("text") or "") for item in value if isinstance(item, dict) and item.get("type") == "text"]
        if len(text_parts) == len(value):
            value = "\n".join(text_parts)
        else:
            return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def _looks_like_base64(text: str) -> bool:
    compact = "".join(text.split())
    if len(compact) < 512 or len(compact) % 4:
        return False
    if not BASE64_RE.fullmatch(compact):
        return False
    try:
        base64.b64decode(compact, validate=True)
    except ValueError:
        return False
    return True


def _redact_blobs(value: Any) -> tuple[Any, int]:
    if isinstance(value, str) and _looks_like_base64(value):
        return {
            "omitted": "base64_blob",
            "encoded_chars": len(value),
            "sha256": hashlib.sha256(value.encode()).hexdigest(),
        }, 1
    if isinstance(value, list):
        output, count = [], 0
        for item in value:
            clean, changed = _redact_blobs(item)
            output.append(clean)
            count += changed
        return output, count
    if isinstance(value, dict):
        output, count = {}, 0
        for key, item in value.items():
            clean, changed = _redact_blobs(item)
            output[str(key)] = clean
            count += changed
        return output, count
    return value, 0


def compact_observation(
    value: Any,
    max_chars: int,
    *,
    preserve_skill_text: bool = False,
    retained_evidence: list[Any] | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    original = json.dumps(value, ensure_ascii=False, default=str)
    redacted, blob_count = _redact_blobs(value)
    redacted_text = json.dumps(redacted, ensure_ascii=False, default=str)
    # Scientific structured observations retain candidate/metric/unit relationships.
    # Only opaque blobs are redacted; the final token gate handles large records.
    if isinstance(redacted, (dict, list)):
        return redacted, ({'method': 'blob_redaction', 'original_size_chars': len(original),
                           'blob_count': blob_count} if blob_count else None)
    limit = max_chars * 4 if preserve_skill_text else max_chars
    if len(redacted_text) <= limit:
        if not blob_count:
            return redacted, None
        return redacted, {
            "method": "blob_redaction",
            "original_size_chars": len(original),
            "blob_count": blob_count,
        }
    text = str(redacted)
    preview = max(256, min(max_chars // 2, 2000))
    compacted = {
        "compacted": True,
        "original_size_chars": len(original),
        "text_head": text[:preview],
        "text_tail": text[-preview:] if len(text) > preview else "",
    }
    method = "text_head_tail"
    if retained_evidence:
        compacted["downstream_evidence"] = retained_evidence
    return compacted, {
        "method": method,
        "original_size_chars": len(original),
        "blob_count": blob_count,
    }


def _status(payload: Any, event_is_error: bool) -> tuple[str, bool]:
    body = payload if isinstance(payload, dict) else {}
    status = str(body.get("status") or body.get("state") or "").strip().lower()
    meaningful_error = body.get("error") not in (None, "", False, [], {})
    is_error = event_is_error or body.get("is_error") is True or status in {
        "error", "failed", "failure", "timeout", "timed_out", "invalid",
    } or meaningful_error
    return ("error" if is_error else status or "success"), is_error


def _unwrap_final(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"<answer>\s*([\s\S]*?)\s*</answer>", stripped, flags=re.IGNORECASE)
    stripped = match.group(1).strip() if match else stripped
    fenced = re.fullmatch(r"```(?:json|JSON)?\s*\n([\s\S]*?)\n```", stripped)
    return fenced.group(1).strip() if fenced else stripped


def _group_assistant_responses(events: list[dict[str, Any]]) -> list[AssistantResponse]:
    grouped: OrderedDict[str, AssistantResponse] = OrderedDict()
    for event in events:
        if str(event.get("type") or "").lower() != "assistant":
            continue
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        message_id = str(message.get("id") or "").strip()
        if not message_id:
            raise SemanticConstructionError(f"assistant event at line {event.get('_line_no')} lacks message.id")
        response = grouped.setdefault(message_id, AssistantResponse(message_id, int(event.get("_line_no") or -1)))
        for item in _content_items(event):
            kind = str(item.get("type") or "")
            if kind == "tool_use":
                call_id = str(item.get("id") or "").strip()
                if not call_id:
                    raise SemanticConstructionError(f"tool_use in {message_id} lacks id")
                if call_id in response.seen_tool_ids:
                    continue
                response.seen_tool_ids.add(call_id)
            elif kind in {"thinking", "text"}:
                prose = str(item.get("thinking") if kind == "thinking" else item.get("text") or "")
                fingerprint = (kind, prose)
                if fingerprint in response.seen_prose:
                    continue
                response.seen_prose.add(fingerprint)
            response.content.append(item)
    return list(grouped.values())


def _index_results(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for event in events:
        if str(event.get("type") or "").lower() != "user":
            continue
        for item in _content_items(event):
            if str(item.get("type") or "") != "tool_result":
                continue
            call_id = str(item.get("tool_use_id") or "").strip()
            if not call_id:
                raise SemanticConstructionError(f"tool_result at line {event.get('_line_no')} lacks tool_use_id")
            if call_id in indexed:
                raise SemanticConstructionError(f"duplicate tool_result for {call_id}")
            indexed[call_id] = item
    return indexed


def _has_real_pwd_call(responses: list[AssistantResponse]) -> bool:
    for response in responses:
        for item in response.content:
            if item.get("type") != "tool_use" or _canonical_tool_name(str(item.get("name") or "")) != "Bash":
                continue
            arguments = item.get("input") if isinstance(item.get("input"), dict) else {}
            command = str(arguments.get("command") or "").strip()
            if command == "pwd":
                return True
    return False


def _mixed_l1_targets(raw_name: str, arguments: dict[str, Any]) -> list[str]:
    if _canonical_tool_name(raw_name) != "Bash":
        return []
    command = str(arguments.get("command") or "").replace("\\", "/")
    if not SKILL_RUNTIME_RE.search(command):
        return []
    targets: list[str] = []
    for match in L1_SKILL_PATH_RE.finditer(command):
        path = match.group("path")
        marker = "L1_tools/"
        relative = marker + path.split(marker, 1)[1]
        if relative not in targets:
            targets.append(relative)
    return targets


def _extract_l1_result_text(workspace: Path, target: str, result_content: Any) -> str:
    relative = target.split("L1_tools/", 1)[1]
    source = workspace / ".claude" / "skills" / "L1_tools" / relative
    if not source.is_file():
        raise SemanticConstructionError(f"mixed Bash L1 snapshot is missing: {source}")
    expected = source.read_text(encoding="utf-8").rstrip()
    result_text = str(result_content or "")
    start = result_text.find(expected)
    if start < 0:
        raise SemanticConstructionError(f"mixed Bash result lacks L1 section for: {target}")
    return result_text[start : start + len(expected)]


def _scalar_leaves(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [leaf for child in value.values() for leaf in _scalar_leaves(child)]
    if isinstance(value, list):
        return [leaf for child in value for leaf in _scalar_leaves(child)]
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return [value]
    return []


def _downstream_evidence(value: Any, downstream: str) -> list[Any]:
    retained: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for scalar in _scalar_leaves(value):
        rendered = str(scalar)
        if isinstance(scalar, str):
            if len(rendered) < 3 or len(rendered) > 2048 or _looks_like_base64(rendered):
                continue
            reused = rendered in downstream
        else:
            reused = bool(re.search(rf"(?<![\w.]){re.escape(rendered)}(?![\w.])", downstream))
        fingerprint = (type(scalar).__name__, rendered)
        if reused and fingerprint not in seen:
            retained.append(scalar)
            seen.add(fingerprint)
            if len(retained) >= 100:
                break
    return retained


def build_semantic_trajectory(
    events: list[dict[str, Any]],
    *,
    record_id: str,
    user_task: str,
    workspace: Path,
    task_type: str,
    source_session_sha256: str,
    max_observation_chars: int = 6000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    responses = _group_assistant_responses(events)
    results = _index_results(events)
    path_normalizer = TrajectoryPathNormalizer(
        workspace=workspace,
        source_id=record_id,
        preserve_pwd_chain=_has_real_pwd_call(responses),
    )
    output: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    derived_calls: list[dict[str, Any]] = []
    retained_hist: Counter[str] = Counter()

    for response_index, response in enumerate(responses):
        raw_calls = [item for item in response.content if str(item.get("type") or "") == "tool_use"]
        retained_calls: list[dict[str, Any]] = []
        call_details: list[tuple[dict[str, Any], bool, dict[str, Any] | None, Any | None]] = []
        for item in raw_calls:
            raw_name = str(item.get("name") or "")
            arguments = item.get("input") if isinstance(item.get("input"), dict) else {}
            call_id = str(item.get("id") or "")
            l1_targets = _mixed_l1_targets(raw_name, arguments)
            if l1_targets:
                raw_result = results.get(call_id)
                if raw_result is None:
                    raise SemanticConstructionError(f"mixed Bash call {call_id} has no tool_result")
                derived_ids: list[str] = []
                for ordinal, target in enumerate(l1_targets, 1):
                    derived_id = f"{call_id}__l1_{ordinal:02d}"
                    derived_ids.append(derived_id)
                    call = {
                        "name": "Read",
                        "arguments": {"file_path": f".agents/skills/{target.split('L1_tools/', 1)[1]}"},
                        "source_tool_use_id": derived_id,
                    }
                    retained_calls.append(call)
                    call_details.append(
                        (
                            call,
                            True,
                            None,
                            _extract_l1_result_text(workspace, target, raw_result.get("content")),
                        )
                    )
                    retained_hist["Read"] += 1
                derived_calls.append(
                    {
                        "source_message_id": response.message_id,
                        "source_tool_use_id": call_id,
                        "derived_tool_use_ids": derived_ids,
                        "operation": "mixed_bash_to_l1_reads",
                        "l1_targets": [f"skills/{target}" for target in l1_targets],
                    }
                )
                dropped.append(
                    {
                        "source_message_id": response.message_id,
                        "source_tool_use_id": call_id,
                        "raw_name": raw_name,
                        "reason": "mixed_teacher_bash_projected_to_l1_reads",
                    }
                )
                continue

            canonical, reason = _drop_reason(raw_name, arguments)
            if reason:
                dropped.append({
                    "source_message_id": response.message_id,
                    "source_tool_use_id": call_id,
                    "raw_name": raw_name,
                    "reason": reason,
                })
                continue
            assert canonical is not None
            call = {
                "name": canonical,
                "arguments": path_normalizer.normalize(arguments),
                "source_tool_use_id": call_id,
            }
            retained_calls.append(call)
            call_details.append(
                (
                    call,
                    any(L1_RE.search(text.replace("\\", "/")) for text in _walk_strings(arguments)),
                    results.get(call_id),
                    None,
                )
            )
            retained_hist[canonical] += 1

        thinking: list[str] = []
        visible: list[str] = []
        for item in response.content:
            kind = str(item.get("type") or "")
            if kind == "thinking":
                text = str(item.get("thinking") or "").strip()
                if text:
                    thinking.append(text)
            elif kind == "text":
                text = str(item.get("text") or "").strip()
                if text:
                    visible.append(text)

        if raw_calls:
            if not retained_calls:
                dropped.append({
                    "source_message_id": response.message_id,
                    "reason": "all_calls_filtered",
                })
                continue
            reasoning = path_normalizer.normalize("\n\n".join(thinking + visible))
            output.append({
                "type": "assistant_decision",
                "source_message_id": response.message_id,
                "reasoning": reasoning,
                "tool_calls": retained_calls,
                "final_answer": None,
            })
            downstream = json.dumps(
                path_normalizer.normalize(
                    [item.content for item in responses[response_index + 1 :]]
                ),
                ensure_ascii=False,
                default=str,
            )
            for call, is_l1_read, raw_result, content_override in call_details:
                call_id = call["source_tool_use_id"]
                if raw_result is None and content_override is None:
                    raise SemanticConstructionError(f"retained call {call_id} has no tool_result")
                result = raw_result or {}
                raw_content = content_override if content_override is not None else result.get("content")
                parsed = path_normalizer.normalize(_parse_content(raw_content))
                retained_evidence = _downstream_evidence(parsed, downstream)
                compacted, compaction = compact_observation(
                    parsed,
                    max_observation_chars,
                    preserve_skill_text=is_l1_read,
                    retained_evidence=retained_evidence,
                )
                status, is_error = _status(parsed, bool(result.get("is_error")))
                observation = {
                    "type": "tool_observation",
                    "name": call["name"],
                    "source_tool_use_id": call_id,
                    "status": status,
                    "is_error": is_error,
                    "content": compacted,
                }
                if compaction:
                    observation["compaction"] = compaction
                output.append(observation)
            continue

        if response_index != len(responses) - 1:
            if thinking or visible:
                raise SemanticConstructionError(
                    f"non-terminal assistant response {response.message_id} has no tool calls"
                )
            continue
        final_answer = _unwrap_final("\n\n".join(visible))
        if not final_answer:
            raise SemanticConstructionError("terminal assistant response has no visible final answer")
        output.append({
            "type": "assistant_decision",
            "source_message_id": response.message_id,
            "reasoning": path_normalizer.normalize("\n\n".join(thinking)),
            "tool_calls": [],
            "final_answer": path_normalizer.normalize(final_answer),
        })

    if not output or output[-1].get("final_answer") in (None, ""):
        raise SemanticConstructionError("trajectory has no terminal assistant decision")
    if not any(event.get("tool_calls") for event in output if event.get("type") == "assistant_decision"):
        raise SemanticConstructionError("trajectory has no retained tool call")

    record = {
        "schema_version": SEMANTIC_SCHEMA_VERSION,
        "id": record_id,
        "user_task": path_normalizer.normalize(user_task),
        "events": output,
        "metadata": {
            "task_type": task_type,
            "source_session_sha256": source_session_sha256,
        },
    }
    findings = semantic_schema_findings(record)
    if findings:
        raise SemanticConstructionError(f"semantic schema validation failed: {findings}")
    decision_count = sum(event["type"] == "assistant_decision" for event in output)
    call_groups = [
        len(event["tool_calls"])
        for event in output
        if event["type"] == "assistant_decision" and event["tool_calls"]
    ]
    return record, {
        "assistant_response_count": len(responses),
        "retained_decision_count": decision_count,
        "retained_tool_call_count": sum(call_groups),
        "retained_tool_call_groups": call_groups,
        "retained_tool_hist": dict(retained_hist),
        "dropped": dropped,
        "derived_calls": derived_calls,
        "path_normalization": path_normalizer.audit(),
    }
