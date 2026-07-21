from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

try:
    from pipeline.cleaning.artifacts import inspect_observation_status, sanitize_artifact_paths
except ImportError:
    from cleaning.artifacts import inspect_observation_status, sanitize_artifact_paths


MOLCLAW_PREFIX = "mcp__molclaw-scp__"
CANONICAL_SYSTEM_PROMPT = """You are a scientific agent operating under the canonical ReAct protocol.
Use only real MolClaw calls from the recorded execution. Write scientific reasoning inside
<thought>...</thought>, calls inside <tool_call>...</tool_call>, recorded results inside
<observation>...</observation>, and the grounded result inside <final_answer>...</final_answer>.
Never invent a tool result. Observations must come from execution history, and the final answer
must be supported by the task result and recorded observations."""
IMPORTANT_OBSERVATION_KEYS = (
    "status",
    "ok",
    "error",
    "message",
    "msg",
    "summary",
    "score",
    "scores",
    "value",
    "count",
    "pocket_count",
    "top_pocket",
    "ranking",
    "result",
    "results",
    "artifact",
    "output_path",
    "output_file",
)
OUTPUT_ARTIFACT_KEYS = ("output_file", "output_path", "artifact")
DELIVERABLE_FILE_RE = re.compile(
    r"(?i)(?:<artifact:[^>]+>|(?<![\w.-])[\w.-]+\.(?:pdb|cif|mmcif|sdf|mol|mol2|pdbqt|csv|tsv|json|npy|npz|pt|pkl))"
)
ARTIFACT_REF_RE = re.compile(r"<artifact:[^>]+>")


def bare_tool_name(raw_name: str) -> str | None:
    name = str(raw_name or "").strip()
    if name.startswith(MOLCLAW_PREFIX):
        return name[len(MOLCLAW_PREFIX) :].strip() or None
    return None


def _parse_content(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def _compact_observation(value: Any, max_chars: int) -> tuple[Any, bool, int]:
    serialized = json.dumps(value, ensure_ascii=False, default=str)
    original_chars = len(serialized)
    if original_chars <= max_chars:
        return value, False, original_chars
    if isinstance(value, dict):
        summary = {key: value[key] for key in IMPORTANT_OBSERVATION_KEYS if key in value}
        omitted = sorted(str(key) for key in value if key not in summary)
        compacted: dict[str, Any] = {
            "compacted": True,
            "original_size_chars": original_chars,
            "summary": summary,
        }
        if omitted:
            compacted["omitted_keys"] = omitted[:50]
        return compacted, True, original_chars
    if isinstance(value, list):
        return {
            "compacted": True,
            "original_size_chars": original_chars,
            "item_count": len(value),
            "items_preview": value[:3],
        }, True, original_chars
    return {
        "compacted": True,
        "original_size_chars": original_chars,
        "text_preview": str(value)[: min(max_chars, 1000)],
    }, True, original_chars


def _content_items(event: dict[str, Any]) -> list[dict[str, Any]]:
    message = event.get("message") if isinstance(event.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    if isinstance(content, str) and content.strip():
        return [{"type": "text", "text": content.strip()}]
    return []


def _raw_system_prompt(events: list[dict[str, Any]]) -> str:
    for event in events:
        if str(event.get("type") or "").lower() != "system":
            continue
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = "\n".join(
                str(item.get("text") or item.get("thinking") or "").strip()
                for item in content
                if isinstance(item, dict)
            ).strip()
        else:
            text = ""
        if all(tag in text for tag in ("<thought>", "<tool_call>", "<observation>", "<final_answer>")):
            return text
    return ""


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _answer_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        for key in ("ranking", "ranked_smiles", "selected_smiles", "answer_smiles", "answer", "result"):
            if key in value:
                values = _answer_values(value[key])
                if values:
                    return values
    return []


def _evidence_summary(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for observation in observations[-8:]:
        content = observation.get("content")
        item: dict[str, Any] = {
            "tool_name": observation["tool_name"],
            "status": observation["status"],
        }
        if isinstance(content, dict):
            key_values = {
                key: content[key]
                for key in ("summary", "message", "msg", "score", "scores", "value", "count", "artifact", "output_path")
                if key in content
            }
            if key_values:
                item["key_values"] = key_values
        evidence.append(item)
    return evidence


def _artifact_refs(value: Any) -> list[str]:
    return ARTIFACT_REF_RE.findall(_serialize(value))


def _values_for_key(value: Any, key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(value, dict):
        for item_key, item in value.items():
            if str(item_key) == key:
                values.append(item)
            values.extend(_values_for_key(item, key))
    elif isinstance(value, list):
        for item in value:
            values.extend(_values_for_key(item, key))
    return values


def _primary_output_artifact(observations: list[dict[str, Any]]) -> str | None:
    preferred: list[str] = []
    fallback: list[str] = []
    for observation in observations:
        if observation.get("is_error"):
            continue
        content = observation.get("content")
        fallback.extend(_artifact_refs(content))
        for key in OUTPUT_ARTIFACT_KEYS:
            for value in _values_for_key(content, key):
                preferred.extend(_artifact_refs(value))
    candidates = preferred or fallback
    return candidates[-1] if candidates else None


def _is_file_delivery_answer(value: Any) -> bool:
    return bool(DELIVERABLE_FILE_RE.search(_serialize(value)))


def build_final_payload(task: str, final_answer: Any, final_text: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
    value = final_answer
    if value in (None, "", []):
        value = final_text.strip()
    values = _answer_values(value)
    summary = final_text.strip()
    evidence = _evidence_summary(observations)
    if task == "vs":
        payload: dict[str, Any] = {
            "task_type": "vs",
            "ranked_smiles": values,
            "selected_smiles": values[0] if values else "",
        }
    elif task == "ac":
        payload = {"task_type": "ac", "answer_smiles": values[0] if values else ""}
    elif task == "pf":
        payload = {"task_type": "pf", "selected_smiles": values}
    else:
        primary_artifact = _primary_output_artifact(observations)
        result = (
            primary_artifact
            if task in {"kg", "e2e"} and primary_artifact and _is_file_delivery_answer(value)
            else value
        )
        payload = {"task_type": task, "result": result}
    if summary:
        payload["summary"] = summary
    if evidence:
        payload["evidence"] = evidence
    return payload


def reconstruct_react_messages(
    events: list[dict[str, Any]],
    *,
    question_text: str,
    final_answer: Any,
    task: str = "kg",
    max_observation_chars: int = 6000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path_map: dict[str, str] = {}
    system_prompt = _raw_system_prompt(events) or CANONICAL_SYSTEM_PROMPT
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sanitize_artifact_paths(system_prompt, path_map), "step_loss_mask": 0},
        {"role": "user", "content": sanitize_artifact_paths(question_text, path_map), "step_loss_mask": 0},
    ]
    retained_calls: dict[str, dict[str, str]] = {}
    observed_calls: set[str] = set()
    tool_hist: Counter[str] = Counter()
    observations_for_final: list[dict[str, Any]] = []
    raw_tool_names: dict[str, str] = {}
    dropped_non_molclaw_calls = 0
    orphan_observations = 0
    compacted_observations = 0
    error_status_conflicts: list[dict[str, Any]] = []
    assistant_text_after_last_observation: list[str] = []

    def append_turn(role: str, content: str, loss_mask: int) -> None:
        text = content.strip()
        if not text:
            return
        if len(messages) > 2 and messages[-1]["role"] == role:
            messages[-1]["content"] = f"{messages[-1]['content']}\n{text}"
        else:
            messages.append({"role": role, "content": text, "step_loss_mask": loss_mask})

    for event_index, event in enumerate(events):
        event_type = str(event.get("type") or "").lower()
        if event_type == "assistant":
            rendered: list[str] = []
            event_texts: list[str] = []
            for item in _content_items(event):
                item_type = str(item.get("type") or "")
                if item_type in {"thinking", "text"}:
                    text = item.get("thinking") if item_type == "thinking" else item.get("text")
                    if isinstance(text, str) and text.strip():
                        clean = sanitize_artifact_paths(text.strip(), path_map)
                        event_texts.append(clean)
                        rendered.append(f"<thought>{clean}</thought>")
                elif item_type == "tool_use":
                    raw_name = str(item.get("name") or "")
                    name = bare_tool_name(raw_name)
                    call_id = str(item.get("id") or f"event-{event_index}-call-{len(retained_calls)}")
                    if not name:
                        dropped_non_molclaw_calls += 1
                        continue
                    arguments = item.get("input") if isinstance(item.get("input"), dict) else {}
                    arguments = sanitize_artifact_paths(arguments, path_map)
                    retained_calls[call_id] = {"name": name, "raw_name": raw_name}
                    raw_tool_names[raw_name] = name
                    tool_hist[name] += 1
                    rendered.append(
                        f"<tool_call>{_serialize({'tool_name': name, 'arguments': arguments})}</tool_call>"
                    )
            if rendered:
                append_turn("assistant", "\n".join(rendered), 1)
            if event_texts and retained_calls:
                assistant_text_after_last_observation.extend(event_texts)
        elif event_type == "user":
            rendered_observations: list[str] = []
            for item in _content_items(event):
                if str(item.get("type") or "") != "tool_result":
                    continue
                call_id = str(item.get("tool_use_id") or "")
                call = retained_calls.get(call_id)
                if not call:
                    orphan_observations += 1
                    continue
                parsed = sanitize_artifact_paths(_parse_content(item.get("content")), path_map)
                raw_is_error = bool(item.get("is_error"))
                status_inspection = inspect_observation_status(parsed, event_is_error=raw_is_error)
                if status_inspection["conflict"]:
                    error_status_conflicts.append(
                        {
                            "event_index": event_index,
                            "tool_use_id": call_id,
                            "tool_name": call["name"],
                            "event_is_error": raw_is_error,
                            "content_status": status_inspection["content_status"],
                        }
                    )
                compacted, changed, original_chars = _compact_observation(parsed, max_observation_chars)
                compacted_observations += int(changed)
                observation = {
                    "tool_name": call["name"],
                    "status": status_inspection["status"],
                    "is_error": status_inspection["is_error"],
                    "content": compacted,
                }
                if changed:
                    observation["compaction"] = {
                        "method": "structured_summary",
                        "original_size_chars": original_chars,
                    }
                observations_for_final.append(observation)
                observed_calls.add(call_id)
                rendered_observations.append(
                    f'<observation tool_name="{call["name"]}">{_serialize(observation)}</observation>'
                )
            if rendered_observations:
                append_turn("user", "\n".join(rendered_observations), 0)
                assistant_text_after_last_observation = []
        elif event_type == "result":
            result_text = str(event.get("result") or "").strip()
            if result_text:
                assistant_text_after_last_observation.append(sanitize_artifact_paths(result_text, path_map))

    final_text = "\n\n".join(assistant_text_after_last_observation).strip()
    payload = sanitize_artifact_paths(build_final_payload(task, final_answer, final_text, observations_for_final), path_map)
    messages.append(
        {
            "role": "assistant",
            "content": f"<final_answer>{_serialize(payload)}</final_answer>",
            "step_loss_mask": 1,
        }
    )
    messages = sanitize_artifact_paths(messages, path_map)
    resolved_final_answer = final_answer
    if resolved_final_answer in (None, "", []):
        resolved_final_answer = payload.get("result") or payload.get("ranked_smiles") or payload.get("answer_smiles")
    return messages, {
        "molclaw_usage_count": len(retained_calls),
        "molclaw_usage_computation_count": 1,
        "tool_name_hist": dict(tool_hist),
        "observed_tool_call_count": len(observed_calls),
        "missing_observation_count": len(set(retained_calls) - observed_calls),
        "dropped_non_molclaw_call_count": dropped_non_molclaw_calls,
        "orphan_observation_count": orphan_observations,
        "compacted_observation_count": compacted_observations,
        "error_status_conflicts": error_status_conflicts,
        "raw_tool_name_map": raw_tool_names,
        "artifact_mappings": path_map,
        "used_raw_system_prompt": bool(_raw_system_prompt(events)),
        "resolved_final_answer": resolved_final_answer,
        "final_payload": payload,
    }
