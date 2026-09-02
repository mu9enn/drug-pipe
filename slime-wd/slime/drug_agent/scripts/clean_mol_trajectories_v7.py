#!/usr/bin/env python3
"""Clean v6 molecular ReAct trajectories for the v7 offline release."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from drug_agent.protocol.react_protocol import parse_react_sequence
from drug_agent.protocol.toolrl_turn import serialize_decision, split_assistant_segments
from drug_agent.utils import normalize_tool_name


ARTIFACT_REF_RE = re.compile(r"<artifact:[^>\r\n]+>")
FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)
LOCAL_ARTIFACT_RE = re.compile(r"<artifact:local/([^>\r\n]+)>")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _artifact_refs(text: str) -> set[str]:
    return set(ARTIFACT_REF_RE.findall(text))


def _unwrap_artifact_ref(ref: str) -> str:
    payload = ref[len("<artifact:") : -1]
    return payload.split("/", 1)[1] if "/" in payload else payload


def _replace_refs(value: Any, refs: set[str]) -> Any:
    if isinstance(value, str):
        out = value
        for ref in refs:
            out = out.replace(ref, _unwrap_artifact_ref(ref))
        return out
    if isinstance(value, list):
        return [_replace_refs(item, refs) for item in value]
    if isinstance(value, dict):
        return {key: _replace_refs(item, refs) for key, item in value.items()}
    return value


def _observation_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    if message.get("role") != "user":
        return []
    content = str(message.get("content") or "")
    if not content.lstrip().startswith("<observation"):
        return []
    parsed = parse_react_sequence(content, role="user")
    if not parsed.get("ok"):
        raise ValueError(str(parsed.get("error_message") or "invalid observation message"))
    blocks = parsed.get("blocks") if isinstance(parsed.get("blocks"), list) else []
    return [block for block in blocks if block.get("kind") == "observation"]


def _failed_observation_for(message: dict[str, Any], tool_name: str) -> bool:
    blocks = _observation_blocks(message)
    if not blocks:
        return False
    expected = normalize_tool_name(tool_name)
    for block in blocks:
        if normalize_tool_name(block.get("tool_name")) != expected:
            return False
        payload = block.get("payload") if isinstance(block.get("payload"), dict) else {}
        failed = (
            payload.get("is_error") is True
            or payload.get("ok") is False
            or str(payload.get("status") or "").lower() == "error"
        )
        if not failed:
            return False
    return True


def _clean_final(payload: dict[str, Any], stats: Counter[str]) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    task_type = str(out.get("task_type") or "").lower()
    if task_type in {"pf", "vs"} and isinstance(out.get("selected_smiles"), str):
        out["selected_smiles"] = [out["selected_smiles"]]
        stats[f"{task_type}_selected_smiles_normalized"] += 1

    evidence = out.get("evidence")
    if isinstance(evidence, list):
        deduplicated = []
        seen: set[str] = set()
        for item in evidence:
            key = _canonical(item)
            if key in seen:
                stats["evidence_duplicates_removed"] += 1
                continue
            seen.add(key)
            deduplicated.append(item)
        out["evidence"] = deduplicated
    return out


def _preclean_legacy_finals(content: str, stats: Counter[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        payload = json.loads(match.group(1))
        if not isinstance(payload, dict):
            raise ValueError("final_answer must contain an object")
        return "<final_answer>" + _canonical(_clean_final(payload, stats)) + "</final_answer>"

    return FINAL_ANSWER_RE.sub(replace, content)


def _clean_assistant_content(content: str, stats: Counter[str]) -> tuple[str, tuple[str, str] | None]:
    content, local_count = LOCAL_ARTIFACT_RE.subn(lambda match: match.group(1), content)
    stats["local_artifact_refs_unwrapped"] += local_count
    content = _preclean_legacy_finals(content, stats)
    segments = split_assistant_segments(content)
    single_call_key: tuple[str, str] | None = None
    action_segments = [segment for segment in segments if segment.get("is_action")]
    if len(action_segments) == 1 and len(action_segments[0].get("tool_calls") or []) == 1:
        call = action_segments[0]["tool_calls"][0]
        single_call_key = (
            normalize_tool_name(call.get("tool_name")),
            _canonical(call.get("arguments") if isinstance(call.get("arguments"), dict) else {}),
        )

    rendered = []
    for segment in segments:
        if not segment.get("is_action"):
            rendered.append("<thought>" + "\n\n".join(segment.get("thoughts") or []) + "</thought>")
            continue
        final_answer = segment.get("final_answer")
        if isinstance(final_answer, dict):
            final_answer = _clean_final(final_answer, stats)
        rendered.append(
            serialize_decision(
                thoughts=segment.get("thoughts") or [],
                tool_calls=segment.get("tool_calls") or [],
                final_answer=final_answer,
            )
        )
    return "\n".join(rendered), single_call_key


def _sanitize_unresolved_artifacts(
    content: str,
    available_artifacts: set[str],
    stats: Counter[str],
) -> str:
    segments = split_assistant_segments(content)
    rendered = []
    for segment in segments:
        thoughts = []
        for thought in segment.get("thoughts") or []:
            missing = _artifact_refs(thought) - available_artifacts
            stats["unresolved_artifact_refs_unwrapped_from_prose"] += len(missing)
            thoughts.append(_replace_refs(thought, missing))

        calls = []
        for source_call in segment.get("tool_calls") or []:
            call = copy.deepcopy(source_call)
            arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            for name, value in list(arguments.items()):
                refs = _artifact_refs(json.dumps(value, ensure_ascii=False))
                missing = refs - available_artifacts
                if not missing:
                    continue
                arguments[name] = _replace_refs(value, missing)
                stats["unresolved_artifact_refs_unwrapped_from_actions"] += len(missing)
            call["arguments"] = arguments
            calls.append(call)

        final_answer = segment.get("final_answer")
        if final_answer is not None:
            missing = _artifact_refs(json.dumps(final_answer, ensure_ascii=False)) - available_artifacts
            final_answer = _replace_refs(final_answer, missing)
            stats["unresolved_artifact_refs_unwrapped_from_prose"] += len(missing)

        if segment.get("is_action"):
            rendered.append(
                serialize_decision(
                    thoughts=thoughts,
                    tool_calls=calls,
                    final_answer=final_answer,
                )
            )
        else:
            rendered.append("<thought>" + "\n\n".join(thoughts) + "</thought>")
    return "\n".join(rendered)


def clean_trajectory(record: dict[str, Any]) -> tuple[dict[str, Any] | None, Counter[str], list[str]]:
    stats: Counter[str] = Counter()
    out = copy.deepcopy(record)
    messages = out.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("trajectory must contain messages")

    cleaned_messages: list[dict[str, Any]] = []
    active_failed_key: tuple[str, str] | None = None
    identical_failed_attempts = 0
    index = 0
    while index < len(messages):
        message = copy.deepcopy(messages[index])
        if not isinstance(message, dict):
            raise ValueError(f"message {index} is not an object")
        if _observation_blocks(message):
            message["step_loss_mask"] = 0
            stats["observation_messages_masked"] += 1

        if message.get("role") != "assistant":
            cleaned_messages.append(message)
            index += 1
            continue

        message["content"], call_key = _clean_assistant_content(str(message.get("content") or ""), stats)
        next_message = messages[index + 1] if index + 1 < len(messages) and isinstance(messages[index + 1], dict) else None
        failed = bool(
            call_key
            and next_message
            and _failed_observation_for(next_message, call_key[0])
        )
        if failed:
            if call_key == active_failed_key:
                identical_failed_attempts += 1
            else:
                active_failed_key = call_key
                identical_failed_attempts = 1
            if identical_failed_attempts > 2:
                stats["identical_failed_retry_pairs_removed"] += 1
                index += 2
                continue
        else:
            active_failed_key = None
            identical_failed_attempts = 0

        cleaned_messages.append(message)
        index += 1

    available_artifacts: set[str] = set()
    for message in cleaned_messages:
        if _observation_blocks(message):
            available_artifacts.update(_artifact_refs(str(message.get("content") or "")))
            continue
        if message.get("role") == "assistant":
            message["content"] = _sanitize_unresolved_artifacts(
                str(message.get("content") or ""), available_artifacts, stats
            )

    out["messages"] = cleaned_messages
    return out, stats, []


def clean_dataset(input_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise ValueError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    totals: Counter[str] = Counter()
    dropped: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            totals["records_input"] += 1
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} is not an object")
            cleaned, stats, unresolved = clean_trajectory(record)
            totals.update(stats)
            if cleaned is None:
                dropped.append({"id": record.get("id"), "unresolved_artifacts": unresolved})
                continue
            target.write(json.dumps(cleaned, ensure_ascii=False, separators=(",", ":")) + "\n")
            totals["records_output"] += 1

    report = {
        "schema_version": "drug_agent_mol_trajectory_cleanup_v1",
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "counts": dict(sorted(totals.items())),
        "dropped": dropped,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(clean_dataset(args.input, args.output, args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
