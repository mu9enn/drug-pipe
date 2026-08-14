from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.decision_extractor import iter_react_decisions, parse_assistant_decision
from drug_agent.protocol.react_protocol import parse_react_sequence
from drug_agent.tools.local_tools import LOCAL_TOOL_NAMES
from drug_agent.toolrl.parse_tool_calls import (
    default_molclaw_allowlist,
    is_supported_decision_name,
    supported_training_tool_names,
)
from drug_agent.utils import ensure_dir, read_jsonl, to_jsonable, write_json, write_jsonl


def _iter_json_files(path: Path) -> Iterable[Path]:
    if path.is_dir():
        for suffix in ("*.json", "*.jsonl"):
            yield from sorted(path.glob(suffix))
    else:
        yield path


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if path.is_dir():
        for file_path in _iter_json_files(path):
            records.extend(_load_records(file_path))
        return records

    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        raise ValueError(f"Unsupported JSON payload in {path}")
    raise ValueError(f"Unsupported input file: {path}")


def _infer_task_type(record: dict[str, Any], source_path: str) -> str | None:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    if isinstance(metadata.get("task"), str) and metadata["task"].strip():
        return metadata["task"].strip()
    for candidate in (record.get("task_type"), record.get("task")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    record_id = str(record.get("id") or "")
    match = re.search(r"(?:mcp_sft|react)_(?P<task_type>vs|ac|pf|kg|e2e)_", record_id)
    if match:
        return match.group("task_type")

    path_name = Path(source_path).name
    match = re.search(r"(?:mcp_sft|react)_(?P<task_type>vs|ac|pf|kg|e2e)_", path_name)
    if match:
        return match.group("task_type")
    return None


def _parse_target_assistant(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    parsed = parse_assistant_decision(content)
    tool_calls = parsed.get("tool_calls") if isinstance(parsed.get("tool_calls"), list) else []
    return {
        "role": "assistant",
        "content": content,
        "parsed": to_jsonable(parsed),
        "tool_call_count": len(tool_calls),
    }


def _canonical_call_set(tool_calls: list[dict[str, Any]]) -> str | None:
    """Return an order-insensitive identity for an exact set of tool calls."""
    if not tool_calls:
        return None
    payloads = []
    for call in tool_calls:
        payloads.append(
            json.dumps(
                {
                    "tool_name": str(call.get("tool_name") or ""),
                    "arguments": to_jsonable(call.get("arguments") if isinstance(call.get("arguments"), dict) else {}),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return json.dumps(sorted(payloads), ensure_ascii=False, separators=(",", ":"))


def _observation_outcome(messages: list[dict[str, Any]], assistant_index: int) -> dict[str, Any]:
    """Classify the result immediately following one assistant action.

    The classification is deliberately conservative.  A retry is only called
    redundant when the earlier call has an explicit usable success; unknown,
    timeout, and error results remain trainable.
    """
    payloads: list[dict[str, Any]] = []
    for message in messages[assistant_index + 1 :]:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            break
        if message.get("role") != "user":
            continue
        parsed = parse_react_sequence(str(message.get("content") or ""), role="user")
        if not parsed.get("ok"):
            continue
        for block in parsed.get("blocks") or []:
            if block.get("kind") == "observation" and isinstance(block.get("payload"), dict):
                payloads.append(block["payload"])

    if not payloads:
        return {"status": "missing", "usable_success": False, "observation_count": 0}
    failure_tokens = {"error", "failed", "failure", "timeout", "timed_out", "cancelled"}
    explicit_failure = False
    explicit_success = False
    for payload in payloads:
        status = str(payload.get("status") or "").strip().lower()
        explicit_failure = explicit_failure or payload.get("ok") is False or status in failure_tokens
        explicit_success = explicit_success or payload.get("ok") is True or status in {"ok", "success", "succeeded", "completed"}
    usable = bool(explicit_success and not explicit_failure)
    return {
        "status": "success" if usable else ("failure" if explicit_failure else "unknown"),
        "usable_success": usable,
        "observation_count": len(payloads),
    }


def _decision_annotations(messages: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Annotate valid trajectory decisions before they are expanded to rows."""
    decisions = [
        item
        for item in iter_react_decisions(messages)
        if item["parse"].get("ok") and item.get("decision_type") in {"tool_call", "final_answer"}
    ]
    seen_calls: dict[str, tuple[int, int]] = {}
    annotations: dict[int, dict[str, Any]] = {}
    successful_state_change_ordinals: set[int] = set()
    trajectory_has_no_progress_repeat = False
    for ordinal, decision in enumerate(decisions):
        assistant_index = int(decision["assistant_index"])
        decision_type = str(decision.get("decision_type") or "")
        if decision_type == "final_answer":
            role = "final"
        else:
            role = "tool_step"

        signature = _canonical_call_set(
            [item for item in (decision.get("tool_calls") or []) if isinstance(item, dict)]
        )
        previous = seen_calls.get(signature) if signature is not None else None
        repeat_of = previous[0] if previous is not None else None
        prior_ordinal = previous[1] if previous is not None else None
        prior_usable = bool(
            repeat_of is not None
            and annotations.get(repeat_of, {}).get("observation_usable_success")
        )
        intervening_state_change = bool(
            prior_ordinal is not None
            and any(prior_ordinal < item < ordinal for item in successful_state_change_ordinals)
        )
        no_progress_repeat = bool(previous is not None and prior_usable and not intervening_state_change)
        if no_progress_repeat:
            trajectory_has_no_progress_repeat = True

        outcome = (
            _observation_outcome(messages, assistant_index)
            if decision_type == "tool_call"
            else {"status": "not_applicable", "usable_success": False, "observation_count": 0}
        )
        if outcome["usable_success"]:
            successful_state_change_ordinals.add(ordinal)
        if signature is not None:
            # Always point later retries at the most recent occurrence.  A
            # failed retry therefore cannot inherit success from an older call.
            seen_calls[signature] = (assistant_index, ordinal)
        annotations[assistant_index] = {
            "decision_role": role,
            "is_initial_step": ordinal == 0,
            "decision_ordinal": ordinal,
            "trajectory_decision_count": len(decisions),
            "is_no_progress_repeat": no_progress_repeat,
            "repeat_of_assistant_index": repeat_of,
            "repeat_prior_usable_success": prior_usable,
            "repeat_intervening_state_change": intervening_state_change,
            "observation_status": outcome["status"],
            "observation_usable_success": outcome["usable_success"],
            "observation_count": outcome["observation_count"],
        }

    for annotation in annotations.values():
        annotation["trajectory_has_no_progress_repeat"] = trajectory_has_no_progress_repeat
    return annotations


def _build_sample(
    *,
    record: dict[str, Any],
    message_index: int,
    prompt_messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
    parsed_assistant: dict[str, Any],
    decision_annotation: dict[str, Any],
    source_path: str,
) -> dict[str, Any]:
    decision_type = str(parsed_assistant.get("decision_type") or "")
    target_tool_calls = [item for item in (parsed_assistant.get("target_tool_calls") or []) if isinstance(item, dict)]
    target_final_answer = parsed_assistant.get("final_answer") if decision_type == "final_answer" else None
    tool_names = [str(item.get("tool_name") or "") for item in target_tool_calls]
    tool_names_raw = [str(item.get("tool_name_raw") or "") for item in target_tool_calls]

    target_assistant = _parse_target_assistant(assistant_message)
    label = {
        "schema_version": "toolrl_step_v3",
        "source_id": record.get("id"),
        "source_path": source_path,
        "assistant_index": message_index,
        "assistant_role": assistant_message.get("role"),
        "assistant_content": assistant_message.get("content"),
        "tool_call_count": len(target_tool_calls),
        "decision_type": decision_type,
        "target_tool_calls": target_tool_calls,
        "target_final_answer": target_final_answer,
        "target_assistant": target_assistant,
    }
    metadata = {
        "schema_version": "toolrl_step_v3",
        "protocol": "react_json",
        "source_id": record.get("id"),
        "source_path": source_path,
        "assistant_index": message_index,
        "task_id": record.get("id"),
        "task_type": _infer_task_type(record, source_path),
        "prompt_message_count": len(prompt_messages),
        "target_tool_call_count": len(target_tool_calls),
        "decision_type": decision_type,
        "tool_names": tool_names,
        "tool_names_raw": tool_names_raw,
        "allowed_tool_names": sorted(supported_training_tool_names(), key=str.casefold),
        "target_assistant": target_assistant,
        "target_tool_calls": target_tool_calls,
        "target_final_answer": target_final_answer,
        **decision_annotation,
        "raw_record_keys": sorted([str(k) for k in record.keys()]),
    }

    return {
        "prompt": prompt_messages,
        "label": label,
        "metadata": metadata,
        "target_assistant": target_assistant,
        "target_tool_calls": target_tool_calls,
        "target_final_answer": target_final_answer,
    }


def _compact_preview(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    label = sample.get("label") if isinstance(sample.get("label"), dict) else {}
    tool_names = metadata.get("tool_names") if isinstance(metadata.get("tool_names"), list) else []
    return {
        "source_id": metadata.get("source_id"),
        "task_type": metadata.get("task_type"),
        "assistant_index": metadata.get("assistant_index"),
        "decision_type": metadata.get("decision_type"),
        "prompt_message_count": metadata.get("prompt_message_count"),
        "target_tool_call_count": metadata.get("target_tool_call_count"),
        "tool_names": tool_names[:5],
        "label_tool_call_count": label.get("tool_call_count"),
    }


def convert_react_to_toolrl_steps(
    input_path: Path,
    output_path: Path,
    *,
    skipped_report_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    records = _load_records(input_path)
    allowlist = default_molclaw_allowlist()

    output_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    counts = Counter()
    per_task_type = defaultdict(int)
    per_tool_name = Counter()
    local_tool_names = set(LOCAL_TOOL_NAMES)

    for record_idx, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            counts["skip_no_messages"] += 1
            skipped_rows.append(
                {
                    "source": str(input_path),
                    "record_index": record_idx,
                    "source_id": record.get("id"),
                    "skip_reason": "no_messages",
                    "details": {},
                }
            )
            continue

        annotations = _decision_annotations(messages)
        if any(item.get("trajectory_has_no_progress_repeat") for item in annotations.values()):
            counts["trajectories_with_no_progress_repeat"] += 1

        for decision in iter_react_decisions(messages):
            message_index = int(decision["assistant_index"])
            message = decision["target_assistant"]
            parsed = decision["parse"]
            if not parsed.get("ok"):
                counts["skip_parse_failed"] += 1
                skipped_rows.append(
                    {
                        "source": str(input_path),
                        "record_index": record_idx,
                        "source_id": record.get("id"),
                        "assistant_index": message_index,
                        "skip_reason": "assistant_parse_failed",
                        "details": {
                            "error_message": parsed.get("error"),
                        },
                    }
                )
                continue

            parsed_tool_calls = [item for item in parsed.get("tool_calls", []) if isinstance(item, dict)]
            unsupported_tool_calls = [
                item
                for item in parsed_tool_calls
                if not is_supported_decision_name(str(item.get("tool_name") or ""), allowlist)
            ]
            target_tool_calls = [
                {**item, "keep": True}
                for item in parsed_tool_calls
                if is_supported_decision_name(str(item.get("tool_name") or ""), allowlist)
            ]
            decision_type = str(decision.get("decision_type") or "")
            if decision_type == "tool_call" and unsupported_tool_calls:
                counts["skip_unsupported_tool_calls"] += 1
                skipped_rows.append(
                    {
                        "source": str(input_path),
                        "record_index": record_idx,
                        "source_id": record.get("id"),
                        "assistant_index": message_index,
                        "skip_reason": "unsupported_tool_calls",
                        "details": {
                            "unsupported_tool_names": [
                                item.get("tool_name_raw") or item.get("tool_name")
                                for item in unsupported_tool_calls
                            ],
                            "has_final_answer": False,
                        },
                    }
                )
                continue
            if decision_type == "tool_call" and not target_tool_calls:
                counts["skip_no_supported_tool_calls"] += 1
                skipped_rows.append(
                    {
                        "source": str(input_path),
                        "record_index": record_idx,
                        "source_id": record.get("id"),
                        "assistant_index": message_index,
                        "skip_reason": "no_supported_tool_calls",
                        "details": {"has_final_answer": False},
                    }
                )
                continue

            prompt_messages = decision["state_messages"]
            if not prompt_messages:
                counts["skip_empty_prompt"] += 1
                skipped_rows.append(
                    {
                        "source": str(input_path),
                        "record_index": record_idx,
                        "source_id": record.get("id"),
                        "assistant_index": message_index,
                        "skip_reason": "empty_prompt",
                        "details": {},
                    }
                )
                continue

            sample = _build_sample(
                record=record,
                message_index=message_index,
                prompt_messages=prompt_messages,
                assistant_message=message,
                parsed_assistant={
                    **parsed,
                    "decision_type": decision_type,
                    "final_answer": decision.get("final_answer"),
                    "target_tool_calls": target_tool_calls,
                },
                decision_annotation=annotations.get(
                    message_index,
                    {
                        "decision_role": "final" if decision_type == "final_answer" else "tool_step",
                        "is_initial_step": False,
                        "decision_ordinal": -1,
                        "trajectory_decision_count": len(annotations),
                        "is_no_progress_repeat": False,
                        "repeat_of_assistant_index": None,
                        "repeat_prior_usable_success": False,
                        "repeat_intervening_state_change": False,
                        "trajectory_has_no_progress_repeat": False,
                        "observation_status": "unknown",
                        "observation_usable_success": False,
                        "observation_count": 0,
                    },
                ),
                source_path=str(input_path),
            )
            output_rows.append(sample)
            counts["kept"] += 1
            counts[f"kept_{decision_type}"] += 1
            counts[f"kept_role_{sample['metadata']['decision_role']}"] += 1
            counts["kept_no_progress_repeat"] += bool(sample["metadata"]["is_no_progress_repeat"])
            if decision_type == "tool_call":
                names = [str(item.get("tool_name") or "") for item in target_tool_calls]
                per_tool_name.update(names)
                local_count = sum(name in local_tool_names for name in names)
                molclaw_count = len(names) - local_count
                counts["target_tool_call_total"] += len(names)
                counts["local_target_tool_call_total"] += local_count
                counts["molclaw_target_tool_call_total"] += molclaw_count
                counts["kept_tool_call_with_local"] += local_count > 0
                counts["kept_tool_call_mixed_local_molclaw"] += local_count > 0 and molclaw_count > 0
            per_task_type[str(sample["metadata"].get("task_type") or "unknown")] += 1

    ensure_dir(output_path.parent)
    write_jsonl(output_path, output_rows)
    if skipped_report_path is not None:
        write_jsonl(skipped_report_path, skipped_rows)

    report = {
        "ok": True,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "skipped_report_path": str(skipped_report_path) if skipped_report_path else None,
        "counts": dict(counts),
        "per_task_type": dict(per_task_type),
        "per_tool_name": dict(sorted(per_tool_name.items(), key=lambda item: item[0].casefold())),
        "kept_rows": len(output_rows),
        "skipped_rows": len(skipped_rows),
        "sample_preview": [_compact_preview(sample) for sample in output_rows[:3]],
    }
    if report_path is not None:
        write_json(report_path, report)
    return report


def _default_output_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    output_jsonl = output_dir / "toolrl_steps.jsonl"
    skipped_report = output_dir / "skipped_report.jsonl"
    report = output_dir / "report.json"
    return output_jsonl, skipped_report, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert cleaned ReAct JSON/JSONL into step-level ToolRL JSONL")
    parser.add_argument("--input", type=str, required=True, help="Input JSON directory / JSON file / JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output ToolRL JSONL path")
    parser.add_argument("--skipped-report", type=str, default=None, help="Optional skipped sample JSONL path")
    parser.add_argument("--report", type=str, default=None, help="Optional summary JSON path")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    skipped_path = Path(args.skipped_report).expanduser().resolve() if args.skipped_report else None
    report_path = Path(args.report).expanduser().resolve() if args.report else None

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    report = convert_react_to_toolrl_steps(
        input_path,
        output_path,
        skipped_report_path=skipped_path,
        report_path=report_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
