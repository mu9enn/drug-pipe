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

from drug_agent.protocol.react_protocol import parse_react_sequence
from drug_agent.toolrl.parse_tool_calls import default_molclaw_allowlist, parse_tool_calls
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


def _copy_message(message: dict[str, Any]) -> dict[str, Any]:
    out = {"role": message.get("role"), "content": message.get("content")}
    if "name" in message:
        out["name"] = message["name"]
    return out


def _infer_task_type(record: dict[str, Any], source_path: str) -> str | None:
    for candidate in (record.get("task_type"), record.get("task")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    record_id = str(record.get("id") or "")
    match = re.search(r"mcp_sft_(?P<task_type>[a-z]+)_", record_id)
    if match:
        return match.group("task_type")

    path_name = Path(source_path).name
    match = re.search(r"mcp_sft_(?P<task_type>[a-z]+)_", path_name)
    if match:
        return match.group("task_type")
    return None


def _parse_target_assistant(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    parsed = parse_tool_calls(content if isinstance(content, str) else "", role="assistant", keep_non_molclaw=True)
    tool_calls = parsed.get("molclaw_tool_calls") if isinstance(parsed.get("molclaw_tool_calls"), list) else []
    return {
        "role": "assistant",
        "content": content,
        "parsed": to_jsonable(parsed),
        "tool_call_count": len(tool_calls),
    }


def _build_sample(
    *,
    record: dict[str, Any],
    message_index: int,
    prompt_messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
    parsed_assistant: dict[str, Any],
    source_path: str,
) -> dict[str, Any]:
    target_tool_calls = [item for item in (parsed_assistant.get("molclaw_tool_calls") or []) if isinstance(item, dict)]
    tool_names = [str(item.get("tool_name") or "") for item in target_tool_calls]
    tool_names_raw = [str(item.get("tool_name_raw") or "") for item in target_tool_calls]

    target_assistant = _parse_target_assistant(assistant_message)
    label = {
        "schema_version": "toolrl_step_v1",
        "source_id": record.get("id"),
        "source_path": source_path,
        "assistant_index": message_index,
        "assistant_role": assistant_message.get("role"),
        "assistant_content": assistant_message.get("content"),
        "tool_call_count": len(target_tool_calls),
        "target_tool_calls": target_tool_calls,
        "target_assistant": target_assistant,
    }
    metadata = {
        "schema_version": "toolrl_step_v1",
        "protocol": "react_json",
        "source_id": record.get("id"),
        "source_path": source_path,
        "assistant_index": message_index,
        "task_id": record.get("id"),
        "task_type": _infer_task_type(record, source_path),
        "prompt_message_count": len(prompt_messages),
        "target_tool_call_count": len(target_tool_calls),
        "tool_names": tool_names,
        "tool_names_raw": tool_names_raw,
        "allowed_tool_names": sorted(default_molclaw_allowlist()),
        "target_assistant": target_assistant,
        "target_tool_calls": target_tool_calls,
        "raw_record_keys": sorted([str(k) for k in record.keys()]),
    }

    return {
        "prompt": prompt_messages,
        "label": label,
        "metadata": metadata,
        "target_assistant": target_assistant,
        "target_tool_calls": target_tool_calls,
    }


def _compact_preview(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    label = sample.get("label") if isinstance(sample.get("label"), dict) else {}
    tool_names = metadata.get("tool_names") if isinstance(metadata.get("tool_names"), list) else []
    return {
        "source_id": metadata.get("source_id"),
        "task_type": metadata.get("task_type"),
        "assistant_index": metadata.get("assistant_index"),
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

        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                counts["skip_non_string_assistant"] += 1
                continue

            parsed = parse_tool_calls(content, role="assistant", allowed_tool_names=allowlist, keep_non_molclaw=True)
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
                            "error_type": parsed.get("error_type"),
                            "error_message": parsed.get("error_message"),
                        },
                    }
                )
                continue

            target_tool_calls = parsed.get("molclaw_tool_calls") if isinstance(parsed.get("molclaw_tool_calls"), list) else []
            target_tool_calls = [item for item in target_tool_calls if isinstance(item, dict)]
            if not target_tool_calls:
                counts["skip_no_molclaw_tool_calls"] += 1
                skipped_rows.append(
                    {
                        "source": str(input_path),
                        "record_index": record_idx,
                        "source_id": record.get("id"),
                        "assistant_index": message_index,
                        "skip_reason": "no_molclaw_tool_calls",
                        "details": {
                            "non_molclaw_tool_call_count": int(parsed.get("non_molclaw_tool_call_count") or 0),
                            "has_final_answer": bool(parsed.get("has_final_answer")),
                        },
                    }
                )
                continue

            prompt_messages = [_copy_message(item) for item in messages[:message_index]]
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
                parsed_assistant=parsed,
                source_path=str(input_path),
            )
            output_rows.append(sample)
            counts["kept"] += 1
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
