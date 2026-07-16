from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from drug_agent.constants import RAW_TASK_TYPES
from drug_agent.utils import (
    bool_from_any,
    is_mcp_tool_name,
    normalize_tool_name,
    parse_numbered_json,
    read_csv_rows,
)


def discover_raw_files(input_root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for task_type in RAW_TASK_TYPES:
        out[task_type] = sorted((input_root / task_type).glob("*.jsonl"))
    return out


def basename_to_task_type(input_root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for task_type, files in discover_raw_files(input_root).items():
        for fp in files:
            mapping[fp.name] = task_type
    return mapping


def parse_question_payload_from_raw(raw_obj: dict[str, Any]) -> dict[str, Any] | None:
    msg = raw_obj.get("message")
    if not isinstance(msg, dict):
        return None
    if msg.get("role") != "user":
        return None
    content = msg.get("content")
    if not isinstance(content, list):
        return None

    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "tool_result":
            continue
        text = item.get("content")
        if not isinstance(text, str):
            continue
        parsed = parse_numbered_json(text)
        if not isinstance(parsed, dict):
            continue
        if "question_text" in parsed or "answer" in parsed or "task" in parsed:
            return parsed
    return None


def parse_assistant_tool_calls(raw_obj: dict[str, Any]) -> list[dict[str, Any]]:
    if raw_obj.get("type") != "assistant":
        return []
    msg = raw_obj.get("message")
    if not isinstance(msg, dict):
        return []
    content = msg.get("content")
    if not isinstance(content, list):
        return []

    calls: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "tool_use":
            continue
        name = item.get("name")
        args = item.get("input")
        if not isinstance(name, str):
            continue
        calls.append(
            {
                "tool_name_raw": name,
                "tool_name": normalize_tool_name(name),
                "is_mcp": is_mcp_tool_name(name),
                "arguments": args if isinstance(args, dict) else {},
                "tool_use_id": item.get("id"),
            }
        )
    return calls


def parse_assistant_text(raw_obj: dict[str, Any]) -> str | None:
    if raw_obj.get("type") != "assistant":
        return None
    msg = raw_obj.get("message")
    if not isinstance(msg, dict):
        return None
    content = msg.get("content")
    if not isinstance(content, list):
        return None

    texts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
            texts.append(item["text"])
    if not texts:
        return None
    return "\n".join(texts)


def parse_raw_trajectory_file(path: Path) -> dict[str, Any]:
    event_counts = Counter()
    tool_use_counts = Counter()
    tool_use_counts_raw = Counter()
    mcp_tool_use_counts = Counter()
    non_mcp_tool_use_counts = Counter()

    question_payload: dict[str, Any] | None = None
    final_result: dict[str, Any] | None = None
    last_assistant_text: str | None = None

    line_count = 0
    task_ids = set()
    task_types = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            line_count += 1
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue

            typ = obj.get("type")
            event_counts[typ] += 1

            if isinstance(obj.get("task_id"), str):
                task_ids.add(obj["task_id"])
            if isinstance(obj.get("task_type"), str):
                task_types.add(obj["task_type"])

            if question_payload is None:
                q = parse_question_payload_from_raw(obj)
                if q is not None:
                    question_payload = q

            calls = parse_assistant_tool_calls(obj)
            for call in calls:
                raw_name = call["tool_name_raw"]
                name = call["tool_name"]
                tool_use_counts_raw[raw_name] += 1
                tool_use_counts[name] += 1
                if call["is_mcp"]:
                    mcp_tool_use_counts[name] += 1
                else:
                    non_mcp_tool_use_counts[name] += 1

            text = parse_assistant_text(obj)
            if text:
                last_assistant_text = text

            if typ == "result":
                final_result = {
                    "subtype": obj.get("subtype"),
                    "is_error": obj.get("is_error"),
                    "result": obj.get("result"),
                    "stop_reason": obj.get("stop_reason"),
                    "num_turns": obj.get("num_turns"),
                }

    question_text = None
    question_answer = None
    question_task = None
    if isinstance(question_payload, dict):
        question_text = question_payload.get("question_text")
        question_answer = question_payload.get("answer")
        question_task = question_payload.get("task")

    return {
        "source_path": str(path),
        "basename": path.name,
        "raw_task_type": path.parent.name,
        "line_count": line_count,
        "event_counts": dict(event_counts),
        "tool_use_counts": dict(tool_use_counts),
        "tool_use_counts_raw": dict(tool_use_counts_raw),
        "mcp_tool_use_counts": dict(mcp_tool_use_counts),
        "non_mcp_tool_use_counts": dict(non_mcp_tool_use_counts),
        "question_payload": question_payload,
        "question_text": question_text,
        "question_answer": question_answer,
        "question_task": question_task,
        "last_assistant_text": last_assistant_text,
        "final_result": final_result,
        "task_ids_seen": sorted(task_ids),
        "task_types_seen": sorted(task_types),
    }


def load_usage_summary_by_basename(csv_path: Path) -> dict[str, dict[str, Any]]:
    if not csv_path.exists():
        return {}
    rows = read_csv_rows(csv_path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        copied_path = row.get("copied_path") or ""
        basename = Path(copied_path).name if copied_path else ""
        if not basename:
            continue
        out[basename] = {
            "task": row.get("task"),
            "status": row.get("status"),
            "is_accepted": bool_from_any(row.get("is_accepted")),
            "answer_hit_pass": bool_from_any(row.get("answer_hit_pass")),
            "vs_top3_hit_num": row.get("vs_top3_hit_num"),
            "vs_top10_hit_num": row.get("vs_top10_hit_num"),
            "ac_is_correct": bool_from_any(row.get("ac_is_correct")),
            "pf_precision": row.get("pf_precision"),
            "pf_recall": row.get("pf_recall"),
            "pf_f1": row.get("pf_f1"),
            "pf_is_correct": bool_from_any(row.get("pf_is_correct")),
            "molclaw_usage_count": row.get("molclaw_usage_count"),
            "original_path": row.get("original_path"),
            "copied_path": copied_path,
            "run_dir": row.get("run_dir"),
        }
    return out


def load_sft_rows_by_id(sft_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not sft_path.exists():
        return out
    with sft_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            sample_id = obj.get("id")
            if isinstance(sample_id, str) and sample_id:
                out[sample_id] = obj
    return out


def index_rl_rows(rl_prompt_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not rl_prompt_path.exists():
        return rows
    with rl_prompt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows
