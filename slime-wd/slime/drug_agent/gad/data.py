from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from drug_agent.decision_extractor import iter_react_decisions
from drug_agent.tools.local_tools import LOCAL_TOOL_NAMES
from drug_agent.toolrl.parse_tool_calls import (
    default_molclaw_allowlist,
    is_supported_decision_name,
    supported_training_tool_names,
)
from drug_agent.utils import read_jsonl, write_json, write_jsonl

UNSUPPORTED_TEACHER_TOOLS = {
    "askuserquestion",
    "notebookedit",
    "skill",
    "task",
    "todowrite",
    "webfetch",
    "websearch",
}
SUPPORTED_LOCAL_TOOLS = {name.casefold() for name in LOCAL_TOOL_NAMES}


def _infer_task_type(record: dict[str, Any]) -> str | None:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    for candidate in (metadata.get("task"), record.get("task_type"), record.get("task")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().lower()
    match = re.search(
        r"(?:mcp_sft|react)_(?P<task_type>vs|ac|pf|kg|e2e)_",
        str(record.get("id") or ""),
    )
    return match.group("task_type") if match else None


def _partition_supported_tool_calls(
    tool_info: dict[str, Any],
    allowed_tool_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep MolClaw and supported local calls; reject teacher-only orchestration.

    Canonical SFT contains the exact assistant decision. GAD must preserve all
    executable calls in that decision, including mixed MolClaw/local turns, in
    original order. Only other MCP servers and unsupported teacher tools are
    rejected.
    """
    supported: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for call in tool_info.get("tool_calls") or []:
        raw = str(call.get("tool_name_raw") or "")
        bare = str(call.get("tool_name") or "").strip().lower()
        is_other_mcp = raw.startswith("mcp__") and not raw.startswith("mcp__molclaw-scp__")
        if is_other_mcp or bare in UNSUPPORTED_TEACHER_TOOLS:
            rejected.append(call)
        elif is_supported_decision_name(raw or bare, allowed_tool_names):
            supported.append(call)
        else:
            rejected.append(call)
    return supported, rejected


def convert_records(records: list[dict[str, Any]], source: str = "") -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    counts = Counter()
    per_tool_name = Counter()
    allowlist = default_molclaw_allowlist()
    allowed_tool_names = sorted(supported_training_tool_names(allowlist), key=str.casefold)
    for record_index, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list):
            skipped.append({"record_index": record_index, "skip_reason": "missing_messages"})
            continue
        for decision in iter_react_decisions(messages):
            assistant_index = int(decision["assistant_index"])
            response = decision["target_assistant"]["content"]
            parsed = decision["parse"]
            if not parsed.get("ok"):
                skipped.append({"record_index": record_index, "assistant_index": assistant_index, "skip_reason": "parse_failed"})
                counts["skip_parse_failed"] += 1
                continue
            tool_info = {"tool_calls": decision["tool_calls"]}
            target_calls, rejected_calls = _partition_supported_tool_calls(tool_info, allowlist)
            if rejected_calls:
                skipped.append(
                    {
                        "record_index": record_index,
                        "assistant_index": assistant_index,
                        "skip_reason": "unsupported_tool",
                        "tool_names": [call.get("tool_name_raw") or call.get("tool_name") for call in rejected_calls],
                    }
                )
                counts["skip_unsupported_tool"] += 1
                continue
            decision_type = decision["decision_type"]
            if decision_type == "tool_call" and not target_calls:
                skipped.append({"record_index": record_index, "assistant_index": assistant_index, "skip_reason": "no_decision"})
                counts["skip_no_decision"] += 1
                continue
            state = decision["state_messages"]
            if not state:
                skipped.append({"record_index": record_index, "assistant_index": assistant_index, "skip_reason": "invalid_state_boundary"})
                counts["skip_invalid_state_boundary"] += 1
                continue
            sample_id = f"{record.get('id') or record_index}:assistant:{assistant_index}"
            label = {
                "teacher_response": response,
                "decision_type": decision_type,
                "target_tool_calls": target_calls,
            }
            metadata = {
                "schema_version": "drug_agent_gad_step_v1",
                "sample_id": sample_id,
                "source_id": record.get("id"),
                "task_type": _infer_task_type(record),
                "source_path": source,
                "assistant_index": assistant_index,
                "decision_type": decision_type,
                "teacher_response": response,
                "target_tool_calls": target_calls,
                "allowed_tool_names": allowed_tool_names,
                # slime replaces Sample.prompt with rendered chat-template text,
                # while the discriminator still needs the original message state.
                "state_messages": state,
            }
            rows.append(
                {
                    "prompt": state,
                    "state_messages": state,
                    "teacher_response": response,
                    "label": label,
                    "metadata": metadata,
                }
            )
            counts[f"kept_{decision_type}"] += 1
            if decision_type == "tool_call":
                names = [str(item.get("tool_name") or "") for item in target_calls]
                per_tool_name.update(names)
                local_count = sum(name.casefold() in SUPPORTED_LOCAL_TOOLS for name in names)
                molclaw_count = len(names) - local_count
                counts["target_tool_call_total"] += len(names)
                counts["local_target_tool_call_total"] += local_count
                counts["molclaw_target_tool_call_total"] += molclaw_count
                counts["kept_tool_call_with_local"] += local_count > 0
                counts["kept_tool_call_mixed_local_molclaw"] += local_count > 0 and molclaw_count > 0
    report = {
        "ok": True,
        "counts": dict(counts),
        "per_tool_name": dict(sorted(per_tool_name.items(), key=lambda item: item[0].casefold())),
        "kept": len(rows),
        "skipped": len(skipped),
    }
    return rows, skipped, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert cleaned ReAct SFT trajectories to aligned GAD decision states")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skipped-report", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    records = read_jsonl(args.input)
    rows, skipped, report = convert_records(records, source=args.input)
    write_jsonl(args.output, rows)
    write_jsonl(args.skipped_report, skipped)
    report |= {"input": args.input, "output": args.output, "skipped_report": args.skipped_report}
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
