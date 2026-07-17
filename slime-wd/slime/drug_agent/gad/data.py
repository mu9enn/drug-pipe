from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from drug_agent.decision_extractor import iter_react_decisions
from drug_agent.utils import read_jsonl, write_json, write_jsonl

NON_MOLCLAW_LOCAL_TOOLS = {
    "askuserquestion",
    "bash",
    "edit",
    "glob",
    "grep",
    "notebookedit",
    "read",
    "skill",
    "task",
    "todowrite",
    "webfetch",
    "websearch",
    "write",
}


def _partition_cleaned_tool_calls(tool_info: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify calls from the cleaned ReAct source without a mini allowlist.

    The upstream cleaned SFT contract retains MolClaw calls and serializes many
    of them as bare names. A mini online-RL allowlist must not silently remove
    valid GAD decisions. Explicit non-MolClaw MCP prefixes and known local
    engineering tools are still rejected.
    """
    molclaw: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for call in tool_info.get("tool_calls") or []:
        raw = str(call.get("tool_name_raw") or "")
        bare = str(call.get("tool_name") or "").strip().lower()
        is_molclaw_mcp = raw.startswith(("mcp__molclaw-scp__", "mcp__molclaw-vs__"))
        is_other_mcp = raw.startswith("mcp__") and not is_molclaw_mcp
        if is_other_mcp or bare in NON_MOLCLAW_LOCAL_TOOLS:
            rejected.append(call)
        else:
            molclaw.append(call)
    return molclaw, rejected


def convert_records(records: list[dict[str, Any]], source: str = "") -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    counts = Counter()
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
            target_calls, rejected_calls = _partition_cleaned_tool_calls(tool_info)
            if rejected_calls:
                skipped.append(
                    {
                        "record_index": record_index,
                        "assistant_index": assistant_index,
                        "skip_reason": "non_molclaw_tool",
                        "tool_names": [call.get("tool_name_raw") or call.get("tool_name") for call in rejected_calls],
                    }
                )
                counts["skip_non_molclaw_tool"] += 1
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
                "source_path": source,
                "assistant_index": assistant_index,
                "decision_type": decision_type,
                "teacher_response": response,
                "target_tool_calls": target_calls,
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
    report = {"ok": True, "counts": dict(counts), "kept": len(rows), "skipped": len(skipped)}
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
