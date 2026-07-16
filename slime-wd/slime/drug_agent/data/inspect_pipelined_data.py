from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.constants import PIPELINED_DATA, SCHEMA_REPORT_DEFAULT, SFT_OUTPUTS_ANSWER_HIT
from drug_agent.data.common import (
    discover_raw_files,
    load_sft_rows_by_id,
    load_usage_summary_by_basename,
    parse_raw_trajectory_file,
)
from drug_agent.utils import write_json

EXPECTED_KEYS = [
    "task_id",
    "task_type",
    "instruction_or_question_or_prompt",
    "inputs",
    "allowed_tools",
    "trajectory",
    "final_answer_or_label_or_score",
    "metadata",
]


def _status(present: bool, derivable: bool) -> str:
    if present:
        return "present"
    if derivable:
        return "derivable"
    return "missing"


def inspect_task_type(task_type: str, files: list[Path], usage_by_basename: dict[str, dict[str, Any]]) -> dict[str, Any]:
    event_counter = Counter()
    tool_counter = Counter()
    mcp_tool_counter = Counter()

    sample_records = []
    total_lines = 0
    question_count = 0
    final_result_count = 0

    key_presence = {k: Counter() for k in EXPECTED_KEYS}

    for idx, fp in enumerate(files):
        record = parse_raw_trajectory_file(fp)
        total_lines += record["line_count"]
        event_counter.update(record["event_counts"])
        tool_counter.update(record["tool_use_counts"])
        mcp_tool_counter.update(record["mcp_tool_use_counts"])

        has_question = bool(record.get("question_text"))
        has_final = bool(record.get("final_result"))
        if has_question:
            question_count += 1
        if has_final:
            final_result_count += 1

        usage = usage_by_basename.get(record["basename"])

        key_presence["task_id"][_status(bool(record["task_ids_seen"]), bool(usage))] += 1
        key_presence["task_type"][_status(True, True)] += 1
        key_presence["instruction_or_question_or_prompt"][_status(has_question, False)] += 1
        question_payload = record.get("question_payload")
        if not isinstance(question_payload, dict):
            question_payload = {}
        key_presence["inputs"][_status(isinstance(question_payload.get("candidates"), list), has_question)] += 1
        key_presence["allowed_tools"][_status(bool(record["tool_use_counts"]), True)] += 1
        key_presence["trajectory"][_status(record["line_count"] > 0, False)] += 1
        key_presence["final_answer_or_label_or_score"][_status(has_final, bool(usage))] += 1
        key_presence["metadata"][_status(True, False)] += 1

        if idx < 3:
            sample_records.append(
                {
                    "source_path": record["source_path"],
                    "line_count": record["line_count"],
                    "event_counts": record["event_counts"],
                    "question_text_preview": (record.get("question_text") or "")[:220],
                    "question_answer_preview": record.get("question_answer"),
                    "final_result_preview": (record.get("final_result", {}).get("result") or "")[:220]
                    if isinstance(record.get("final_result", {}).get("result"), str)
                    else record.get("final_result"),
                    "top_tools": dict(Counter(record["tool_use_counts"]).most_common(8)),
                    "usage_summary": usage,
                }
            )

    return {
        "task_type": task_type,
        "num_files": len(files),
        "num_lines": total_lines,
        "files_with_question": question_count,
        "files_with_final_result": final_result_count,
        "event_type_distribution": dict(event_counter),
        "top_tools": dict(tool_counter.most_common(30)),
        "top_mcp_tools": dict(mcp_tool_counter.most_common(30)),
        "schema_presence": {k: dict(v) for k, v in key_presence.items()},
        "samples": sample_records,
    }


def build_sft_basename_index(sft_rows_by_id: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for sample_id, row in sft_rows_by_id.items():
        metadata = row.get("metadata", {})
        trajectory_path = metadata.get("trajectory_path") if isinstance(metadata, dict) else None
        if not isinstance(trajectory_path, str):
            continue
        basename = Path(trajectory_path).name
        out.setdefault(basename, []).append(sample_id)
    return out


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# pipelined_data Schema Report",
        "",
        f"- input_root: `{report['input_root']}`",
        f"- total_files: {report['total_files']}",
        f"- total_lines: {report['total_lines']}",
        f"- usage_summary_rows: {report['usage_summary_rows']}",
        f"- sft_answer_hit_rows: {report['sft_answer_hit_rows']}",
        f"- basename_join_with_usage: {report['basename_join_with_usage']}",
        f"- basename_join_with_sft_answer_hit: {report['basename_join_with_sft']}",
        "",
    ]

    for item in report["by_task_type"]:
        lines.append(f"## {item['task_type']}")
        lines.append("")
        lines.append(f"- files: {item['num_files']}")
        lines.append(f"- lines: {item['num_lines']}")
        lines.append(f"- files_with_question: {item['files_with_question']}")
        lines.append(f"- files_with_final_result: {item['files_with_final_result']}")
        lines.append("")
        lines.append("### Event Distribution")
        for k, v in sorted(item["event_type_distribution"].items()):
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("### Schema Presence")
        for key, dist in item["schema_presence"].items():
            lines.append(f"- {key}: {dist}")
        lines.append("")
        lines.append("### Top MCP Tools")
        for k, v in list(item["top_mcp_tools"].items())[:15]:
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("### Sample Rows")
        for s in item["samples"]:
            lines.append(f"- source: `{s['source_path']}`")
            lines.append(f"  - line_count: {s['line_count']}")
            lines.append(f"  - event_counts: {s['event_counts']}")
            lines.append(f"  - question_text_preview: {json.dumps(s['question_text_preview'], ensure_ascii=False)}")
            lines.append(f"  - question_answer_preview: {json.dumps(s['question_answer_preview'], ensure_ascii=False)}")
            lines.append(f"  - final_result_preview: {json.dumps(s['final_result_preview'], ensure_ascii=False)}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect ac/pf/vs pipelined_data schema and joins")
    parser.add_argument("--input-root", type=str, default=str(PIPELINED_DATA))
    parser.add_argument("--output", type=str, default=str(SCHEMA_REPORT_DEFAULT))
    parser.add_argument("--sft-path", type=str, default=None)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_path = Path(args.output)

    files_by_type = discover_raw_files(input_root)
    usage_by_basename = load_usage_summary_by_basename(input_root / "molclaw_usage_summary.csv")

    sft_path = Path(args.sft_path) if args.sft_path else input_root / "sft_outputs_answer_hit/mcp_sft_all.jsonl"
    if not sft_path.exists():
        sft_path = SFT_OUTPUTS_ANSWER_HIT / "mcp_sft_all.jsonl"

    sft_rows_by_id = load_sft_rows_by_id(sft_path)
    sft_basename_index = build_sft_basename_index(sft_rows_by_id)

    reports = []
    total_files = 0
    total_lines = 0

    raw_basenames = set()
    for task_type, files in files_by_type.items():
        total_files += len(files)
        for fp in files:
            raw_basenames.add(fp.name)
        task_report = inspect_task_type(task_type, files, usage_by_basename)
        total_lines += task_report["num_lines"]
        reports.append(task_report)

    join_usage = len(raw_basenames & set(usage_by_basename.keys()))
    join_sft = len(raw_basenames & set(sft_basename_index.keys()))

    full_report = {
        "input_root": str(input_root),
        "output": str(output_path),
        "total_files": total_files,
        "total_lines": total_lines,
        "usage_summary_rows": len(usage_by_basename),
        "sft_answer_hit_rows": len(sft_rows_by_id),
        "basename_join_with_usage": join_usage,
        "basename_join_with_sft": join_sft,
        "by_task_type": reports,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(full_report), encoding="utf-8")

    write_json(output_path.with_suffix(output_path.suffix + ".json"), full_report)

    print(json.dumps({
        "output": str(output_path),
        "total_files": total_files,
        "total_lines": total_lines,
        "basename_join_with_usage": join_usage,
        "basename_join_with_sft": join_sft,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
