from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from drug_agent.decision_extractor import iter_react_decisions
from drug_agent.gad.data import convert_records as convert_gad_records
from drug_agent.toolrl.convert_react_to_toolrl_steps import convert_react_to_toolrl_steps
from drug_agent.toolrl.parse_tool_calls import default_molclaw_allowlist, is_molclaw_decision_name
from drug_agent.utils import read_jsonl, write_json, write_jsonl


def _eligible(record: dict) -> bool:
    allowlist = default_molclaw_allowlist()
    saw_tool = False
    saw_final = False
    for decision in iter_react_decisions(record.get("messages") or []):
        if not decision["parse"].get("ok"):
            continue
        if decision["decision_type"] == "final_answer":
            saw_final = True
        if any(
            is_molclaw_decision_name(str(call.get("tool_name") or ""), allowlist)
            for call in decision.get("tool_calls") or []
        ):
            saw_tool = True
    return saw_tool and saw_final


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate reviewed, pretty JSON format examples from canonical data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    rows = read_jsonl(Path(args.input))
    candidates = [row for row in rows if _eligible(row)]
    if not candidates:
        raise ValueError("no canonical record contains both ToolRL tool and final decisions")
    selected = min(candidates, key=lambda row: len(json.dumps(row, ensure_ascii=False, sort_keys=True)))
    root = Path(args.output_root)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        source_jsonl = tmp_root / "canonical.jsonl"
        write_jsonl(source_jsonl, [selected])
        toolrl_jsonl = tmp_root / "toolrl.jsonl"
        toolrl_skipped = tmp_root / "toolrl.skipped.jsonl"
        toolrl_report = convert_react_to_toolrl_steps(
            source_jsonl,
            toolrl_jsonl,
            skipped_report_path=toolrl_skipped,
        )
        toolrl_rows = read_jsonl(toolrl_jsonl)
        toolrl_skipped_rows = read_jsonl(toolrl_skipped)
        gad_rows, gad_skipped, gad_report = convert_gad_records([selected], source="format_example")

    write_json(root / "source/canonical_react.json", selected)
    write_json(root / "sft/sft_messages.json", selected)
    write_json(root / "toolrl/toolrl_steps.json", toolrl_rows)
    write_json(root / "toolrl/report.json", {
        "source_id": selected.get("id"),
        "counts": toolrl_report.get("counts"),
        "kept_rows": len(toolrl_rows),
    })
    write_jsonl(root / "toolrl/skipped.jsonl", toolrl_skipped_rows)
    write_json(root / "gad/gad_steps.json", gad_rows)
    write_json(root / "gad/report.json", {
        "source_id": selected.get("id"),
        **gad_report,
    })
    write_jsonl(root / "gad/skipped.jsonl", gad_skipped)
    print(json.dumps({
        "ok": True,
        "source_id": selected.get("id"),
        "toolrl_rows": len(toolrl_rows),
        "gad_rows": len(gad_rows),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
