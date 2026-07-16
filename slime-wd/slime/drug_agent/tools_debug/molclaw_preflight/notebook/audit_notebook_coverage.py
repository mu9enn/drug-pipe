#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any

NOTEBOOK_FILES = [
    "drug_api_test.ipynb",
    "drug_api_test_sxy.ipynb",
]

MANUAL_ALIAS_MAP = {
    "run_chai1_prediction": "chai1_predict",
    "run_diffdock_auto": None,
    "equiscore_extract_pocket": "equiscore_pocket",
    "design_peptide_binder_evobind": "evobind_tool",
    "run_hdock_docking": "hdock_tool",
    "prolif_analyze_docking": "prolif_docking",
    "run_openawsem_simulation": "openawsem_sim",
    "run_openmm_protein_md": "protein_openmm_md",
    "run_proteinmpnn_design": "proteinmpnn_tool",
    "run_bioemu_sampling": "run_bioemu",
    "run_mmpbsa_calculation": "run_mmpbsa",
    "run_mmpbsa_propro": "gmx_mmpbsa_propro",
    "prepare_protein_ligand_complex": "prepare_complex",
    "fix_pdb_dock": None,
    "boltz_binding_affinity": "pred_binding_affinity_boltz2",
    "run_fpocket": "fpocket_toolkit",
    "search_uniprot_id": None,
    "visualize_complex": None,
    "read_fasta_file": None,
}

CALL_TOOL_PATTERN = re.compile(r"call_tool\s*\(\s*\"([^\"]+)\"", re.S)
UNKNOWN_TOOL_PATTERN = re.compile(r"Unknown tool:\s*([^'\n\"]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit notebook tool coverage against drugsda schema")
    parser.add_argument("--notebook-dir", default=".", help="Directory containing notebooks and schema")
    parser.add_argument("--out-json", default="coverage_report.json", help="Output JSON path")
    parser.add_argument("--out-md", default="coverage_report.md", help="Output markdown path")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_notebook_calls(nb_path: Path) -> list[dict[str, Any]]:
    nb = read_json(nb_path)
    rows: list[dict[str, Any]] = []

    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        tools = CALL_TOOL_PATTERN.findall(source)
        if not tools:
            continue

        execution_count = cell.get("execution_count")
        outputs = cell.get("outputs", [])
        has_error_output = any(o.get("output_type") == "error" for o in outputs)
        stream_text = "\n".join(
            "".join(o.get("text", []))
            for o in outputs
            if o.get("output_type") == "stream"
        )
        unknown_tool = None
        match = UNKNOWN_TOOL_PATTERN.search(stream_text)
        if match:
            unknown_tool = match.group(1).strip()

        for tool_name in tools:
            rows.append(
                {
                    "notebook": nb_path.name,
                    "cell_index": idx,
                    "tool_name": tool_name,
                    "execution_count": execution_count,
                    "executed": execution_count is not None,
                    "has_error_output": has_error_output,
                    "unknown_tool_return": unknown_tool,
                    "source_preview": source.strip().splitlines()[0] if source.strip() else "",
                }
            )

    return rows


def suggest_alias(tool_name: str, schema_names: list[str]) -> dict[str, Any]:
    manual = MANUAL_ALIAS_MAP.get(tool_name)
    fuzzy = difflib.get_close_matches(tool_name, schema_names, n=3, cutoff=0.35)
    suggestion = manual if manual in schema_names else (fuzzy[0] if fuzzy else None)
    return {
        "notebook_tool_name": tool_name,
        "manual_suggestion": manual,
        "fuzzy_candidates": fuzzy,
        "best_suggestion": suggestion,
    }


def aggregate_execution_notes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_tool.setdefault(row["tool_name"], []).append(row)

    notes: dict[str, Any] = {}
    for tool_name, entries in by_tool.items():
        notes[tool_name] = {
            "call_count": len(entries),
            "executed_call_count": sum(1 for e in entries if e["executed"]),
            "error_output_count": sum(1 for e in entries if e["has_error_output"]),
            "unknown_tool_return_count": sum(1 for e in entries if e.get("unknown_tool_return")),
            "details": entries,
        }
    return notes


def build_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]

    lines = [
        "# Notebook Coverage Report",
        "",
        "## Counts",
        "",
        f"- schema_tool_count: **{summary['schema_tool_count']}**",
        f"- notebook_called_tool_count: **{summary['notebook_called_tool_count']}**",
        f"- matched_schema_tested_count: **{summary['matched_schema_tested_count']}**",
        f"- schema_untested_count: **{summary['schema_untested_count']}**",
        f"- notebook_called_not_in_schema_count: **{summary['notebook_called_not_in_schema_count']}**",
        "",
        "## Baseline Check",
        "",
        "- expected baseline: schema=81, notebook_called=52, matched=33, untested=48, drift=19",
        f"- current baseline: schema={summary['schema_tool_count']}, notebook_called={summary['notebook_called_tool_count']}, matched={summary['matched_schema_tested_count']}, untested={summary['schema_untested_count']}, drift={summary['notebook_called_not_in_schema_count']}",
        "",
        "## Schema Matched Tools",
        "",
    ]

    for name in report["matched_schema_tools"]:
        note = report["execution_notes_by_tool"].get(name, {})
        lines.append(
            f"- `{name}` (calls={note.get('call_count', 0)}, executed={note.get('executed_call_count', 0)}, unknown_return={note.get('unknown_tool_return_count', 0)})"
        )

    lines.extend([
        "",
        "## Schema Untested Tools",
        "",
    ])
    for name in report["schema_untested_tools"]:
        lines.append(f"- `{name}`")

    lines.extend([
        "",
        "## Notebook Drift Tools (Not In Schema)",
        "",
        "| notebook_tool | best_suggestion | manual_suggestion | fuzzy_candidates |",
        "|---|---|---|---|",
    ])

    for item in report["drift_tool_alias_suggestions"]:
        lines.append(
            "| {tool} | {best} | {manual} | {fuzzy} |".format(
                tool=item["notebook_tool_name"],
                best=item.get("best_suggestion") or "",
                manual=item.get("manual_suggestion") or "",
                fuzzy=", ".join(item.get("fuzzy_candidates", [])),
            )
        )

    lines.extend([
        "",
        "## Execution Trace Notes",
        "",
        "- Execution status is based on `execution_count` and cell outputs from notebook JSON.",
        "- `unknown_tool_return_count` counts cells whose stream output contains `Unknown tool:`.",
    ])

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    nb_dir = Path(args.notebook_dir).resolve()

    schema_path = nb_dir / "drugsda_tools_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"schema not found: {schema_path}")

    schema = read_json(schema_path)
    schema_tools = schema.get("tools", [])
    schema_names = sorted({str(t.get("name")) for t in schema_tools if t.get("name")})
    schema_set = set(schema_names)

    all_calls: list[dict[str, Any]] = []
    for nb_name in NOTEBOOK_FILES:
        nb_path = nb_dir / nb_name
        if not nb_path.exists():
            continue
        all_calls.extend(collect_notebook_calls(nb_path))

    notebook_called_tools = sorted({row["tool_name"] for row in all_calls})
    notebook_set = set(notebook_called_tools)

    matched_schema_tools = sorted(notebook_set & schema_set)
    schema_untested_tools = sorted(schema_set - notebook_set)
    notebook_called_not_in_schema = sorted(notebook_set - schema_set)

    execution_notes_by_tool = aggregate_execution_notes(all_calls)

    drift_suggestions = [
        suggest_alias(name, schema_names)
        for name in notebook_called_not_in_schema
    ]

    report = {
        "meta": {
            "notebook_dir": str(nb_dir),
            "schema_path": str(schema_path),
            "schema_server_url": schema.get("server_url"),
            "schema_saved_at": schema.get("saved_at"),
        },
        "summary": {
            "schema_tool_count": len(schema_set),
            "notebook_called_tool_count": len(notebook_set),
            "matched_schema_tested_count": len(matched_schema_tools),
            "schema_untested_count": len(schema_untested_tools),
            "notebook_called_not_in_schema_count": len(notebook_called_not_in_schema),
        },
        "matched_schema_tools": matched_schema_tools,
        "schema_untested_tools": schema_untested_tools,
        "notebook_called_not_in_schema": notebook_called_not_in_schema,
        "drift_tool_alias_suggestions": drift_suggestions,
        "execution_notes_by_tool": execution_notes_by_tool,
        "raw_call_rows": all_calls,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(report), encoding="utf-8")

    print(f"saved json: {out_json}")
    print(f"saved md: {out_md}")
    print(
        "baseline: schema={schema}, notebook_called={called}, matched={matched}, untested={untested}, drift={drift}".format(
            schema=report["summary"]["schema_tool_count"],
            called=report["summary"]["notebook_called_tool_count"],
            matched=report["summary"]["matched_schema_tested_count"],
            untested=report["summary"]["schema_untested_count"],
            drift=report["summary"]["notebook_called_not_in_schema_count"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
