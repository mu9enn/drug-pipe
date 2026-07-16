#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build auto cases for remaining tools not covered by existing case files")
    parser.add_argument("--schema-json", required=True, help="drugsda_tools_schema.json path")
    parser.add_argument("--existing-case-files", nargs="+", required=True, help="existing case json files")
    parser.add_argument("--out-case-file", required=True, help="output case json path")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_type(prop: dict[str, Any]) -> str:
    t = prop.get("type")
    if isinstance(t, str):
        return t
    if isinstance(t, list):
        for item in t:
            if isinstance(item, str) and item != "null":
                return item
    if "anyOf" in prop and isinstance(prop["anyOf"], list):
        for item in prop["anyOf"]:
            if isinstance(item, dict) and isinstance(item.get("type"), str) and item.get("type") != "null":
                return str(item["type"])
    return "unknown"


def sample_for_field(tool_name: str, field: str, prop: dict[str, Any]) -> Any:
    lname = field.lower()
    t = detect_type(prop)

    if "default" in prop:
        return prop["default"]

    if "enum" in prop and isinstance(prop["enum"], list) and prop["enum"]:
        return prop["enum"][0]

    if field == "file_base64_string":
        return "aGVsbG8gd29ybGQK"
    if field == "file_name":
        return "test_molclaw_tools_auto.txt"
    if field == "target_smiles":
        return "CCO"
    if field == "candidate_smiles_list":
        return ["CCN", "CCC", "c1ccccc1"]
    if field == "smiles_list":
        return ["CCO", "CCN"]
    if field == "smiles":
        return "CCO"
    if field == "ligand_smi":
        return "CCO"
    if field == "compound_names":
        return ["aspirin", "caffeine"]
    if field == "sequences":
        return ["MKTIIALSYIFCLVFA", "ACDEFGHIKLMNPQRSTVWY"]
    if field in {"sequence", "seq"}:
        return "MKTIIALSYIFCLVFA"
    if field == "name":
        return "chainA"
    if field == "gene_name":
        return "TP53"
    if field == "organism":
        return "9606"
    if field == "sort_by":
        return "length"
    if field == "uniprot_id":
        return "P04637"
    if field == "pdb_id":
        return "6VKV"
    if field == "group":
        return "C3"
    if field == "warheads":
        return "CCO.CCN"
    if field == "warhead_pair_name":
        return "pair_demo"
    if field == "scaffold_name":
        return "benzene"
    if field == "scaffold":
        return "c1ccccc1"
    if field == "template":
        return "ACDEFG"
    if field == "peptide":
        return "ACDEFG"

    if "mode" == lname:
        if tool_name == "foldx_tool":
            return "repair"
        if tool_name == "interaction_visualizer":
            return "protein_ligand"
        if tool_name == "pulchura_rebuild":
            return "single"
        if tool_name == "chai1_predict":
            return "sequence"
        return "default"

    path_keywords = [
        "path",
        "file",
        "dir",
        "pdb",
        "cif",
        "sdf",
        "fasta",
        "topology",
        "trajectory",
        "work_dir",
        "run_dir",
    ]
    if any(k in lname for k in path_keywords):
        if t == "array":
            if "fasta" in lname:
                return ["__AUTO__:fasta_file"]
            if "cif" in lname:
                return ["__AUTO__:cif_file"]
            if "sdf" in lname or "ligand" in lname:
                return ["__AUTO__:sdf_file"]
            if "pocket_dir" in lname or "work_dir" in lname or lname.endswith("dir"):
                return ["__AUTO__:dir"]
            return ["__AUTO__:pdb_file"]

        if "fasta" in lname:
            return "__AUTO__:fasta_file"
        if "cif" in lname:
            return "__AUTO__:cif_file"
        if "ligand" in lname:
            return "__AUTO__:sdf_file"
        if "sdf" in lname:
            return "__AUTO__:sdf_file"
        if "pocket_dir" in lname or "work_dir" in lname or lname.endswith("dir"):
            return "__AUTO__:dir"
        return "__AUTO__:pdb_file"

    if field == "protein" and t == "array":
        return [{"id": "A", "sequence": "MKTIIALSYIFCLVFA"}]

    if t == "boolean":
        if lname == "dry_run":
            return True
        return False

    if t == "integer":
        if isinstance(prop.get("minimum"), int):
            return max(1, int(prop["minimum"]))
        return 1

    if t == "number":
        if isinstance(prop.get("minimum"), (int, float)):
            return max(1.0, float(prop["minimum"]))
        return 1.0

    if t == "array":
        items = prop.get("items", {}) if isinstance(prop.get("items"), dict) else {}
        item_type = detect_type(items)
        if item_type == "string":
            return ["A"]
        if item_type == "number":
            return [1.0]
        if item_type == "integer":
            return [1]
        if item_type == "object":
            return [{"id": "A", "sequence": "MKTIIALSYIFCLVFA"}]
        return ["value"]

    if t == "object":
        return {}

    if t == "string" or t == "unknown":
        return "value"

    return "value"


def build_case(tool: dict[str, Any], idx: int) -> dict[str, Any]:
    tool_name = tool["name"]
    input_schema = tool.get("inputSchema", {}) if isinstance(tool.get("inputSchema"), dict) else {}
    props = input_schema.get("properties", {}) if isinstance(input_schema.get("properties"), dict) else {}
    required = input_schema.get("required", []) if isinstance(input_schema.get("required"), list) else []

    args: dict[str, Any] = {}
    for field in required:
        prop = props.get(field, {}) if isinstance(props.get(field), dict) else {}
        args[field] = sample_for_field(tool_name, field, prop)

    if "dry_run" in props and "dry_run" not in args:
        args["dry_run"] = True

    # Optional but commonly needed fields for stability.
    if tool_name == "calculate_morgan_fingerprint_similarity":
        if "radius" in props:
            args.setdefault("radius", 2)
        if "nBits" in props:
            args.setdefault("nBits", 2048)
    if tool_name == "calculate_common_fragments":
        if "radius" in props:
            args.setdefault("radius", 2)
        if "nBits" in props:
            args.setdefault("nBits", 2048)
    if tool_name == "chai1_predict":
        args.setdefault("mode", "sequence")
        args.setdefault("seq", "MKTIIALSYIFCLVFA")
        args.setdefault("name", "chainA")
        args.setdefault("samples", 1)
        args["dry_run"] = True
    if tool_name == "run_bioemu":
        args.setdefault("sequence", "MKTIIALSYIFCLVFA")
        args.setdefault("num_samples", 1)
        args["dry_run"] = True

    case = {
        "case_id": f"remaining_{idx:03d}_{tool_name}",
        "tool_name": tool_name,
        "arguments": args,
        "notes": "auto-generated from schema (remaining tool coverage)",
        "source": "auto_remaining",
    }
    return case


def main() -> int:
    args = parse_args()

    schema = load_json(Path(args.schema_json))
    tools = schema.get("tools", [])

    covered_tools: set[str] = set()
    for case_file in args.existing_case_files:
        data = load_json(Path(case_file))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and isinstance(item.get("tool_name"), str):
                    covered_tools.add(item["tool_name"])

    remaining_tools = [
        t for t in tools
        if isinstance(t, dict)
        and isinstance(t.get("name"), str)
        and t["name"] not in covered_tools
    ]

    cases = [build_case(tool, i + 1) for i, tool in enumerate(sorted(remaining_tools, key=lambda x: x["name"]))]

    out_path = Path(args.out_case_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved remaining cases: {out_path}")
    print(f"covered_tools={len(covered_tools)}")
    print(f"remaining_tools={len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
