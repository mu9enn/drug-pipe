from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft7Validator

from drug_agent.tools.local_tools import LOCAL_TOOL_NAMES
from drug_agent.utils import normalize_tool_name, read_jsonl, write_json, write_jsonl


TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
OBSERVATION_RE = re.compile(
    r"<observation\s+tool_name=(?P<quote>[\"'])(?P<name>[^\"']+)(?P=quote)>(?P<body>.*?)</observation>",
    re.DOTALL,
)
OVERLAP_PATH = Path(__file__).resolve().parents[1] / "evaluation/molbench_exclusions.json"
UNRESOLVED_REMOVAL = "retrieve_protein_structure_by_chembl_id"


def _prompt_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _bool(value: Any) -> Any:
    if isinstance(value, str):
        if value.strip().lower() in {"true", "1", "yes"}:
            return True
        if value.strip().lower() in {"false", "0", "no"}:
            return False
    return value


def _same(arguments: dict[str, Any]) -> dict[str, Any]:
    return dict(arguments)


def _sequence(arguments: dict[str, Any]) -> dict[str, Any]:
    identifier = next(
        (arguments.get(key) for key in ("identifier", "gene_name", "uniprot_id", "protein_id", "query") if arguments.get(key)),
        None,
    )
    return {"identifier": identifier, "organism": arguments.get("organism") or arguments.get("species") or "Homo sapiens"}


def _fix_pdb(arguments: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    input_path = arguments.get("input_path") or arguments.get("input_pdb") or arguments.get("pdb_file")
    if input_path is not None:
        output["input_path"] = input_path
    for key in (
        "output_path", "add_hydrogens", "ph", "remove_heterogens", "remove_water",
        "replace_nonstandard", "keep_chains", "add_missing_residues", "dry_run",
    ):
        if key in arguments:
            output[key] = _bool(arguments[key])
    return output


def _fpocket(arguments: dict[str, Any]) -> dict[str, Any]:
    output = dict(arguments)
    if "pdb_file" not in output:
        for key in ("input_path", "pdb_file_path"):
            if key in output:
                output["pdb_file"] = output.pop(key)
                break
    return output


def _json_list(value: Any) -> Any:
    original = value
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return original
    return value if isinstance(value, list) else original


def _normalize_current_arguments(
    tool_name: str, arguments: dict[str, Any], schema: dict[str, Any]
) -> dict[str, Any]:
    output = dict(arguments)
    if tool_name.startswith("calculate_mol_") and "smiles_list" in output:
        output["smiles_list"] = _json_list(output["smiles_list"])
    elif tool_name == "pred_binding_affinity_boltz2" and "protein" in output:
        output["protein"] = _json_list(output["protein"])
    elif tool_name == "retrieve_protein_structure_by_gene_name":
        if "gene_name" not in output and output.get("identifier"):
            output["gene_name"] = output.pop("identifier")
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if "organism" in properties:
            output.setdefault("organism", "Homo sapiens")
        if "sort_by" in properties:
            output.setdefault("sort_by", "resolution")
    elif tool_name == "retrieve_protein_structure_by_uniprot_id":
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if "sort_by" in properties:
            output.setdefault("sort_by", "resolution")
    elif tool_name == "retrieve_protein_structure_by_pdb_id":
        output.pop("sort_by", None)
    elif tool_name == "server_file_to_base64":
        # Historical clients sent output-oriented fields that never affected
        # which server file was read.
        output.pop("file_name", None)
        output.pop("file_base64_string", None)
    elif tool_name == "equiscore_pocket":
        if "docking_result" not in output and "ligand_file" in output:
            output["docking_result"] = output.pop("ligand_file")
        if "receptor_pdb" not in output and "receptor_file" in output:
            output["receptor_pdb"] = output.pop("receptor_file")
        if "pocket_cutoff" not in output and "pocket_radius" in output:
            output["pocket_cutoff"] = output.pop("pocket_radius")
    elif tool_name == "interaction_visualizer":
        if "ligand_path" not in output and "ligand" in output:
            output["ligand_path"] = output.pop("ligand")
        if "receptor_path" not in output and "receptor" in output:
            output["receptor_path"] = output.pop("receptor")
        if "mode" not in output and output.get("ligand_path") and output.get("receptor_path"):
            output["mode"] = "protein_ligand"
    elif tool_name == "prolif_docking":
        if "protein_path" not in output and "receptor_file" in output:
            output["protein_path"] = output.pop("receptor_file")
        if "ligand_paths" not in output and "ligand_file" in output:
            output["ligand_paths"] = [output.pop("ligand_file")]
        elif "ligand_paths" in output:
            output["ligand_paths"] = _json_list(output["ligand_paths"])
        output.setdefault("ligand_format", "pdbqt")
        output.pop("dry_run", None)
    elif tool_name == "retrieve_protein_sequence":
        output = _sequence(output)
    elif tool_name == "molecule_docking_quickvina_fullprocess":
        if "pdb_file_path" not in output:
            for key in ("receptor_path", "protein_path", "pdb_file"):
                if key in output:
                    output["pdb_file_path"] = output.pop(key)
                    break

    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    if schema.get("additionalProperties") is False:
        # Only discard known, operational compatibility fields. Any other
        # unknown field remains and causes rejection below.
        for key in ("description",):
            if key not in properties:
                output.pop(key, None)
    for key, prop in properties.items():
        if key not in output or not isinstance(prop, dict):
            continue
        expected = prop.get("type")
        if expected == "boolean":
            output[key] = _bool(output[key])
        elif expected == "array":
            output[key] = _json_list(output[key])
    return output


Adapter = tuple[str, Callable[[dict[str, Any]], dict[str, Any]]]
MIGRATIONS: dict[str, Adapter] = {
    "calculate_mol_structure_complexity_metrics": ("calculate_mol_structure_complexity", _same),
    "calculate_mol_topo": ("calculate_mol_topology", _same),
    "calculate_mol_topoiogy": ("calculate_mol_topology", _same),
    "calculate_mol_topology_metrics": ("calculate_mol_topology", _same),
    "calculate_mol_tpsa": ("calculate_mol_topology", _same),
    "evaluate_protein_sequence_retrieve": ("retrieve_protein_sequence", _sequence),
    "fetch_protein_sequence": ("retrieve_protein_sequence", _sequence),
    "retrieve_protein_sequence_by_gene_name": ("retrieve_protein_sequence", _sequence),
    "retrieve_protein_sequence_by_uniprot_id": ("retrieve_protein_sequence", _sequence),
    "retrieve_protein_sequence_retrieve": ("retrieve_protein_sequence", _sequence),
    "fpocket": ("fpocket_toolkit", _fpocket),
    "pdbfixer": ("fix_pdb", _fix_pdb),
}


def _load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
        rows = payload.get("tools") if isinstance(payload, dict) else payload
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError("tool catalog must contain a tools list")
    catalog = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("executor") == "local_sandbox":
            continue
        name = normalize_tool_name(row.get("name"))
        if name:
            catalog[name] = row
    if not catalog:
        raise ValueError("tool catalog contains no MCP tools")
    return catalog


def _adapt_call(name: str, arguments: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any], bool]:
    bare = normalize_tool_name(name)
    migrated = False
    if bare not in catalog:
        target = MIGRATIONS.get(bare)
        if target is None:
            raise ValueError(f"tool_not_in_live_catalog:{bare}")
        bare, adapter = target
        if bare not in catalog:
            raise ValueError(f"migration_target_not_in_live_catalog:{bare}")
        arguments = adapter(arguments)
        migrated = True
    schema = catalog[bare].get("input_schema") or catalog[bare].get("inputSchema") or {}
    normalized = _normalize_current_arguments(bare, arguments, schema)
    migrated = migrated or normalized != arguments
    arguments = normalized
    errors = sorted(Draft7Validator(schema).iter_errors(arguments), key=lambda item: list(item.path))
    if errors:
        raise ValueError(f"arguments_not_valid_for_live_schema:{bare}:{errors[0].message}")
    return bare, arguments, migrated or bare != name


def _rewrite_message(
    content: str, role: str, catalog: dict[str, dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    changes: list[dict[str, Any]] = []

    def call_replace(match: re.Match[str]) -> str:
        payload = json.loads(match.group(1).strip())
        if not isinstance(payload, dict):
            raise ValueError("tool_call_payload_not_object")
        old = normalize_tool_name(payload.get("tool_name"))
        if old in LOCAL_TOOL_NAMES:
            return match.group(0)
        arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
        new, adapted, changed = _adapt_call(old, arguments, catalog)
        if changed:
            changes.append({"kind": "tool_call", "old_tool": old, "new_tool": new, "arguments_changed": adapted != arguments})
        payload["tool_name"] = new
        payload["arguments"] = adapted
        return f"<tool_call>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</tool_call>"

    rewritten = TOOL_CALL_RE.sub(call_replace, content) if role == "assistant" else content

    def observation_replace(match: re.Match[str]) -> str:
        old = normalize_tool_name(match.group("name"))
        if old in LOCAL_TOOL_NAMES or old in catalog:
            new = old
        elif old in MIGRATIONS:
            new = MIGRATIONS[old][0]
        else:
            raise ValueError(f"observation_tool_not_in_live_catalog:{old}")
        body = match.group("body")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and normalize_tool_name(payload.get("tool_name")) == old:
            payload["tool_name"] = new
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if old != new:
            changes.append({"kind": "observation", "old_tool": old, "new_tool": new})
        return f'<observation tool_name="{new}">{body}</observation>'

    if role == "user" and rewritten.lstrip().startswith("<observation"):
        rewritten = OBSERVATION_RE.sub(observation_replace, rewritten)
    return rewritten, changes


def migrate_records(input_path: Path, catalog_path: Path, output_root: Path) -> dict[str, Any]:
    catalog = _load_catalog(catalog_path)
    overlap_hashes = set(json.loads(OVERLAP_PATH.read_text(encoding="utf-8"))["molbench_ms2_prompt_sha256"])
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    counts = Counter()
    for index, record in enumerate(read_jsonl(input_path)):
        source_id = record.get("id")
        messages = record.get("messages") if isinstance(record.get("messages"), list) else []
        user = next((m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "user"), "")
        if _prompt_hash(str(user)) in overlap_hashes:
            reason = "benchmark_prompt_overlap"
            rejected.append({"source_id": source_id, "record_index": index, "reason": reason, "record": record})
            audit.append({"source_id": source_id, "status": "rejected", "reason": reason})
            counts[reason] += 1
            continue
        if any(UNRESOLVED_REMOVAL in str(m.get("content") or "") for m in messages if isinstance(m, dict)):
            reason = "removed_tool_without_safe_equivalent"
            rejected.append({"source_id": source_id, "record_index": index, "reason": reason, "record": record})
            audit.append({"source_id": source_id, "status": "rejected", "reason": reason, "tool_name": UNRESOLVED_REMOVAL})
            counts[reason] += 1
            continue

        migrated = dict(record)
        migrated_messages = []
        record_changes: list[dict[str, Any]] = []
        try:
            for message in messages:
                item = dict(message)
                if isinstance(item.get("content"), str):
                    item["content"], changes = _rewrite_message(
                        item["content"], str(item.get("role") or ""), catalog
                    )
                    record_changes.extend(changes)
                migrated_messages.append(item)
        except Exception as exc:
            reason = str(exc)
            rejected.append({"source_id": source_id, "record_index": index, "reason": reason, "record": record})
            audit.append({"source_id": source_id, "status": "rejected", "reason": reason})
            counts["unresolved_live_schema"] += 1
            continue
        migrated["messages"] = migrated_messages
        accepted.append(migrated)
        status = "migrated" if record_changes else "unchanged"
        audit.append({"source_id": source_id, "status": status, "changes": record_changes})
        counts[status] += 1

    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "react_trajectories.jsonl"
    rejected_path = output_root / "migration_rejected.jsonl"
    audit_path = output_root / "migration_audit.jsonl"
    write_jsonl(output, accepted)
    write_jsonl(rejected_path, rejected)
    write_jsonl(audit_path, audit)
    report = {
        "schema_version": "drug_agent_live_tool_migration_v1",
        "input": str(input_path), "tool_catalog": str(catalog_path), "output": str(output),
        "input_count": len(accepted) + len(rejected), "accepted_count": len(accepted),
        "rejected_count": len(rejected), "counts": dict(counts),
        "rejected_path": str(rejected_path), "audit_path": str(audit_path),
    }
    write_json(output_root / "migration_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate canonical ReAct data to a captured live MolClaw catalog")
    parser.add_argument("--input", required=True)
    parser.add_argument("--tool-catalog", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    report = migrate_records(Path(args.input), Path(args.tool_catalog), Path(args.output_root))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
