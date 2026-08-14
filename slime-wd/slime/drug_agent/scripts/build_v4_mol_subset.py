#!/usr/bin/env python3
"""Build the auditable 365-trajectory MolBench subset from canonical v4."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("id"):
                raise ValueError(f"invalid row at {path}:{line_number}")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-root", required=True, type=Path)
    parser.add_argument("--mol-source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    source = args.v4_root / "react_trajectories.jsonl"
    catalog = args.v4_root / "tool_catalog.json"
    for path in (source, catalog, args.mol_source):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.output_root.exists():
        raise SystemExit(f"output already exists: {args.output_root}")

    v4_rows = read_jsonl(source)
    mol_rows = read_jsonl(args.mol_source)
    if len(v4_rows) != 605 or len(mol_rows) != 365:
        raise SystemExit(f"unexpected source counts: v4={len(v4_rows)} mol={len(mol_rows)}")
    mol_ids = [str(row["id"]) for row in mol_rows]
    if len(set(mol_ids)) != 365:
        raise SystemExit("MolBench source IDs are not unique")
    v4_by_id = {str(row["id"]): row for row in v4_rows}
    missing = [record_id for record_id in mol_ids if record_id not in v4_by_id]
    changed = [record_id for record_id, row in zip(mol_ids, mol_rows) if v4_by_id.get(record_id) != row]
    if missing or changed:
        raise SystemExit(f"v4 membership mismatch: missing={missing[:5]} changed={changed[:5]}")

    # Preserve v4 order and copy the original v4 JSONL bytes for selected rows.
    selected_ids = set(mol_ids)
    selected = [row for row in v4_rows if str(row["id"]) in selected_ids]
    if len(selected) != 365 or [str(row["id"]) for row in selected] != mol_ids:
        raise SystemExit("v4 subset order differs from the canonical MolBench source")

    args.output_root.mkdir(parents=True)
    output = args.output_root / "react_trajectories.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    shutil.copy2(catalog, args.output_root / "tool_catalog.json")

    ids_path = args.output_root / "source_ids.jsonl"
    with ids_path.open("w", encoding="utf-8") as handle:
        for index, record_id in enumerate(mol_ids):
            handle.write(json.dumps({"source_index": index, "id": record_id}, separators=(",", ":")) + "\n")

    manifest = {
        "schema_version": "drug_agent_subset_manifest_v1",
        "dataset_version": "live_tool_catalog_v4_mol",
        "description": "The 365 canonical MolBench-derived trajectories selected from frozen v4 by canonical-v2 IDs.",
        "parent": {
            "path": str(args.v4_root.resolve()),
            "records": 605,
            "react_sha256": sha256(source),
        },
        "membership_authority": {
            "path": str(args.mol_source.resolve()),
            "records": 365,
            "react_sha256": sha256(args.mol_source),
            "selection": "exact ID join followed by full JSON-object equality",
        },
        "canonical_react": {
            "path": "react_trajectories.jsonl",
            "records": 365,
            "unique_ids": 365,
            "sha256": sha256(output),
        },
        "source_ids": {"path": "source_ids.jsonl", "records": 365, "sha256": sha256(ids_path)},
        "tool_catalog": {"path": "tool_catalog.json", "sha256": sha256(args.output_root / "tool_catalog.json")},
        "excluded_parent_records": 240,
        "source_files_unchanged": True,
    }
    (args.output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "README.md").write_text(
        "# Live Tool Catalog v4 MolBench subset\n\n"
        "This dataset contains exactly the 365 canonical MolBench-derived trajectories from v4. "
        "Membership is established by the canonical-v2 ID set and full row equality, not by an "
        "unchecked positional slice. The frozen 605-row v4 and 365-row source are unchanged.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
