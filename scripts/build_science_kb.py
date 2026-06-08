#!/usr/bin/env python3
"""Build the fixed local MolClaw Stage3 Science-KB from existing local datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from molclaw_kg.science_kb import initialize_database  # noqa: E402

csv.field_size_limit(sys.maxsize)


def provenance(path: Path, row_id: str, source: str, version: str) -> str:
    return json.dumps(
        {
            "source_database": source,
            "source_version": version,
            "source_path": str(path.resolve()),
            "source_row_id": row_id,
        },
        ensure_ascii=False,
    )


def record_id(prefix: str, *parts: str) -> str:
    payload = "::".join(parts)
    return f"{prefix}::{hashlib.sha256(payload.encode()).hexdigest()[:20]}"


def import_cara(conn: sqlite3.Connection, sequences_path: Path, activities_path: Path, max_pairs: int) -> dict:
    source = "ChEMBL-CARA"
    version = "ChEMBL30"
    target_sequences: dict[str, str] = {}
    with sequences_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            target_id = str(row.get("Target ChEMBL ID") or "").strip()
            sequence = str(row.get("Target Sequence") or "").strip()
            if target_id and len(sequence) >= 40:
                target_sequences.setdefault(target_id, sequence)

    protein_count = compound_count = pair_count = 0
    seen_proteins: set[str] = set()
    seen_compounds: set[str] = set()
    with activities_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader, start=1):
            if pair_count >= max_pairs:
                break
            protein_id = str(row.get("Target ChEMBL ID") or "").strip()
            compound_id = str(row.get("Molecule ChEMBL ID") or "").strip()
            smiles = str(row.get("Smiles") or "").strip()
            if not protein_id or not compound_id or not smiles or protein_id not in target_sequences:
                continue

            if protein_id not in seen_proteins:
                conn.execute(
                    "INSERT OR IGNORE INTO proteins VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        record_id("protein", source, protein_id),
                        source,
                        version,
                        protein_id,
                        None,
                        None,
                        protein_id,
                        None,
                        target_sequences[protein_id],
                        "[]",
                        provenance(sequences_path, protein_id, source, version),
                    ),
                )
                seen_proteins.add(protein_id)
                protein_count += 1

            if compound_id not in seen_compounds:
                conn.execute(
                    "INSERT OR IGNORE INTO compounds VALUES (?,?,?,?,?,?,?)",
                    (
                        record_id("compound", source, compound_id),
                        source,
                        version,
                        compound_id,
                        compound_id,
                        smiles,
                        provenance(activities_path, str(idx), source, version),
                    ),
                )
                seen_compounds.add(compound_id)
                compound_count += 1

            try:
                value = float(row.get("pChEMBL Value") or "")
            except Exception:
                value = None
            conn.execute(
                "INSERT OR IGNORE INTO target_ligand_pairs VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    record_id("pair", source, protein_id, compound_id, str(idx)),
                    source,
                    version,
                    protein_id,
                    compound_id,
                    str(row.get("Value Type") or "pChEMBL"),
                    value,
                    "pChEMBL",
                    provenance(activities_path, str(idx), source, version),
                ),
            )
            pair_count += 1
            if pair_count % 1000 == 0:
                conn.commit()
    conn.commit()
    return {"proteins": protein_count, "compounds": compound_count, "target_ligand_pairs": pair_count}


def import_uniprot(conn: sqlite3.Connection, path: Path) -> dict:
    source, version = "UniProtKB", "reviewed-human"
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for idx, row in enumerate(csv.DictReader(f, delimiter="\t"), start=1):
            accession = str(row.get("Entry") or "").strip()
            sequence = str(row.get("Sequence") or "").strip()
            if not accession or len(sequence) < 40:
                continue
            gene_names = str(row.get("Gene Names") or "").split()
            pdb_ids = [x.strip() for x in str(row.get("PDB") or "").split(";") if x.strip()]
            before = conn.total_changes
            conn.execute(
                "INSERT OR IGNORE INTO proteins VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record_id("protein", source, accession), source, version, accession, accession,
                    gene_names[0] if gene_names else None, str(row.get("Protein names") or accession),
                    "Homo sapiens", sequence, json.dumps(pdb_ids),
                    provenance(path, str(idx), source, version),
                ),
            )
            count += int(conn.total_changes > before)
    conn.commit()
    return {"proteins": count, "compounds": 0, "target_ligand_pairs": 0}


def import_gtopdb(
    conn: sqlite3.Connection,
    interactions_path: Path,
    ligands_path: Path,
    uniprot_mapping_path: Path,
    max_pairs: int,
) -> dict:
    source, version = "GtoPdb", "2026.1"
    ligands: dict[str, dict[str, str]] = {}
    with ligands_path.open("r", encoding="utf-8-sig", newline="") as f:
        next(f, None)  # Version comment.
        for row in csv.DictReader(f, delimiter="\t"):
            ligand_id, smiles = str(row.get("Ligand ID") or "").strip(), str(row.get("SMILES") or "").strip()
            if ligand_id and smiles:
                ligands[ligand_id] = row
    target_uniprot: dict[str, str] = {}
    with uniprot_mapping_path.open("r", encoding="utf-8-sig", newline="") as f:
        next(f, None)  # Version comment.
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("Species") == "Human" and row.get("GtoPdb IUPHAR ID") and row.get("UniProtKB ID"):
                target_uniprot[str(row["GtoPdb IUPHAR ID"])] = str(row["UniProtKB ID"])

    counts = {"proteins": 0, "compounds": 0, "target_ligand_pairs": 0}
    seen_compounds: set[str] = set()
    with interactions_path.open("r", encoding="utf-8-sig", newline="") as f:
        next(f, None)  # Version comment.
        for idx, row in enumerate(csv.DictReader(f, delimiter="\t"), start=1):
            if counts["target_ligand_pairs"] >= max_pairs:
                break
            if str(row.get("Target Species") or "") != "Human":
                continue
            ligand_id = str(row.get("Ligand ID") or "").strip()
            ligand = ligands.get(ligand_id)
            target_id = str(row.get("Target ID") or "").strip()
            uniprot = str(row.get("Target UniProt ID") or target_uniprot.get(target_id) or "").strip()
            if not ligand or not uniprot:
                continue
            compound_id = f"GtoPdb:{ligand_id}"
            if compound_id not in seen_compounds:
                before = conn.total_changes
                conn.execute(
                    "INSERT OR IGNORE INTO compounds VALUES (?,?,?,?,?,?,?)",
                    (
                        record_id("compound", source, compound_id), source, version, compound_id,
                        str(ligand.get("Name") or row.get("Ligand") or compound_id), str(ligand["SMILES"]),
                        provenance(ligands_path, ligand_id, source, version),
                    ),
                )
                counts["compounds"] += int(conn.total_changes > before)
                seen_compounds.add(compound_id)
            before = conn.total_changes
            conn.execute(
                "INSERT OR IGNORE INTO proteins VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record_id("protein", source, uniprot), source, version, uniprot, uniprot,
                    str(row.get("Target Gene Symbol") or "") or None, str(row.get("Target") or uniprot),
                    "Homo sapiens", None, "[]", provenance(interactions_path, str(idx), source, version),
                ),
            )
            counts["proteins"] += int(conn.total_changes > before)
            raw_value = str(row.get("Original Affinity Median nm") or row.get("Affinity Median") or "").strip()
            try:
                activity_value = float(raw_value)
            except ValueError:
                activity_value = None
            conn.execute(
                "INSERT OR IGNORE INTO target_ligand_pairs VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    record_id("pair", source, uniprot, compound_id, str(idx)), source, version,
                    uniprot, compound_id, str(row.get("Original Affinity Units") or row.get("Affinity Units") or ""),
                    activity_value, "nM" if row.get("Original Affinity Median nm") else str(row.get("Affinity Units") or ""),
                    provenance(interactions_path, str(idx), source, version),
                ),
            )
            counts["target_ligand_pairs"] += 1
    conn.commit()
    return counts


def import_bindingdb(conn: sqlite3.Connection, path: Path, max_pairs: int) -> dict:
    source, version = "BindingDB", path.stem
    counts = {"proteins": 0, "compounds": 0, "target_ligand_pairs": 0}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        for idx, row in enumerate(csv.DictReader(f, delimiter="\t"), start=1):
            if counts["target_ligand_pairs"] >= max_pairs:
                break
            sequence = str(row.get("BindingDB Target Chain Sequence") or "").strip()
            uniprot = str(row.get("UniProt (SwissProt) Primary ID of Target Chain") or row.get("UniProt (TrEMBL) Primary ID of Target Chain") or "").strip()
            target_name = str(row.get("Target Name") or row.get("UniProt (SwissProt) Recommended Name of Target Chain") or "").strip()
            smiles = str(row.get("Ligand SMILES") or "").strip()
            compound_id = str(row.get("ChEMBL ID of Ligand") or row.get("BindingDB MonomerID") or "").strip()
            if not sequence or len(sequence) < 40 or not smiles or not compound_id:
                continue
            protein_id = uniprot or f"BDBTARGET-{hashlib.sha256((target_name + sequence).encode()).hexdigest()[:16]}"
            pdb_ids = [x.strip() for x in str(row.get("PDB ID(s) of Target Chain") or "").replace(",", " ").split() if x.strip()]
            before = conn.total_changes
            conn.execute(
                "INSERT OR IGNORE INTO proteins VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record_id("protein", source, protein_id), source, version, protein_id, uniprot or None,
                    None, target_name or protein_id,
                    str(row.get("Target Source Organism According to Curator or DataSource") or "") or None,
                    sequence, json.dumps(pdb_ids), provenance(path, str(idx), source, version),
                ),
            )
            if conn.total_changes > before:
                counts["proteins"] += 1
            before = conn.total_changes
            conn.execute(
                "INSERT OR IGNORE INTO compounds VALUES (?,?,?,?,?,?,?)",
                (
                    record_id("compound", source, compound_id), source, version, compound_id,
                    str(row.get("BindingDB Ligand Name") or compound_id), smiles,
                    provenance(path, str(idx), source, version),
                ),
            )
            if conn.total_changes > before:
                counts["compounds"] += 1
            activity_type, activity_value = None, None
            for col in ["Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)"]:
                match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(row.get(col) or ""))
                if not match:
                    continue
                try:
                    parsed = float(match.group(0))
                except ValueError:
                    continue
                if 0 <= parsed <= 1e9:
                    activity_value, activity_type = parsed, col.removesuffix(" (nM)")
                    break
            conn.execute(
                "INSERT OR IGNORE INTO target_ligand_pairs VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    record_id("pair", source, protein_id, compound_id, str(idx)), source, version,
                    protein_id, compound_id, activity_type, activity_value, "nM",
                    provenance(path, str(idx), source, version),
                ),
            )
            counts["target_ligand_pairs"] += 1
            if counts["target_ligand_pairs"] % 1000 == 0:
                conn.commit()
    conn.commit()
    return counts


def main() -> None:
    raw = PROJECT_ROOT / "science_kb" / "raw"
    default_data = PROJECT_ROOT.parent / "mol-pipeline" / "get-molbench" / "data" / "CARA" / "Task"
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", default=str(PROJECT_ROOT / "science_kb/processed/science_kb.sqlite"))
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "science_kb/manifests/science_kb_manifest.json"))
    parser.add_argument("--uniprot", default=str(raw / "uniprot_reviewed_human.tsv"))
    parser.add_argument("--gtopdb-interactions", default=str(raw / "gtopdb_interactions.tsv"))
    parser.add_argument("--gtopdb-ligands", default=str(raw / "gtopdb_ligands.tsv"))
    parser.add_argument("--gtopdb-uniprot-mapping", default=str(raw / "gtopdb_uniprot_mapping.tsv"))
    parser.add_argument("--max-gtopdb-pairs", type=int, default=20000)
    parser.add_argument("--cara-sequences", default="")
    parser.add_argument("--cara-activities", default="")
    parser.add_argument("--max-pairs", type=int, default=20000)
    repository_bindingdb = raw / "BindingDB_All_202409.tsv"
    default_bindingdb = os.environ.get("BINDINGDB_PATH", "")
    if not default_bindingdb and repository_bindingdb.is_file():
        default_bindingdb = str(repository_bindingdb)
    parser.add_argument(
        "--bindingdb",
        default=default_bindingdb,
        help="Optional BindingDB TSV. Defaults to BINDINGDB_PATH or science_kb/raw/BindingDB_All_202409.tsv.",
    )
    parser.add_argument("--max-bindingdb-pairs", type=int, default=20000)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite).resolve()
    manifest_path = Path(args.manifest).resolve()
    if args.replace and sqlite_path.exists():
        sqlite_path.unlink()

    conn = initialize_database(sqlite_path)
    counts, sources = {}, []
    uniprot_path = Path(args.uniprot).resolve()
    if not uniprot_path.is_file():
        raise FileNotFoundError(uniprot_path)
    counts["uniprot"] = import_uniprot(conn, uniprot_path)
    sources.append({"source_database": "UniProtKB", "source_version": "reviewed-human", "path": str(uniprot_path)})
    gtopdb_paths = [
        Path(args.gtopdb_interactions).resolve(), Path(args.gtopdb_ligands).resolve(),
        Path(args.gtopdb_uniprot_mapping).resolve(),
    ]
    if not all(path.is_file() for path in gtopdb_paths):
        raise FileNotFoundError(f"missing GtoPdb files: {[str(x) for x in gtopdb_paths if not x.is_file()]}")
    counts["gtopdb"] = import_gtopdb(conn, *gtopdb_paths, max(1, args.max_gtopdb_pairs))
    sources.append({"source_database": "GtoPdb", "source_version": "2026.1", "paths": [str(x) for x in gtopdb_paths]})
    if args.cara_sequences and args.cara_activities:
        sequences_path, activities_path = Path(args.cara_sequences).resolve(), Path(args.cara_activities).resolve()
        counts["cara"] = import_cara(conn, sequences_path, activities_path, max(1, args.max_pairs))
        sources.append({"source_database": "ChEMBL-CARA", "source_version": "ChEMBL30", "sequence_path": str(sequences_path), "activity_path": str(activities_path)})
    bindingdb_path = Path(args.bindingdb).resolve() if args.bindingdb else None
    if bindingdb_path and bindingdb_path.is_file():
        counts["bindingdb"] = import_bindingdb(conn, bindingdb_path, max(1, args.max_bindingdb_pairs))
    conn.close()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if bindingdb_path and bindingdb_path.is_file():
        sources.append({"source_database": "BindingDB", "source_version": bindingdb_path.stem, "path": str(bindingdb_path)})
    manifest = {
        "schema_version": "science_kb_lite_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sqlite_path": str(sqlite_path),
        "sources": sources,
        "counts": counts,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
