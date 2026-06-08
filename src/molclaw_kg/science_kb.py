from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS proteins (
  record_id TEXT PRIMARY KEY,
  source_database TEXT NOT NULL,
  source_version TEXT NOT NULL,
  protein_id TEXT NOT NULL,
  uniprot_accession TEXT,
  gene_name TEXT,
  protein_name TEXT,
  organism TEXT,
  sequence TEXT,
  pdb_ids_json TEXT NOT NULL DEFAULT '[]',
  provenance_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_proteins_id ON proteins(protein_id);
CREATE INDEX IF NOT EXISTS idx_proteins_uniprot ON proteins(uniprot_accession);
CREATE INDEX IF NOT EXISTS idx_proteins_gene ON proteins(gene_name);

CREATE TABLE IF NOT EXISTS compounds (
  record_id TEXT PRIMARY KEY,
  source_database TEXT NOT NULL,
  source_version TEXT NOT NULL,
  compound_id TEXT NOT NULL,
  compound_name TEXT,
  canonical_smiles TEXT NOT NULL,
  provenance_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_compounds_id ON compounds(compound_id);
CREATE INDEX IF NOT EXISTS idx_compounds_name ON compounds(compound_name);

CREATE TABLE IF NOT EXISTS target_ligand_pairs (
  record_id TEXT PRIMARY KEY,
  source_database TEXT NOT NULL,
  source_version TEXT NOT NULL,
  protein_id TEXT NOT NULL,
  compound_id TEXT NOT NULL,
  activity_type TEXT,
  activity_value REAL,
  activity_unit TEXT,
  provenance_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pairs_protein ON target_ligand_pairs(protein_id);
CREATE INDEX IF NOT EXISTS idx_pairs_compound ON target_ligand_pairs(compound_id);
"""


@dataclass(frozen=True)
class ScienceKBPaths:
    sqlite_path: Path
    manifest_path: Path


def connect_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"Science-KB SQLite missing: {path}")
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_SQL)
    return conn


def _loads(value: Any, default: Any) -> Any:
    try:
        return json.loads(value) if value else default
    except Exception:
        return default


def row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    rec = dict(row)
    for key in ["pdb_ids_json", "provenance_json"]:
        if key in rec:
            new_key = key.removesuffix("_json")
            rec[new_key] = _loads(rec.pop(key), [] if key == "pdb_ids_json" else {})
    return rec


class ScienceKB:
    def __init__(self, sqlite_path: Path, manifest_path: Path | None = None):
        self.sqlite_path = sqlite_path.resolve()
        self.manifest_path = manifest_path.resolve() if manifest_path else None
        self.conn = connect_readonly(self.sqlite_path)
        self.manifest = (
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest_path and self.manifest_path.is_file()
            else {}
        )

    def close(self) -> None:
        self.conn.close()

    def _query(self, sql: str, params: Iterable[Any], limit: int) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 50))
        rows = self.conn.execute(f"{sql} LIMIT ?", (*params, safe_limit)).fetchall()
        return [row_to_record(r) for r in rows]

    def search_proteins(self, query: str = "", require_sequence: bool = False, require_structure: bool = False, limit: int = 10) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        params: list[Any] = []
        if query.strip():
            q = f"%{query.strip()}%"
            clauses.append("(protein_id LIKE ? OR uniprot_accession LIKE ? OR gene_name LIKE ? OR protein_name LIKE ?)")
            params.extend([q, q, q, q])
        if require_sequence:
            clauses.append("sequence IS NOT NULL AND length(sequence) >= 40")
        if require_structure:
            clauses.append("pdb_ids_json IS NOT NULL AND pdb_ids_json != '[]'")
        return self._query(
            f"SELECT * FROM proteins WHERE {' AND '.join(clauses)} ORDER BY protein_id",
            params,
            limit,
        )

    def get_protein(self, identifier: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM proteins WHERE record_id=? OR protein_id=? OR uniprot_accession=? OR gene_name=? LIMIT 1",
            (identifier, identifier, identifier, identifier),
        ).fetchone()
        return row_to_record(row) if row else None

    def search_compounds(self, query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        clauses = ["canonical_smiles IS NOT NULL", "canonical_smiles != ''"]
        params: list[Any] = []
        if query.strip():
            q = f"%{query.strip()}%"
            clauses.append("(compound_id LIKE ? OR compound_name LIKE ? OR canonical_smiles LIKE ?)")
            params.extend([q, q, q])
        return self._query(
            f"SELECT * FROM compounds WHERE {' AND '.join(clauses)} ORDER BY compound_id",
            params,
            limit,
        )

    def get_compound(self, identifier: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM compounds WHERE record_id=? OR compound_id=? OR compound_name=? LIMIT 1",
            (identifier, identifier, identifier),
        ).fetchone()
        return row_to_record(row) if row else None

    def find_target_ligand_pairs(
        self,
        protein_id: str = "",
        compound_id: str = "",
        limit: int = 10,
        include_sequence: bool = False,
    ) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        params: list[Any] = []
        if protein_id:
            clauses.append("p.protein_id=?")
            params.append(protein_id)
        if compound_id:
            clauses.append("c.compound_id=?")
            params.append(compound_id)
        records = self._query(
            f"""
            SELECT x.*, p.uniprot_accession, p.gene_name, p.protein_name, p.sequence,
                   p.pdb_ids_json, c.compound_name, c.canonical_smiles
            FROM target_ligand_pairs x
            JOIN proteins p ON p.protein_id=x.protein_id
            JOIN compounds c ON c.compound_id=x.compound_id
            WHERE {' AND '.join(clauses)}
            ORDER BY x.activity_value DESC
            """,
            params,
            limit,
        )
        if not include_sequence:
            for record in records:
                record.pop("sequence", None)
        return records

    def find_proteins_with_structures(self, query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        return self.search_proteins(query=query, require_structure=True, limit=limit)

    def validate_records(self, record_ids: list[str]) -> dict[str, Any]:
        found: dict[str, dict[str, Any]] = {}
        for rid in sorted(set(str(x) for x in record_ids if str(x).strip())):
            for table in ["proteins", "compounds", "target_ligand_pairs"]:
                row = self.conn.execute(f"SELECT * FROM {table} WHERE record_id=? LIMIT 1", (rid,)).fetchone()
                if row:
                    found[rid] = {"table": table, "record": row_to_record(row)}
                    break
        missing = sorted(set(record_ids) - set(found))
        return {"valid": not missing, "found": found, "missing": missing, "manifest": self.manifest}
