from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .science_kb import ScienceKB


def build_server(sqlite_path: Path, manifest_path: Path, trace_path: Path | None = None) -> FastMCP:
    kb = ScienceKB(sqlite_path, manifest_path)
    mcp = FastMCP(
        "molclaw-science-kb",
        instructions="Read-only, provenance-preserving scientific grounding database for MolClaw Stage3.",
    )

    def traced(tool: str, args: dict[str, Any], result: Any) -> Any:
        if trace_path:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"tool": tool, "arguments": args, "result": result}, ensure_ascii=False) + "\n")
        return result

    @mcp.tool()
    def search_proteins(query: str = "", require_sequence: bool = False, require_structure: bool = False, limit: int = 3) -> list[dict[str, Any]]:
        """Search a few real proteins. Use get_protein after selecting one."""
        args = {"query": query, "require_sequence": require_sequence, "require_structure": require_structure, "limit": limit}
        return traced("search_proteins", args, kb.search_proteins(**args))

    @mcp.tool()
    def get_protein(identifier: str) -> dict[str, Any] | None:
        """Get a real protein by record ID, protein ID, UniProt accession, or gene name."""
        return traced("get_protein", {"identifier": identifier}, kb.get_protein(identifier))

    @mcp.tool()
    def search_compounds(query: str = "", limit: int = 3) -> list[dict[str, Any]]:
        """Search a few real compounds. Use get_compound after selecting one."""
        args = {"query": query, "limit": limit}
        return traced("search_compounds", args, kb.search_compounds(**args))

    @mcp.tool()
    def get_compound(identifier: str) -> dict[str, Any] | None:
        """Get a real compound by record ID, compound ID, or name."""
        return traced("get_compound", {"identifier": identifier}, kb.get_compound(identifier))

    @mcp.tool()
    def find_target_ligand_pairs(
        protein_id: str = "",
        compound_id: str = "",
        limit: int = 3,
        include_sequence: bool = False,
    ) -> list[dict[str, Any]]:
        """Find a few real pairs. Set include_sequence only after selecting a pair."""
        args = {
            "protein_id": protein_id,
            "compound_id": compound_id,
            "limit": limit,
            "include_sequence": include_sequence,
        }
        return traced("find_target_ligand_pairs", args, kb.find_target_ligand_pairs(**args))

    @mcp.tool()
    def find_proteins_with_structures(query: str = "", limit: int = 3) -> list[dict[str, Any]]:
        """Find proteins that have PDB cross-references."""
        args = {"query": query, "limit": limit}
        return traced("find_proteins_with_structures", args, kb.find_proteins_with_structures(**args))

    @mcp.tool()
    def validate_grounding_records(record_ids: list[str]) -> dict[str, Any]:
        """Validate that all grounding record IDs exist and return their provenance."""
        validation = kb.validate_records(record_ids)
        compact_found = {}
        for record_id, item in validation["found"].items():
            record = item["record"]
            compact_found[record_id] = {
                "table": item["table"],
                "record_id": record_id,
                "source_database": record.get("source_database"),
                "source_version": record.get("source_version"),
                "protein_id": record.get("protein_id"),
                "compound_id": record.get("compound_id"),
                "uniprot_accession": record.get("uniprot_accession"),
                "provenance": record.get("provenance"),
            }
        result = {
            "valid": validation["valid"],
            "found": compact_found,
            "missing": validation["missing"],
            "manifest_schema_version": validation["manifest"].get("schema_version"),
        }
        return traced("validate_grounding_records", {"record_ids": record_ids}, result)

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", default=os.getenv("MOLCLAW_SCIENCE_KB_SQLITE", "science_kb/processed/science_kb.sqlite"))
    parser.add_argument("--manifest", default=os.getenv("MOLCLAW_SCIENCE_KB_MANIFEST", "science_kb/manifests/science_kb_manifest.json"))
    parser.add_argument("--trace", default=os.getenv("MOLCLAW_SCIENCE_KB_TRACE", ""))
    args = parser.parse_args()
    server = build_server(
        Path(args.sqlite).resolve(),
        Path(args.manifest).resolve(),
        Path(args.trace).resolve() if args.trace else None,
    )
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
