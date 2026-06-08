from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, write_jsonl
from .settings import ProjectConfig


def build_provenance_sidecar(config: ProjectConfig) -> dict[str, Any]:
    edges = read_jsonl(config.paths.run_dir / "graph_all.jsonl")
    prov_rows = []

    now = datetime.now(timezone.utc).isoformat()

    for e in edges:
        edge_id = e["edge_id"]
        pair_id = str(e.get("pair_id") or f"pair::{e.get('source_tool')}__to__{e.get('target_tool')}")

        prov_rows.append(
            {
                "prov_id": f"prov::{edge_id}",
                "entity": {
                    "id": f"entity::{edge_id}",
                    "type": "prov:Entity",
                    "label": "FinalEdge",
                },
                "activity": {
                    "id": f"activity::{pair_id}",
                    "type": "prov:Activity",
                    "label": "PairwiseAdjudicationAndScoring",
                    "time": now,
                },
                "agent": {
                    "id": "agent::molclaw_kg_pipeline",
                    "type": "prov:Agent",
                    "label": "MolClawKGPipeline",
                },
                "plan": {
                    "id": "plan::edge_ontology_v1",
                    "type": "prov:Plan",
                    "label": "edge_type_v1 + rules_v1",
                },
                "wasGeneratedBy": f"activity::{pair_id}",
                "wasAttributedTo": "agent::molclaw_kg_pipeline",
                "used": [
                    "entity::tool_snapshot",
                    "entity::tool_cards",
                    "entity::doc_chunks",
                    "entity::candidate_pairs",
                ],
                "specialization": {
                    "edge_type": e.get("edge_type"),
                    "view": e.get("view"),
                    "confidence": e.get("confidence_calibrated"),
                },
            }
        )

    out = config.paths.run_dir / "provenance_sidecar.jsonl"
    write_jsonl(out, prov_rows)
    return {"provenance_count": len(prov_rows), "output": str(out)}
