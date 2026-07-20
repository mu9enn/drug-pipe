from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from jsonschema import ValidationError, validate

from .io_utils import read_jsonl, write_json, write_jsonl
from .schemas import ADJUDICATION_SCHEMA
from .settings import ProjectConfig


CANONICAL_EDGE_SCHEMA_VERSION = "canonical_edge_v1"


def canonicalize_adjudication(
    row: dict[str, Any],
    *,
    source_authority: str = "claude_pair_adjudication",
) -> dict[str, Any]:
    if not bool(row.get("response_schema_ok", False)):
        raise ValueError(str(row.get("response_schema_error") or "response_schema_ok is false"))
    adjudication_payload = {
        key: row[key]
        for key in ADJUDICATION_SCHEMA["properties"]
        if key in row
    }
    validate(instance=adjudication_payload, schema=ADJUDICATION_SCHEMA)

    pair_id = str(row["pair_id"])
    source_tool = str(row.get("source_tool") or "")
    target_tool = str(row.get("target_tool") or "")
    if not source_tool or not target_tool:
        raise ValueError("source_tool and target_tool are required")

    raw_confidence = float(row["agent_confidence"])
    edge_types = deepcopy(row["edge_types"])
    evidence_ids = sorted(
        {
            evidence_id
            for edge_type in edge_types
            if isinstance(edge_type, dict)
            for evidence_id in (edge_type.get("evidence_ids") or [])
            if isinstance(evidence_id, str) and evidence_id
        }
    )
    return {
        "schema_version": CANONICAL_EDGE_SCHEMA_VERSION,
        "pair_id": pair_id,
        "source_tool": source_tool,
        "target_tool": target_tool,
        "source_stage": row.get("source_stage"),
        "target_stage": row.get("target_stage"),
        "relation_status": row["relation_status"],
        "direct_transition": row["direct_transition"],
        "edge_types": edge_types,
        "context": row["context"],
        "satisfied_mappings": deepcopy(row["satisfied_mappings"]),
        "unsatisfied_required_inputs": deepcopy(row["unsatisfied_required_inputs"]),
        "negative_reason": row["negative_reason"],
        "evidence_ids": evidence_ids,
        "evidence_refs": deepcopy(row["evidence_refs"]),
        "rationale": row["rationale"],
        "confidence_raw": raw_confidence,
        "agent_model": row["agent_model"],
        "source_authority": source_authority,
        "source_created_at_utc": row.get("created_at_utc"),
        "source_cache_key": row.get("cache_key"),
    }


def canonicalize_rows(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical_by_pair: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    for line_number, row in enumerate(rows, start=1):
        try:
            canonical = canonicalize_adjudication(row)
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            rejected.append(
                {
                    "line_number": line_number,
                    "pair_id": row.get("pair_id"),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        canonical_by_pair[canonical["pair_id"]] = canonical
    return [canonical_by_pair[key] for key in sorted(canonical_by_pair)], rejected


def build_canonical_edges(config: ProjectConfig) -> dict[str, Any]:
    source_path = config.paths.run_dir / "pair_adjudications.jsonl"
    canonical, rejected = canonicalize_rows(read_jsonl(source_path))

    out_path = config.paths.run_dir / "canonical_edges.jsonl"
    rejected_path = config.paths.run_dir / "canonical_edges_rejected.jsonl"
    write_jsonl(out_path, canonical)
    write_jsonl(rejected_path, rejected)

    summary = {
        "schema_version": CANONICAL_EDGE_SCHEMA_VERSION,
        "source": str(source_path),
        "canonical_count": len(canonical),
        "rejected_count": len(rejected),
        "semantic_repairs": 0,
        "output": str(out_path),
        "rejected_output": str(rejected_path),
    }
    write_json(config.paths.run_dir / "canonical_edges_meta.json", summary)
    return summary
