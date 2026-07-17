from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from .canonical_edges import CANONICAL_EDGE_SCHEMA_VERSION, canonicalize_adjudication
from .canonical_outputs import canonical_edge_to_decision, project_graph
from .io_utils import ensure_dir, sha256_file, stable_hash_obj, write_json, write_jsonl


def _read_jsonl_tolerant(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected.append(
                    {
                        "source_file": path.name,
                        "line_number": line_number,
                        "reason": f"JSONDecodeError: {exc}",
                    }
                )
                continue
            if not isinstance(value, dict):
                rejected.append(
                    {
                        "source_file": path.name,
                        "line_number": line_number,
                        "reason": "row is not a JSON object",
                    }
                )
                continue
            rows.append(value)
    return rows, rejected


def _edge_type_signature(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _raw_scored_conflicts(raw: dict[str, Any], scored: dict[str, Any]) -> list[str]:
    changed: list[str] = []
    for field in [
        "relation_status",
        "direct_transition",
        "context",
        "satisfied_mappings",
        "unsatisfied_required_inputs",
        "negative_reason",
        "evidence_refs",
    ]:
        if raw.get(field) != scored.get(field):
            changed.append(field)
    if [_edge_type_signature(x) for x in raw.get("edge_types", [])] != [
        _edge_type_signature(x) for x in scored.get("edge_types", [])
    ]:
        changed.append("edge_types")
    if float(raw.get("agent_confidence", 0.0)) != float(scored.get("confidence_raw", 0.0)):
        changed.append("confidence_raw")
    return changed


def _canonical_from_legacy_scored(row: dict[str, Any]) -> dict[str, Any]:
    required = ["pair_id", "source_tool", "target_tool", "relation_status", "direct_transition", "edge_types"]
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if row["relation_status"] not in {"valid", "negative", "uncertain", "alternative"}:
        raise ValueError(f"unsupported relation_status: {row['relation_status']!r}")
    if not isinstance(row["edge_types"], list):
        raise ValueError("edge_types must be a list")
    raw_confidence = float(row.get("confidence_raw", row.get("confidence_calibrated", 0.0)))
    calibrated = float(row.get("confidence_calibrated", raw_confidence))
    return {
        "schema_version": CANONICAL_EDGE_SCHEMA_VERSION,
        "pair_id": str(row["pair_id"]),
        "source_tool": str(row["source_tool"]),
        "target_tool": str(row["target_tool"]),
        "source_stage": row.get("source_stage"),
        "target_stage": row.get("target_stage"),
        "relation_status": row["relation_status"],
        "direct_transition": bool(row["direct_transition"]),
        "edge_types": deepcopy(row["edge_types"]),
        "context": str(row.get("context") or ""),
        "satisfied_mappings": deepcopy(row.get("satisfied_mappings") or []),
        "unsatisfied_required_inputs": deepcopy(row.get("unsatisfied_required_inputs") or []),
        "negative_reason": row.get("negative_reason"),
        "evidence_ids": deepcopy(row.get("evidence_ids") or []),
        "evidence_refs": deepcopy(row.get("evidence_refs") or []),
        "rationale": str(row.get("rationale") or ""),
        "confidence_raw": raw_confidence,
        "confidence_calibrated": calibrated,
        "agent_model": str(row.get("agent_model") or "legacy_scored_unknown"),
        "source_authority": "legacy_scored_supplement",
        "source_created_at_utc": row.get("created_at_utc"),
        "source_cache_key": None,
    }


def _graph_conflicts(canonical: dict[str, Any], graph_rows: list[dict[str, Any]]) -> list[str]:
    changed: list[str] = []
    expected_types = Counter(_edge_type_signature(item.get("type")) for item in canonical.get("edge_types", []))
    observed_types = Counter(_edge_type_signature(item.get("edge_type")) for item in graph_rows)
    if expected_types != observed_types:
        changed.append("edge_types")
    statuses = {item.get("relation_status") for item in graph_rows}
    if statuses != {canonical.get("relation_status")}:
        changed.append("relation_status")
    raw_values = {float(item.get("confidence_raw", 0.0)) for item in graph_rows}
    if raw_values != {float(canonical.get("confidence_raw", 0.0))}:
        changed.append("confidence_raw")
    return changed


def migrate_historical_kg(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    ensure_dir(output_dir)

    input_paths = {
        "pair_adjudications": source_dir / "pair_adjudications.jsonl",
        "scored_edges": source_dir / "scored_edges.jsonl",
        "graph_all": source_dir / "graph_all.jsonl",
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"historical KG inputs missing: {missing}")

    raw_rows, rejected = _read_jsonl_tolerant(input_paths["pair_adjudications"])
    scored_rows, scored_rejected = _read_jsonl_tolerant(input_paths["scored_edges"])
    graph_rows, graph_rejected = _read_jsonl_tolerant(input_paths["graph_all"])
    rejected.extend(scored_rejected)
    rejected.extend(graph_rejected)

    raw_by_pair = {str(row.get("pair_id")): row for row in raw_rows if row.get("pair_id")}
    scored_by_pair = {str(row.get("pair_id")): row for row in scored_rows if row.get("pair_id")}
    graph_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in graph_rows:
        if row.get("pair_id"):
            graph_by_pair[str(row["pair_id"])].append(row)

    canonical_by_pair: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for pair_id, raw in sorted(raw_by_pair.items()):
        try:
            canonical_by_pair[pair_id] = canonicalize_adjudication(raw)
        except Exception as exc:
            rejected.append(
                {
                    "source_file": "pair_adjudications.jsonl",
                    "pair_id": pair_id,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        scored = scored_by_pair.get(pair_id)
        if scored is not None:
            fields = _raw_scored_conflicts(raw, scored)
            if fields:
                conflicts.append(
                    {
                        "pair_id": pair_id,
                        "comparison": "pair_adjudications_vs_scored_edges",
                        "conflicting_fields": fields,
                        "resolution": "pair_adjudications",
                    }
                )

    for pair_id, scored in sorted(scored_by_pair.items()):
        if pair_id in canonical_by_pair or pair_id in raw_by_pair:
            continue
        try:
            canonical_by_pair[pair_id] = _canonical_from_legacy_scored(scored)
        except Exception as exc:
            rejected.append(
                {
                    "source_file": "scored_edges.jsonl",
                    "pair_id": pair_id,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    for pair_id, canonical in sorted(canonical_by_pair.items()):
        observed = graph_by_pair.get(pair_id)
        if not observed:
            conflicts.append(
                {
                    "pair_id": pair_id,
                    "comparison": "canonical_edge_vs_graph_all",
                    "conflicting_fields": ["missing_graph_view"],
                    "resolution": "canonical_edge",
                }
            )
            continue
        fields = _graph_conflicts(canonical, observed)
        if fields:
            conflicts.append(
                {
                    "pair_id": pair_id,
                    "comparison": "canonical_edge_vs_graph_all",
                    "conflicting_fields": fields,
                    "resolution": "canonical_edge",
                    "historical_graph_edge_types": [row.get("edge_type") for row in observed],
                    "canonical_edge_types": [row.get("type") for row in canonical.get("edge_types", [])],
                }
            )

    canonical_rows = [canonical_by_pair[key] for key in sorted(canonical_by_pair)]
    decisions = [canonical_edge_to_decision(row) for row in canonical_rows]
    graph = project_graph(decisions, output_dir.name)
    tool_cards_path = source_dir / "tool_cards.jsonl"
    if tool_cards_path.is_file():
        tool_catalog, tool_rejected = _read_jsonl_tolerant(tool_cards_path)
        rejected.extend(tool_rejected)
    else:
        tool_ids = sorted(
            {
                str(row[field])
                for row in decisions
                for field in ("source_tool", "target_tool")
                if row.get(field)
            }
        )
        tool_catalog = [{"tool_id": tool_id, "source": "historical_edge_reference"} for tool_id in tool_ids]

    write_jsonl(output_dir / "tool_catalog.jsonl", tool_catalog)
    write_jsonl(output_dir / "edge_decisions.jsonl", decisions)
    write_jsonl(output_dir / "graph.jsonl", graph)
    issues = [
        *({"kind": "historical_conflict", **row} for row in conflicts),
        *({"kind": "migration_rejection", **row} for row in rejected),
    ]
    if issues:
        write_jsonl(output_dir / "issues.jsonl", issues)

    source_hash = stable_hash_obj(
        {name: sha256_file(path) for name, path in sorted(input_paths.items())}
    )
    semantic_changes = [
        {
            "kind": "historical_graph_projection_ignored",
            "pair_count": len(
                {
                    row["pair_id"]
                    for row in conflicts
                    if row["comparison"] == "canonical_edge_vs_graph_all"
                }
            ),
            "reason": "graph_all cannot override pair adjudication semantics",
        }
    ]
    report = {
        "source": str(source_dir),
        "source_hash": source_hash,
        "target_contract": "tool_kg_edge_decision_v1",
        "input_count": len(raw_rows) + len(scored_rows) + len(graph_rows),
        "output_count": len(decisions),
        "rejected_count": len(rejected),
        "conflict_count": len(conflicts),
        "semantic_changes": semantic_changes,
        "details": {
            "pair_adjudication_rows": len(raw_rows),
            "scored_edge_rows": len(scored_rows),
            "graph_rows": len(graph_rows),
            "raw_authority_count": sum(
                row.get("source_authority") == "claude_pair_adjudication" for row in canonical_rows
            ),
            "legacy_scored_supplement_count": sum(
                row.get("source_authority") == "legacy_scored_supplement" for row in canonical_rows
            ),
        },
        "outputs": {
            "tool_catalog": "tool_catalog.jsonl",
            "edge_decisions": "edge_decisions.jsonl",
            "graph": "graph.jsonl",
            "issues": "issues.jsonl" if issues else None,
        },
    }
    write_json(output_dir / "run_manifest.json", report)
    return report
