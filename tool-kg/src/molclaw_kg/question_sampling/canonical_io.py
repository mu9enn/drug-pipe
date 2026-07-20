from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..io_utils import read_json, read_jsonl, write_json


def load_canonical_sampling_inputs(
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    results_dir = run_dir / "results"
    paths = {
        "graph": results_dir / "graph.jsonl",
        "tool_catalog": results_dir / "tool_catalog.jsonl",
        "edge_decisions": results_dir / "edge_decisions.jsonl",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    graph = read_jsonl(paths["graph"])
    cards = read_jsonl(paths["tool_catalog"])
    decisions = read_jsonl(paths["edge_decisions"])
    decision_context: list[dict[str, Any]] = []
    for decision in decisions:
        for edge in decision.get("edge_types") or []:
            if not isinstance(edge, dict) or not edge.get("type"):
                continue
            decision_context.append(
                {
                    "pair_id": decision.get("pair_id"),
                    "source_tool": decision.get("source_tool"),
                    "target_tool": decision.get("target_tool"),
                    "edge_type": edge.get("type"),
                    "satisfied_mappings": decision.get("satisfied_inputs") or [],
                    "unsatisfied_required_inputs": decision.get("unsatisfied_inputs") or [],
                    "evidence": edge.get("evidence_ids") or decision.get("evidence") or [],
                    "source_authority": decision.get("source_authority"),
                }
            )
    return graph, cards, decision_context


def _simple_expected_trajectory(row: dict[str, Any]) -> dict[str, Any] | None:
    tools = row.get("hidden_toolchain_nodes")
    edges = row.get("hidden_toolchain_edges")
    if not isinstance(tools, list) or not tools:
        return None
    node_ids = {str(tool): f"tool::{index:02d}::{tool}" for index, tool in enumerate(tools, start=1)}
    graph_edges = []
    for index, edge in enumerate(edges if isinstance(edges, list) else [], start=1):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source_tool") or "")
        target = str(edge.get("target_tool") or "")
        if source not in node_ids or target not in node_ids:
            continue
        graph_edges.append(
            {
                "edge_id": f"edge::toolchain::{index:02d}",
                "source": node_ids[source],
                "target": node_ids[target],
                "relation": edge.get("edge_type") or "workflow_transition",
            }
        )
    payload = row.get("question_payload") if isinstance(row.get("question_payload"), dict) else {}
    return {
        "schema_version": "trajectory_v2_graph",
        "workflow_graph": {
            "nodes": [
                {"node_id": node_ids[str(tool)], "type": "tool", "label": str(tool), "tool_id": str(tool)}
                for tool in tools
            ],
            "edges": graph_edges,
        },
        "execution_plan": {
            "topological_order": [node_ids[str(tool)] for tool in tools],
            "tool_order": [str(tool) for tool in tools],
        },
        "final_deliverable": payload.get("expected_output"),
    }


def canonical_task(row: dict[str, Any], run_id: str) -> dict[str, Any]:
    task_id = str(row.get("sample_id") or row.get("id") or "")
    expected = row.get("expected_trajectory")
    if not isinstance(expected, dict):
        expected = _simple_expected_trajectory(row)
    return {
        "schema_version": "tool_kg_task_v1",
        "id": task_id,
        "task_type": row.get("task_type") or "toolchain_derived",
        "public_question_text": row.get("public_question_text"),
        "question_payload": row.get("question_payload") or {},
        "toolchain_nodes": row.get("toolchain_nodes") or row.get("hidden_toolchain_nodes") or [],
        "toolchain_edges": row.get("toolchain_edges") or row.get("hidden_toolchain_edges") or [],
        "walk_hops": row.get("walk_hops"),
        "start_tool": row.get("start_tool")
        or ((row.get("hidden_toolchain_nodes") or [None])[0]),
        "end_tool": row.get("end_tool")
        or ((row.get("hidden_toolchain_nodes") or [None])[-1]),
        "expected_trajectory": expected,
        "grounded_initial_inputs": row.get("grounded_initial_inputs") or [],
        "grounding_refs": row.get("grounding_refs") or [],
        "grounding_sources": row.get("grounding_sources") or [],
        "source_run_id": run_id,
    }


def update_manifest_tasks(
    run_dir: Path,
    task_count: int,
    sampling_profile_meta: dict[str, Any] | None = None,
) -> None:
    manifest_path = run_dir / "results" / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    manifest.setdefault("counts", {})["tasks"] = int(task_count)
    manifest.setdefault("outputs", {})["tasks"] = "tasks.jsonl"
    if sampling_profile_meta:
        manifest["stage3_sampling"] = sampling_profile_meta
    write_json(manifest_path, manifest)
