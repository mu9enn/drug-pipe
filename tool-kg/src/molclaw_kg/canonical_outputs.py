from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, sha256_file, write_json, write_jsonl
from .settings import ProjectConfig
from .stage_taxonomy import resolve_stage_taxonomy_path


def canonical_edge_to_decision(row: dict[str, Any]) -> dict[str, Any]:
    source_authority = str(row.get("source_authority") or "")
    if source_authority == "claude_pair_adjudication":
        source_authority = "claude_adjudication"
    edge_types = row.get("edge_types") if isinstance(row.get("edge_types"), list) else []
    eligible = (
        source_authority == "claude_adjudication"
        and row.get("relation_status") == "valid"
        and bool(row.get("direct_transition"))
        and bool(edge_types)
    )
    return {
        "schema_version": "tool_kg_edge_decision_v1",
        "pair_id": row.get("pair_id"),
        "source_tool": row.get("source_tool"),
        "target_tool": row.get("target_tool"),
        "source_stage": row.get("source_stage"),
        "target_stage": row.get("target_stage"),
        "relation_status": row.get("relation_status"),
        "direct_transition": bool(row.get("direct_transition")),
        "edge_type": edge_types[0].get("type") if len(edge_types) == 1 and isinstance(edge_types[0], dict) else None,
        "edge_types": edge_types,
        "satisfied_inputs": row.get("satisfied_mappings") or [],
        "unsatisfied_inputs": row.get("unsatisfied_required_inputs") or [],
        "confidence": float(row.get("confidence_calibrated", row.get("confidence_raw", 0.0))),
        "confidence_raw": float(row.get("confidence_raw", 0.0)),
        "evidence": row.get("evidence_refs") or row.get("evidence_ids") or [],
        "rationale": row.get("rationale") or "",
        "negative_reason": row.get("negative_reason"),
        "source_authority": source_authority,
        "eligible_for_sampling": eligible,
        "agent_model": row.get("agent_model"),
        "source_created_at_utc": row.get("source_created_at_utc"),
    }


def project_graph(decisions: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    graph: list[dict[str, Any]] = []
    for decision in decisions:
        if decision.get("relation_status") != "valid" or not decision.get("eligible_for_sampling"):
            continue
        for index, edge_type in enumerate(decision.get("edge_types") or [], start=1):
            if not isinstance(edge_type, dict) or not edge_type.get("type"):
                continue
            graph.append(
                {
                    "schema_version": "tool_kg_graph_edge_v1",
                    "edge_id": f"{decision['pair_id']}::edge::{index:02d}",
                    "pair_id": decision["pair_id"],
                    "source_tool": decision["source_tool"],
                    "target_tool": decision["target_tool"],
                    "source_stage": decision.get("source_stage"),
                    "target_stage": decision.get("target_stage"),
                    "edge_type": edge_type["type"],
                    "direct_transition": decision["direct_transition"],
                    "relation_status": decision["relation_status"],
                    "confidence": decision["confidence"],
                    "source_slot": edge_type.get("source_slot"),
                    "target_slot": edge_type.get("target_slot_or_precondition"),
                    "evidence": edge_type.get("evidence_ids") or decision.get("evidence") or [],
                    "source_authority": decision["source_authority"],
                    "eligible_for_sampling": True,
                    "run_id": run_id,
                }
            )
    return graph


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _config_hashes(config: ProjectConfig) -> dict[str, str]:
    paths = {
        "taxonomy": resolve_stage_taxonomy_path(config.paths.root),
        "edge_ontology": config.paths.configs / "edge_ontology.yaml",
    }
    return {name: sha256_file(path) for name, path in paths.items() if path.is_file()}


def _collect_issues(run_dir: Path) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    sources = {
        "tool_card_alert": run_dir / "tool_card_alerts.jsonl",
        "adjudication_alert": run_dir / "pair_adjudication_alerts.jsonl",
        "canonical_rejection": run_dir / "canonical_edges_rejected.jsonl",
    }
    for kind, path in sources.items():
        for row in read_jsonl(path):
            issues.append({"kind": kind, **row})
    return issues


def publish_canonical_outputs(config: ProjectConfig) -> dict[str, Any]:
    run_dir = config.paths.run_dir
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    tool_catalog = read_jsonl(run_dir / "tool_cards.jsonl")
    canonical_rows = read_jsonl(run_dir / "canonical_edges.jsonl")
    if not tool_catalog:
        raise RuntimeError("tool_cards.jsonl is missing or empty")
    if not canonical_rows:
        raise RuntimeError("canonical_edges.jsonl is missing or empty")

    decisions = [canonical_edge_to_decision(row) for row in canonical_rows]
    decisions.sort(key=lambda row: str(row.get("pair_id") or ""))
    graph = project_graph(decisions, run_dir.name)

    write_jsonl(results_dir / "tool_catalog.jsonl", tool_catalog)
    write_jsonl(results_dir / "edge_decisions.jsonl", decisions)
    write_jsonl(results_dir / "graph.jsonl", graph)

    issues = _collect_issues(run_dir)
    issues_path = results_dir / "issues.jsonl"
    if issues:
        write_jsonl(issues_path, issues)
    elif issues_path.exists():
        issues_path.unlink()

    manifest = {
        "schema_version": "tool_kg_run_manifest_v1",
        "run_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(config.paths.root),
        "taxonomy_path": str(resolve_stage_taxonomy_path(config.paths.root)),
        "config_hashes": _config_hashes(config),
        "inputs": {
            "tool_snapshot": str(run_dir / "tool_snapshot.jsonl"),
            "skills_root": str(config.runtime.skills_root),
        },
        "claude": {
            "model": config.runtime.model_name,
            "server_url": config.runtime.server_url,
        },
        "counts": {
            "tools": len(tool_catalog),
            "edge_decisions": len(decisions),
            "sampling_eligible_decisions": sum(bool(row["eligible_for_sampling"]) for row in decisions),
            "graph_edges": len(graph),
            "issues": len(issues),
        },
        "outputs": {
            "tool_catalog": "tool_catalog.jsonl",
            "edge_decisions": "edge_decisions.jsonl",
            "graph": "graph.jsonl",
            "tasks": "tasks.jsonl" if (results_dir / "tasks.jsonl").is_file() else None,
            "issues": "issues.jsonl" if issues else None,
        },
    }
    write_json(results_dir / "run_manifest.json", manifest)
    return {
        "results_dir": str(results_dir),
        "tool_count": len(tool_catalog),
        "edge_decision_count": len(decisions),
        "graph_edge_count": len(graph),
        "issue_count": len(issues),
    }
