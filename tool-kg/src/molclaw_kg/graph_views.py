from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .io_utils import read_jsonl, write_json, write_jsonl
from .models import FinalEdge
from .settings import ProjectConfig


def _edge_id(
    source_tool: str,
    target_tool: str,
    edge_type: str | None,
    source_slot: str | None,
    target_slot: str | None,
) -> str:
    raw = f"{source_tool}|{edge_type or ''}|{target_tool}|{source_slot or ''}|{target_slot or ''}".encode("utf-8")
    return "edge::" + hashlib.md5(raw).hexdigest()[:16]


def build_graph_views(config: ProjectConfig) -> dict[str, Any]:
    canonical = read_jsonl(config.paths.run_dir / "canonical_edges.jsonl")
    cards = {x["tool_id"]: x for x in read_jsonl(config.paths.run_dir / "tool_cards.jsonl")}

    configs_dir = getattr(config.paths, "configs", None)
    profile_path = (
        Path(configs_dir) / "legacy" / "graph_views_v1.yaml"
        if configs_dir is not None
        else Path()
    )
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8")) if profile_path.is_file() else {}
    th = (profile or {}).get("thresholds", {})
    core_min = float(th.get("core_min", 0.80))
    expanded_min = float(th.get("expanded_min", 0.55))
    final_edges: list[FinalEdge] = []
    debug_rows: list[dict[str, Any]] = []

    for row in canonical:
        s_tool = row["source_tool"]
        t_tool = row["target_tool"]
        pair_id = str(row.get("pair_id") or f"pair::{s_tool}__to__{t_tool}")
        relation_status = str(row["relation_status"])
        conf = float(row.get("confidence_raw", 0.0))
        src_stage = row.get("source_stage") or cards.get(s_tool, {}).get("primary_stage", "simulation_prediction")
        tgt_stage = row.get("target_stage") or cards.get(t_tool, {}).get("primary_stage", "simulation_prediction")

        if relation_status == "negative":
            view = "negative"
        elif relation_status == "alternative":
            view = "expanded" if conf >= expanded_min else "uncertain"
        elif relation_status == "uncertain":
            view = "uncertain"
        else:  # valid
            if conf >= core_min:
                view = "core"
            elif conf >= expanded_min:
                view = "expanded"
            else:
                view = "uncertain"

        edge_types = row.get("edge_types", [])
        if not edge_types:
            edge_types = [
                {
                    "type": None,
                    "source_slot": None,
                    "target_slot_or_precondition": None,
                    "confidence": conf,
                    "evidence_ids": row.get("evidence_ids", []),
                }
            ]

        for et in edge_types:
            etype = et.get("type")

            edge_id = _edge_id(s_tool, t_tool, etype, et.get("source_slot"), et.get("target_slot_or_precondition"))
            final_edges.append(
                FinalEdge(
                    edge_id=edge_id,
                    pair_id=pair_id,
                    source_tool=s_tool,
                    target_tool=t_tool,
                    edge_type=etype,
                    direct_transition=bool(row.get("direct_transition", False)),
                    source_slot=et.get("source_slot"),
                    target_slot=et.get("target_slot_or_precondition"),
                    stage_src=src_stage,
                    stage_tgt=tgt_stage,
                    relation_status=relation_status,  # type: ignore[arg-type]
                    confidence_raw=float(row.get("confidence_raw", conf)),
                    confidence_calibrated=conf,
                    view=view,  # type: ignore[arg-type]
                    evidence_ids=sorted(set(et.get("evidence_ids", []) or row.get("evidence_ids", []))),
                    negative_reason=row.get("negative_reason"),
                    created_at=str(row.get("source_created_at_utc") or ""),
                    run_id=config.paths.run_dir.name,
                )
            )
            debug_rows.append(
                {
                    "edge_id": edge_id,
                    "pair_id": pair_id,
                    "source_tool": s_tool,
                    "target_tool": t_tool,
                    "edge_type": etype,
                    "context": str(row.get("context") or ""),
                    "satisfied_mappings": row.get("satisfied_mappings", []),
                    "unsatisfied_required_inputs": row.get("unsatisfied_required_inputs", []),
                    "evidence_refs": row.get("evidence_refs", []),
                    "agent_conf": row.get("confidence_raw"),
                    "rationale": row.get("rationale", ""),
                    "source_authority": row.get("source_authority"),
                    "source_created_at_utc": row.get("source_created_at_utc"),
                }
            )

    all_rows = [e.model_dump() for e in final_edges]

    by_view = defaultdict(list)
    for r in all_rows:
        by_view[r["view"]].append(r)

    write_jsonl(config.paths.run_dir / "graph_all.jsonl", all_rows)
    write_jsonl(config.paths.run_dir / "edge_debug_sidecar.jsonl", debug_rows)
    for view in ["core", "expanded", "uncertain", "negative"]:
        write_jsonl(config.paths.run_dir / f"graph_{view}.jsonl", by_view.get(view, []))

    summary = {
        "source": str(config.paths.run_dir / "canonical_edges.jsonl"),
        "edge_count_all": len(all_rows),
        "semantic_rewrites": 0,
        "view_counts": {k: len(v) for k, v in sorted(by_view.items())},
        "outputs": {
            "all": str(config.paths.run_dir / "graph_all.jsonl"),
            "core": str(config.paths.run_dir / "graph_core.jsonl"),
            "expanded": str(config.paths.run_dir / "graph_expanded.jsonl"),
            "uncertain": str(config.paths.run_dir / "graph_uncertain.jsonl"),
            "negative": str(config.paths.run_dir / "graph_negative.jsonl"),
            "edge_debug_sidecar": str(config.paths.run_dir / "edge_debug_sidecar.jsonl"),
        },
    }
    write_json(config.paths.run_dir / "graph_views_meta.json", summary)
    return summary
