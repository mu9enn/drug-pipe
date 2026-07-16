from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_json, read_jsonl, write_json, write_jsonl
from .relation_utils import context_from_legacy_fields, normalize_edge_types, normalize_relation_status
from .settings import ProjectConfig


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _confidence_raw(adj: dict[str, Any]) -> tuple[float, dict[str, float]]:
    s_agent = _clip01(float(adj.get("agent_confidence", 0.5)))
    return s_agent, {"agent": s_agent}


def _apply_calibration(raw: float, calibration_bins: list[tuple[float, float]] | None = None) -> float:
    if not calibration_bins:
        return raw
    selected = calibration_bins[0][1]
    for th, val in calibration_bins:
        if raw >= th:
            selected = val
    return _clip01(selected)


def score_edges(config: ProjectConfig, calibration_file: Path | None = None) -> dict[str, Any]:
    adjud = read_jsonl(config.paths.run_dir / "pair_adjudications.jsonl")

    calibration_bins = None
    if calibration_file and calibration_file.exists():
        obj = read_json(calibration_file)
        bins = obj.get("bins") if isinstance(obj, dict) else None
        if isinstance(bins, list):
            calibration_bins = [(float(x["threshold"]), float(x["value"])) for x in bins if "threshold" in x and "value" in x]

    scored_rows = []
    for a in adjud:
        pid = a["pair_id"]
        relation_status = normalize_relation_status(a.get("relation_status", a.get("decision")))
        edge_types = normalize_edge_types(a.get("edge_types", []))
        edge_evidence_ids = sorted(
            {
                eid
                for et in edge_types
                if isinstance(et, dict)
                for eid in (et.get("evidence_ids") or [])
                if isinstance(eid, str) and eid.strip()
            }
        )

        raw, comp = _confidence_raw(a)
        cal = _apply_calibration(raw, calibration_bins)

        scored_rows.append(
            {
                "pair_id": pid,
                "source_tool": a.get("source_tool"),
                "target_tool": a.get("target_tool"),
                "relation_status": relation_status,
                "direct_transition": a.get("direct_transition"),
                "edge_types": edge_types,
                "context": str(a.get("context") or context_from_legacy_fields(a)),
                "satisfied_mappings": a.get("satisfied_mappings", []),
                "unsatisfied_required_inputs": a.get("unsatisfied_required_inputs", []),
                "negative_reason": a.get("negative_reason"),
                "evidence_ids": edge_evidence_ids,
                "evidence_refs": a.get("evidence_refs", []),
                "confidence_raw": round(raw, 6),
                "confidence_calibrated": round(cal, 6),
                "components": comp,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_id": config.paths.run_dir.name,
            }
        )

    out = config.paths.run_dir / "scored_edges.jsonl"
    write_jsonl(out, scored_rows)

    summary = {
        "scored_count": len(scored_rows),
        "avg_conf_raw": round(sum(r["confidence_raw"] for r in scored_rows) / max(1, len(scored_rows)), 6),
        "avg_conf_calibrated": round(sum(r["confidence_calibrated"] for r in scored_rows) / max(1, len(scored_rows)), 6),
        "output": str(out),
    }
    write_json(config.paths.run_dir / "scoring_meta.json", summary)
    return summary
