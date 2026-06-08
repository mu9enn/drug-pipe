from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, write_csv, write_json
from .settings import ProjectConfig


def _view_rank(view: str) -> int:
    order = {"core": 4, "expanded": 3, "negative": 2, "uncertain": 1}
    return order.get(str(view), 0)


def _serialize_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _build_debug_context_index(sidecar_rows: list[dict[str, Any]]) -> dict[str, str]:
    by_pair: dict[str, list[str]] = defaultdict(list)
    for r in sidecar_rows:
        pid = str(r.get("pair_id") or "").strip()
        ctx = str(r.get("context") or "").strip()
        if not pid or not ctx:
            continue
        if ctx not in by_pair[pid]:
            by_pair[pid].append(ctx)
    return {k: " | ".join(v) for k, v in by_pair.items()}


def _aggregate_pair_rows(rows: list[dict[str, Any]], pair_contexts: dict[str, str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        src = str(r.get("source_tool") or "")
        tgt = str(r.get("target_tool") or "")
        pair_id = str(r.get("pair_id") or f"pair::{src}__to__{tgt}")
        grouped[(src, tgt, pair_id)].append(r)

    out: list[dict[str, Any]] = []
    for (src, tgt, pair_id), rs in sorted(grouped.items()):
        rs_sorted = sorted(rs, key=lambda x: float(x.get("confidence_calibrated", 0.0)), reverse=True)
        edge_types = [str(x.get("edge_type") or "") for x in rs_sorted]
        edge_confidences = [round(float(x.get("confidence_calibrated", 0.0)), 6) for x in rs_sorted]
        relation_statuses = [str(x.get("relation_status") or "uncertain") for x in rs_sorted]

        max_conf = max(edge_confidences) if edge_confidences else 0.0
        min_conf = min(edge_confidences) if edge_confidences else 0.0
        view = max((str(x.get("view") or "uncertain") for x in rs_sorted), key=_view_rank)

        metadata = {
            "pair_id": pair_id,
            "context": pair_contexts.get(pair_id, ""),
            "relation_statuses": sorted(set(relation_statuses)),
        }

        out.append(
            {
                "Source": src,
                "Target": tgt,
                "pair_id": pair_id,
                "edge_types": _serialize_json(edge_types),
                "edge_confidences": _serialize_json(edge_confidences),
                "max_confidence": round(max_conf, 6),
                "min_confidence": round(min_conf, 6),
                "view": view,
                "metadata": _serialize_json(metadata),
            }
        )

    return out


def export_csv(config: ProjectConfig) -> dict[str, str]:
    rows = read_jsonl(config.paths.run_dir / "graph_all.jsonl")
    sidecar_rows = read_jsonl(config.paths.run_dir / "edge_debug_sidecar.jsonl")
    pair_contexts = _build_debug_context_index(sidecar_rows)

    edge_flat = []
    for r in rows:
        edge_flat.append(
            {
                "edge_id": r.get("edge_id"),
                "source_tool": r.get("source_tool"),
                "target_tool": r.get("target_tool"),
                "pair_id": r.get("pair_id"),
                "relation_status": r.get("relation_status"),
                "edge_type": r.get("edge_type"),
                "direct_transition": r.get("direct_transition"),
                "stage_src": r.get("stage_src"),
                "stage_tgt": r.get("stage_tgt"),
                "confidence_raw": r.get("confidence_raw"),
                "confidence_calibrated": r.get("confidence_calibrated"),
                "view": r.get("view"),
                "negative_reason": r.get("negative_reason"),
                "evidence_count": len(r.get("evidence_ids", [])),
            }
        )
    edge_out = config.paths.run_dir / "graph_all.csv"
    write_csv(edge_out, edge_flat)

    pair_rows = _aggregate_pair_rows(rows, pair_contexts)
    pair_out = config.paths.run_dir / f"{config.paths.run_dir.name}.csv"
    write_csv(pair_out, pair_rows)
    return {"edge_csv": str(edge_out), "pair_csv": str(pair_out)}


def export_graphml(config: ProjectConfig) -> dict[str, str]:
    rows = read_jsonl(config.paths.run_dir / "graph_all.jsonl")
    ns = "http://graphml.graphdrawing.org/xmlns"
    ET.register_namespace("", ns)
    graphml = ET.Element(f"{{{ns}}}graphml")

    node_keys = [("label", "string")]
    edge_keys = [
        ("edge_type", "string"),
        ("direct_transition", "boolean"),
        ("view", "string"),
        ("confidence", "double"),
        ("relation_status", "string"),
        ("pair_id", "string"),
    ]

    for key, typ in node_keys:
        ET.SubElement(graphml, f"{{{ns}}}key", id=key, **{"for": "node", "attr.name": key, "attr.type": typ})
    for key, typ in edge_keys:
        ET.SubElement(graphml, f"{{{ns}}}key", id=key, **{"for": "edge", "attr.name": key, "attr.type": typ})

    graph = ET.SubElement(graphml, f"{{{ns}}}graph", edgedefault="directed", id="molclaw_tool_graph")

    tools = sorted({r["source_tool"] for r in rows} | {r["target_tool"] for r in rows})
    for t in tools:
        n = ET.SubElement(graph, f"{{{ns}}}node", id=t)
        ET.SubElement(n, f"{{{ns}}}data", key="label").text = t

    for r in rows:
        e = ET.SubElement(graph, f"{{{ns}}}edge", id=r["edge_id"], source=r["source_tool"], target=r["target_tool"])
        ET.SubElement(e, f"{{{ns}}}data", key="edge_type").text = str(r.get("edge_type"))
        ET.SubElement(e, f"{{{ns}}}data", key="direct_transition").text = str(bool(r.get("direct_transition", False))).lower()
        ET.SubElement(e, f"{{{ns}}}data", key="view").text = str(r.get("view"))
        ET.SubElement(e, f"{{{ns}}}data", key="confidence").text = str(float(r.get("confidence_calibrated", 0.0)))
        ET.SubElement(e, f"{{{ns}}}data", key="relation_status").text = str(r.get("relation_status", "uncertain"))
        ET.SubElement(e, f"{{{ns}}}data", key="pair_id").text = str(r.get("pair_id", ""))

    out = config.paths.run_dir / "graph_all.graphml"
    ET.ElementTree(graphml).write(out, encoding="utf-8", xml_declaration=True)
    return {"graphml": str(out)}


def export_artifacts(config: ProjectConfig) -> dict[str, str]:
    a = export_csv(config)
    b = export_graphml(config)
    result = {**a, **b}
    write_json(config.paths.run_dir / "export_meta.json", result)
    return result
