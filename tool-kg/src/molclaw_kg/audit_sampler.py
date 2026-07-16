from __future__ import annotations

import random
from pathlib import Path

from .io_utils import read_jsonl, write_csv, write_json
from .settings import ProjectConfig


def sample_for_audit(config: ProjectConfig, seed: int = 20260517) -> dict[str, int]:
    random.seed(seed)

    views = {
        "core": read_jsonl(config.paths.run_dir / "graph_core.jsonl"),
        "expanded": read_jsonl(config.paths.run_dir / "graph_expanded.jsonl"),
        "uncertain": read_jsonl(config.paths.run_dir / "graph_uncertain.jsonl"),
        "negative": read_jsonl(config.paths.run_dir / "graph_negative.jsonl"),
    }

    quotas = {
        "core": 120,
        "expanded": 80,
        "uncertain": 60,
        "negative": 40,
    }

    sampled = []
    for view, rows in views.items():
        k = min(quotas[view], len(rows))
        if k <= 0:
            continue
        pick = random.sample(rows, k=k)
        for r in pick:
            sampled.append(
                {
                    "view": view,
                    "edge_id": r.get("edge_id"),
                    "source_tool": r.get("source_tool"),
                    "target_tool": r.get("target_tool"),
                    "edge_type": r.get("edge_type"),
                    "confidence": r.get("confidence_calibrated"),
                    "pair_id": r.get("pair_id") or f"pair::{r.get('source_tool')}__to__{r.get('target_tool')}",
                    "valid_label": "",
                    "edge_type_label": "",
                    "direct_transition_label": "",
                    "io_mapping_complete": "",
                    "evidence_sufficiency": "",
                    "stage_ok": "",
                    "negative_reason_label": "",
                    "reviewer_id": "",
                    "adjudication_note": "",
                }
            )

    out = config.paths.run_dir / "audit_sample.csv"
    write_csv(out, sampled)
    summary = {
        "sample_count": len(sampled),
        "output": str(out),
        "view_breakdown": {k: min(quotas[k], len(v)) for k, v in views.items()},
    }
    write_json(config.paths.run_dir / "audit_sample_meta.json", summary)
    return {"sample_count": len(sampled)}
