from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import write_json
from .settings import ProjectConfig


PREFIXES = (
    "mcp__molclaw-scp__",
    "mcp__molclaw-vs__",
)


def _strip_prefix(name: str) -> str:
    for p in PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return ""


def _extract_pairs_from_jsonl(path: Path) -> list[tuple[str, str]]:
    seq = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            msg = obj.get("message", {}) if isinstance(obj, dict) else {}
            content = msg.get("content", []) if isinstance(msg, dict) else []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    name = item.get("name")
                    if isinstance(name, str):
                        s = _strip_prefix(name)
                        if s:
                            seq.append(s)
    return list(zip(seq, seq[1:]))


def evaluate_against_logs(config: ProjectConfig) -> dict[str, Any]:
    graph_rows = []
    for fn in ["graph_core.jsonl", "graph_expanded.jsonl", "graph_all.jsonl"]:
        p = config.paths.run_dir / fn
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        graph_rows.append(json.loads(line))

    graph_pairs = {(r["source_tool"], r["target_tool"]) for r in graph_rows}

    log_root = config.runtime.logs_root
    files = sorted(log_root.rglob("*.jsonl")) if log_root.exists() else []

    pair_counter = Counter()
    for fp in files:
        for pr in _extract_pairs_from_jsonl(fp):
            pair_counter[pr] += 1

    if not pair_counter:
        summary = {
            "log_pair_count": 0,
            "graph_pair_count": len(graph_pairs),
            "coverage": None,
            "note": "no log pairs found",
        }
        write_json(config.paths.run_dir / "log_evaluation.json", summary)
        return summary

    total_obs = sum(pair_counter.values())
    covered_obs = sum(c for p, c in pair_counter.items() if p in graph_pairs)
    unique_cov = sum(1 for p in pair_counter if p in graph_pairs)

    misses = sorted(
        [
            {"source_tool": p[0], "target_tool": p[1], "count": c}
            for p, c in pair_counter.items()
            if p not in graph_pairs
        ],
        key=lambda x: -x["count"],
    )

    summary = {
        "log_file_count": len(files),
        "log_unique_pair_count": len(pair_counter),
        "log_total_pair_observations": total_obs,
        "graph_pair_count": len(graph_pairs),
        "coverage_unique_pair": unique_cov / len(pair_counter),
        "coverage_observation_weighted": covered_obs / total_obs,
        "top_missed_pairs": misses[:100],
    }

    write_json(config.paths.run_dir / "log_evaluation.json", summary)
    return summary
