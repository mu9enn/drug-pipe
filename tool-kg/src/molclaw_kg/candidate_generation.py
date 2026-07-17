from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any

from .io_utils import read_jsonl, write_json, write_jsonl
from .models import CandidatePair
from .settings import ProjectConfig
from .stage_taxonomy import load_stage_taxonomy


def _interface_slots(card: dict[str, Any], direction: str) -> list[dict[str, Any]]:
    if direction == "output":
        keys = ("connectable_outputs", "outputs", "side_effects")
    else:
        keys = ("connectable_inputs", "inputs", "preconditions")
    for key in keys:
        rows = card.get(key)
        if isinstance(rows, list) and rows:
            return [row for row in rows if isinstance(row, dict)]
    return []


def generate_candidates(config: ProjectConfig) -> dict[str, Any]:
    cards = read_jsonl(config.paths.run_dir / "tool_cards.jsonl")

    taxonomy = load_stage_taxonomy(config.stage_taxonomy_path)
    tool_map = {c["tool_id"]: c for c in cards}
    tools = sorted(tool_map.keys())
    all_pairs = list(itertools.permutations(tools, 2))
    alternative_pairs: set[tuple[str, str]] = set()
    for src, tgt, relation, _cluster_id in taxonomy.alternative_pairs():
        if relation == "alternative_to" and src in tool_map and tgt in tool_map:
            alternative_pairs.add((src, tgt))

    candidates: list[CandidatePair] = []
    excluded: list[dict[str, Any]] = []
    for a, b in all_pairs:
        src_card, tgt_card = tool_map[a], tool_map[b]
        has_output_interface = bool(_interface_slots(src_card, "output"))
        has_input_interface = bool(_interface_slots(tgt_card, "input"))
        reasons: list[str] = []
        if has_output_interface and has_input_interface:
            reasons.append("potential_io_interface")
        if (a, b) in alternative_pairs:
            reasons.append("taxonomy_pair_requires_adjudication")
        if not reasons:
            excluded.append(
                {
                    "pair_id": f"pair::{a}__to__{b}",
                    "source_tool": a,
                    "target_tool": b,
                    "reason": "no_declared_output_or_input_interface",
                    "recall_risk": "documentation-only or control-flow relations may be missed",
                }
            )
            continue
        pid = f"pair::{a}__to__{b}"
        candidates.append(
            CandidatePair(
                pair_id=pid,
                source_tool=a,
                target_tool=b,
                source_stage=str(src_card.get("primary_stage", "simulation_prediction")),
                target_stage=str(tgt_card.get("primary_stage", "simulation_prediction")),
                proposal_reasons=reasons,
                recall_risk=None,
            )
        )

    out = config.paths.run_dir / "candidate_pairs.jsonl"
    write_jsonl(out, [c.model_dump() for c in candidates])
    write_jsonl(config.paths.run_dir / "candidate_pairs_excluded.jsonl", excluded)

    source_stats = defaultdict(int)
    for c in candidates:
        for s in c.proposal_reasons:
            source_stats[s] += 1

    write_json(
        config.paths.run_dir / "candidate_meta.json",
        {
            "candidate_count": len(candidates),
            "all_ordered_pair_count": len(all_pairs),
            "excluded_count": len(excluded),
            "source_breakdown": dict(sorted(source_stats.items())),
            "semantic_decisions_made": 0,
            "recall_risk": (
                "Pairs with no declared source output interface or target input interface are excluded; "
                "documentation-only or control-flow relations may be missed."
            ),
            "path": str(out),
        },
    )

    return {
        "candidate_count": len(candidates),
        "path": str(out),
    }
