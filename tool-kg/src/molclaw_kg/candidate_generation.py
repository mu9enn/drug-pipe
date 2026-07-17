from __future__ import annotations

import itertools
from collections import defaultdict

from .io_utils import read_jsonl, write_json, write_jsonl
from .models import CandidatePair
from .settings import ProjectConfig
from .stage_taxonomy import load_stage_taxonomy


def _cached_pair_ids(config: ProjectConfig) -> set[str]:
    cached: set[str] = set()
    for row in read_jsonl(config.paths.run_dir / "pair_adjudications.jsonl"):
        pair_id = str(row.get("pair_id") or "").strip()
        if pair_id:
            cached.add(pair_id)
    for row in read_jsonl(config.paths.run_dir / "pairwise_cache.jsonl"):
        response = row.get("response") if isinstance(row.get("response"), dict) else {}
        pair_id = str(response.get("pair_id") or "").strip()
        if pair_id:
            cached.add(pair_id)
    return cached


def generate_candidates(config: ProjectConfig) -> dict[str, Any]:
    cards = read_jsonl(config.paths.run_dir / "tool_cards.jsonl")

    taxonomy = load_stage_taxonomy(config.stage_taxonomy_path)
    tool_map = {c["tool_id"]: c for c in cards}
    tools = sorted(tool_map.keys())
    all_pairs = list(itertools.permutations(tools, 2))
    alternative_pairs = {
        (src, tgt): (relation, cluster_id)
        for src, tgt, relation, cluster_id in taxonomy.alternative_pairs()
        if src in tool_map and tgt in tool_map
    }

    candidates: list[CandidatePair] = []
    excluded: list[dict[str, Any]] = []
    taxonomy_allowed_count = 0
    for a, b in all_pairs:
        src_card, tgt_card = tool_map[a], tool_map[b]
        source_stage = str(src_card["primary_stage"])
        target_stage = str(tgt_card["primary_stage"])
        taxonomy_allowed = taxonomy.is_transition_allowed(source_stage, target_stage)
        alternative = alternative_pairs.get((a, b))
        reasons: list[str] = []
        if taxonomy_allowed:
            reasons.append("taxonomy_allowed_transition")
            taxonomy_allowed_count += 1
        if alternative is not None:
            relation, cluster_id = alternative
            reasons.append(f"taxonomy_alternative:{relation}:{cluster_id}")
        if not reasons:
            excluded.append(
                {
                    "pair_id": f"pair::{a}__to__{b}",
                    "source_tool": a,
                    "target_tool": b,
                    "source_stage": source_stage,
                    "target_stage": target_stage,
                    "reason": "taxonomy_direction_not_scheduled",
                }
            )
            continue
        pid = f"pair::{a}__to__{b}"
        candidates.append(
            CandidatePair(
                pair_id=pid,
                source_tool=a,
                target_tool=b,
                source_stage=source_stage,
                target_stage=target_stage,
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

    cached_pair_ids = _cached_pair_ids(config)
    scheduled_pair_ids = {candidate.pair_id for candidate in candidates}
    cached_count = len(scheduled_pair_ids & cached_pair_ids)
    new_call_count = len(scheduled_pair_ids - cached_pair_ids)
    write_json(
        config.paths.run_dir / "candidate_meta.json",
        {
            "total_tools": len(tools),
            "all_directed_pairs": len(all_pairs),
            "taxonomy_allowed_pairs": taxonomy_allowed_count,
            "alternative_pairs": len(set(alternative_pairs) & set(all_pairs)),
            "cached_pairs": cached_count,
            "new_claude_calls": new_call_count,
            "candidate_count": len(candidates),
            "all_ordered_pair_count": len(all_pairs),
            "excluded_count": len(excluded),
            "source_breakdown": dict(sorted(source_stats.items())),
            "semantic_decisions_made": 0,
            "scheduling_authority": "stage_taxonomy",
            "tool_interfaces_used_as_gate": False,
            "path": str(out),
        },
    )

    return {
        "candidate_count": len(candidates),
        "cached_pairs": cached_count,
        "new_claude_calls": new_call_count,
        "path": str(out),
    }
