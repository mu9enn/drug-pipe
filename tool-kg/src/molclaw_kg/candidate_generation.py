from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any

from .io_utils import read_jsonl, write_json, write_jsonl
from .models import CandidatePair
from .settings import ProjectConfig
from .stage_taxonomy import load_stage_taxonomy


def _build_semantic_relations(sem_cfg: dict[str, Any]):
    sub = {tuple(x) for x in sem_cfg.get("subtype_pairs", [])}
    conv = {tuple(x) for x in sem_cfg.get("convertible_pairs", [])}
    # symmetrize
    sub |= {(b, a) for a, b in list(sub)}
    conv |= {(b, a) for a, b in list(conv)}
    return sub, conv


def _sem_score(a: str, b: str, sem_cfg: dict[str, Any], sub, conv) -> float:
    comp = sem_cfg.get("compatibility", {})
    if not a or not b or a == "unknown" or b == "unknown":
        return 0.35
    if a == b:
        return float(comp.get("exact", 1.0))
    if (a, b) in sub:
        return float(comp.get("subtype", 0.85))
    if (a, b) in conv:
        return float(comp.get("convertible", 0.55))
    return float(comp.get("incompatible", 0.0))


def _fmt_score(fmt_a: str, fmt_b: str, sem_cfg: dict[str, Any]) -> float:
    if fmt_a == "unknown" or fmt_b == "unknown":
        return 0.5
    if fmt_a == fmt_b:
        return 1.0
    compat = sem_cfg.get("format_compatibility", {})
    if fmt_a in compat and fmt_b in compat.get(fmt_a, []):
        return 0.7
    return 0.2


def _name_score(name_a: str, name_b: str) -> float:
    ta = set(name_a.lower().replace("_", " ").split())
    tb = set(name_b.lower().replace("_", " ").split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union


def _compat_slot(out_slot: dict[str, Any], in_slot: dict[str, Any], sem_cfg: dict[str, Any], sub, conv) -> float:
    sem = _sem_score(out_slot.get("semantic_type", "unknown"), in_slot.get("semantic_type", "unknown"), sem_cfg, sub, conv)
    fmt = _fmt_score(out_slot.get("format", "unknown"), in_slot.get("format", "unknown"), sem_cfg)
    name = _name_score(out_slot.get("name", ""), in_slot.get("name", ""))

    # simplified weighted combination
    score = 0.55 * sem + 0.30 * fmt + 0.15 * name
    return max(0.0, min(1.0, score))


def _schema_pair_score(src_card: dict[str, Any], dst_card: dict[str, Any], sem_cfg: dict[str, Any]) -> tuple[float, list[str]]:
    sub, conv = _build_semantic_relations(sem_cfg)

    out_slots = (src_card.get("outputs") or []) + (src_card.get("side_effects") or [])
    in_slots = dst_card.get("inputs") or []
    req_slots = [s for s in in_slots if s.get("required")]
    target_slots = req_slots if req_slots else in_slots

    if not target_slots:
        return 0.4, []

    coverage = []
    matched: list[str] = []

    for ins in target_slots:
        best = 0.0
        best_out_name = None
        for outs in out_slots:
            sc = _compat_slot(outs, ins, sem_cfg, sub, conv)
            if sc > best:
                best = sc
                best_out_name = outs.get("name")
        coverage.append(best)
        if best_out_name and best >= 0.55:
            matched.append(f"{best_out_name}->{ins.get('name')}")

    avg_cov = sum(coverage) / len(coverage)
    miss = sum(1 for x in coverage if x < 0.45) / len(coverage)
    score = avg_cov - 0.25 * miss
    return max(0.0, min(1.0, score)), matched


def _suggest_edge_type(src_card: dict[str, Any], dst_card: dict[str, Any], matched_slots: list[str], schema_score: float) -> str:
    st = str(src_card.get("primary_stage", ""))
    dt = str(dst_card.get("primary_stage", ""))
    st_l = st.lower()
    dt_l = dt.lower()

    if "reporting" in st_l:
        return "reports_or_summarizes"
    if "binding_site" in st_l or "pocket" in src_card.get("tool_id", ""):
        return "parameterizes_for"
    if "convert" in src_card.get("tool_id", ""):
        return "converts_format_for"
    if "preparation" in st_l or "postprocessing" in st_l or "mapping" in st_l:
        return "preprocesses_for"
    if "filtering" in st_l:
        return "filters_candidates_for"
    if "rescoring" in st_l or "screening" in st_l or "ranking" in st_l:
        return "ranks_or_scores_for"
    if "validation" in st_l:
        return "validates_output_of"
    if "pose_generation" in st_l and ("rescoring" in dt_l or "screening" in dt_l):
        return "generates_full_input_for" if schema_score >= 0.70 else "generates_partial_input_for"
    return "generates_full_input_for" if schema_score >= 0.70 else "generates_partial_input_for"


def generate_candidates(config: ProjectConfig) -> dict[str, Any]:
    cards = read_jsonl(config.paths.run_dir / "tool_cards.jsonl")

    sem_cfg = config.semantic_types
    taxonomy = load_stage_taxonomy(config.stage_taxonomy_path)
    rule_cfg = config.rules
    threshold_schema = float(rule_cfg.get("thresholds", {}).get("schema_candidate_min", 0.45))
    high_schema = float(rule_cfg.get("thresholds", {}).get("schema_high_priority", 0.70))

    tool_map = {c["tool_id"]: c for c in cards}
    tools = sorted(tool_map.keys())
    all_pairs = list(itertools.permutations(tools, 2))

    pair_cache: dict[tuple[str, str], dict[str, Any]] = {}
    schema_cache: dict[tuple[str, str], tuple[float, list[str]]] = {}

    # 1) schema-based candidate generation over all ordered pairs
    for a, b in all_pairs:
        ca, cb = tool_map[a], tool_map[b]
        schema_score, matched = _schema_pair_score(ca, cb, sem_cfg)
        schema_cache[(a, b)] = (schema_score, matched)
        if schema_score >= threshold_schema:
            pair_cache[(a, b)] = {
                "source_tool": a,
                "target_tool": b,
                "source_stage": ca.get("primary_stage", "simulation_prediction"),
                "target_stage": cb.get("primary_stage", "simulation_prediction"),
                "source": ["schema_match"],
                "schema_score": schema_score,
                "suggested_edge_types": [_suggest_edge_type(ca, cb, matched, schema_score)],
                "negative_reason": None,
            }

    # 2) negative candidate generation for uncovered pairs
    for a, b in all_pairs:
        if (a, b) in pair_cache:
            continue
        ca, cb = tool_map[a], tool_map[b]
        s_stage = ca.get("primary_stage", "simulation_prediction")
        t_stage = cb.get("primary_stage", "simulation_prediction")
        reason = None
        if s_stage != t_stage and not taxonomy.is_transition_allowed(str(s_stage), str(t_stage)):
            reason = "stage_transition_not_allowed"
        else:
            sc = schema_cache[(a, b)][0]
            if sc < 0.15:
                reason = "requires_intermediate"
        if reason:
            pair_cache[(a, b)] = {
                "source_tool": a,
                "target_tool": b,
                "source_stage": s_stage,
                "target_stage": t_stage,
                "source": ["negative_generator"],
                "schema_score": 0.0,
                "suggested_edge_types": [],
                "negative_reason": reason,
            }

    # 3) deterministic alternative candidates from taxonomy clusters
    for src, tgt, relation, cluster_id in taxonomy.alternative_pairs():
        if relation != "alternative_to":
            continue
        if src not in tool_map or tgt not in tool_map:
            continue
        if (src, tgt) not in pair_cache:
            pair_cache[(src, tgt)] = {
                "source_tool": src,
                "target_tool": tgt,
                "source_stage": tool_map[src].get("primary_stage", "simulation_prediction"),
                "target_stage": tool_map[tgt].get("primary_stage", "simulation_prediction"),
                "source": ["alternative_cluster"],
                "schema_score": 0.0,
                "suggested_edge_types": ["alternative_to"],
                "negative_reason": None,
                "cluster_id": cluster_id,
            }
        else:
            rec = pair_cache[(src, tgt)]
            rec.setdefault("source", [])
            if "alternative_cluster" not in rec["source"]:
                rec["source"].append("alternative_cluster")
            rec.setdefault("suggested_edge_types", [])
            if "alternative_to" not in rec["suggested_edge_types"]:
                rec["suggested_edge_types"].append("alternative_to")

    # 4) finalize candidates
    candidates: list[CandidatePair] = []
    for a, b in sorted(pair_cache):
        rec = pair_cache[(a, b)]
        pid = f"pair::{a}__to__{b}"
        c = CandidatePair(
            pair_id=pid,
            source_tool=rec["source_tool"],
            target_tool=rec["target_tool"],
            source_stage=rec["source_stage"],
            target_stage=rec["target_stage"],
            source=rec["source"],
            schema_score=float(rec["schema_score"]),
            suggested_edge_types=sorted(set(rec["suggested_edge_types"])),
            negative_reason=rec.get("negative_reason"),
        )

        if c.schema_score >= high_schema and "schema_high_priority" not in c.source:
            c.source.append("schema_high_priority")

        candidates.append(c)

    out = config.paths.run_dir / "candidate_pairs.jsonl"
    write_jsonl(out, [c.model_dump() for c in candidates])

    source_stats = defaultdict(int)
    for c in candidates:
        for s in c.source:
            source_stats[s] += 1

    write_json(
        config.paths.run_dir / "candidate_meta.json",
        {
            "candidate_count": len(candidates),
            "all_ordered_pair_count": len(all_pairs),
            "source_breakdown": dict(sorted(source_stats.items())),
            "path": str(out),
        },
    )

    return {
        "candidate_count": len(candidates),
        "path": str(out),
    }
