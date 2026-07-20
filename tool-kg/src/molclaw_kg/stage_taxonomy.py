from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import os

from .io_utils import read_json


@dataclass(frozen=True)
class StageTaxonomy:
    path: Path
    raw: dict[str, Any]
    stage_order: list[str]
    stages: dict[str, Any]
    tool_stage_map: dict[str, Any]
    allowed_stage_transitions: dict[str, list[str]]
    same_stage_policy: dict[str, Any]
    alternative_clusters: dict[str, Any]
    edge_type_stage_policy: dict[str, Any]
    pruning_policy: dict[str, Any]
    coverage_policy: dict[str, Any]

    def allowed_stages(self) -> set[str]:
        return set(self.stages.keys())

    def get_primary_stage(self, tool_id: str) -> str:
        info = self.tool_stage_map.get(tool_id)
        if isinstance(info, str):
            stage = info.strip()
            if stage not in self.allowed_stages():
                raise ValueError(f"invalid primary_stage={stage} for tool_id={tool_id}")
            return stage
        if not isinstance(info, dict):
            raise KeyError(f"tool_id is not mapped in stage taxonomy: {tool_id}")
        stage = str(info.get("primary_stage", "")).strip()
        if not stage:
            raise ValueError(f"primary_stage is missing for tool_id={tool_id}")
        if stage not in self.allowed_stages():
            raise ValueError(f"invalid primary_stage={stage} for tool_id={tool_id}")
        return stage

    def get_scheduling_stages(self, tool_id: str) -> list[str]:
        info = self.tool_stage_map.get(tool_id)
        if isinstance(info, str):
            return []
        if not isinstance(info, dict):
            raise KeyError(f"tool_id is not mapped in stage taxonomy: {tool_id}")
        values = info.get("scheduling_stages", info.get("secondary_stages", [])) or []
        out: list[str] = []
        for x in values:
            s = str(x).strip()
            if s:
                if s not in self.allowed_stages():
                    raise ValueError(f"invalid scheduling_stage={s} for tool_id={tool_id}")
                if s not in out:
                    out.append(s)
        return out

    def stage_order_index(self, stage: str) -> int:
        try:
            return self.stage_order.index(stage)
        except ValueError as exc:
            raise KeyError(f"unknown stage={stage}") from exc

    def validate_tool_coverage(self, tool_ids: Iterable[str]) -> None:
        expected = {str(x).strip() for x in tool_ids if str(x).strip()}
        mapped = set(self.tool_stage_map.keys())
        missing = sorted(expected - mapped)
        extra = sorted(mapped - expected)
        if missing:
            raise ValueError(f"stage taxonomy missing mappings for tools: {missing}")
        if extra:
            raise ValueError(f"stage taxonomy contains extra mappings not in snapshot: {extra}")

        exp_count = self.coverage_policy.get("expected_tool_count")
        if exp_count is not None:
            try:
                exp_n = int(exp_count)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"coverage_policy.expected_tool_count must be int: {exp_count}") from exc
            if exp_n != len(expected):
                raise ValueError(f"expected_tool_count={exp_n} but snapshot has {len(expected)} tools")

    def is_transition_allowed(self, source_stage: str, target_stage: str) -> bool:
        if source_stage == target_stage:
            transition_policy = str(
                (self.same_stage_policy or {}).get("transition_edges", "forbid")
            ).strip()
            return transition_policy not in {"forbid", "deny", "false", "0"}
        allowed = self.allowed_stage_transitions.get(source_stage, [])
        return target_stage in allowed

    def alternative_pairs(self) -> list[tuple[str, str, str, str]]:
        pairs: list[tuple[str, str, str, str]] = []
        for cluster_id, spec in self.alternative_clusters.items():
            if not isinstance(spec, dict):
                continue
            relation = str(spec.get("relation", "alternative_to")).strip() or "alternative_to"
            tools = [str(x).strip() for x in (spec.get("tools") or []) if str(x).strip()]
            for i in range(len(tools)):
                for j in range(len(tools)):
                    if i == j:
                        continue
                    pairs.append((tools[i], tools[j], relation, str(cluster_id)))
        return pairs


def load_stage_taxonomy(path: Path) -> StageTaxonomy:
    raw = read_json(path)
    stage_order = raw.get("stage_order")
    stages = raw.get("pruning_stages") or raw.get("stages")
    tool_stage_map = raw.get("tool_pruning_stage_map") or raw.get("tool_stage_map")
    allowed_stage_transitions = raw.get("allowed_stage_transitions", {}) or {}
    same_stage_policy = raw.get("same_pruning_stage_transition_policy", {}) or {}
    alternative_clusters = raw.get("alternative_clusters", {}) or {}
    edge_type_stage_policy = raw.get("edge_type_stage_policy", {}) or {}
    pruning_policy = raw.get("stage_pruning_policy", {}) or {}
    coverage_policy = raw.get("coverage_policy", {}) or {}

    if not isinstance(stage_order, list) or not stage_order:
        raise ValueError("stage_order must be a non-empty list")
    if not isinstance(stages, dict) or not stages:
        raise ValueError("stages must be a non-empty mapping")
    if not isinstance(tool_stage_map, dict):
        raise ValueError("tool_stage_map must be a mapping")
    if not isinstance(allowed_stage_transitions, dict):
        raise ValueError("allowed_stage_transitions must be a mapping")
    if not isinstance(same_stage_policy, dict):
        raise ValueError("same_pruning_stage_transition_policy must be a mapping")
    if not isinstance(alternative_clusters, dict):
        raise ValueError("alternative_clusters must be a mapping")
    if not isinstance(edge_type_stage_policy, dict):
        raise ValueError("edge_type_stage_policy must be a mapping")

    allowed = set(stages.keys())
    missing_stage_defs = [s for s in stage_order if s not in allowed]
    if missing_stage_defs:
        raise ValueError(f"stage_order contains unknown stages: {missing_stage_defs}")

    for tool_id, info in tool_stage_map.items():
        if isinstance(info, str):
            if info not in allowed:
                raise ValueError(f"tool_stage_map[{tool_id}] has unknown primary_stage={info}")
            continue
        if not isinstance(info, dict):
            raise ValueError(f"tool_stage_map[{tool_id}] must be an object")
        primary = str(info.get("primary_stage", "")).strip()
        if not primary:
            raise ValueError(f"tool_stage_map[{tool_id}] missing primary_stage")
        if primary not in allowed:
            raise ValueError(f"tool_stage_map[{tool_id}] has unknown primary_stage={primary}")
        secondaries = info.get("secondary_stages", []) or []
        if not isinstance(secondaries, list):
            raise ValueError(f"tool_stage_map[{tool_id}].secondary_stages must be a list")
        for s in secondaries:
            ss = str(s).strip()
            if ss and ss not in allowed:
                raise ValueError(f"tool_stage_map[{tool_id}] has unknown secondary_stage={ss}")

    for src, targets in allowed_stage_transitions.items():
        if src not in allowed:
            raise ValueError(f"allowed_stage_transitions has unknown source stage={src}")
        if not isinstance(targets, list):
            raise ValueError(f"allowed_stage_transitions[{src}] must be a list")
        for tgt in targets:
            tt = str(tgt).strip()
            if tt not in allowed:
                raise ValueError(f"allowed_stage_transitions has unknown target stage={src}->{tt}")

    for cluster_id, spec in alternative_clusters.items():
        if not isinstance(spec, dict):
            raise ValueError(f"alternative_clusters[{cluster_id}] must be an object")
        tools = spec.get("tools", []) or []
        if not isinstance(tools, list) or len(tools) < 2:
            raise ValueError(f"alternative_clusters[{cluster_id}] must include at least two tools")
        for tool_id in tools:
            t = str(tool_id).strip()
            if t not in tool_stage_map:
                raise ValueError(f"alternative_clusters[{cluster_id}] references unknown tool={t}")

    return StageTaxonomy(
        path=path.resolve(),
        raw=raw,
        stage_order=[str(s) for s in stage_order],
        stages=stages,
        tool_stage_map=tool_stage_map,
        allowed_stage_transitions={str(k): [str(x) for x in (v or [])] for k, v in allowed_stage_transitions.items()},
        same_stage_policy=same_stage_policy,
        alternative_clusters=alternative_clusters,
        edge_type_stage_policy=edge_type_stage_policy,
        pruning_policy=pruning_policy,
        coverage_policy=coverage_policy,
    )


_DEFAULT_TAXONOMY: StageTaxonomy | None = None


def _default_taxonomy_path() -> Path:
    return Path(os.getenv("MOLCLAW_STAGE_TAXONOMY_JSON", "configs/stage_taxonomy.json")).resolve()


def get_default_stage_taxonomy() -> StageTaxonomy:
    global _DEFAULT_TAXONOMY
    if _DEFAULT_TAXONOMY is None:
        _DEFAULT_TAXONOMY = load_stage_taxonomy(_default_taxonomy_path())
    return _DEFAULT_TAXONOMY


def allowed_stages(taxonomy: StageTaxonomy | None = None) -> set[str]:
    t = taxonomy or get_default_stage_taxonomy()
    return t.allowed_stages()


def get_primary_stage(tool_id: str, taxonomy: StageTaxonomy | None = None) -> str:
    t = taxonomy or get_default_stage_taxonomy()
    return t.get_primary_stage(tool_id)


def get_secondary_stages(tool_id: str, taxonomy: StageTaxonomy | None = None) -> list[str]:
    t = taxonomy or get_default_stage_taxonomy()
    return t.get_scheduling_stages(tool_id)


def get_scheduling_stages(tool_id: str, taxonomy: StageTaxonomy | None = None) -> list[str]:
    t = taxonomy or get_default_stage_taxonomy()
    return t.get_scheduling_stages(tool_id)


def validate_tool_coverage(tool_ids: Iterable[str], taxonomy: StageTaxonomy | None = None) -> None:
    t = taxonomy or get_default_stage_taxonomy()
    t.validate_tool_coverage(tool_ids)


def stage_order_index(stage: str, taxonomy: StageTaxonomy | None = None) -> int:
    t = taxonomy or get_default_stage_taxonomy()
    return t.stage_order_index(stage)
