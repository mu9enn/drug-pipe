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
    version: str
    display_order: list[str]
    stages: dict[str, Any]
    tool_stage_map: dict[str, Any]
    allowed_transitions: dict[str, list[str]]
    alternative_clusters: dict[str, Any]

    def allowed_stages(self) -> set[str]:
        return set(self.stages)

    def _tool_info(self, tool_id: str) -> dict[str, Any]:
        info = self.tool_stage_map.get(tool_id)
        if not isinstance(info, dict):
            raise KeyError(f"tool_id is not mapped in stage taxonomy: {tool_id}")
        return info

    def get_primary_stage(self, tool_id: str) -> str:
        stage = str(self._tool_info(tool_id).get("primary_stage") or "").strip()
        if not stage:
            raise ValueError(f"primary_stage is missing for tool_id={tool_id}")
        if stage not in self.allowed_stages():
            raise ValueError(f"invalid primary_stage={stage} for tool_id={tool_id}")
        return stage

    def get_scheduling_stages(self, tool_id: str) -> list[str]:
        info = self._tool_info(tool_id)
        primary = self.get_primary_stage(tool_id)
        values = info.get("scheduling_stages")
        if values is None:
            values = [primary]
        if not isinstance(values, list):
            raise ValueError(f"scheduling_stages must be a list for tool_id={tool_id}")
        out: list[str] = []
        for value in [primary, *values]:
            stage = str(value).strip()
            if not stage:
                continue
            if stage not in self.allowed_stages():
                raise ValueError(f"invalid scheduling_stage={stage} for tool_id={tool_id}")
            if stage not in out:
                out.append(stage)
        return out

    def validate_tool_coverage(self, tool_ids: Iterable[str]) -> None:
        expected = {str(value).strip() for value in tool_ids if str(value).strip()}
        mapped = set(self.tool_stage_map)
        missing = sorted(expected - mapped)
        extra = sorted(mapped - expected)
        if missing:
            raise ValueError(f"stage taxonomy missing mappings for tools: {missing}")
        if extra:
            raise ValueError(f"stage taxonomy contains extra mappings not in snapshot: {extra}")

    def is_transition_allowed(self, source_stage: str, target_stage: str) -> bool:
        return target_stage in self.allowed_transitions.get(source_stage, [])

    def supporting_stage_pairs(
        self,
        source_tool: str,
        target_tool: str,
    ) -> list[list[str]]:
        return [
            [source_stage, target_stage]
            for source_stage in self.get_scheduling_stages(source_tool)
            for target_stage in self.get_scheduling_stages(target_tool)
            if self.is_transition_allowed(source_stage, target_stage)
        ]

    def alternative_pairs(self) -> list[tuple[str, str, str]]:
        pairs: list[tuple[str, str, str]] = []
        for cluster_id, spec in self.alternative_clusters.items():
            if not isinstance(spec, dict):
                continue
            tools = [str(value).strip() for value in spec.get("tools") or [] if str(value).strip()]
            for source in tools:
                for target in tools:
                    if source != target:
                        pairs.append((source, target, str(cluster_id)))
        return pairs


def load_stage_taxonomy(path: Path) -> StageTaxonomy:
    raw = read_json(path)
    allowed_top_level = {
        "version",
        "description",
        "stages",
        "display_order",
        "tool_stage_map",
        "allowed_transitions",
        "alternative_clusters",
    }
    extra_top_level = sorted(set(raw) - allowed_top_level)
    if extra_top_level:
        raise ValueError(f"unsupported stage taxonomy fields: {extra_top_level}")
    version = str(raw.get("version") or "").strip()
    display_order = raw.get("display_order")
    stages = raw.get("stages")
    tool_stage_map = raw.get("tool_stage_map")
    allowed_transitions = raw.get("allowed_transitions")
    alternative_clusters = raw.get("alternative_clusters", {}) or {}
    if version != "stage_taxonomy_v2":
        raise ValueError(f"unsupported stage taxonomy version: {version!r}")
    if not isinstance(display_order, list) or not display_order:
        raise ValueError("display_order must be a non-empty list")
    if not isinstance(stages, dict) or not stages:
        raise ValueError("stages must be a non-empty mapping")
    if not isinstance(tool_stage_map, dict):
        raise ValueError("tool_stage_map must be a mapping")
    if not isinstance(allowed_transitions, dict):
        raise ValueError("allowed_transitions must be a mapping")
    if not isinstance(alternative_clusters, dict):
        raise ValueError("alternative_clusters must be a mapping")

    allowed = set(stages)
    if len(display_order) != len(set(display_order)):
        raise ValueError("display_order contains duplicate stages")
    unknown_display = [stage for stage in display_order if stage not in allowed]
    if unknown_display:
        raise ValueError(f"display_order contains unknown stages: {unknown_display}")
    missing_display = sorted(allowed - set(display_order))
    if missing_display:
        raise ValueError(f"display_order omits stages: {missing_display}")
    for stage, spec in stages.items():
        if not isinstance(spec, dict) or set(spec) != {"definition"}:
            raise ValueError(f"stages[{stage}] must contain only definition")
        if not str(spec.get("definition") or "").strip():
            raise ValueError(f"stages[{stage}].definition is required")
    for source, targets in allowed_transitions.items():
        if source not in allowed:
            raise ValueError(f"allowed_transitions has unknown source stage={source}")
        if not isinstance(targets, list):
            raise ValueError(f"allowed_transitions[{source}] must be a list")
        if len(targets) != len(set(targets)):
            raise ValueError(f"allowed_transitions[{source}] contains duplicates")
        for target in targets:
            if target not in allowed:
                raise ValueError(f"allowed_transitions has unknown target stage={source}->{target}")
    taxonomy = StageTaxonomy(
        path=path.resolve(),
        raw=raw,
        version=version,
        display_order=[str(stage) for stage in display_order],
        stages=stages,
        tool_stage_map=tool_stage_map,
        allowed_transitions={
            str(source): [str(target) for target in targets or []]
            for source, targets in allowed_transitions.items()
        },
        alternative_clusters=alternative_clusters,
    )
    for tool_id, spec in tool_stage_map.items():
        if not isinstance(spec, dict):
            raise ValueError(f"tool_stage_map[{tool_id}] must be an object")
        if set(spec) != {"primary_stage", "scheduling_stages"}:
            raise ValueError(
                f"tool_stage_map[{tool_id}] must contain primary_stage and scheduling_stages only"
            )
        taxonomy.get_primary_stage(str(tool_id))
        taxonomy.get_scheduling_stages(str(tool_id))
    for cluster_id, spec in alternative_clusters.items():
        if not isinstance(spec, dict):
            raise ValueError(f"alternative_clusters[{cluster_id}] must be an object")
        if set(spec) != {"tools"}:
            raise ValueError(f"alternative_clusters[{cluster_id}] must contain only tools")
        tools = spec.get("tools") or []
        if not isinstance(tools, list) or len(tools) < 2:
            raise ValueError(f"alternative_clusters[{cluster_id}] must include at least two tools")
        if len(tools) != len(set(tools)):
            raise ValueError(f"alternative_clusters[{cluster_id}] contains duplicate tools")
        for tool_id in tools:
            if str(tool_id) not in tool_stage_map:
                raise ValueError(
                    f"alternative_clusters[{cluster_id}] references unknown tool={tool_id}"
                )
    return taxonomy


def resolve_stage_taxonomy_path(project_root: Path | None = None) -> Path:
    configured = os.getenv("MOLCLAW_STAGE_TAXONOMY_JSON")
    if configured:
        path = Path(configured)
        if not path.is_absolute() and project_root is not None:
            path = project_root / path
        return path.resolve()
    if project_root is not None:
        return (project_root / "configs" / "stage_taxonomy.json").resolve()
    return Path("configs/stage_taxonomy.json").resolve()
