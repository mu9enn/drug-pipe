from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import yaml

from .constants import DEFAULT_SERVER_URL, DEFAULT_LOGS_ROOT


@dataclass
class ProjectPaths:
    root: Path
    configs: Path
    runs: Path
    run_dir: Path


@dataclass
class RuntimeConfig:
    server_url: str
    api_key: str
    skills_root: Path
    logs_root: Path
    model_name: str = "claude-cc-v1"


@dataclass
class ProjectConfig:
    paths: ProjectPaths
    runtime: RuntimeConfig
    edge_ontology: dict[str, Any]
    stage_taxonomy: dict[str, Any]
    stage_taxonomy_path: Path
    semantic_types: dict[str, Any]
    rules: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_project_path(project_root: Path, value: str | None, default: str) -> Path:
    raw = (value or "").strip()
    path = Path(raw or default)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def build_config(project_root: Path, run_id: str | None = None, server_url: str | None = None, api_key: str | None = None, skills_root: str | None = None, logs_root: str | None = None, model_name: str = "claude-cc-v1") -> ProjectConfig:
    project_root = project_root.resolve()
    if run_id is None:
        run_id = "run_latest"

    paths = ProjectPaths(
        root=project_root,
        configs=project_root / "configs",
        runs=project_root / "runs",
        run_dir=project_root / "runs" / run_id,
    )
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    runtime = RuntimeConfig(
        server_url=(
            server_url
            or os.getenv("MOLCLAW_SCP_SERVER_URL")
            or os.getenv("MOLCLAW_SCP_MCP_URL")
            or DEFAULT_SERVER_URL
        ),
        api_key=api_key or os.getenv("MOLCLAW_SCP_API_KEY", ""),
        skills_root=_resolve_project_path(project_root, skills_root or os.getenv("MOLCLAW_SKILLS_ROOT"), "skills_full"),
        logs_root=_resolve_project_path(project_root, logs_root or os.getenv("MOLCLAW_LOGS_ROOT"), DEFAULT_LOGS_ROOT),
        model_name=model_name,
    )
    stage_taxonomy_path = _resolve_project_path(
        project_root,
        os.getenv("MOLCLAW_STAGE_TAXONOMY_JSON"),
        "configs/stage_taxonomy.json",
    )

    return ProjectConfig(
        paths=paths,
        runtime=runtime,
        edge_ontology=_load_yaml(paths.configs / "edge_ontology_v1.yaml"),
        stage_taxonomy=_load_json(stage_taxonomy_path),
        stage_taxonomy_path=stage_taxonomy_path,
        semantic_types=_load_yaml(paths.configs / "semantic_types_v1.yaml"),
        rules=_load_yaml(paths.configs / "rules_v1.yaml"),
    )
