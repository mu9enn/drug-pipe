from __future__ import annotations

import shutil
from pathlib import Path

from .settings import ProjectConfig


def scene_root(config: ProjectConfig, scene: str) -> Path:
    root = getattr(config.runtime, "workdir_skills_root", None)
    if root is None:
        return Path()
    return Path(root) / scene


def load_scene_prompt(config: ProjectConfig, scene: str) -> str:
    path = scene_root(config, scene) / "system_prompt.md"
    if not path.is_file():
        raise FileNotFoundError(f"workdir scene prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def install_scene(config: ProjectConfig, workdir: Path, scene: str) -> None:
    """Install only the selected scene's Claude skill payload."""
    configured_root = getattr(config.runtime, "workdir_skills_root", None)
    if configured_root is None:
        return  # Keep lightweight test fixtures and external legacy callers compatible.

    workdir.mkdir(parents=True, exist_ok=True)
    source_claude = scene_root(config, scene) / ".claude"
    if not source_claude.is_dir():
        raise FileNotFoundError(f"workdir scene skills not found: {source_claude}")
    shutil.copytree(source_claude, workdir / ".claude", dirs_exist_ok=True)
