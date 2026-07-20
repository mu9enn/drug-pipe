from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..io_utils import sha256_file
from ..settings import ProjectConfig


@dataclass(frozen=True)
class ResolvedSamplingConfig:
    profile_name: str
    values: dict[str, Any]
    config_path: Path
    config_sha256: str
    prompt_path: Path
    prompt_sha256: str

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "sampling_profile": self.profile_name,
            "resolved_sampling_config": self.values,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "prompt_path": str(self.prompt_path),
            "prompt_sha256": self.prompt_sha256,
        }


def resolve_sampling_profile(
    config: ProjectConfig,
    profile_name: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> ResolvedSamplingConfig:
    path = config.paths.configs / "question_sampling_v2.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles = raw.get("profiles")
    if not isinstance(profiles, dict) or profile_name not in profiles:
        raise ValueError(f"unknown sampling profile: {profile_name}")
    selected = profiles[profile_name]
    if not isinstance(selected, dict):
        raise ValueError(f"sampling profile must be an object: {profile_name}")
    values = dict(selected)
    for key, value in (overrides or {}).items():
        if value is not None:
            values[key] = value
    mode = str(values.get("mode") or "").strip()
    if mode not in {"simple_toolchain_question", "dag_closure", "linear_debug"}:
        raise ValueError(f"invalid sampling mode in profile {profile_name}: {mode}")
    prompt_value = str(values.get("prompt") or "").strip()
    if not prompt_value:
        raise ValueError(f"sampling profile missing prompt: {profile_name}")
    prompt_path = (config.paths.configs / prompt_value).resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(prompt_path)
    return ResolvedSamplingConfig(
        profile_name=profile_name,
        values=values,
        config_path=path.resolve(),
        config_sha256=sha256_file(path),
        prompt_path=prompt_path,
        prompt_sha256=sha256_file(prompt_path),
    )
