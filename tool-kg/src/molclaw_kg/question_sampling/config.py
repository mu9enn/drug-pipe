from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..io_utils import sha256_file, stable_hash_obj
from ..settings import ProjectConfig


@dataclass(frozen=True)
class ResolvedSamplingConfig:
    profile_name: str
    values: dict[str, Any]
    config_path: Path
    config_sha256: str
    profile_sha256: str
    cli_overrides: dict[str, Any]
    prompt_path: Path
    prompt_sha256: str
    prompt_hashes: dict[str, str]

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "sampling_profile": self.profile_name,
            "resolved_sampling_config": self.values,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "profile_sha256": self.profile_sha256,
            "cli_overrides": self.cli_overrides,
            "prompt_path": str(self.prompt_path),
            "prompt_sha256": self.prompt_sha256,
            "prompt_hashes": self.prompt_hashes,
        }


_COMMON_FIELDS = {"mode", "prompt", "min_hops", "max_hops", "random_seed"}
_MODE_FIELDS = {
    "simple_toolchain_question": _COMMON_FIELDS
    | {
        "json_repair_prompt",
        "target_successes",
        "max_attempts",
        "science_kb_topk",
        "grounding_selection",
        "max_repeat_target",
        "max_repeat_compound",
        "json_repair_rounds",
        "semantic_repair_rounds",
        "partial_edge_policy",
        "tool_leak_policy",
    },
    "dag_closure": _COMMON_FIELDS
    | {
        "repair_prompt",
        "sample_size",
        "partial_policy",
        "edge_profile",
        "max_repair_rounds",
    },
}
_REQUIRED_FIELDS = {
    "simple_toolchain_question": {
        "mode",
        "prompt",
        "json_repair_prompt",
        "target_successes",
        "max_attempts",
        "min_hops",
        "max_hops",
        "science_kb_topk",
        "grounding_selection",
        "max_repeat_target",
        "max_repeat_compound",
        "json_repair_rounds",
        "semantic_repair_rounds",
    },
    "dag_closure": {
        "mode",
        "prompt",
        "repair_prompt",
        "sample_size",
        "min_hops",
        "max_hops",
        "partial_policy",
        "edge_profile",
        "max_repair_rounds",
    },
}


def _validate_resolved_profile(profile_name: str, values: dict[str, Any]) -> str:
    mode = str(values.get("mode") or "").strip()
    if mode not in _MODE_FIELDS:
        raise ValueError(f"invalid sampling mode in profile {profile_name}: {mode}")
    unknown = sorted(set(values) - _MODE_FIELDS[mode])
    if unknown:
        raise ValueError(f"unsupported fields in sampling profile {profile_name}: {unknown}")
    missing = sorted(_REQUIRED_FIELDS[mode] - set(values))
    if missing:
        raise ValueError(f"sampling profile {profile_name} is missing fields: {missing}")
    integer_fields = {
        "sample_size",
        "target_successes",
        "max_attempts",
        "min_hops",
        "max_hops",
        "science_kb_topk",
        "max_repeat_target",
        "max_repeat_compound",
        "json_repair_rounds",
        "semantic_repair_rounds",
        "max_repair_rounds",
    }
    for field in integer_fields.intersection(values):
        value = values[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"sampling profile {profile_name}.{field} must be an integer")
        minimum = 0 if field in {"json_repair_rounds", "semantic_repair_rounds", "max_repair_rounds"} else 1
        if value < minimum:
            raise ValueError(f"sampling profile {profile_name}.{field} must be >= {minimum}")
    random_seed = values.get("random_seed")
    if random_seed is not None and (
        isinstance(random_seed, bool) or not isinstance(random_seed, int)
    ):
        raise ValueError(f"sampling profile {profile_name}.random_seed must be an integer or null")
    if int(values["max_hops"]) < int(values["min_hops"]):
        raise ValueError(f"sampling profile {profile_name} has max_hops < min_hops")
    return mode


def resolve_sampling_profile(
    config: ProjectConfig,
    profile_name: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> ResolvedSamplingConfig:
    path = config.paths.configs / "question_sampling.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("version") != "question_sampling_profiles_v1":
        raise ValueError(f"unsupported sampling config version: {raw.get('version')!r}")
    profiles = raw.get("profiles")
    if not isinstance(profiles, dict) or profile_name not in profiles:
        raise ValueError(f"unknown sampling profile: {profile_name}")
    selected = profiles[profile_name]
    if not isinstance(selected, dict):
        raise ValueError(f"sampling profile must be an object: {profile_name}")
    explicit_overrides = {
        key: value for key, value in (overrides or {}).items() if value is not None
    }
    if "mode" in explicit_overrides:
        raise ValueError("sampling mode is profile-owned; select another named profile")
    values = {**selected, **explicit_overrides}
    _validate_resolved_profile(profile_name, values)
    prompt_value = str(values.get("prompt") or "").strip()
    if not prompt_value:
        raise ValueError(f"sampling profile missing prompt: {profile_name}")
    prompt_path = (config.paths.configs / prompt_value).resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(prompt_path)
    prompt_paths = {"generation": prompt_path}
    for key, label in [("json_repair_prompt", "json_repair"), ("repair_prompt", "semantic_repair")]:
        configured = str(values.get(key) or "").strip()
        if not configured:
            continue
        configured_path = (config.paths.configs / configured).resolve()
        if not configured_path.is_file():
            raise FileNotFoundError(configured_path)
        prompt_paths[label] = configured_path
    prompt_hashes = {
        label: sha256_file(prompt_file)
        for label, prompt_file in prompt_paths.items()
    }
    return ResolvedSamplingConfig(
        profile_name=profile_name,
        values=values,
        config_path=path.resolve(),
        config_sha256=sha256_file(path),
        profile_sha256=stable_hash_obj(selected),
        cli_overrides=explicit_overrides,
        prompt_path=prompt_path,
        prompt_sha256=prompt_hashes["generation"],
        prompt_hashes=prompt_hashes,
    )
