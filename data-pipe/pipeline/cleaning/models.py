from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


CLEANING_DIR = Path(__file__).resolve().parent
SCHEMA_DIR = CLEANING_DIR / "schemas"
EXAMPLE_DIR = CLEANING_DIR / "examples"
PROMPT_DIR = CLEANING_DIR / "prompts"
DRUG_PIPE_ROOT = CLEANING_DIR.parents[2]
WORKDIR_SKILLS_DIR = DRUG_PIPE_ROOT / "workdir-skills"
LLM_CLEAN_SCENE_DIR = WORKDIR_SKILLS_DIR / "drug-trajectory-prose-curation"
LLM_CLEAN_SKILL_DIR = (
    LLM_CLEAN_SCENE_DIR / ".claude/skills/clean-drug-trajectory"
)
LLM_CLEAN_SYSTEM_PROMPT = LLM_CLEAN_SCENE_DIR / "system_prompt.md"
LLM_CLEAN_USER_PROMPT = LLM_CLEAN_SCENE_DIR / "user_prompt.md"
SEMANTIC_SCHEMA_VERSION = "drug_agent_semantic_trajectory_v1"
QWEN35_SFT_SCHEMA_VERSION = "drug_agent_qwen35_sft_v1"
TOOLRL_SCHEMA_VERSION = "drug_agent_toolrl_decision_v1"
PATCH_SCHEMA_VERSION = "semantic_reasoning_patch_v1"


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict[str, Any]:
    value = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"schema is not an object: {name}")
    return value


def schema_findings(value: Any, schema_name: str) -> list[str]:
    validator = Draft202012Validator(load_schema(schema_name))
    findings: list[str] = []
    for error in sorted(validator.iter_errors(value), key=lambda item: list(item.absolute_path)):
        location = "/".join(str(part) for part in error.absolute_path) or "$"
        findings.append(f"schema:{location}:{error.message}")
    return findings


def semantic_schema_findings(value: Any) -> list[str]:
    return schema_findings(value, "semantic_trajectory_v1.schema.json")


def qwen35_sft_schema_findings(value: Any) -> list[str]:
    return schema_findings(value, "qwen35_sft_v1.schema.json")


def toolrl_schema_findings(value: Any) -> list[str]:
    return schema_findings(value, "toolrl_decision_v1.schema.json")


def patch_schema_findings(value: Any) -> list[str]:
    return schema_findings(value, "semantic_reasoning_patch_v1.schema.json")
