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
REACT_SCHEMA_VERSION = "drug_agent_sft_react_json_v1"
PATCH_SCHEMA_VERSION = "llm_clean_patch_v1"


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


def react_schema_findings(value: Any) -> list[str]:
    return schema_findings(value, "react_trajectory_v1.schema.json")


def patch_schema_findings(value: Any) -> list[str]:
    return schema_findings(value, "llm_clean_patch_v1.schema.json")
