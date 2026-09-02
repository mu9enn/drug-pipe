from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from drug_agent.tools.local_tools import LocalToolError, LocalToolExecutor


_WORKSPACE_ONLY_LOCAL_TOOLS = {
    "Read": (
        "Read a UTF-8 file from the task workspace.",
        {"file_path": "A workspace/... filesystem path; resource handles are not accepted."},
    ),
    "Grep": (
        "Search text files in the task workspace.",
        {"path": "A workspace/... filesystem path; resource handles are not accepted."},
    ),
    "Glob": (
        "List files matching a glob in the task workspace.",
        {"path": "A workspace/... filesystem path; resource handles are not accepted."},
    ),
}


def workspace_only_tool_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove the optional L1 document surface from slime-raw local tool schemas."""

    result = copy.deepcopy(specs)
    for spec in result:
        replacement = _WORKSPACE_ONLY_LOCAL_TOOLS.get(str(spec.get("name") or ""))
        if replacement is None or spec.get("executor") != "local_sandbox":
            continue
        description, property_descriptions = replacement
        spec["description"] = description
        schema = spec.get("input_schema")
        properties = schema.get("properties") if isinstance(schema, dict) else None
        if not isinstance(properties, dict):
            continue
        for property_name, property_description in property_descriptions.items():
            prop = properties.get(property_name)
            if isinstance(prop, dict):
                prop["description"] = property_description
    return result


class SlimeRawLocalToolExecutor(LocalToolExecutor):
    """Expose only task-workspace file operations, with no mounted skill catalog."""

    def __init__(self, workspace: str | Path, unused_l1_skills_root: str | Path) -> None:
        super().__init__(workspace, unused_l1_skills_root)

    def _path(self, raw: str, *, write: bool = False) -> Path:
        normalized = str(raw).strip().replace("\\", "/")
        if normalized == "skills/L1_tools" or normalized.startswith("skills/L1_tools/"):
            raise LocalToolError("slime-raw exposes only task-workspace filesystem paths")
        return super()._path(raw, write=write)
