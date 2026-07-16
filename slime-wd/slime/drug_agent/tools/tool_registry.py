from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from drug_agent.tools.tool_executor import MCPToolExecutor
from drug_agent.utils import normalize_tool_name


DEFAULT_ALLOWLIST_PATH = Path(__file__).resolve().parent / "allowlist_v0.json"


def load_allowlist(path: str | Path | None = None) -> set[str]:
    p = Path(path) if path is not None else DEFAULT_ALLOWLIST_PATH
    if not p.exists():
        return set()

    obj = json.loads(p.read_text(encoding="utf-8"))
    values = []
    if isinstance(obj, list):
        values = obj
    elif isinstance(obj, dict):
        values = obj.get("allowed_tools", [])

    out: set[str] = set()
    for item in values:
        if isinstance(item, str) and item.strip():
            out.add(normalize_tool_name(item))
    return out


class ToolRegistry:
    """Thin MCP tool registry with allowlist and basic argument checks."""

    def __init__(
        self,
        executor: MCPToolExecutor,
        *,
        allowlist: set[str] | None = None,
        allow_all: bool = False,
    ) -> None:
        self.executor = executor
        self.allowlist = set(allowlist or [])
        self.allow_all = bool(allow_all)

        self._tool_specs: list[dict[str, Any]] = []
        self._tool_map: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_env(cls, executor: MCPToolExecutor | None = None) -> "ToolRegistry":
        allow_all = os.environ.get("DRUG_AGENT_ALLOW_ALL", "0").strip().lower() in {"1", "true", "yes", "on"}
        allowlist_path = os.environ.get("DRUG_AGENT_ALLOWLIST_PATH")
        allowlist = load_allowlist(allowlist_path)

        return cls(
            executor=executor or MCPToolExecutor(connect_on_init=False),
            allowlist=allowlist,
            allow_all=allow_all,
        )

    def list_tools(self, force_refresh: bool = False) -> list[dict[str, Any]]:
        if self._tool_specs and not force_refresh:
            return self._tool_specs

        specs = self.executor.list_tools()
        normalized_specs: list[dict[str, Any]] = []
        tool_map: dict[str, dict[str, Any]] = {}

        for spec in specs:
            raw_name = spec.get("name")
            if not isinstance(raw_name, str) or not raw_name.strip():
                continue

            bare_name = normalize_tool_name(raw_name)
            norm = {
                "name": bare_name,
                "raw_name": raw_name,
                "description": spec.get("description", ""),
                "input_schema": spec.get("input_schema") or {},
            }
            normalized_specs.append(norm)
            tool_map[bare_name] = norm

        self._tool_specs = normalized_specs
        self._tool_map = tool_map
        return self._tool_specs

    def load_tool_schema(self, tool_name: str) -> dict[str, Any] | None:
        bare_name = normalize_tool_name(tool_name)
        if bare_name not in self._tool_map:
            self.list_tools(force_refresh=True)
        spec = self._tool_map.get(bare_name)
        if not spec:
            return None

        schema = spec.get("input_schema")
        return schema if isinstance(schema, dict) else {}

    def validate_tool_name(self, tool_name: str, allowed_tools: list[str] | None = None) -> tuple[bool, str | None]:
        bare_name = normalize_tool_name(tool_name)
        if not bare_name:
            return False, "empty_tool_name"

        if allowed_tools is not None:
            normalized_allowed = {normalize_tool_name(x) for x in allowed_tools if isinstance(x, str)}
            if bare_name not in normalized_allowed:
                return False, "tool_not_in_sample_allowed_tools"

        if (not self.allow_all) and self.allowlist and bare_name not in self.allowlist:
            return False, "tool_not_in_allowlist"

        if bare_name not in self._tool_map:
            self.list_tools(force_refresh=True)
        if bare_name not in self._tool_map:
            return False, "tool_not_found_in_registry"

        return True, None

    def validate_arguments_basic(self, tool_name: str, arguments: Any) -> tuple[bool, str | None]:
        if not isinstance(arguments, dict):
            return False, "arguments_not_object"

        schema = self.load_tool_schema(tool_name)
        if not schema:
            return True, None

        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in arguments:
                    return False, f"missing_required_argument:{key}"

        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, value in arguments.items():
                prop = properties.get(key)
                if not isinstance(prop, dict):
                    continue
                expected_type = prop.get("type")
                if expected_type == "string" and not isinstance(value, str):
                    return False, f"invalid_type:{key}:expected_string"
                if expected_type == "number" and not isinstance(value, (int, float)):
                    return False, f"invalid_type:{key}:expected_number"
                if expected_type == "integer" and not isinstance(value, int):
                    return False, f"invalid_type:{key}:expected_integer"
                if expected_type == "boolean" and not isinstance(value, bool):
                    return False, f"invalid_type:{key}:expected_boolean"
                if expected_type == "array" and not isinstance(value, list):
                    return False, f"invalid_type:{key}:expected_array"
                if expected_type == "object" and not isinstance(value, dict):
                    return False, f"invalid_type:{key}:expected_object"

        return True, None
