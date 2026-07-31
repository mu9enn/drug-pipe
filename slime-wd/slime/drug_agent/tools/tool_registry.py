from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from jsonschema import Draft7Validator

from drug_agent.tools.tool_executor import MCPToolExecutor
from drug_agent.tools.local_tools import LOCAL_TOOL_SPECS, LocalToolExecutor, is_local_tool
from drug_agent.utils import normalize_tool_name


def catalog_sha256(specs: list[dict[str, Any]]) -> str:
    """Order-independent fingerprint of the runtime's complete tool authority."""
    normalized = sorted(specs, key=lambda item: str(item.get("name") or ""))
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ToolRegistry:
    """Registry and dispatcher for MolClaw MCP and sandboxed local tools."""

    def __init__(
        self,
        executor: MCPToolExecutor,
        *,
        include_local_tools: bool = False,
    ) -> None:
        self.executor = executor
        self.include_local_tools = bool(include_local_tools)

        self._tool_specs: list[dict[str, Any]] = []
        self._tool_map: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_env(cls, executor: MCPToolExecutor | None = None) -> "ToolRegistry":
        return cls(
            executor=executor or MCPToolExecutor(connect_on_init=False),
            include_local_tools=os.environ.get("DRUG_AGENT_ENABLE_LOCAL_TOOLS", "1").strip().lower()
            not in {"0", "false", "no", "off"},
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

        if self.include_local_tools:
            for spec in LOCAL_TOOL_SPECS:
                norm = {**spec, "raw_name": spec["name"], "executor": "local_sandbox"}
                normalized_specs.append(norm)
                tool_map[spec["name"]] = norm

        self._tool_specs = normalized_specs
        self._tool_map = tool_map
        return self._tool_specs

    def install_catalog(self, specs: list[dict[str, Any]]) -> None:
        """Install a run-authoritative catalog on an isolated task executor."""
        self._tool_specs = [dict(spec) for spec in specs]
        self._tool_map = {
            str(spec["name"]): spec
            for spec in self._tool_specs
            if isinstance(spec.get("name"), str) and spec.get("name")
        }

    def load_tool_schema(self, tool_name: str) -> dict[str, Any] | None:
        bare_name = normalize_tool_name(tool_name)
        if self.include_local_tools and is_local_tool(bare_name):
            for spec in LOCAL_TOOL_SPECS:
                if spec["name"] == bare_name:
                    schema = spec.get("input_schema")
                    return schema if isinstance(schema, dict) else {}
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

        if is_local_tool(bare_name):
            if not self.include_local_tools:
                return False, "local_tools_disabled"
            return True, None

        if bare_name not in self._tool_map:
            self.list_tools(force_refresh=True)
        if bare_name not in self._tool_map:
            return False, "tool_not_found_in_registry"

        return True, None

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        local_executor: LocalToolExecutor | None = None,
    ) -> dict[str, Any]:
        bare_name = normalize_tool_name(tool_name)
        if is_local_tool(bare_name):
            if not self.include_local_tools:
                raise RuntimeError("local tools are disabled")
            if local_executor is None:
                raise RuntimeError("a per-task LocalToolExecutor is required")
            return local_executor.execute(bare_name, arguments)
        return self.executor.execute(bare_name, arguments)

    def validate_arguments(self, tool_name: str, arguments: Any) -> tuple[bool, str | None]:
        if not isinstance(arguments, dict):
            return False, "arguments_not_object"

        schema = self.load_tool_schema(tool_name)
        if not schema:
            return True, None

        errors = sorted(Draft7Validator(schema).iter_errors(arguments), key=lambda item: list(item.path))
        if errors:
            error = errors[0]
            location = ".".join(str(part) for part in error.absolute_path) or "$"
            return False, f"json_schema:{location}:{error.message}"
        return True, None

    # Transitional method name for callers outside the online mainline. It now
    # performs complete JSON Schema validation rather than the historical
    # required/type subset.
    def validate_arguments_basic(self, tool_name: str, arguments: Any) -> tuple[bool, str | None]:
        return self.validate_arguments(tool_name, arguments)
