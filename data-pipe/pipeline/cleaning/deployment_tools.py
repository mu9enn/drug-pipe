from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOCAL_TOOLS = frozenset({"Read", "Write", "Edit", "Bash", "Grep", "Glob"})
TOOL_VISIBILITY_CHOICES = ("all", "trajectory-plus-distractors")


@dataclass(frozen=True)
class DeploymentToolSet:
    """The exact tool set visible to the student at deployment/rollout time."""

    tools: tuple[dict[str, Any], ...]
    source_path: Path
    sha256: str

    @property
    def by_name(self) -> dict[str, dict[str, Any]]:
        return {str(tool["name"]): tool for tool in self.tools}

    def resolve_raw_name(self, raw_name: str) -> str | None:
        name = str(raw_name or "").strip()
        candidates = [name]
        if name.startswith("mcp__") and "__" in name[5:]:
            candidates.append(name.rsplit("__", 1)[-1])
        for tool in self.tools:
            canonical = str(tool["name"])
            raw = str(tool.get("raw_name") or canonical)
            if any(candidate in {canonical, raw} for candidate in candidates):
                return canonical
        return None

    def require_public_name(self, raw_name: str) -> str:
        resolved = self.resolve_raw_name(raw_name)
        if resolved is None:
            raise ValueError(f"tool absent from deployment tool-set: {raw_name}")
        return resolved

    def qwen_tools(self, names: set[str] | None = None) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"],
                },
            }
            for tool in self.tools
            if names is None or str(tool["name"]) in names
        ]

    def visible_tool_names(self, record: dict[str, Any], policy: str) -> set[str]:
        if policy not in TOOL_VISIBILITY_CHOICES:
            raise ValueError(f"unsupported tool visibility policy: {policy}")
        available = set(self.by_name)
        local_public = {
            public
            for name in LOCAL_TOOLS
            if (public := self.resolve_raw_name(name)) is not None
        }
        missing_local = {
            name for name in LOCAL_TOOLS if self.resolve_raw_name(name) is None
        }
        if missing_local:
            raise ValueError(f"deployment tool-set lacks required local tools: {sorted(missing_local)}")
        used = {
            self.require_public_name(str(call["name"]))
            for event in record.get("events") or []
            if event.get("type") == "assistant_decision"
            for call in event.get("tool_calls") or []
        }
        missing_used = used - available
        if missing_used:
            raise ValueError(f"semantic calls tools absent from deployment tool-set: {sorted(missing_used)}")
        if policy == "all":
            return available

        used_molclaw = used - local_public
        candidates = sorted(available - local_public - used_molclaw)
        sample_id = str(record.get("id") or "")
        candidates.sort(
            key=lambda name: (
                hashlib.sha256(f"{sample_id}\0{name}".encode("utf-8")).hexdigest(),
                name,
            )
        )
        distractors = set(candidates[: len(used_molclaw)])
        return local_public | used_molclaw | distractors


def load_deployment_tool_set(path: Path) -> DeploymentToolSet:
    source = path.expanduser().resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    serialized = json.dumps(payload, ensure_ascii=False)
    legacy_markers = [marker for marker in ("resource://", "<artifact:") if marker in serialized]
    if legacy_markers:
        raise ValueError(
            "deployment tool-set contains obsolete path-protocol annotations: "
            + ", ".join(legacy_markers)
        )
    rows = payload.get("tools") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("deployment tool-set must contain a non-empty tools list")
    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("deployment tool-set entries must be objects")
        function = row.get("function") if isinstance(row.get("function"), dict) else row
        name = str(function.get("name") or "").strip()
        schema = function.get("parameters", function.get("input_schema"))
        if not name or not isinstance(schema, dict):
            raise ValueError(f"tool entry lacks name/input schema: {row!r}")
        if name in seen:
            raise ValueError(f"duplicate deployment tool: {name}")
        seen.add(name)
        tools.append(
            {
                "name": name,
                "raw_name": str(row.get("raw_name") or name),
                "description": str(function.get("description") or ""),
                "input_schema": schema,
            }
        )
    return DeploymentToolSet(tuple(tools), source, hashlib.sha256(raw).hexdigest())
