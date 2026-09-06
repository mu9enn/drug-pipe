from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrajectoryPathNormalizer:
    """Normalize only paths whose provenance is the Claude attempt workspace.

    Other absolute paths are task/MCP data and remain byte-for-byte unchanged.
    This prevents collection-host workdirs from becoming training tokens without
    inventing a model-specific artifact URI.
    """

    workspace: Path
    source_id: str
    preserve_pwd_chain: bool = False
    _mappings: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def pwd_root(self) -> str:
        return "/" + hashlib.sha256(self.source_id.encode("utf-8")).hexdigest()[:20]

    def _remember(self, source: str, target: str, kind: str) -> str:
        if source != target:
            self._mappings[source] = {"source": source, "target": target, "kind": kind}
        return target

    def normalize_text(self, text: str) -> str:
        workspace = str(self.workspace.resolve()).rstrip("/")
        l1_source = f"{workspace}/.claude/skills/L1_tools"
        value = text

        if l1_source in value:
            value = value.replace(
                l1_source,
                self._remember(l1_source, "skills/L1_tools", "l1_runtime_path"),
            )
        if ".claude/skills/L1_tools" in value:
            value = value.replace(
                ".claude/skills/L1_tools",
                self._remember(
                    ".claude/skills/L1_tools",
                    "skills/L1_tools",
                    "l1_runtime_path",
                ),
            )

        workspace_child = workspace + "/"
        workspace_root_pattern = re.compile(re.escape(workspace) + r"(?![A-Za-z0-9._/-])")
        if workspace_child in value or workspace_root_pattern.search(value):
            if self.preserve_pwd_chain:
                replacement = self._remember(workspace, self.pwd_root, "observed_pwd_root")
                value = value.replace(workspace_child, replacement + "/")
                value = workspace_root_pattern.sub(replacement, value)
            else:
                self._remember(workspace, ".", "attempt_workdir")
                value = value.replace(workspace_child, "")
                value = workspace_root_pattern.sub(".", value)
        return value

    def normalize(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.normalize_text(value)
        if isinstance(value, list):
            return [self.normalize(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self.normalize(item) for key, item in value.items()}
        return value

    def audit(self) -> dict[str, Any]:
        return {
            "workspace": str(self.workspace.resolve()),
            "preserve_pwd_chain": self.preserve_pwd_chain,
            "pwd_root": self.pwd_root if self.preserve_pwd_chain else None,
            "mappings": list(self._mappings.values()),
        }
