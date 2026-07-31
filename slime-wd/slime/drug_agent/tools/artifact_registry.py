from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Any


ARTIFACT_RE = re.compile(r"^<artifact:([A-Za-z0-9._/-]+)>$")
ABSOLUTE_PATH_RE = re.compile(r"(?<![\w<:/])/(?!/)(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+")
_TRAILING = ".,;:"


def _namespace(path: str) -> str:
    lowered = path.lower()
    name = PurePosixPath(path).name
    if "fpocket" in lowered:
        return "fpocket"
    if "dock" in lowered or name.endswith(".pdbqt"):
        return "docking"
    if "boltz" in lowered:
        return "boltz"
    if "pdbfix" in lowered or "fix_pdb" in lowered:
        return "pdbfixer"
    if PurePosixPath(name).suffix.lower() in {".pdb", ".cif", ".mmcif", ".mol", ".mol2", ".sdf"}:
        return "structure"
    return "local"


class ArtifactRegistry:
    """Per-task raw-path/canonical-reference mapping.

    Raw server paths are kept only in this registry's audit snapshot. Values
    returned to the model contain canonical references, while later call
    arguments are resolved back immediately before execution.
    """

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self._raw_to_ref: dict[str, str] = {}
        self._ref_to_raw: dict[str, str] = {}

    def register(self, raw_path: str, *, namespace: str | None = None) -> str:
        normalized = raw_path.strip().rstrip(_TRAILING).replace("\\", "/")
        if normalized in self._raw_to_ref:
            return self._raw_to_ref[normalized]
        name = PurePosixPath(normalized).name or "result"
        ref = f"<artifact:{namespace or _namespace(normalized)}/{name}>"
        if ref in self._ref_to_raw and self._ref_to_raw[ref] != normalized:
            stem = ref[:-1]
            ref = f"{stem}-{hashlib.sha256(normalized.encode()).hexdigest()[:8]}>"
        self._raw_to_ref[normalized] = ref
        self._ref_to_raw[ref] = normalized
        return ref

    def register_local(self, relative_path: str) -> str:
        normalized = relative_path.strip().replace("\\", "/").lstrip("./")
        raw = str((self.workspace / normalized).resolve(strict=False))
        ref = f"<artifact:local/{normalized}>"
        self._raw_to_ref[raw] = ref
        self._ref_to_raw[ref] = normalized
        return ref

    def canonicalize(
        self,
        value: Any,
        *,
        local_result: bool = False,
        register_unknown_paths: bool = True,
    ) -> Any:
        if isinstance(value, str):
            exact = ARTIFACT_RE.fullmatch(value.strip())
            if exact:
                if register_unknown_paths or exact.group(0) in self._ref_to_raw:
                    return value
                name = PurePosixPath(exact.group(1)).name or "result"
                return f"<artifact:unavailable/{name}>"
            if local_result and value and not value.startswith("/") and not value.startswith("skills/"):
                return self.register_local(value)

            def replace(match: re.Match[str]) -> str:
                raw = match.group(0)
                core = raw.rstrip(_TRAILING)
                if core in self._raw_to_ref:
                    ref = self._raw_to_ref[core]
                elif register_unknown_paths:
                    ref = self.register(core)
                else:
                    ref = f"<artifact:unavailable/{PurePosixPath(core).name or 'result'}>"
                return ref + raw[len(core) :]

            return ABSOLUTE_PATH_RE.sub(replace, value)
        if isinstance(value, list):
            return [self.canonicalize(item, register_unknown_paths=register_unknown_paths) for item in value]
        if isinstance(value, tuple):
            return [self.canonicalize(item, register_unknown_paths=register_unknown_paths) for item in value]
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                is_local_path = local_result and str(key) in {"path", "artifact", "output_file", "output_path"}
                out[str(key)] = self.canonicalize(
                    item,
                    local_result=is_local_path,
                    register_unknown_paths=register_unknown_paths,
                )
            return out
        return value

    def resolve(self, value: Any) -> Any:
        if isinstance(value, str):
            exact = ARTIFACT_RE.fullmatch(value.strip())
            if exact:
                ref = exact.group(0)
                if ref in self._ref_to_raw:
                    raw = self._ref_to_raw[ref]
                    if raw.startswith("/"):
                        return raw
                    return str((self.workspace / raw).resolve(strict=False))
                logical = exact.group(1)
                if logical.startswith("local/"):
                    relative = PurePosixPath(logical[len("local/") :])
                    if relative.is_absolute() or ".." in relative.parts:
                        return value
                    candidate = (self.workspace / Path(*relative.parts)).resolve(strict=False)
                    try:
                        candidate.relative_to(self.workspace)
                    except ValueError:
                        return value
                    return str(candidate)
            return value
        if isinstance(value, list):
            return [self.resolve(item) for item in value]
        if isinstance(value, tuple):
            return [self.resolve(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self.resolve(item) for key, item in value.items()}
        return value

    def audit_snapshot(self) -> dict[str, Any]:
        return {
            "workspace": str(self.workspace),
            "mappings": [
                {"reference": ref, "raw_path": raw}
                for ref, raw in sorted(self._ref_to_raw.items())
            ],
        }
