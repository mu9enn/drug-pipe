from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any


ARTIFACT_RE = re.compile(r"<artifact:[^>]+>")
ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w<])(?:/root|/home|/tmp|/mnt|/workspace)(?:/[^\s<>\"'{}\[\],)]*)?"
)
RELATIVE_PATH_RE = re.compile(
    r"(?<![:/A-Za-z0-9])(?:\.\.?/)+(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+"
    r"\.(?:json|pdb|cif|mmcif|sdf|mol2|pdbqt|csv|tsv|txt|npy|npz|pt|pkl)"
)
ERROR_STATUSES = {"error", "failed", "failure", "timeout", "timed_out", "invalid"}
SUCCESS_STATUSES = {"ok", "success", "succeeded", "complete", "completed", "partial_success"}


def _artifact_reference(raw_path: str) -> str:
    path = str(raw_path).rstrip(".,;:")
    normalized = path.replace("\\", "/")
    name = PurePosixPath(normalized).name or "result"
    lowered = normalized.lower()
    if "fpocket" in lowered:
        namespace = "fpocket"
    elif "dock" in lowered:
        namespace = "docking"
    elif "boltz" in lowered:
        namespace = "boltz"
    elif "pdbfix" in lowered:
        namespace = "pdbfixer"
    elif PurePosixPath(name).suffix.lower() in {
        ".pdb",
        ".cif",
        ".mmcif",
        ".pdbqt",
        ".mol",
        ".mol2",
        ".sdf",
    }:
        namespace = "structure"
    else:
        namespace = "local"
    return f"<artifact:{namespace}/{name}>"


def sanitize_artifact_paths(value: Any, path_map: dict[str, str]) -> Any:
    """Replace absolute and explicit relative file paths using one pure rule."""
    if isinstance(value, str):
        protected: list[str] = []

        def protect(match: re.Match[str]) -> str:
            protected.append(match.group(0))
            return f"__DRUG_PIPE_ARTIFACT_{len(protected) - 1}__"

        text = ARTIFACT_RE.sub(protect, value)

        def replace(match: re.Match[str]) -> str:
            raw = match.group(0)
            path_map.setdefault(raw, _artifact_reference(raw))
            return path_map[raw]

        text = ABSOLUTE_PATH_RE.sub(replace, text)
        text = RELATIVE_PATH_RE.sub(replace, text)
        for index, artifact in enumerate(protected):
            text = text.replace(f"__DRUG_PIPE_ARTIFACT_{index}__", artifact)
        return text
    if isinstance(value, list):
        return [sanitize_artifact_paths(item, path_map) for item in value]
    if isinstance(value, tuple):
        return [sanitize_artifact_paths(item, path_map) for item in value]
    if isinstance(value, dict):
        return {str(key): sanitize_artifact_paths(item, path_map) for key, item in value.items()}
    return value


def _meaningful_error(value: Any) -> bool:
    if value in (None, "", False, [], {}):
        return False
    text = str(value).strip().lower()
    return not (text.startswith("no error") or text.startswith("without error"))


def inspect_observation_status(
    payload: Any,
    *,
    event_is_error: bool | None = None,
) -> dict[str, Any]:
    """Interpret status signals without mutating the observation payload."""
    outer = payload if isinstance(payload, dict) else {}
    content = outer.get("content") if isinstance(outer.get("content"), dict) else outer
    outer_status = str(outer.get("status") or "").strip().lower()
    content_status = str(content.get("status") or content.get("state") or "").strip().lower()
    payload_is_error = outer.get("is_error") if isinstance(outer.get("is_error"), bool) else None

    error_signals = [
        outer_status in ERROR_STATUSES,
        content_status in ERROR_STATUSES,
        payload_is_error is True,
        event_is_error is True,
        _meaningful_error(content.get("error")),
    ]
    success_signals = [
        outer_status in SUCCESS_STATUSES,
        content_status in SUCCESS_STATUSES,
        payload_is_error is False,
        event_is_error is False,
    ]
    is_error = any(error_signals)
    status = "error" if is_error else content_status or outer_status or "success"
    return {
        "is_error": is_error,
        "status": status,
        "conflict": any(error_signals) and any(success_signals),
        "outer_status": outer_status or None,
        "content_status": content_status or None,
        "payload_is_error": payload_is_error,
        "event_is_error": event_is_error,
    }
