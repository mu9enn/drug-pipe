from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from drug_agent.tools.artifact_registry import ArtifactRegistry


SERVER_FILE_TO_BASE64 = "server_file_to_base64"
DEFAULT_MAX_DOWNLOAD_BYTES = 10_000_000
_BASE64_KEYS = ("base64_string", "file_base64_string")
_WRAPPER_KEYS = ("result", "content", "data", "structuredContent", "structured_content")


class ServerFileMaterializationError(ValueError):
    pass


def _max_download_bytes() -> int:
    raw = os.environ.get("DRUG_AGENT_MAX_SERVER_FILE_DOWNLOAD_BYTES", "").strip()
    if not raw:
        return DEFAULT_MAX_DOWNLOAD_BYTES
    try:
        value = int(raw)
    except ValueError as exc:
        raise ServerFileMaterializationError(
            "DRUG_AGENT_MAX_SERVER_FILE_DOWNLOAD_BYTES must be an integer"
        ) from exc
    if value <= 0:
        raise ServerFileMaterializationError(
            "DRUG_AGENT_MAX_SERVER_FILE_DOWNLOAD_BYTES must be positive"
        )
    return value


def _json_mapping(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return value
    return decoded


def _find_base64_payload(value: Any, *, inherited_name: str | None = None) -> tuple[str, str | None] | None:
    value = _json_mapping(value)
    if isinstance(value, dict):
        file_name = value.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            file_name = inherited_name
        for key in _BASE64_KEYS:
            encoded = value.get(key)
            if isinstance(encoded, str) and encoded.strip():
                return encoded, file_name
        for key in _WRAPPER_KEYS:
            if key in value:
                found = _find_base64_payload(value[key], inherited_name=file_name)
                if found is not None:
                    return found
        for key, item in value.items():
            if key in _WRAPPER_KEYS or key in _BASE64_KEYS:
                continue
            if isinstance(item, (dict, list)):
                found = _find_base64_payload(item, inherited_name=file_name)
                if found is not None:
                    return found
    elif isinstance(value, list):
        for item in value:
            found = _find_base64_payload(item, inherited_name=inherited_name)
            if found is not None:
                return found
    return None


def _safe_file_name(server_name: str | None, source_path: Any) -> str:
    candidates = [server_name, source_path if isinstance(source_path, str) else None]
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        normalized = candidate.strip().replace("\\", "/")
        name = PurePosixPath(normalized).name
        if name and name not in {".", ".."} and "\x00" not in name:
            return name
    return "download.bin"


def _decode_base64(encoded: str, *, max_bytes: int) -> bytes:
    compact = "".join(encoded.split())
    if not compact:
        raise ServerFileMaterializationError("MCP result contains an empty base64 payload")
    # Reject oversized payloads before allocating the decoded byte buffer.
    estimated_bytes = (len(compact) // 4) * 3
    if compact.endswith("=="):
        estimated_bytes -= 2
    elif compact.endswith("="):
        estimated_bytes -= 1
    if estimated_bytes > max_bytes:
        raise ServerFileMaterializationError(
            f"decoded server file exceeds the {max_bytes}-byte client limit"
        )
    try:
        decoded = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ServerFileMaterializationError("MCP result contains invalid base64") from exc
    if len(decoded) > max_bytes:
        raise ServerFileMaterializationError(
            f"decoded server file exceeds the {max_bytes}-byte client limit"
        )
    return decoded


def _atomic_write(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def materialize_server_file_result(
    tool_result: dict[str, Any],
    *,
    execution_args: dict[str, Any],
    artifact_registry: ArtifactRegistry,
) -> dict[str, Any]:
    """Decode a server_file_to_base64 result without exposing base64 to the model."""

    if not bool(tool_result.get("ok")):
        return tool_result

    sanitized = dict(tool_result)
    metadata = dict(sanitized.get("metadata") or {})
    # This raw MCP copy contains the same large base64 payload. It is neither
    # model-visible nor useful after the verified local artifact is created.
    metadata.pop("raw", None)
    sanitized["metadata"] = metadata

    try:
        found = _find_base64_payload(sanitized.get("result"))
        if found is None:
            raise ServerFileMaterializationError("MCP result is missing base64_string")
        encoded, server_name = found
        payload = _decode_base64(encoded, max_bytes=_max_download_bytes())
        file_name = _safe_file_name(server_name, execution_args.get("file_path"))
        target = (artifact_registry.workspace / file_name).resolve(strict=False)
        try:
            relative = target.relative_to(artifact_registry.workspace)
        except ValueError as exc:
            raise ServerFileMaterializationError("download target escapes the task workspace") from exc
        _atomic_write(target, payload)
        digest = hashlib.sha256(payload).hexdigest()
        result = {
            "status": "success",
            "artifact": artifact_registry.register_local(relative.as_posix()),
            "bytes_written": len(payload),
            "sha256": digest,
        }
        sanitized["result"] = result
        metadata["server_file_materialization"] = {
            "status": "success",
            "bytes_written": len(payload),
            "sha256": digest,
        }
        return sanitized
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        sanitized.update(
            {
                "ok": False,
                "result": None,
                "error": {"type": "ServerFileMaterializationError", "message": message},
                "tool_execution_success": False,
                "tool_semantic_success": False,
                "semantic_unknown": False,
            }
        )
        metadata["server_file_materialization"] = {"status": "error", "message": message}
        return sanitized
