from __future__ import annotations

from typing import Any

from drug_agent.utils import to_jsonable

ERROR_STATUS_VALUES = {
    "error",
    "failed",
    "failure",
    "invalid",
    "timeout",
    "exception",
}
SUCCESS_STATUS_VALUES = {
    "ok",
    "success",
    "succeeded",
    "completed",
    "done",
}


def _norm_status(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip().lower()
        if text:
            return text
    return None


def _extract_status(parsed_payload: Any, raw_payload: Any) -> str | None:
    candidates: list[Any] = []
    if isinstance(parsed_payload, dict):
        candidates.extend(
            [
                parsed_payload.get("status"),
                (parsed_payload.get("result") or {}).get("status") if isinstance(parsed_payload.get("result"), dict) else None,
                (parsed_payload.get("structuredContent") or {}).get("status")
                if isinstance(parsed_payload.get("structuredContent"), dict)
                else None,
            ]
        )
    if isinstance(raw_payload, dict):
        candidates.extend(
            [
                raw_payload.get("status"),
                (raw_payload.get("structuredContent") or {}).get("status")
                if isinstance(raw_payload.get("structuredContent"), dict)
                else None,
            ]
        )

    for value in candidates:
        status = _norm_status(value)
        if status is not None:
            return status
    return None


def evaluate_tool_success(
    *,
    transport_ok: bool,
    tool_schema_valid: bool,
    parsed_payload: Any,
    raw_payload: Any,
    unknown_as_failure: bool = True,
) -> dict[str, Any]:
    parsed_payload = to_jsonable(parsed_payload)
    raw_payload = to_jsonable(raw_payload)

    if not transport_ok:
        return {
            "transport_ok": False,
            "tool_schema_valid": bool(tool_schema_valid),
            "tool_execution_success": False,
            "tool_semantic_success": False,
            "semantic_unknown": False,
            "semantic_error_type": "TransportError",
            "semantic_error_message": "MCP transport failed",
            "status": None,
        }

    if not tool_schema_valid:
        return {
            "transport_ok": True,
            "tool_schema_valid": False,
            "tool_execution_success": False,
            "tool_semantic_success": False,
            "semantic_unknown": False,
            "semantic_error_type": "ToolValidationError",
            "semantic_error_message": "tool schema validation failed",
            "status": None,
        }

    is_error = raw_payload.get("isError") if isinstance(raw_payload, dict) else None
    status = _extract_status(parsed_payload, raw_payload)

    if is_error is True:
        return {
            "transport_ok": True,
            "tool_schema_valid": True,
            "tool_execution_success": False,
            "tool_semantic_success": False,
            "semantic_unknown": False,
            "semantic_error_type": "ToolRuntimeError",
            "semantic_error_message": "MCP result has isError=true",
            "status": status,
        }

    if status in ERROR_STATUS_VALUES:
        return {
            "transport_ok": True,
            "tool_schema_valid": True,
            "tool_execution_success": False,
            "tool_semantic_success": False,
            "semantic_unknown": False,
            "semantic_error_type": "ToolStatusError",
            "semantic_error_message": f"tool status indicates failure: {status}",
            "status": status,
        }

    if status in SUCCESS_STATUS_VALUES:
        return {
            "transport_ok": True,
            "tool_schema_valid": True,
            "tool_execution_success": True,
            "tool_semantic_success": True,
            "semantic_unknown": False,
            "semantic_error_type": None,
            "semantic_error_message": None,
            "status": status,
        }

    semantic_success = not unknown_as_failure
    return {
        "transport_ok": True,
        "tool_schema_valid": True,
        "tool_execution_success": True,
        "tool_semantic_success": bool(semantic_success),
        "semantic_unknown": True,
        "semantic_error_type": "ToolSemanticUnknown" if unknown_as_failure else None,
        "semantic_error_message": "tool semantic status is unknown",
        "status": status,
    }


def make_validation_failed_result(
    *,
    tool_name: str,
    message: str,
    tool_reason: str | None = None,
    args_reason: str | None = None,
) -> dict[str, Any]:
    tool_success = evaluate_tool_success(
        transport_ok=False,
        tool_schema_valid=False,
        parsed_payload=None,
        raw_payload=None,
    )
    return {
        "ok": False,
        "tool_name": tool_name,
        "result": None,
        "error": {
            "type": "ToolValidationError",
            "message": message,
        },
        "latency_sec": 0.0,
        "transport_ok": tool_success["transport_ok"],
        "tool_schema_valid": tool_success["tool_schema_valid"],
        "tool_execution_success": tool_success["tool_execution_success"],
        "tool_semantic_success": tool_success["tool_semantic_success"],
        "semantic_unknown": tool_success["semantic_unknown"],
        "metadata": {
            "tool_reason": tool_reason,
            "args_reason": args_reason,
            "tool_success": tool_success,
        },
    }

