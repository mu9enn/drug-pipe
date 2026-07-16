from __future__ import annotations

import json
import re
from typing import Any

PROTOCOL_AUTO = "auto"
PROTOCOL_ACTION_JSON = "action_json"
PROTOCOL_REACT_JSON = "react_json"
SUPPORTED_SFT_PROTOCOLS = {
    PROTOCOL_AUTO,
    PROTOCOL_ACTION_JSON,
    PROTOCOL_REACT_JSON,
}

_FENCE_RE = re.compile(r"^\s*```(?:[a-zA-Z0-9_-]+)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_BLOCK_RE = re.compile(
    r"""
    \s*(?:
        <thought>(?P<thought>.*?)</thought>
      | <tool_call>(?P<tool_call>.*?)</tool_call>
      | <final_answer>(?P<final_answer>.*?)</final_answer>
      | <observation\s+tool_name=(?:"(?P<observation_tool_name_dq>[^"]+)"|'(?P<observation_tool_name_sq>[^']+)')>(?P<observation>.*?)</observation>
    )\s*
    """,
    re.DOTALL | re.VERBOSE,
)


def _strip_markdown_fence(text: str) -> tuple[str, bool, bool]:
    if not isinstance(text, str):
        return "", False, False

    candidate = text.strip()
    match = _FENCE_RE.fullmatch(candidate)
    if not match:
        return text, False, False

    inner = match.group(1) or ""
    return inner, True, bool(inner.strip())


def _json_object(text: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, "ReactJSONDecodeError", f"invalid JSON: {exc.msg}"
    if not isinstance(payload, dict):
        return None, "ReactSchemaError", "top-level JSON must be an object"
    return payload, None, None


def _validate_tool_call(payload: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    tool_name = payload.get("tool_name")
    arguments = payload.get("arguments")
    if not isinstance(tool_name, str) or not tool_name.strip():
        return False, "ReactSchemaError", "`tool_name` must be a non-empty string"
    if not isinstance(arguments, dict):
        return False, "ReactSchemaError", "`arguments` must be an object"
    return True, None, None


def _validate_final_answer(payload: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    answer = payload.get("answer")
    if not isinstance(answer, dict):
        return False, "ReactSchemaError", "`final_answer.answer` must be an object"

    if "summary" not in answer or not isinstance(answer.get("summary"), str):
        return False, "ReactSchemaError", "`answer.summary` must be a string"
    if "evidence" not in answer or not isinstance(answer.get("evidence"), list):
        return False, "ReactSchemaError", "`answer.evidence` must be a list"
    if "result" not in answer or not isinstance(answer.get("result"), dict):
        return False, "ReactSchemaError", "`answer.result` must be an object"

    ranked = answer.get("ranked_molecules")
    if ranked is not None and not isinstance(ranked, list):
        return False, "ReactSchemaError", "`answer.ranked_molecules` must be a list when provided"

    return True, None, None


def _validate_observation(payload: dict[str, Any], tool_name: str) -> tuple[bool, str | None, str | None]:
    if not tool_name.strip():
        return False, "ReactSchemaError", "`observation tool_name` must be a non-empty string"

    if not any(key in payload for key in ("ok", "status", "content", "metadata")):
        return False, "ReactSchemaError", "`observation` must contain at least one of ok/status/content/metadata"

    return True, None, None


def parse_react_sequence(text: str, *, role: str | None = None) -> dict[str, Any]:
    """Parse one ReAct-style message.

    The function accepts:
    - assistant messages containing one or more tagged blocks
    - user messages that are either plain prompt text or one or more observation blocks

    Plain text is only accepted for user prompt turns. Any other untagged text
    outside supported tags is treated as a parse failure.
    """

    if not isinstance(text, str):
        return {
            "ok": False,
            "error_type": "ReactTypeError",
            "error_message": "message content must be a string",
            "blocks": [],
            "fence_wrappers_stripped": 0,
            "fence_inner_content_preserved": 0,
        }

    stripped = text.strip()
    if not stripped:
        return {
            "ok": False,
            "error_type": "ReactFormatError",
            "error_message": "message content is empty",
            "blocks": [],
            "fence_wrappers_stripped": 0,
            "fence_inner_content_preserved": 0,
        }

    if role == "user":
        user_text = stripped.lstrip()
        if not user_text.startswith("<observation"):
            return {
                "ok": True,
                "mode": "plain_user_prompt",
                "blocks": [
                    {
                        "kind": "plain_user_text",
                        "text": stripped,
                    }
                ],
                "fence_wrappers_stripped": 0,
                "fence_inner_content_preserved": 0,
            }

    if role == "assistant" and "<" not in stripped:
        return {
            "ok": False,
            "error_type": "ReactFormatError",
            "error_message": "assistant message must use tagged ReAct blocks",
            "blocks": [],
            "fence_wrappers_stripped": 0,
            "fence_inner_content_preserved": 0,
        }

    blocks: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    fence_wrappers_stripped = 0
    fence_inner_content_preserved = 0

    pos = 0
    saw_tag = False
    while pos < len(text):
        match = _BLOCK_RE.match(text, pos)
        if not match:
            remainder = text[pos:]
            if remainder.strip() == "":
                break
            if role == "user" and not saw_tag:
                return {
                    "ok": True,
                    "mode": "plain_user_prompt",
                    "blocks": [
                        {
                            "kind": "plain_user_text",
                            "text": stripped,
                        }
                    ],
                    "fence_wrappers_stripped": 0,
                    "fence_inner_content_preserved": 0,
                }
            return {
                "ok": False,
                "error_type": "ReactFormatError",
                "error_message": "content outside supported ReAct tags",
                "blocks": blocks,
                "errors": errors,
                "fence_wrappers_stripped": fence_wrappers_stripped,
                "fence_inner_content_preserved": fence_inner_content_preserved,
            }

        saw_tag = True
        kind = "thought"
        raw_body = ""
        tool_name = None
        if match.group("tool_call") is not None:
            kind = "tool_call"
            raw_body = match.group("tool_call") or ""
        elif match.group("final_answer") is not None:
            kind = "final_answer"
            raw_body = match.group("final_answer") or ""
        elif match.group("observation") is not None:
            kind = "observation"
            raw_body = match.group("observation") or ""
            tool_name = match.group("observation_tool_name_dq") or match.group("observation_tool_name_sq")
        else:
            raw_body = match.group("thought") or ""

        clean_body, fence_wrapped, fence_inner_preserved = _strip_markdown_fence(raw_body)
        if fence_wrapped:
            fence_wrappers_stripped += 1
        if fence_inner_preserved:
            fence_inner_content_preserved += 1

        block: dict[str, Any] = {
            "kind": kind,
            "raw_body": raw_body,
            "body": clean_body,
            "fence_wrapped": fence_wrapped,
            "fence_inner_preserved": fence_inner_preserved,
        }

        if kind == "thought":
            if not clean_body.strip():
                return {
                    "ok": False,
                    "error_type": "ReactSchemaError",
                    "error_message": "`thought` body must be non-empty",
                    "blocks": blocks,
                    "errors": errors + [block],
                    "fence_wrappers_stripped": fence_wrappers_stripped,
                    "fence_inner_content_preserved": fence_inner_content_preserved,
                }
            block["text"] = clean_body
        elif kind == "tool_call":
            payload, error_type, error_message = _json_object(clean_body.strip())
            if payload is None:
                return {
                    "ok": False,
                    "error_type": error_type,
                    "error_message": error_message,
                    "blocks": blocks,
                    "errors": errors + [block],
                    "fence_wrappers_stripped": fence_wrappers_stripped,
                    "fence_inner_content_preserved": fence_inner_content_preserved,
                }
            ok, error_type, error_message = _validate_tool_call(payload)
            if not ok:
                return {
                    "ok": False,
                    "error_type": error_type,
                    "error_message": error_message,
                    "blocks": blocks,
                    "errors": errors + [block],
                    "fence_wrappers_stripped": fence_wrappers_stripped,
                    "fence_inner_content_preserved": fence_inner_content_preserved,
                }
            block["payload"] = payload
            block["tool_name"] = payload.get("tool_name")
            block["arguments"] = payload.get("arguments")
        elif kind == "final_answer":
            payload, error_type, error_message = _json_object(clean_body.strip())
            if payload is None:
                return {
                    "ok": False,
                    "error_type": error_type,
                    "error_message": error_message,
                    "blocks": blocks,
                    "errors": errors + [block],
                    "fence_wrappers_stripped": fence_wrappers_stripped,
                    "fence_inner_content_preserved": fence_inner_content_preserved,
                }
            ok, error_type, error_message = _validate_final_answer(payload)
            if not ok:
                return {
                    "ok": False,
                    "error_type": error_type,
                    "error_message": error_message,
                    "blocks": blocks,
                    "errors": errors + [block],
                    "fence_wrappers_stripped": fence_wrappers_stripped,
                    "fence_inner_content_preserved": fence_inner_content_preserved,
                }
            block["payload"] = payload
        elif kind == "observation":
            payload, error_type, error_message = _json_object(clean_body.strip())
            if payload is None:
                return {
                    "ok": False,
                    "error_type": error_type,
                    "error_message": error_message,
                    "blocks": blocks,
                    "errors": errors + [block],
                    "fence_wrappers_stripped": fence_wrappers_stripped,
                    "fence_inner_content_preserved": fence_inner_content_preserved,
                }
            ok, error_type, error_message = _validate_observation(payload, tool_name or "")
            if not ok:
                return {
                    "ok": False,
                    "error_type": error_type,
                    "error_message": error_message,
                    "blocks": blocks,
                    "errors": errors + [block],
                    "fence_wrappers_stripped": fence_wrappers_stripped,
                    "fence_inner_content_preserved": fence_inner_content_preserved,
                }
            block["payload"] = payload
            block["tool_name"] = tool_name
        else:
            return {
                "ok": False,
                "error_type": "ReactFormatError",
                "error_message": f"unsupported react block: {kind}",
                "blocks": blocks,
                "errors": errors + [block],
                "fence_wrappers_stripped": fence_wrappers_stripped,
                "fence_inner_content_preserved": fence_inner_content_preserved,
            }

        blocks.append(block)
        pos = match.end()

    if not blocks:
        if role == "user" and "<" not in stripped:
            return {
                "ok": True,
                "mode": "plain_user_prompt",
                "blocks": [
                    {
                        "kind": "plain_user_text",
                        "text": stripped,
                    }
                ],
                "fence_wrappers_stripped": 0,
                "fence_inner_content_preserved": 0,
            }
        return {
            "ok": False,
            "error_type": "ReactFormatError",
            "error_message": "no supported ReAct blocks found",
            "blocks": [],
            "fence_wrappers_stripped": fence_wrappers_stripped,
            "fence_inner_content_preserved": fence_inner_content_preserved,
        }

    return {
        "ok": True,
        "mode": "tagged",
        "blocks": blocks,
        "fence_wrappers_stripped": fence_wrappers_stripped,
        "fence_inner_content_preserved": fence_inner_content_preserved,
    }


def detect_sft_protocol(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    schema_version = str(metadata.get("schema_version") or "").strip().lower()
    explicit_protocol = str(metadata.get("protocol") or "").strip().lower()

    if explicit_protocol in {PROTOCOL_ACTION_JSON, PROTOCOL_REACT_JSON}:
        return explicit_protocol
    if "react" in schema_version:
        return PROTOCOL_REACT_JSON
    if "action" in schema_version or "legacy" in schema_version:
        return PROTOCOL_ACTION_JSON

    messages = record.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            stripped = content.strip()
            if any(tag in stripped for tag in ("<thought>", "<tool_call>", "<final_answer>", "<observation")):
                return PROTOCOL_REACT_JSON
            if stripped.startswith("{") and '"type"' in stripped:
                return PROTOCOL_ACTION_JSON

    return PROTOCOL_ACTION_JSON
