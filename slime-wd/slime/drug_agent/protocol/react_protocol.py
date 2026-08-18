from __future__ import annotations

import json
import re
from typing import Any

PROTOCOL_AUTO = "auto"
PROTOCOL_REACT_JSON = "react_json"
SUPPORTED_SFT_PROTOCOLS = {
    PROTOCOL_AUTO,
    PROTOCOL_REACT_JSON,
}

_FENCE_RE = re.compile(r"^\s*```(?:[a-zA-Z0-9_-]+)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_EMPTY_THINK_ENVELOPE_RE = re.compile(r"\A\s*<think>\s*</think>\s*")
_QWEN_END_ENVELOPE_RE = re.compile(r"\s*<\|im_end\|>\s*\Z")
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


def _json_objects(text: str) -> tuple[list[dict[str, Any]] | None, str | None, str | None]:
    """Decode whitespace-separated JSON objects without accepting arrays or commas."""
    decoder = json.JSONDecoder()
    payloads: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        try:
            payload, end = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError as exc:
            return None, "ReactJSONDecodeError", f"invalid JSON object sequence: {exc.msg}"
        if not isinstance(payload, dict):
            return None, "ReactSchemaError", "each tool_call entry must be a JSON object"
        payloads.append(payload)
        cursor = end
    if not payloads:
        return None, "ReactSchemaError", "tool_call container must contain at least one JSON object"
    return payloads, None, None


def _validate_tool_call(payload: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    tool_name = payload.get("tool_name")
    arguments = payload.get("arguments")
    if not isinstance(tool_name, str) or not tool_name.strip():
        return False, "ReactSchemaError", "`tool_name` must be a non-empty string"
    if not isinstance(arguments, dict):
        return False, "ReactSchemaError", "`arguments` must be an object"
    return True, None, None


def _validate_final_answer(payload: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    # Data-Pipe's canonical ReAct contract is task-specific and intentionally
    # does not reuse the removed online action-JSON {"answer": ...} envelope.
    if "summary" in payload and not isinstance(payload.get("summary"), str):
        return False, "ReactSchemaError", "canonical `final_answer.summary` must be a string when provided"
    if not isinstance(payload.get("evidence"), list):
        return False, "ReactSchemaError", "canonical `final_answer.evidence` must be a list"
    task_type = str(payload.get("task_type") or "").lower()
    if task_type == "vs":
        if not isinstance(payload.get("ranked_smiles"), list):
            return False, "ReactSchemaError", "VS `final_answer.ranked_smiles` must be a list"
        if not isinstance(payload.get("selected_smiles"), str):
            return False, "ReactSchemaError", "VS `final_answer.selected_smiles` must be a string"
    elif task_type == "ac":
        if not isinstance(payload.get("answer_smiles"), str):
            return False, "ReactSchemaError", "AC `final_answer.answer_smiles` must be a string"
    elif task_type == "pf":
        if not isinstance(payload.get("selected_smiles"), list):
            return False, "ReactSchemaError", "PF `final_answer.selected_smiles` must be a list"
    elif task_type in {"kg", "e2e"}:
        if "result" not in payload:
            return False, "ReactSchemaError", f"{task_type.upper()} `final_answer.result` is required"
    elif task_type == "mol_edit":
        if not isinstance(payload.get("output_smiles"), str):
            return False, "ReactSchemaError", "mol_edit `final_answer.output_smiles` must be a string"
    elif task_type in {"mol_opt", "mol_opt_physchem"}:
        if not isinstance(payload.get("optimized_smiles"), str):
            return False, "ReactSchemaError", "mol_opt `final_answer.optimized_smiles` must be a string"
    else:
        return False, "ReactSchemaError", "canonical `final_answer.task_type` is unsupported"
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

    # Qwen's chat template can prepend this empty transport envelope even when
    # native thinking is disabled. It is not part of Drug-Pipe's canonical
    # ReAct protocol. Strip exactly one leading empty envelope; a non-empty
    # native <think> block remains unsupported and is rejected below.
    if role == "assistant":
        text = _EMPTY_THINK_ENVELOPE_RE.sub("", text, count=1)
        text = _QWEN_END_ENVELOPE_RE.sub("", text, count=1)

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
            payloads, error_type, error_message = _json_objects(clean_body.strip())
            if payloads is None:
                return {
                    "ok": False,
                    "error_type": error_type,
                    "error_message": error_message,
                    "blocks": blocks,
                    "errors": errors + [block],
                    "fence_wrappers_stripped": fence_wrappers_stripped,
                    "fence_inner_content_preserved": fence_inner_content_preserved,
                }
            for payload in payloads:
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
            block["payloads"] = payloads
            # Preserve the legacy single-call block shape for downstream code.
            if len(payloads) == 1:
                block["payload"] = payloads[0]
                block["tool_name"] = payloads[0].get("tool_name")
                block["arguments"] = payloads[0].get("arguments")
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


def parse_runtime_decision(text: str, *, strict_toolrl_turn: bool = False) -> dict[str, Any]:
    """Parse one model generation into the canonical runtime decision shape."""
    parsed = parse_react_sequence(text, role="assistant")
    if not parsed.get("ok"):
        return {
            "ok": False,
            "decision_type": None,
            "thoughts": [],
            "tool_calls": [],
            "final_answer": None,
            "error_type": parsed.get("error_type"),
            "error_message": parsed.get("error_message"),
            "raw_text": text,
        }

    blocks = parsed.get("blocks") if isinstance(parsed.get("blocks"), list) else []
    thoughts = [str(block.get("body") or "") for block in blocks if block.get("kind") == "thought"]
    tool_calls = []
    for block in blocks:
        if block.get("kind") != "tool_call":
            continue
        payloads = block.get("payloads")
        if not isinstance(payloads, list):
            payload = block.get("payload")
            payloads = [payload] if isinstance(payload, dict) else []
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            tool_calls.append(
                {
                    "tool_name": str(payload.get("tool_name") or ""),
                    "arguments": payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {},
                    "raw_payload": payload,
                }
            )
    finals = [block.get("payload") for block in blocks if block.get("kind") == "final_answer"]
    unsupported = [block.get("kind") for block in blocks if block.get("kind") not in {"thought", "tool_call", "final_answer"}]
    error = None
    kinds = [block.get("kind") for block in blocks]
    tool_containers = sum(kind == "tool_call" for kind in kinds)
    if unsupported:
        error = f"unsupported assistant block(s): {unsupported}"
    elif strict_toolrl_turn and tool_containers > 1:
        error = "toolrl_turn_v1 requires exactly one tool_call container"
    elif strict_toolrl_turn and tool_calls and kinds not in (["tool_call"], ["thought", "tool_call"]):
        error = "toolrl_turn_v1 tool decision must be optional thought followed by one tool_call container"
    elif strict_toolrl_turn and finals and kinds not in (["final_answer"], ["thought", "final_answer"]):
        error = "toolrl_turn_v1 final decision must be optional thought followed by one final_answer"
    elif tool_calls and finals:
        error = "tool_call and final_answer cannot appear in the same assistant generation"
    elif len(finals) > 1:
        error = "assistant generation must contain at most one final_answer"
    elif not tool_calls and not finals:
        error = "assistant generation must contain a tool_call or final_answer"
    if error:
        return {
            "ok": False,
            "decision_type": None,
            "thoughts": thoughts,
            "tool_calls": tool_calls,
            "final_answer": finals[-1] if finals else None,
            "error_type": "ReactDecisionError",
            "error_message": error,
            "raw_text": text,
        }
    return {
        "ok": True,
        "decision_type": "tool_call" if tool_calls else "final_answer",
        "thoughts": thoughts,
        "tool_calls": tool_calls,
        "final_answer": finals[-1] if finals else None,
        "error_type": None,
        "error_message": None,
        "raw_text": text,
    }


def project_final_answer(payload: dict[str, Any], task_type: str | None = None) -> Any:
    """Project canonical terminal payload to the task-facing benchmark answer."""
    if not isinstance(payload, dict):
        raise ValueError("final_answer payload must be an object")
    resolved_task = str(payload.get("task_type") or task_type or "").strip().lower()
    if resolved_task == "vs":
        return payload.get("ranked_smiles")
    if resolved_task == "ac":
        return payload.get("answer_smiles")
    if resolved_task == "pf":
        return payload.get("selected_smiles")
    if resolved_task in {"kg", "e2e"}:
        return payload.get("result")
    if resolved_task == "mol_edit":
        return payload.get("output_smiles")
    if resolved_task in {"mol_opt", "mol_opt_physchem"}:
        return payload.get("optimized_smiles")
    raise ValueError(f"unsupported final_answer task_type: {resolved_task or '<missing>'}")


def final_answer_matches_task(payload: Any, task_type: str) -> bool:
    """Require the generated terminal schema to match the active benchmark task."""
    return isinstance(payload, dict) and str(payload.get("task_type") or "").lower() == str(task_type or "").lower()


def detect_sft_protocol(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    schema_version = str(metadata.get("schema_version") or "").strip().lower()
    explicit_protocol = str(metadata.get("protocol") or "").strip().lower()

    if explicit_protocol == PROTOCOL_REACT_JSON:
        return explicit_protocol
    if "react" in schema_version:
        return PROTOCOL_REACT_JSON

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
    return PROTOCOL_REACT_JSON
