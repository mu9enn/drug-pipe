from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from drug_agent.protocol.action_schema import (
    ACTION_FINAL_ANSWER,
    ACTION_TOOL_CALL,
    ALLOWED_ACTION_TYPES,
    REQUIRED_FINAL_ANSWER_SUBFIELDS,
)


@dataclass
class ParseResult:
    ok: bool
    action_type: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    answer: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    raw_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action_type": self.action_type,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "answer": self.answer,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "raw_text": self.raw_text,
        }


def _err(raw_text: str, error_type: str, error_message: str) -> ParseResult:
    return ParseResult(
        ok=False,
        raw_text=raw_text,
        error_type=error_type,
        error_message=error_message,
    )


def parse_action(raw_text: str) -> ParseResult:
    if not isinstance(raw_text, str):
        return _err(str(raw_text), "ActionTypeError", "raw_text must be a string")

    text = raw_text.strip()
    if text.startswith("```"):
        return _err(raw_text, "ActionFormatError", "markdown fenced block is not allowed")
    if text.startswith("<"):
        return _err(raw_text, "ActionFormatError", "XML-like format is not allowed")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return _err(raw_text, "ActionJSONDecodeError", f"invalid JSON: {exc.msg}")

    if not isinstance(payload, dict):
        return _err(raw_text, "ActionSchemaError", "top-level JSON must be an object")

    action_type = payload.get("type")
    if action_type not in ALLOWED_ACTION_TYPES:
        return _err(raw_text, "ActionSchemaError", f"unsupported action type: {action_type}")

    if action_type == ACTION_TOOL_CALL:
        tool_name = payload.get("tool_name")
        arguments = payload.get("arguments")
        if not isinstance(tool_name, str) or not tool_name.strip():
            return _err(raw_text, "ActionSchemaError", "`tool_name` must be a non-empty string")
        if not isinstance(arguments, dict):
            return _err(raw_text, "ActionSchemaError", "`arguments` must be an object")
        return ParseResult(
            ok=True,
            action_type=ACTION_TOOL_CALL,
            tool_name=tool_name.strip(),
            arguments=arguments,
            raw_text=raw_text,
        )

    answer = payload.get("answer")
    if not isinstance(answer, dict):
        return _err(raw_text, "ActionSchemaError", "`final_answer.answer` must be an object")

    for key in REQUIRED_FINAL_ANSWER_SUBFIELDS:
        if key not in answer:
            return _err(raw_text, "ActionSchemaError", f"`answer.{key}` is required")

    if not isinstance(answer.get("summary"), str):
        return _err(raw_text, "ActionSchemaError", "`answer.summary` must be a string")
    if not isinstance(answer.get("evidence"), list):
        return _err(raw_text, "ActionSchemaError", "`answer.evidence` must be a list")
    if not isinstance(answer.get("result"), dict):
        return _err(raw_text, "ActionSchemaError", "`answer.result` must be an object")

    ranked = answer.get("ranked_molecules")
    if ranked is not None and not isinstance(ranked, list):
        return _err(raw_text, "ActionSchemaError", "`answer.ranked_molecules` must be a list when provided")

    return ParseResult(
        ok=True,
        action_type=ACTION_FINAL_ANSWER,
        answer=answer,
        raw_text=raw_text,
    )


def _run_self_test() -> int:
    tests = [
        (
            "valid_tool_call",
            '{"type":"tool_call","tool_name":"is_valid_smiles","arguments":{"smiles":"CCO"}}',
            True,
        ),
        (
            "valid_final_answer",
            '{"type":"final_answer","answer":{"summary":"done","evidence":[],"result":{},"ranked_molecules":[]}}',
            True,
        ),
        (
            "markdown_fenced_invalid",
            '```json\n{"type":"tool_call","tool_name":"x","arguments":{}}\n```',
            False,
        ),
        ("malformed_json_invalid", '{"type":"tool_call"', False),
        (
            "arguments_not_object_invalid",
            '{"type":"tool_call","tool_name":"x","arguments":"oops"}',
            False,
        ),
    ]

    ok_all = True
    for name, raw, expected_ok in tests:
        result = parse_action(raw)
        passed = result.ok == expected_ok
        ok_all = ok_all and passed
        print(f"[{name}] expected_ok={expected_ok} got_ok={result.ok} passed={passed}")
        if not passed:
            print("  result:", json.dumps(result.to_dict(), ensure_ascii=False))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(_run_self_test())
