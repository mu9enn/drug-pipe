from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NativeParserConfig:
    reasoning_parser: str
    tool_parser: str


def resolve_parser_config(
    *,
    reasoning_parser: str | None = None,
    tool_parser: str | None = None,
) -> NativeParserConfig:
    reasoning = (reasoning_parser or os.environ.get("DRUG_AGENT_NATIVE_REASONING_PARSER") or "").strip()
    tool = (tool_parser or os.environ.get("DRUG_AGENT_NATIVE_TOOL_PARSER") or "").strip()
    if not reasoning or not tool:
        raise RuntimeError(
            "native Qwen parser configuration is required: set "
            "DRUG_AGENT_NATIVE_REASONING_PARSER and DRUG_AGENT_NATIVE_TOOL_PARSER "
            "after validating them against the active checkpoint/SGLang version"
        )
    return NativeParserConfig(reasoning, tool)


def _sglang_tools(tools_schema: list[dict[str, Any]]) -> list[Any]:
    from sglang.srt.entrypoints.openai.protocol import Function, Tool

    output = []
    for tool in tools_schema:
        function = tool.get("function") if isinstance(tool.get("function"), dict) else None
        if tool.get("type") != "function" or function is None:
            raise ValueError("native parser requires OpenAI/Qwen function-tool schemas")
        output.append(Tool(type="function", function=Function(**function)))
    return output


def parse_qwen_native_completion(
    text: str,
    *,
    tools_schema: list[dict[str, Any]],
    reasoning_parser: str | None = None,
    tool_parser: str | None = None,
) -> dict[str, Any]:
    """Parse one Qwen generation through the active SGLang native parsers.

    There is intentionally no Drug-Pipe XML/JSON fallback. Parser selection is
    a serving configuration and is not encoded in the training row schema.
    """
    config = resolve_parser_config(reasoning_parser=reasoning_parser, tool_parser=tool_parser)
    if not isinstance(text, str) or not text.strip():
        return {
            "ok": False,
            "error_type": "EmptyCompletion",
            "error_message": "completion is empty",
            "reasoning": "",
            "text": "",
            "tool_calls": [],
            "has_tool_call": False,
            "has_final_answer": False,
            "final_answer": None,
            "blocks": [],
        }
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    reasoning, body = ReasoningParser(
        model_type=config.reasoning_parser,
        stream_reasoning=False,
    ).parse_non_stream(text)
    reasoning = str(reasoning or "").strip()
    body = str(body or "")
    parser = FunctionCallParser(
        tools=_sglang_tools(tools_schema),
        tool_call_parser=config.tool_parser,
    )
    calls = []
    visible = body
    if parser.has_tool_call(body):
        try:
            visible, calls = parser.parse_non_stream(body)
        except Exception as exc:
            return {
                "ok": False,
                "error_type": "NativeToolParseError",
                "error_message": str(exc),
                "reasoning": reasoning,
                "text": body.strip(),
                "tool_calls": [],
                "has_tool_call": False,
                "has_final_answer": False,
                "final_answer": None,
                "blocks": ([{"kind": "thought", "body": reasoning}] if reasoning else []),
            }
    parsed_calls: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        try:
            arguments = json.loads(call.parameters or "{}")
        except json.JSONDecodeError as exc:
            return {
                "ok": False,
                "error_type": "NativeArgumentParseError",
                "error_message": f"call {index}: {exc}",
                "reasoning": reasoning,
                "text": str(visible or "").strip(),
                "tool_calls": parsed_calls,
                "has_tool_call": bool(parsed_calls),
                "has_final_answer": False,
                "final_answer": None,
                "blocks": [],
            }
        if not isinstance(arguments, dict):
            return {
                "ok": False,
                "error_type": "NativeArgumentTypeError",
                "error_message": f"call {index} arguments are not an object",
                "reasoning": reasoning,
                "text": str(visible or "").strip(),
                "tool_calls": parsed_calls,
                "has_tool_call": bool(parsed_calls),
                "has_final_answer": False,
                "final_answer": None,
                "blocks": [],
            }
        name = str(call.name or "").strip()
        parsed_calls.append(
            {
                "index": index,
                "tool_name_raw": name,
                "tool_name": name,
                "name": name,
                "arguments": arguments,
            }
        )
    visible = str(visible or "").strip()
    if parsed_calls and visible:
        return {
            "ok": False,
            "error_type": "NativeToolCallSuffix",
            "error_message": "tool-call decision contains visible suffix text",
            "reasoning": reasoning,
            "text": visible,
            "tool_calls": parsed_calls,
            "has_tool_call": True,
            "has_final_answer": False,
            "final_answer": None,
            "blocks": [],
        }
    blocks = ([{"kind": "thought", "body": reasoning}] if reasoning else [])
    blocks.extend({"kind": "tool_call", "payload": call} for call in parsed_calls)
    if not parsed_calls and visible:
        blocks.append({"kind": "final_answer", "payload": visible})
    return {
        "ok": bool(parsed_calls or visible),
        "error_type": None if parsed_calls or visible else "EmptyDecision",
        "error_message": None if parsed_calls or visible else "completion has neither calls nor final content",
        "reasoning": reasoning,
        "text": visible,
        "tool_calls": parsed_calls,
        "has_tool_call": bool(parsed_calls),
        "has_final_answer": bool(visible and not parsed_calls),
        "final_answer": visible if visible and not parsed_calls else None,
        "blocks": blocks,
        "parser_config": {
            "reasoning_parser": config.reasoning_parser,
            "tool_parser": config.tool_parser,
        },
    }
