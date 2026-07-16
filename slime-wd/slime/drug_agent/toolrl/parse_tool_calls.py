from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from drug_agent.protocol.react_protocol import parse_react_sequence
from drug_agent.toolrl.normalization import canonical_tool_name
from drug_agent.tools.tool_registry import load_allowlist
from drug_agent.utils import normalize_tool_name


DEFAULT_ALLOWLIST_PATH = Path(__file__).resolve().parents[1] / "tools" / "allowlist_v0.json"


@dataclass
class ParsedToolCall:
    index: int
    tool_name_raw: str
    tool_name: str
    arguments: dict[str, Any]
    keep: bool
    raw_payload: dict[str, Any]
    block: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "tool_name_raw": self.tool_name_raw,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "keep": self.keep,
            "raw_payload": self.raw_payload,
        }


@lru_cache(maxsize=1)
def default_molclaw_allowlist() -> set[str]:
    return load_allowlist(DEFAULT_ALLOWLIST_PATH)


def _canonical_allowlist_name(tool_name: str | None) -> str:
    return canonical_tool_name(tool_name, None)


def _is_molclaw_tool(tool_name: str, allowed_tool_names: set[str] | None) -> bool:
    bare = normalize_tool_name(tool_name)
    canonical = _canonical_allowlist_name(bare)
    if not canonical:
        return False
    if allowed_tool_names is None:
        allowed_tool_names = default_molclaw_allowlist()
    if allowed_tool_names:
        canonical_allowed = {_canonical_allowlist_name(name) for name in allowed_tool_names}
        return canonical in canonical_allowed
    return tool_name.startswith("mcp__molclaw-scp__") or bare.startswith("mcp__molclaw-scp__")


def parse_tool_calls(
    text: str,
    *,
    role: str = "assistant",
    allowed_tool_names: set[str] | None = None,
    keep_non_molclaw: bool = False,
) -> dict[str, Any]:
    """Parse ReAct content and extract one or more tool calls.

    The parser accepts tagged ReAct assistant messages and returns all
    `tool_call` blocks in order. By default, only MolClaw tools listed in the
    allowlist are marked as keep=True and included in `molclaw_tool_calls`.
    """

    parsed = parse_react_sequence(text, role=role)
    result: dict[str, Any] = {
        "ok": bool(parsed.get("ok")),
        "error_type": parsed.get("error_type"),
        "error_message": parsed.get("error_message"),
        "mode": parsed.get("mode"),
        "fence_wrappers_stripped": int(parsed.get("fence_wrappers_stripped") or 0),
        "fence_inner_content_preserved": int(parsed.get("fence_inner_content_preserved") or 0),
        "blocks": parsed.get("blocks") or [],
        "tool_calls": [],
        "molclaw_tool_calls": [],
        "non_molclaw_tool_calls": [],
    }
    if not result["ok"]:
        return result

    tool_calls: list[ParsedToolCall] = []
    for block_index, block in enumerate(result["blocks"]):
        if not isinstance(block, dict) or block.get("kind") != "tool_call":
            continue
        payload = block.get("payload")
        if not isinstance(payload, dict):
            continue
        tool_name_raw = str(payload.get("tool_name") or "")
        tool_name = canonical_tool_name(tool_name_raw or None, None)
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        keep = _is_molclaw_tool(tool_name_raw or tool_name, allowed_tool_names)
        item = ParsedToolCall(
            index=block_index,
            tool_name_raw=tool_name_raw,
            tool_name=tool_name,
            arguments=arguments,
            keep=keep,
            raw_payload=payload,
            block=block,
        )
        tool_calls.append(item)
        if keep:
            result["molclaw_tool_calls"].append(item.to_dict())
            if keep_non_molclaw:
                result["tool_calls"].append(item.to_dict())
        else:
            result["non_molclaw_tool_calls"].append(item.to_dict())
            if keep_non_molclaw:
                result["tool_calls"].append(item.to_dict())

    if not keep_non_molclaw:
        result["tool_calls"] = result["molclaw_tool_calls"]
    else:
        result["tool_calls"] = [item.to_dict() for item in tool_calls]

    result["tool_call_count"] = len(result["tool_calls"])
    result["molclaw_tool_call_count"] = len(result["molclaw_tool_calls"])
    result["non_molclaw_tool_call_count"] = len(result["non_molclaw_tool_calls"])
    result["has_tool_call"] = result["tool_call_count"] > 0
    result["has_final_answer"] = any(isinstance(block, dict) and block.get("kind") == "final_answer" for block in result["blocks"])
    return result
