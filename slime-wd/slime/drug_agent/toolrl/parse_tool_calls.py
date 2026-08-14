from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

from drug_agent.protocol.react_protocol import parse_react_sequence
from drug_agent.toolrl.normalization import canonical_tool_name
from drug_agent.tools.local_tools import LOCAL_TOOL_NAMES
from drug_agent.utils import normalize_tool_name


_LOCAL_TOOL_NAME_BY_CASEFOLD = {name.casefold(): name for name in LOCAL_TOOL_NAMES}


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


@lru_cache(maxsize=8)
def _load_offline_catalog_names(raw_path: str) -> set[str]:
    payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    rows = payload.get("tools") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("DRUG_AGENT_TOOL_CATALOG must contain a tools list")
    return {
        normalize_tool_name(item.get("name"))
        for item in rows
        if isinstance(item, dict) and item.get("executor") != "local_sandbox" and normalize_tool_name(item.get("name"))
    }


def default_molclaw_allowlist() -> set[str]:
    """Optional offline catalog, never the authority for online execution."""
    raw_path = os.environ.get("DRUG_AGENT_TOOL_CATALOG", "").strip()
    if not raw_path:
        return set()
    return _load_offline_catalog_names(str(Path(raw_path).expanduser().resolve()))


def _canonical_allowlist_name(tool_name: str | None) -> str:
    return canonical_tool_name(tool_name, None)


def canonical_decision_tool_name(tool_name: str | None) -> str:
    """Return the runtime spelling for local tools and canonical MCP spelling otherwise."""
    bare = normalize_tool_name(tool_name)
    local_name = _LOCAL_TOOL_NAME_BY_CASEFOLD.get(bare.casefold())
    return local_name or canonical_tool_name(bare, None)


def _is_molclaw_tool(tool_name: str, allowed_tool_names: set[str] | None) -> bool:
    bare = normalize_tool_name(tool_name)
    if bare.casefold() in _LOCAL_TOOL_NAME_BY_CASEFOLD:
        return False
    canonical = _canonical_allowlist_name(bare)
    if not canonical:
        return False
    if allowed_tool_names is None:
        allowed_tool_names = default_molclaw_allowlist()
    if allowed_tool_names:
        canonical_allowed = {_canonical_allowlist_name(name) for name in allowed_tool_names}
        return canonical in canonical_allowed
    return bare.casefold() not in _LOCAL_TOOL_NAME_BY_CASEFOLD


def is_molclaw_decision_name(tool_name: str, allowed_tool_names: set[str] | None = None) -> bool:
    return _is_molclaw_tool(tool_name, allowed_tool_names)


def is_local_decision_name(tool_name: str | None) -> bool:
    return normalize_tool_name(tool_name).casefold() in _LOCAL_TOOL_NAME_BY_CASEFOLD


def is_supported_decision_name(tool_name: str, allowed_tool_names: set[str] | None = None) -> bool:
    return is_local_decision_name(tool_name) or _is_molclaw_tool(tool_name, allowed_tool_names)


def supported_training_tool_names(allowed_tool_names: set[str] | None = None) -> set[str]:
    if allowed_tool_names is None:
        allowed_tool_names = default_molclaw_allowlist()
    return {canonical_decision_tool_name(name) for name in allowed_tool_names} | set(LOCAL_TOOL_NAMES)


def parse_tool_calls(
    text: str,
    *,
    role: str = "assistant",
    allowed_tool_names: set[str] | None = None,
    keep_non_molclaw: bool = False,
) -> dict[str, Any]:
    """Parse ReAct content and extract one or more tool calls.

    The parser accepts tagged ReAct assistant messages and returns all
    `tool_call` blocks in order. MolClaw, supported local, and unsupported
    calls are classified separately; `keep` remains the legacy MolClaw-only
    flag and must not be used as the training-decision authority.
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
        "supported_tool_calls": [],
        "molclaw_tool_calls": [],
        "local_tool_calls": [],
        "unsupported_tool_calls": [],
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
        tool_name = canonical_decision_tool_name(tool_name_raw or None)
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        keep = _is_molclaw_tool(tool_name_raw or tool_name, allowed_tool_names)
        is_local = is_local_decision_name(tool_name_raw or tool_name)
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
            result["supported_tool_calls"].append(item.to_dict())
            if keep_non_molclaw:
                result["tool_calls"].append(item.to_dict())
        else:
            result["non_molclaw_tool_calls"].append(item.to_dict())
            if is_local:
                result["local_tool_calls"].append(item.to_dict())
                result["supported_tool_calls"].append(item.to_dict())
            else:
                result["unsupported_tool_calls"].append(item.to_dict())
            if keep_non_molclaw:
                result["tool_calls"].append(item.to_dict())

    if not keep_non_molclaw:
        result["tool_calls"] = result["molclaw_tool_calls"]
    else:
        result["tool_calls"] = [item.to_dict() for item in tool_calls]

    result["tool_call_count"] = len(result["tool_calls"])
    result["molclaw_tool_call_count"] = len(result["molclaw_tool_calls"])
    result["local_tool_call_count"] = len(result["local_tool_calls"])
    result["supported_tool_call_count"] = len(result["molclaw_tool_calls"]) + len(result["local_tool_calls"])
    result["unsupported_tool_call_count"] = len(result["unsupported_tool_calls"])
    result["non_molclaw_tool_call_count"] = len(result["non_molclaw_tool_calls"])
    result["has_tool_call"] = result["tool_call_count"] > 0
    final_blocks = [
        block for block in result["blocks"]
        if isinstance(block, dict) and block.get("kind") == "final_answer"
    ]
    result["has_final_answer"] = bool(final_blocks)
    result["final_answer"] = final_blocks[-1].get("payload") if final_blocks else None
    return result
