"""Explicit prompt strategies for ToolRL baseline and Drug-Pipe variants."""

from __future__ import annotations

import copy
import json
from typing import Any


OFFICIAL_CONTRACT = """You are a helpful assistant capable of using the available tools.

Available Tools
{tools}

Output Format
Always include exactly one <thought>...</thought>. Then emit either one
<tool_call> container containing one or more newline-separated JSON objects, or
one <final_answer>...</final_answer>. Do not use commas between tool-call objects
and do not use a JSON array."""


def render_tool_catalog(catalog: dict[str, Any]) -> str:
    tools = catalog.get("tools") if isinstance(catalog.get("tools"), list) else []
    rendered = []
    for index, tool in enumerate(tools, 1):
        if not isinstance(tool, dict):
            continue
        rendered.append(
            f"{index}. Name: {tool.get('name')}\n"
            f"Description: {tool.get('description') or ''}\n"
            f"Parameters: {json.dumps(tool.get('input_schema') or {}, ensure_ascii=False, sort_keys=True)}"
        )
    return "\n".join(rendered)


def apply_prompt_strategy(
    row: dict[str, Any], *, strategy: str, catalog: dict[str, Any] | None = None
) -> dict[str, Any]:
    if strategy not in {"official_catalog", "drug_pipe_skill_discovery"}:
        raise ValueError(f"unsupported prompt strategy: {strategy}")
    out = copy.deepcopy(row)
    metadata = out.get("metadata") if isinstance(out.get("metadata"), dict) else {}
    metadata["prompt_strategy"] = strategy
    out["metadata"] = metadata
    if strategy == "drug_pipe_skill_discovery":
        return out
    if catalog is None:
        raise ValueError("official_catalog strategy requires a tool catalog")
    prompt = out.get("prompt")
    if not isinstance(prompt, list):
        raise ValueError("row prompt must be a message list")
    contract = OFFICIAL_CONTRACT.format(tools=render_tool_catalog(catalog))
    if prompt and isinstance(prompt[0], dict) and prompt[0].get("role") == "system":
        prompt[0]["content"] = contract
    else:
        prompt.insert(0, {"role": "system", "content": contract})
    return out
