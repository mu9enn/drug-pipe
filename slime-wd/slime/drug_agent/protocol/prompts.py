from __future__ import annotations

import json
from typing import Any

from drug_agent.constants import DEFAULT_SYSTEM_PROMPT
from drug_agent.utils import normalize_tool_name


REACT_FORMAT_DOC = (
    "Reasoning: <thought>...</thought>\n"
    "Tool decision: <tool_call>{\"tool_name\":\"...\",\"arguments\":{...}}</tool_call> "
    "(one or more calls are allowed)\n"
    "Terminal decision: <final_answer>{\"task_type\":\"...\", ...}</final_answer>\n"
    "Never mix tool_call and final_answer in the same assistant generation."
)

_FINAL_FIELDS = {
    "vs": {"task_type": "vs", "ranked_smiles": ["full candidate ranking"], "selected_smiles": "top-ranked SMILES", "evidence": []},
    "ac": {"task_type": "ac", "answer_smiles": "selected SMILES", "evidence": []},
    "pf": {"task_type": "pf", "selected_smiles": ["selected SMILES"], "evidence": []},
    "kg": {"task_type": "kg", "result": "task result", "evidence": []},
    "e2e": {"task_type": "e2e", "result": "task result", "evidence": []},
    "mol_edit": {"task_type": "mol_edit", "output_smiles": "modified SMILES", "evidence": []},
    "mol_opt": {"task_type": "mol_opt", "optimized_smiles": "optimized SMILES", "evidence": []},
    "mol_opt_physchem": {"task_type": "mol_opt_physchem", "optimized_smiles": "optimized SMILES", "evidence": []},
}


def build_system_prompt() -> str:
    return DEFAULT_SYSTEM_PROMPT


def fresh_task_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project any conversation-shaped input to the fresh system+question boundary."""
    system = next(
        (dict(item) for item in messages if isinstance(item, dict) and item.get("role") == "system"),
        None,
    )
    user = next(
        (dict(item) for item in messages if isinstance(item, dict) and item.get("role") == "user"),
        None,
    )
    return [item for item in (system, user) if item is not None]


def format_tool_catalog(tool_specs: list[dict[str, Any]]) -> str:
    """Render the live MCP/local schemas without inventing aliases."""
    rows = []
    for spec in sorted(tool_specs, key=lambda item: str(item.get("name") or "")):
        name = str(spec.get("name") or "").strip()
        if not name:
            continue
        rows.append(
            {
                "name": name,
                "description": str(spec.get("description") or "").strip(),
                "input_schema": spec.get("input_schema") if isinstance(spec.get("input_schema"), dict) else {},
            }
        )
    return "Available tools (authoritative names and schemas):\n" + json.dumps(
        rows, ensure_ascii=False, separators=(",", ":")
    )


def format_final_contract(task_type: str) -> str:
    payload = _FINAL_FIELDS.get(str(task_type or "").lower())
    if payload is None:
        raise ValueError(f"Unsupported online task type: {task_type}")
    return (
        "Required terminal payload for this task (replace placeholder values; summary is optional):\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def build_user_prompt_payload(
    *,
    task_id: str,
    task_type: str,
    instruction: str,
    inputs: dict[str, Any],
    allowed_tools: list[str],
    max_steps: int,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "task_type": task_type,
        "instruction": instruction,
        "inputs": inputs,
        "allowed_tools": [normalize_tool_name(x) for x in allowed_tools if isinstance(x, str) and x.strip()],
        "max_steps": max_steps,
        "required_action_format": REACT_FORMAT_DOC,
        "output_constraints": {
            "no_markdown_code_fence": True,
            "canonical_react_xml": True,
            "enable_thinking": False,
        },
    }


def build_user_prompt_text(
    *,
    task_id: str,
    task_type: str,
    instruction: str,
    inputs: dict[str, Any],
    allowed_tools: list[str],
    max_steps: int,
) -> str:
    payload = build_user_prompt_payload(
        task_id=task_id,
        task_type=task_type,
        instruction=instruction,
        inputs=inputs,
        allowed_tools=allowed_tools,
        max_steps=max_steps,
    )
    return "/no_think\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_grpo_prompt_messages(
    *,
    task_id: str,
    task_type: str,
    instruction: str,
    inputs: dict[str, Any],
    allowed_tools: list[str],
    max_steps: int,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": build_user_prompt_text(
                task_id=task_id,
                task_type=task_type,
                instruction=instruction,
                inputs=inputs,
                allowed_tools=allowed_tools,
                max_steps=max_steps,
            ),
        },
    ]
