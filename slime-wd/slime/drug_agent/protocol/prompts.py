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


def build_system_prompt() -> str:
    return DEFAULT_SYSTEM_PROMPT


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
