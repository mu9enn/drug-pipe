from __future__ import annotations

import json
from typing import Any

from drug_agent.constants import DEFAULT_SYSTEM_PROMPT
from drug_agent.protocol.action_schema import ACTION_FORMAT_DOC
from drug_agent.utils import normalize_tool_name


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
        "required_action_format": ACTION_FORMAT_DOC,
        "output_constraints": {
            "json_only": True,
            "no_markdown_code_fence": True,
            "no_xml": True,
            "single_json_object": True,
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
