from __future__ import annotations

from typing import Any

from drug_agent.protocol.react_protocol import parse_react_sequence


PROFILE_NAME = "slime-raw"

SYSTEM_PROMPT = (
    "You are an agent in a strict tool-use evaluation. On every assistant turn, output exactly one "
    "canonical ReAct XML decision: an optional <thought>...</thought> followed by either one "
    "<tool_call>...</tool_call> container or one <final_answer>...</final_answer> container. "
    "A tool_call container contains one or more whitespace-separated JSON objects with exactly the "
    "fields tool_name and arguments. Do not emit observations; the environment executes valid tool "
    "calls and returns <observation> blocks. Do not put content outside the supported XML containers, "
    "do not use Markdown fences, and do not mix tool calls with a final answer."
)


def task_messages(
    prompt: Any,
    *,
    tool_catalog: str,
    final_contract: str,
) -> list[dict[str, str]]:
    """Build the minimal model-neutral prompt for one fresh evaluation task."""

    if isinstance(prompt, str):
        user_content = prompt
    elif isinstance(prompt, list):
        user = next(
            (
                item
                for item in prompt
                if isinstance(item, dict)
                and item.get("role") == "user"
                and isinstance(item.get("content"), str)
            ),
            None,
        )
        if user is None:
            raise ValueError("slime-raw evaluation sample has no fresh user question")
        user_content = user["content"]
    else:
        raise TypeError("slime-raw prompt must be a string or a list of chat messages")

    if not user_content.strip():
        raise ValueError("slime-raw evaluation question is empty")

    system_content = f"{SYSTEM_PROMPT}\n\n{tool_catalog}\n{final_contract}"
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def render_prompt(state: Any, prompt: Any, *, tool_catalog: str, final_contract: str) -> str:
    """Apply the model's native chat template without model-specific switches."""

    messages = task_messages(prompt, tool_catalog=tool_catalog, final_contract=final_contract)
    return state.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _failure(text: Any, error_type: str | None, error_message: str | None) -> dict[str, Any]:
    return {
        "ok": False,
        "decision_type": None,
        "thoughts": [],
        "tool_calls": [],
        "final_answer": None,
        "error_type": error_type,
        "error_message": error_message,
        "raw_text": text,
    }


def parse_decision(text: str) -> dict[str, Any]:
    """Parse exactly the documented protocol, without transport or fence recovery."""

    parsed = parse_react_sequence(text, role=None)
    if not parsed.get("ok"):
        return _failure(text, parsed.get("error_type"), parsed.get("error_message"))
    if parsed.get("fence_wrappers_stripped"):
        return _failure(text, "ReactFormatError", "slime-raw does not accept Markdown-fenced payloads")

    blocks = parsed.get("blocks") if isinstance(parsed.get("blocks"), list) else []
    kinds = [block.get("kind") for block in blocks]
    thoughts = [str(block.get("body") or "") for block in blocks if block.get("kind") == "thought"]
    tool_blocks = [block for block in blocks if block.get("kind") == "tool_call"]
    final_blocks = [block for block in blocks if block.get("kind") == "final_answer"]

    if len(tool_blocks) > 1:
        return _failure(text, "ReactDecisionError", "slime-raw requires exactly one tool_call container")
    if tool_blocks and kinds not in (["tool_call"], ["thought", "tool_call"]):
        return _failure(
            text,
            "ReactDecisionError",
            "slime-raw tool decision must be an optional thought followed by one tool_call container",
        )
    if final_blocks and kinds not in (["final_answer"], ["thought", "final_answer"]):
        return _failure(
            text,
            "ReactDecisionError",
            "slime-raw final decision must be an optional thought followed by one final_answer",
        )
    if not tool_blocks and not final_blocks:
        return _failure(
            text,
            "ReactDecisionError",
            "slime-raw assistant generation must contain a tool_call or final_answer",
        )

    if tool_blocks:
        payloads = tool_blocks[0].get("payloads")
        if not isinstance(payloads, list):
            payload = tool_blocks[0].get("payload")
            payloads = [payload] if isinstance(payload, dict) else []
        tool_calls = [
            {
                "tool_name": str(payload.get("tool_name") or ""),
                "arguments": payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {},
                "raw_payload": payload,
            }
            for payload in payloads
            if isinstance(payload, dict)
        ]
        return {
            "ok": True,
            "decision_type": "tool_call",
            "thoughts": thoughts,
            "tool_calls": tool_calls,
            "final_answer": None,
            "error_type": None,
            "error_message": None,
            "raw_text": text,
        }

    return {
        "ok": True,
        "decision_type": "final_answer",
        "thoughts": thoughts,
        "tool_calls": [],
        "final_answer": final_blocks[0].get("payload"),
        "error_type": None,
        "error_message": None,
        "raw_text": text,
    }
