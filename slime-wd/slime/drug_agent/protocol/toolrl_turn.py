"""Canonical serialization for one ToolRL reasoning/action turn."""

from __future__ import annotations

import copy
import json
from typing import Any

from drug_agent.protocol.react_protocol import parse_react_sequence, parse_runtime_decision


TOOLRL_TURN_PROTOCOL = "toolrl_turn_v1"
SFT_SCHEMA = "drug_agent_sft_toolrl_turn_v1"
SYSTEM_CONTRACT = (
    "Within each reasoning/action segment, put all tool calls inside exactly one "
    "<tool_call>...</tool_call> container as newline-separated JSON objects without commas or a JSON array."
)


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def serialize_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    if not tool_calls:
        raise ValueError("tool decision must contain at least one tool call")
    lines = []
    for call in tool_calls:
        if not isinstance(call, dict):
            raise ValueError("tool call must be an object")
        name = str(call.get("tool_name") or call.get("name") or "").strip()
        arguments = call.get("arguments")
        if arguments is None:
            arguments = call.get("input")
        if not name or not isinstance(arguments, dict):
            raise ValueError("tool call requires a non-empty tool_name and object arguments")
        lines.append(_compact_json({"tool_name": name, "arguments": arguments}))
    return "<tool_call>\n" + "\n".join(lines) + "\n</tool_call>"


def serialize_decision(
    *,
    thoughts: list[str] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    final_answer: Any = None,
) -> str:
    normalized_thoughts = [str(item).strip() for item in (thoughts or []) if str(item).strip()]
    blocks = []
    if normalized_thoughts:
        blocks.append("<thought>" + "\n\n".join(normalized_thoughts) + "</thought>")
    if tool_calls:
        if final_answer is not None:
            raise ValueError("tool calls and final answer cannot coexist")
        blocks.append(serialize_tool_calls(tool_calls))
    elif final_answer is not None:
        blocks.append("<final_answer>" + _compact_json(final_answer) + "</final_answer>")
    else:
        raise ValueError("decision must contain tool calls or a final answer")
    return "\n".join(blocks)


def split_assistant_segments(content: str) -> list[dict[str, Any]]:
    """Split at a new thought only after the preceding segment has an action."""
    parsed = parse_react_sequence(content, role="assistant")
    if not parsed.get("ok"):
        raise ValueError(str(parsed.get("error_message") or "invalid assistant content"))
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] = {"thoughts": [], "tool_calls": [], "final_answer": None}

    def flush() -> None:
        nonlocal current
        if current["thoughts"] or current["tool_calls"] or current["final_answer"] is not None:
            segments.append(current)
        current = {"thoughts": [], "tool_calls": [], "final_answer": None}

    for block in parsed.get("blocks") or []:
        kind = block.get("kind")
        if kind == "thought":
            if current["tool_calls"] or current["final_answer"] is not None:
                flush()
            text = str(block.get("body") or "").strip()
            if text:
                current["thoughts"].append(text)
        elif kind == "tool_call":
            payloads = block.get("payloads")
            if not isinstance(payloads, list):
                payload = block.get("payload")
                payloads = [payload] if isinstance(payload, dict) else []
            current["tool_calls"].extend(payload for payload in payloads if isinstance(payload, dict))
        elif kind == "final_answer":
            if current["tool_calls"]:
                raise ValueError("tool calls and final answer cannot coexist in one reasoning segment")
            current["final_answer"] = block.get("payload")
        else:
            raise ValueError(f"unsupported assistant block in turn source: {kind}")
    flush()
    if not segments:
        raise ValueError("assistant content has no supported segment")
    for index, segment in enumerate(segments):
        segment["segment_index"] = index
        segment["is_action"] = bool(segment["tool_calls"] or segment["final_answer"] is not None)
        segment["content"] = (
            serialize_decision(
                thoughts=segment["thoughts"],
                tool_calls=segment["tool_calls"],
                final_answer=segment["final_answer"],
            )
            if segment["is_action"]
            else "<thought>" + "\n\n".join(segment["thoughts"]) + "</thought>"
        )
    return segments


def normalize_assistant_content(content: str) -> tuple[str, dict[str, Any]]:
    segments = split_assistant_segments(content)
    action_segments = [segment for segment in segments if segment["is_action"]]
    if len(action_segments) != 1 or len(segments) != 1:
        raise ValueError(f"assistant content contains {len(segments)} reasoning/action segments")
    source = action_segments[0]
    normalized = source["content"]
    strict = parse_runtime_decision(normalized, strict_toolrl_turn=True)
    if not strict.get("ok"):
        raise AssertionError(f"canonical serializer emitted invalid content: {strict.get('error_message')}")
    strict["source_thought_count"] = len(source["thoughts"])
    return normalized, strict


def normalize_assistant_message(content: str) -> tuple[str, dict[str, Any], bool]:
    """Normalize every ordered reasoning/action segment without reordering."""
    segments = split_assistant_segments(content)
    action_segments = [segment for segment in segments if segment["is_action"]]
    normalized = "\n".join(segment["content"] for segment in segments)
    summary = {
        "segments": segments,
        "action_segment_count": len(action_segments),
        "multi_segment": len(segments) > 1,
        "thought_only_incomplete_tail_count": sum(
            not segment["is_action"] for segment in segments
        ),
        "source_thought_count": sum(len(segment["thoughts"]) for segment in segments),
        "tool_calls": [call for segment in action_segments for call in segment["tool_calls"]],
    }
    return normalized, summary, bool(action_segments)


def normalize_trajectory(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("trajectory must contain messages")
    out = copy.deepcopy(record)
    out["schema_version"] = SFT_SCHEMA
    out_messages = out["messages"]
    assistant_message_count = assistant_turns = actionless_assistant_messages = 0
    multi_call_turns = thoughtless_turns = merged_thought_blocks = 0
    expanded_decisions = multi_action_segment_turns = interleaved_segment_turns = 0
    thought_only_incomplete_tails = 0
    tool_decisions = final_decisions = 0
    for index, message in enumerate(out_messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} is not an object")
        if message.get("role") == "system":
            content = str(message.get("content") or "").rstrip()
            if SYSTEM_CONTRACT not in content:
                message["content"] = content + "\n" + SYSTEM_CONTRACT
        if message.get("role") != "assistant":
            continue
        assistant_message_count += 1
        normalized, parsed, is_action = normalize_assistant_message(str(message.get("content") or ""))
        message["content"] = normalized
        if not is_action:
            actionless_assistant_messages += 1
            # A thought without a following action is not a valid v6 SFT/RL
            # target.  Preserve it as causal history, but never supervise it.
            message["step_loss_mask"] = 0
            thought_only_incomplete_tails += int(parsed["thought_only_incomplete_tail_count"])
            merged_thought_blocks += max(0, int(parsed.get("source_thought_count") or 0) - 1)
            continue
        action_segments = [segment for segment in parsed["segments"] if segment["is_action"]]
        # The base trajectory retains this message as exact downstream
        # context, but action supervision is supplied by prefix-masked SFT
        # segment records so the model learns one container per target.
        if len(parsed["segments"]) > 1:
            message["step_loss_mask"] = 0
            interleaved_segment_turns += 1
        if len(action_segments) > 1:
            multi_action_segment_turns += 1
        assistant_turns += 1
        expanded_decisions += len(action_segments)
        thought_only_incomplete_tails += int(parsed["thought_only_incomplete_tail_count"])
        tool_decisions += sum(bool(segment["tool_calls"]) for segment in action_segments)
        final_decisions += sum(segment["final_answer"] is not None for segment in action_segments)
        multi_call_turns += sum(len(segment["tool_calls"]) > 1 for segment in action_segments)
        thought_count = int(parsed.get("source_thought_count") or 0)
        thoughtless_turns += sum(not segment["thoughts"] for segment in action_segments)
        merged_thought_blocks += max(0, thought_count - 1)
    audit = {
        "protocol": TOOLRL_TURN_PROTOCOL,
        "assistant_message_count": assistant_message_count,
        "assistant_turns": assistant_turns,
        "expanded_decisions": expanded_decisions,
        # `multi_segment_turns` means a turn that expands to more than one
        # trainable decision.  Keep interleaving (which may end in an
        # incomplete thought-only tail) as a separate count.
        "multi_segment_turns": multi_action_segment_turns,
        "multi_action_segment_turns": multi_action_segment_turns,
        "interleaved_segment_turns": interleaved_segment_turns,
        "thought_only_incomplete_tails": thought_only_incomplete_tails,
        "tool_decisions": tool_decisions,
        "final_decisions": final_decisions,
        "actionless_assistant_messages": actionless_assistant_messages,
        "multi_call_turns": multi_call_turns,
        "thoughtless_turns": thoughtless_turns,
        "thoughtless_actions": thoughtless_turns,
        "merged_thought_blocks": merged_thought_blocks,
        "causal_interleaving_quarantines": 0,
    }
    return out, audit
