#!/usr/bin/env python3
"""Audit ordered reasoning/action segmentation without inventing observations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.protocol.react_protocol import parse_runtime_decision
from drug_agent.protocol.toolrl_turn import split_assistant_segments


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def audit(source_path: Path, normalized_path: Path) -> dict[str, Any]:
    source = _records(source_path)
    normalized = _records(normalized_path)
    if len(source) != len(normalized):
        raise ValueError(f"trajectory count changed: {len(source)} != {len(normalized)}")
    assistant_messages = action_turns = expanded = multi_action_segment = 0
    interleaved_turns = 0
    thoughtless = incomplete_tails = tool_decisions = final_decisions = 0
    multi_call_decisions = observations = 0
    examples: list[dict[str, Any]] = []
    for row_index, (before, after) in enumerate(zip(source, normalized, strict=True)):
        if before.get("id") != after.get("id"):
            raise ValueError(f"source id changed at row {row_index}")
        before_messages = before.get("messages")
        after_messages = after.get("messages")
        if not isinstance(before_messages, list) or not isinstance(after_messages, list):
            raise ValueError(f"missing messages at row {row_index}")
        if len(before_messages) != len(after_messages):
            raise ValueError(f"message boundary count changed for {before.get('id')}")
        for message_index, (left, right) in enumerate(zip(before_messages, after_messages, strict=True)):
            if left.get("role") != right.get("role"):
                raise ValueError(f"role changed for {before.get('id')}:{message_index}")
            role = left.get("role")
            if role == "user":
                if left != right:
                    raise ValueError(f"user task/observation changed for {before.get('id')}:{message_index}")
                observations += "<observation" in str(left.get("content") or "")
                continue
            if role != "assistant":
                continue
            assistant_messages += 1
            source_decision = parse_runtime_decision(str(left.get("content") or ""))
            segments = split_assistant_segments(str(right.get("content") or ""))
            actions = [segment for segment in segments if segment["is_action"]]
            interleaved_turns += len(segments) > 1
            multi_action_segment += len(actions) > 1
            incomplete_tails += sum(not segment["is_action"] for segment in segments)
            if not actions:
                continue
            action_turns += 1
            expanded += len(actions)
            thoughtless += sum(not segment["thoughts"] for segment in actions)
            tool_decisions += sum(bool(segment["tool_calls"]) for segment in actions)
            final_decisions += sum(segment["final_answer"] is not None for segment in actions)
            multi_call_decisions += sum(len(segment["tool_calls"]) > 1 for segment in actions)
            source_calls = [call.get("raw_payload") for call in source_decision.get("tool_calls") or []]
            segmented_calls = [call for segment in actions for call in segment["tool_calls"]]
            if source_calls != segmented_calls:
                raise ValueError(f"tool order/payload changed for {before.get('id')}:{message_index}")
            prefix_parts: list[str] = []
            rendered_segments = []
            action_index = 0
            for segment in segments:
                if segment["is_action"]:
                    prefix = "\n".join(prefix_parts)
                    if prefix:
                        prefix += "\n"
                    parsed = parse_runtime_decision(segment["content"], strict_toolrl_turn=True)
                    if not parsed.get("ok"):
                        raise ValueError(f"invalid segment {before.get('id')}:{message_index}:{action_index}")
                    rendered_segments.append(
                        {
                            "assistant_subturn_index": action_index,
                            "state_assistant_prefix": prefix,
                            "action": segment["content"],
                        }
                    )
                    action_index += 1
                prefix_parts.append(segment["content"])
            if len(segments) > 1:
                examples.append(
                    {
                        "source_id": before.get("id"),
                        "assistant_index": message_index,
                        "source_content": left.get("content"),
                        "segments": rendered_segments,
                        "thought_only_incomplete_tail": [
                            segment["content"] for segment in segments if not segment["is_action"]
                        ],
                    }
                )
    return {
        "schema_version": "toolrl_reasoning_action_segmentation_audit_v2",
        "ok": True,
        "source_path": str(source_path.resolve()),
        "normalized_path": str(normalized_path.resolve()),
        "trajectories": len(source),
        "assistant_turn_count": assistant_messages,
        "assistant_turns_with_action": action_turns,
        "expanded_decision_count": expanded,
        "multi_segment_turn_count": multi_action_segment,
        "multi_action_segment_turn_count": multi_action_segment,
        "interleaved_assistant_turn_count": interleaved_turns,
        "thoughtless_action_count": thoughtless,
        "thought_only_incomplete_tail_count": incomplete_tails,
        "tool_decision_count": tool_decisions,
        "final_decision_count": final_decisions,
        "multi_tool_decision_count": multi_call_decisions,
        "observation_messages_preserved": observations,
        "tool_call_order_preserved": True,
        "no_cross_observation_merge": True,
        "multi_segment_examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--normalized", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.source, args.normalized)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
