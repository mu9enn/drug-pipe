#!/usr/bin/env python3
"""Prove that every ToolRL gold action is an identical SFT supervised target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from drug_agent.protocol.toolrl_turn import split_assistant_segments


def _key(source_id: str, assistant_index: int, subturn_index: int) -> str:
    return f"{source_id}:{assistant_index}:{subturn_index}"


def audit(sft_path: Path, rl_path: Path) -> dict[str, Any]:
    sft_targets: dict[str, str] = {}
    supplemental = 0
    with sft_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            segment = row.get("sft_segment_target")
            if isinstance(segment, dict):
                key = _key(
                    str(segment["source_id"]),
                    int(segment["assistant_index"]),
                    int(segment["assistant_subturn_index"]),
                )
                message = row["messages"][-1]
                loss_start = int(message.get("loss_char_start") or 0)
                target = str(message.get("content") or "")[loss_start:]
                if target != segment["target_action"]:
                    raise ValueError(f"SFT loss span mismatch for {key}")
                sft_targets[key] = target
                supplemental += 1
                continue
            source_id = str(row.get("source_id") or row.get("id") or "")
            for assistant_index, message in enumerate(row.get("messages") or []):
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                if message.get("step_loss_mask", 1) != 1:
                    continue
                segments = split_assistant_segments(str(message.get("content") or ""))
                actions = [item for item in segments if item["is_action"]]
                if len(actions) != 1 or len(segments) != 1:
                    raise ValueError(f"non-canonical supervised SFT target {source_id}:{assistant_index}")
                sft_targets[_key(source_id, assistant_index, 0)] = actions[0]["content"]

    compared = multi_tool = 0
    missing: list[str] = []
    with rl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            label = row.get("label") or {}
            key = _key(
                str(metadata.get("source_id") or ""),
                int(metadata.get("assistant_index") or 0),
                int(metadata.get("assistant_subturn_index") or 0),
            )
            target = str(label.get("assistant_content") or "")
            if key not in sft_targets:
                missing.append(key)
                continue
            if sft_targets[key] != target:
                raise ValueError(f"SFT/ToolRL serializer mismatch for {key}")
            parsed = split_assistant_segments(target)[0]
            multi_tool += len(parsed["tool_calls"]) > 1
            compared += 1
    if missing:
        raise ValueError(f"ToolRL decisions missing SFT target: {missing[:5]}")
    return {
        "schema_version": "sft_toolrl_serializer_parity_v1",
        "ok": True,
        "sft_supervised_action_targets": len(sft_targets),
        "supplemental_prefix_conditioned_targets": supplemental,
        "toolrl_gold_actions_compared": compared,
        "multi_tool_actions_compared": multi_tool,
        "byte_exact_matches": compared,
        "mismatches": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", required=True, type=Path)
    parser.add_argument("--toolrl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.sft, args.toolrl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
