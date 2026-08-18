#!/usr/bin/env python3
"""Write deterministic real before/after examples for a ToolRL v6 release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.protocol.toolrl_turn import normalize_trajectory, split_assistant_segments


def build(source: Path) -> dict:
    singles: list[dict] = []
    multis: list[dict] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            normalized, _ = normalize_trajectory(record)
            for assistant_index, (before, after) in enumerate(
                zip(record.get("messages") or [], normalized.get("messages") or [], strict=True)
            ):
                if before.get("role") != "assistant":
                    continue
                for subturn_index, segment in enumerate(
                    item for item in split_assistant_segments(str(after.get("content") or ""))
                    if item["is_action"]
                ):
                    calls = segment["tool_calls"]
                    if not calls:
                        continue
                    item = {
                        "source_id": record.get("id"),
                        "assistant_index": assistant_index,
                        "assistant_subturn_index": subturn_index,
                        "call_count": len(calls),
                        "tool_names": [call.get("tool_name") for call in calls],
                        "before_assistant_message": before.get("content"),
                        "after_action_segment": segment["content"],
                    }
                    target = multis if len(calls) > 1 else singles
                    limit = 2 if target is multis else 3
                    if len(target) < limit:
                        target.append(item)
            if len(singles) >= 3 and len(multis) >= 2:
                break
    if len(singles) != 3 or len(multis) != 2:
        raise ValueError(f"could not collect 3 single + 2 multi examples: {len(singles)}, {len(multis)}")
    return {"schema_version": "toolrl_turn_serializer_examples_v1", "examples": singles + multis}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "examples": 5}, ensure_ascii=False))


if __name__ == "__main__":
    main()
