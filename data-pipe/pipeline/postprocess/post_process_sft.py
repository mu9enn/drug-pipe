#!/usr/bin/env python3
"""Compatibility entry point for the single canonical ReAct interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report the canonical ReAct dataset; no second cleaning or RL-prompt export is performed."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--summary-csv", default="")
    parser.add_argument("--answer-hit-only", action="store_true")
    parser.add_argument("--tool-role-mode", choices=["user_observation", "tool"], default="user_observation")
    parser.add_argument("--split-multi-tool-calls", action="store_true")
    parser.add_argument("--max-observation-chars", type=int, default=6000)
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    canonical = input_root / "react_trajectories.jsonl"
    if not canonical.is_file():
        raise FileNotFoundError(
            f"canonical dataset not found: {canonical}; run scripts/run_postprocess.sh without --skip-export"
        )
    count = sum(1 for line in canonical.open("r", encoding="utf-8") if line.strip())
    print(
        json.dumps(
            {
                "compatibility_entrypoint": "post_process_sft",
                "canonical_react": str(canonical),
                "count": count,
                "semantic_rewrites": 0,
                "rl_prompt_exported": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
