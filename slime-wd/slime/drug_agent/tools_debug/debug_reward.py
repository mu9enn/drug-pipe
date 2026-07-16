from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from slime.utils.types import Sample

from drug_agent.rollout.reward_func import reward_func


class _Args:
    reward_key = "score"


async def _run(sample: Sample):
    return await reward_func(_Args(), sample)


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug reward function using trajectory row")
    parser.add_argument("--trajectory-jsonl", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.trajectory_jsonl)
    if not path.exists():
        raise FileNotFoundError(path)

    row = None
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == args.index:
                row = json.loads(line)
                break

    if row is None:
        raise IndexError(f"index {args.index} out of range")

    sample = Sample(
        prompt=row.get("prompt", ""),
        response="",
        label=row.get("label"),
        metadata={
            "drug_agent_trace": {
                "task_id": row.get("task_id"),
                "task_type": row.get("task_type"),
                "data_source": row.get("data_source"),
                "actions": row.get("actions", []),
                "observations": row.get("observations", []),
                "final_answer": row.get("final_answer"),
                "done_reason": row.get("done_reason"),
                "num_steps": row.get("num_steps", 0),
                "num_invalid": row.get("num_invalid", 0),
                "num_tool_success": row.get("num_tool_success", 0),
                "num_tool_error": row.get("num_tool_error", 0),
                "truncated": row.get("truncated", False),
            },
            "env_kwargs": {
                "allowed_tools": row.get("allowed_tools", []),
                "max_steps": row.get("num_steps", 0),
            },
        },
    )

    result = asyncio.run(_run(sample))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
