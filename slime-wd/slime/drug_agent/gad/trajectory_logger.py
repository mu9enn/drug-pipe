from __future__ import annotations

import os
from pathlib import Path

from drug_agent.utils import append_jsonl, to_jsonable


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    output = Path(os.environ.get("GAD_TRAJECTORY_LOG", str(Path(args.save) / "gad_trajectories.jsonl")))
    for sample in samples:
        reward = sample.reward if isinstance(sample.reward, dict) else {}
        append_jsonl(
            output,
            {
                "rollout_id": rollout_id,
                "sample_id": (sample.metadata or {}).get("sample_id"),
                "prompt": to_jsonable(sample.prompt),
                "state_messages": to_jsonable((sample.metadata or {}).get("state_messages")),
                "teacher_response": (sample.label or {}).get("teacher_response") if isinstance(sample.label, dict) else None,
                "student_response": sample.response,
                "student_weight_versions": to_jsonable(sample.weight_versions),
                "reward": to_jsonable(reward),
                "gad_reward": to_jsonable((sample.metadata or {}).get("gad_reward")),
            },
        )
    if rollout_extra_metrics is not None:
        rollout_extra_metrics["gad/trajectory_rows"] = len(samples)
        rollout_extra_metrics["gad/trajectory_path"] = str(output)
    return False
