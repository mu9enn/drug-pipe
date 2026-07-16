from __future__ import annotations

import os
from pathlib import Path

from drug_agent.utils import append_jsonl, to_jsonable


async def zero_reward(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        return [0.0 for _ in sample_or_samples]
    return 0.0


def log_negative_cache(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    output = Path(os.environ["GAD_NEGATIVE_CACHE"])
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        label = sample.label if isinstance(sample.label, dict) else {}
        state_messages = metadata.get("state_messages")
        if not isinstance(state_messages, list):
            state_messages = sample.prompt if isinstance(sample.prompt, list) else []
        append_jsonl(
            output,
            {
                "sample_id": metadata.get("sample_id"),
                "state_messages": to_jsonable(state_messages),
                "rendered_prompt": sample.prompt if isinstance(sample.prompt, str) else None,
                "teacher_response": label.get("teacher_response") or metadata.get("teacher_response"),
                "student_response": sample.response,
                "student_weight_versions": to_jsonable(sample.weight_versions),
                "rollout_id": rollout_id,
                "metadata": to_jsonable(metadata),
            },
        )
    if rollout_extra_metrics is not None:
        rollout_extra_metrics["gad/negative_cache_rows"] = len(samples)
        rollout_extra_metrics["gad/negative_cache_path"] = str(output)
    return False
