"""Online learnability selection and audit for Slime GRPO rollouts."""

from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput


def _rewards(args: Any, samples: list[Any]) -> list[float]:
    return [float(sample.get_reward_value(args)) for sample in samples]


def policy_boundary_filter(args: Any, samples: list[Any], **kwargs: Any) -> DynamicFilterOutput:
    rewards = _rewards(args, samples)
    keep = len(rewards) > 1 and torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    if keep:
        reason = None
    elif rewards and all(value >= 0.999 for value in rewards):
        reason = "mastered_all_correct"
    elif rewards and all(value <= -0.399 for value in rewards):
        reason = "too_hard_all_wrong"
    else:
        reason = "zero_reward_variance"
    return DynamicFilterOutput(keep=keep, reason=reason)


def audit_all_groups(args: Any, all_groups: list[list[Any]], data_source: Any, **kwargs: Any) -> None:
    """Append one compact record per attempted group, including filtered ones."""
    output = os.environ.get("TOOLRL_LEARNABILITY_LOG", "").strip()
    if not output:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for group in all_groups:
            flat = group[0] if group and isinstance(group[0], list) else group
            if not flat:
                continue
            rewards = _rewards(args, flat)
            sample = flat[0]
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            row = {
                "source_id": metadata.get("source_id") or metadata.get("task_id"),
                "assistant_index": metadata.get("assistant_index"),
                "decision_role": metadata.get("decision_role"),
                "is_initial_step": bool(metadata.get("is_initial_step")),
                "task_type": metadata.get("task_type"),
                "tool_names": metadata.get("tool_names") or [],
                "rewards": rewards,
                "reward_mean": statistics.fmean(rewards),
                "reward_std": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
                "policy_boundary": len(rewards) > 1 and statistics.stdev(rewards) > 1e-6,
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
