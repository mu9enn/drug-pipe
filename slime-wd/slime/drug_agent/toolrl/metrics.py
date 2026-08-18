"""Decision-role metrics injected into Slime's normal rollout logger."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _component(sample: Any, *keys: str) -> float:
    reward = sample.reward if isinstance(sample.reward, dict) else {}
    components = reward.get("components") if isinstance(reward.get("components"), dict) else {}
    for key in keys:
        value = components.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        # Shared reward fields such as ``format`` live at the top level in
        # both the official and hierarchical schemas.  Keep components as the
        # preferred source, but do not silently report those fields as zero.
        value = reward.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return 0.0


def _reward_stage(sample: Any) -> str:
    reward = sample.reward if isinstance(sample.reward, dict) else {}
    diagnostics = reward.get("diagnostics") if isinstance(reward.get("diagnostics"), dict) else {}
    return str(diagnostics.get("reward_stage") or "")


def _diagnostic(sample: Any, key: str) -> float:
    reward = sample.reward if isinstance(sample.reward, dict) else {}
    diagnostics = reward.get("diagnostics") if isinstance(reward.get("diagnostics"), dict) else {}
    value = diagnostics.get(key)
    return float(value) if isinstance(value, (bool, int, float)) else 0.0


def augment_rollout_metrics(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Mutate extra metrics, then return False so Slime keeps its default log path."""
    if not isinstance(rollout_extra_metrics, dict):
        return False
    by_role: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        by_role[str(metadata.get("decision_role") or "unknown")].append(sample)

    groups: dict[int, list[Any]] = defaultdict(list)
    for sample in samples:
        groups[int(sample.group_index)].append(sample)
    invalid_groups = [group for group in groups.values() if group and all(
        _reward_stage(sample) == "invalid_react_tool_envelope" for sample in group
    )]
    zero_variance_groups = []
    for group in groups.values():
        rewards = [float(sample.get_reward_value(args)) for sample in group]
        if rewards and all(abs(value - rewards[0]) <= 1e-12 for value in rewards[1:]):
            zero_variance_groups.append(group)
    rollout_extra_metrics["rollout/invalid_envelope_group_ratio"] = (
        len(invalid_groups) / len(groups) if groups else 0.0
    )
    rollout_extra_metrics["rollout/zero_variance_group_ratio"] = (
        len(zero_variance_groups) / len(groups) if groups else 0.0
    )

    for role, role_samples in sorted(by_role.items()):
        prefix = f"decision_role/{role}"
        rewards = [float(sample.get_reward_value(args)) for sample in role_samples]
        formats = [_component(sample, "format") for sample in role_samples]
        tool_names = [_component(sample, "tool_name_f1", "tool_name") for sample in role_samples]
        param_names = [_component(sample, "required_argument_coverage", "param_name") for sample in role_samples]
        param_values = [_component(sample, "critical_argument_exact", "param_value") for sample in role_samples]
        configurable = [_component(sample, "configurable_argument_validity") for sample in role_samples]
        terminal_correctness = [_component(sample, "terminal_correctness") for sample in role_samples]
        thought_present = [_diagnostic(sample, "thought_present") for sample in role_samples]
        thought_chars = [_diagnostic(sample, "thought_char_count") for sample in role_samples]
        call_counts = [_diagnostic(sample, "pred_call_count") for sample in role_samples]
        truncated = [
            float(str(getattr(getattr(sample, "status", None), "value", getattr(sample, "status", ""))).lower() == "truncated")
            for sample in role_samples
        ]
        groups: dict[int, list[float]] = defaultdict(list)
        for sample, reward in zip(role_samples, rewards, strict=True):
            groups[int(sample.group_index)].append(reward)
        nonzero_groups = sum(
            1 for values in groups.values() if values and any(abs(value - values[0]) > 1e-12 for value in values[1:])
        )
        rollout_extra_metrics.update(
            {
                f"{prefix}/count": len(role_samples),
                f"{prefix}/reward_mean": _mean(rewards),
                f"{prefix}/reward_std": _population_std(rewards),
                f"{prefix}/format_mean": _mean(formats),
                f"{prefix}/tool_name_mean": _mean(tool_names),
                f"{prefix}/param_name_mean": _mean(param_names),
                f"{prefix}/param_value_mean": _mean(param_values),
                f"{prefix}/configurable_argument_validity_mean": _mean(configurable),
                f"{prefix}/terminal_correctness_mean": _mean(terminal_correctness),
                f"{prefix}/thought_present_ratio": _mean(thought_present),
                f"{prefix}/thought_char_count_mean": _mean(thought_chars),
                f"{prefix}/multi_call_ratio": _mean([float(value > 1) for value in call_counts]),
                f"{prefix}/response_token_mean": _mean([float(getattr(sample, "response_length", 0) or 0) for sample in role_samples]),
                f"{prefix}/truncated_ratio": _mean(truncated),
                f"{prefix}/nonzero_std_group_ratio": nonzero_groups / len(groups) if groups else 0.0,
            }
        )
    return False
