from __future__ import annotations

from types import SimpleNamespace

import pytest

from drug_agent.toolrl.metrics import augment_rollout_metrics


class _Sample:
    def __init__(
        self,
        *,
        role: str,
        group: int,
        value: float,
        components: dict,
        status: str = "completed",
        reward_fields: dict | None = None,
    ):
        self.metadata = {"decision_role": role}
        self.group_index = group
        self.reward = {"score": value, "components": components, **(reward_fields or {})}
        self.status = SimpleNamespace(value=status)

    def get_reward_value(self, args):
        return self.reward["score"]


def test_decision_role_metrics_include_dispersion_accuracy_and_valid_groups():
    samples = [
        _Sample(
            role="tool_step",
            group=0,
            value=-0.5,
            components={"tool_name_f1": 0.0, "param_name": 0.0, "param_value": 0.0},
            status="truncated",
            reward_fields={"format": 0.0},
        ),
        _Sample(
            role="tool_step",
            group=0,
            value=1.0,
            components={"tool_name_f1": 1.0, "param_name": 1.0, "param_value": 1.0},
            reward_fields={"format": 1.0},
        ),
        _Sample(
            role="final",
            group=1,
            value=1.0,
            components={"format": 1.0, "terminal_correctness": 1.0},
        ),
    ]
    metrics = {}
    assert augment_rollout_metrics(0, SimpleNamespace(), samples, metrics, 1.0) is False
    assert metrics["decision_role/tool_step/reward_mean"] == pytest.approx(0.25)
    assert metrics["decision_role/tool_step/reward_std"] == pytest.approx(0.75)
    assert metrics["decision_role/tool_step/nonzero_std_group_ratio"] == 1.0
    assert metrics["decision_role/tool_step/truncated_ratio"] == 0.5
    assert metrics["decision_role/tool_step/format_mean"] == 0.5
    assert metrics["decision_role/final/terminal_correctness_mean"] == 1.0
