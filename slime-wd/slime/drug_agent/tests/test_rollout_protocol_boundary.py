from __future__ import annotations

from drug_agent.rollout.generate_with_drug_agent import _with_forbidden_observation_stop


def test_observation_stop_is_added_without_losing_existing_stops():
    params = _with_forbidden_observation_stop({"stop": ["</final_answer>"], "temperature": 0})
    assert params["stop"] == ["</final_answer>", "<observation"]
    assert params["no_stop_trim"] is True
