from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx
import pytest

from drug_agent.rollout.context_budget import ContextBudgetError, bounded_step_limit, fit_context
from slime.utils.http_utils import _post


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": list(text.encode("utf-8"))}


def _turn(step, *, assistant=10, observation=40, compacted=5):
    return {
        "step": step,
        "assistant_ids": [10 + step] * assistant,
        "observation_ids": [20 + step] * observation,
        "compacted_observation_ids": [30 + step] * compacted,
        "event": {
            "step": step,
            "decision_type": "tool_call",
            "tool_calls": [{"tool_name": f"tool_{step}", "arguments": {"id": step}}],
            "observations": [{"tool_name": f"tool_{step}", "status": "success"}],
        },
    }


def test_step_limit_uses_strictest_active_positive_cap():
    assert bounded_step_limit(0, 128) == 128
    assert bounded_step_limit(64, 128) == 64
    assert bounded_step_limit(0, 0) == 0
    with pytest.raises(ValueError):
        bounded_step_limit(-1, 128)


def test_live_context_microcompacts_old_observations_and_preserves_recent_turn():
    ids, audit = fit_context(
        FakeTokenizer(),
        prefix_ids=[1] * 20,
        turns=[_turn(0), _turn(1), _turn(2)],
        max_prompt_tokens=115,
        keep_recent_turns=1,
    )
    assert len(ids) <= 115
    assert audit["strategy"] == "typed_observation_compaction"
    assert audit["microcompacted_turns"] == 2
    assert ids[-40:] == [22] * 40


def test_live_context_summarizes_complete_older_turns_without_cutting_prefix():
    ids, audit = fit_context(
        FakeTokenizer(),
        prefix_ids=[1] * 20,
        turns=[_turn(index, assistant=80, observation=80, compacted=40) for index in range(4)],
        max_prompt_tokens=420,
        summary_max_tokens=220,
        keep_recent_turns=1,
    )
    assert len(ids) <= 420
    assert ids[:20] == [1] * 20
    assert audit["strategy"] == "typed_observation_then_structured_summary"
    assert audit["summarized_turns"] >= 1


def test_live_context_fails_closed_when_immutable_prefix_is_oversized():
    with pytest.raises(ContextBudgetError):
        fit_context(FakeTokenizer(), prefix_ids=[1] * 101, turns=[], max_prompt_tokens=100)


class FakeClient:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0

    async def post(self, url, json=None, headers=None):
        del json, headers
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        request = httpx.Request("POST", url)
        return httpx.Response(status, request=request, json={"ok": status < 400})


def test_http_400_is_not_retried():
    client = FakeClient([400])
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_post(client, "http://localhost/generate", {}, max_retries=60))
    assert client.calls == 1


def test_http_503_is_retryable():
    client = FakeClient([503, 200])

    async def no_sleep(_seconds):
        return None

    with patch("slime.utils.http_utils.asyncio.sleep", no_sleep):
        result = asyncio.run(_post(client, "http://localhost/generate", {}, max_retries=2))
    assert result == {"ok": True}
    assert client.calls == 2
