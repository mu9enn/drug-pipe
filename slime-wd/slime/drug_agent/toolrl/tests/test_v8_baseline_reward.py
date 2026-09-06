from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from drug_agent.toolrl.molclaw_reward import reward_func


TOOLS = [
    {
        "type": "function",
        "function": {"name": "tool_a", "description": "", "parameters": {"type": "object"}},
    }
]


def score(parsed: dict, label: dict, *, mode: str = "v8_baseline") -> dict:
    sample = SimpleNamespace(
        prompt=[{"role": "user", "content": "task"}],
        response="native checkpoint output",
        label=label,
        metadata={"decision_role": "final", "tools": TOOLS},
    )
    with patch.dict(os.environ, {"TOOLRL_REWARD_MODE": mode}), patch(
        "drug_agent.toolrl.molclaw_reward.parse_qwen_native_completion",
        return_value=parsed,
    ):
        return asyncio.run(reward_func(None, sample))


class NativeRewardTest(unittest.TestCase):
    def test_native_final_answer_exact_match(self) -> None:
        parsed = {
            "ok": True,
            "has_final_answer": True,
            "has_tool_call": False,
            "final_answer": "answer",
            "tool_calls": [],
            "blocks": [{"kind": "thought", "body": "reason"}, {"kind": "final_answer", "payload": "answer"}],
        }
        out = score(
            parsed,
            {"decision_type": "final_answer", "target_final_answer": "answer"},
        )
        self.assertEqual(out["score"], 1.0)
        self.assertEqual(out["components"]["terminal_correctness"], 1.0)

    def test_native_final_answer_mismatch(self) -> None:
        parsed = {
            "ok": True,
            "has_final_answer": True,
            "has_tool_call": False,
            "final_answer": "wrong",
            "tool_calls": [],
            "blocks": [],
        }
        out = score(
            parsed,
            {"decision_type": "final_answer", "target_final_answer": "answer"},
        )
        self.assertEqual(out["score"], -0.5)

    def test_official_mode_scores_native_parallel_calls(self) -> None:
        calls = [
            {"name": "tool_a", "tool_name": "tool_a", "arguments": {"x": 1}},
            {"name": "tool_a", "tool_name": "tool_a", "arguments": {"x": 2}},
        ]
        parsed = {
            "ok": True,
            "has_final_answer": False,
            "has_tool_call": True,
            "final_answer": None,
            "tool_calls": calls,
            "blocks": [{"kind": "tool_call", "payload": item} for item in calls],
        }
        out = score(
            parsed,
            {"decision_type": "tool_call", "target_tool_calls": calls},
            mode="official",
        )
        self.assertEqual(out["score"], 4.0)
        self.assertEqual(out["diagnostics"]["pred_call_count"], 2)


if __name__ == "__main__":
    unittest.main()
