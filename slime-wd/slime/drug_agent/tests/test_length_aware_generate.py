from __future__ import annotations

import asyncio
import sys
import types
import unittest
from unittest.mock import patch

from drug_agent.rollout.length_aware_generate import generate, resolve_response_cap


class LengthAwareGenerateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = types.SimpleNamespace(
            rollout_long_response_len=16384,
            rollout_long_task_types=["vs", "pf"],
        )
        self.params = {"max_new_tokens": 8192}

    @staticmethod
    def sample(metadata: dict, label: dict | None = None):
        return types.SimpleNamespace(metadata=metadata, label=label)

    def test_ordinary_decision_uses_default_cap(self) -> None:
        cap, tier = resolve_response_cap(
            self.args,
            self.sample({"task_type": "ac", "decision_type": "tool_call"}),
            self.params,
        )
        self.assertEqual((cap, tier), (8192, "default"))

    def test_declared_long_terminal_decision_uses_long_cap(self) -> None:
        for task_type in ("vs", "pf"):
            with self.subTest(task_type=task_type):
                cap, tier = resolve_response_cap(
                    self.args,
                    self.sample({"task_type": task_type, "decision_type": "final_answer"}),
                    self.params,
                )
                self.assertEqual((cap, tier), (16384, "long"))

    def test_long_task_tool_call_does_not_receive_long_cap(self) -> None:
        cap, tier = resolve_response_cap(
            self.args,
            self.sample({"task_type": "vs", "decision_type": "tool_call"}),
            self.params,
        )
        self.assertEqual((cap, tier), (8192, "default"))

    def test_explicit_metadata_override_is_bounded_and_auditable(self) -> None:
        cap, tier = resolve_response_cap(
            self.args,
            self.sample({"rollout_max_response_len": 12288}),
            self.params,
        )
        self.assertEqual((cap, tier), (12288, "metadata"))
        with self.assertRaisesRegex(ValueError, "must be in"):
            resolve_response_cap(
                self.args,
                self.sample({"rollout_max_response_len": 32768}),
                self.params,
            )

    def test_gold_response_length_is_not_consulted(self) -> None:
        huge_gold = {"decision_type": "final_answer", "target_assistant": "x" * 100_000}
        cap, tier = resolve_response_cap(
            self.args,
            self.sample({"task_type": "ac"}, label=huge_gold),
            self.params,
        )
        self.assertEqual((cap, tier), (8192, "default"))

    def test_generate_delegates_with_resolved_cap_and_records_tier(self) -> None:
        observed = {}

        async def fake_generate(args, sample, sampling_params):
            observed.update(sampling_params)
            return sample

        fake_module = types.ModuleType("slime.rollout.sglang_rollout")
        fake_module.generate = fake_generate
        sample = self.sample({"task_type": "vs", "decision_type": "final_answer"})
        with patch.dict(sys.modules, {"slime.rollout.sglang_rollout": fake_module}):
            result = asyncio.run(generate(self.args, sample, self.params))
        self.assertIs(result, sample)
        self.assertEqual(observed["max_new_tokens"], 16384)
        self.assertEqual(sample.metadata["resolved_rollout_max_response_len"], 16384)
        self.assertEqual(sample.metadata["rollout_response_length_tier"], "long")


if __name__ == "__main__":
    unittest.main()
