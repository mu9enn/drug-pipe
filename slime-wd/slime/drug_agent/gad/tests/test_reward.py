import asyncio
import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from drug_agent.gad.reward import _rule_components, reward_func


FINAL = '<thought>done</thought><final_answer>{"task_type":"kg","result":{},"evidence":[]}</final_answer>'
TOOL = '<thought>inspect</thought><tool_call>{"tool_name":"is_valid_smiles","arguments":{"smiles_list":["CCO"]}}</tool_call>'


class TestGADRuleComponents(unittest.TestCase):
    def _sample(self, response, decision_type, target_tool_calls=None):
        return SimpleNamespace(
            prompt=[{"role": "user", "content": "task"}],
            response=response,
            label={
                "decision_type": decision_type,
                "target_tool_calls": target_tool_calls or [],
                "teacher_response": FINAL,
            },
            metadata={"state_messages": [{"role": "user", "content": "task"}]},
            weight_versions=[1],
        )

    def test_invalid_final_answer_gets_no_schema_reward(self):
        format_score, schema_score, _ = _rule_components(None, self._sample("plain prose", "final_answer"))
        self.assertEqual(format_score, 0.0)
        self.assertEqual(schema_score, 0.0)

    def test_valid_final_answer_gets_schema_reward(self):
        format_score, schema_score, _ = _rule_components(None, self._sample(FINAL, "final_answer"))
        self.assertEqual(format_score, 1.0)
        self.assertEqual(schema_score, 1.0)

    def test_tool_call_is_not_valid_at_final_answer_step(self):
        _, schema_score, _ = _rule_components(None, self._sample(TOOL, "final_answer"))
        self.assertEqual(schema_score, 0.0)

    def test_rule_mode_does_not_require_discriminator_service(self):
        sample = self._sample(FINAL, "final_answer")
        env = {"GAD_REWARD_MODE": "rule"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("GAD_DISCRIMINATOR_URL", None)
            out = asyncio.run(reward_func(None, sample))
        self.assertEqual(out["score"], 1.0)
        self.assertEqual(out["diagnostics"]["reward_mode"], "rule")


if __name__ == "__main__":
    unittest.main()
