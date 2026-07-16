import unittest
from types import SimpleNamespace

from drug_agent.gad.reward import _rule_components


FINAL = '<thought>done</thought><final_answer>{"answer":{"summary":"ok","evidence":[],"result":{}}}</final_answer>'
TOOL = '<thought>inspect</thought><tool_call>{"tool_name":"is_valid_smiles","arguments":{"smiles_list":["CCO"]}}</tool_call>'


class TestGADRuleComponents(unittest.TestCase):
    def _sample(self, response, decision_type, target_tool_calls=None):
        return SimpleNamespace(
            response=response,
            label={"decision_type": decision_type, "target_tool_calls": target_tool_calls or []},
            metadata={},
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


if __name__ == "__main__":
    unittest.main()
