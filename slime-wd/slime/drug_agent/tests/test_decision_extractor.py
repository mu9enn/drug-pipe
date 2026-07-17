import json
import tempfile
import unittest
from pathlib import Path

from drug_agent.decision_extractor import (
    iter_react_decisions,
    normalize_tool_call,
    parse_assistant_decision,
)
from drug_agent.gad.data import convert_records
from drug_agent.toolrl.convert_react_to_toolrl_steps import convert_react_to_toolrl_steps


TOOL = (
    '<thought>inspect</thought><tool_call>{"name":"mcp__molclaw-scp__is_valid_smiles",'
    '"arguments":{"smiles_list":["CCO"]},"id":"c1"}</tool_call>'
)
OBS = '<observation tool_name="mcp__molclaw-scp__is_valid_smiles">{"ok":true}</observation>'
FINAL = '<final_answer>{"answer":"done"}</final_answer>'


class TestSharedDecisionExtractor(unittest.TestCase):
    def setUp(self):
        self.messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": TOOL},
            {"role": "user", "content": OBS},
            {"role": "assistant", "content": FINAL},
        ]

    def test_parser_accepts_canonical_name_field(self):
        parsed = parse_assistant_decision(TOOL)
        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["tool_calls"][0]["tool_name"], "is_valid_smiles")
        self.assertEqual(
            normalize_tool_call({"tool_name": "mcp__molclaw-vs__dock", "arguments": {}})["tool_name"],
            "dock",
        )

    def test_current_state_never_contains_future_observation(self):
        decisions = list(iter_react_decisions(self.messages))
        self.assertEqual(len(decisions), 2)
        first_state_text = "\n".join(str(item.get("content")) for item in decisions[0]["state_messages"])
        second_state_text = "\n".join(str(item.get("content")) for item in decisions[1]["state_messages"])
        self.assertNotIn(OBS, first_state_text)
        self.assertIn(OBS, second_state_text)
        self.assertNotIn(FINAL, second_state_text)

    def test_gad_consumes_the_shared_state_boundary(self):
        shared = list(iter_react_decisions(self.messages))
        rows, skipped, _ = convert_records([{"id": "x", "messages": self.messages}])
        self.assertFalse(skipped)
        self.assertEqual(rows[0]["state_messages"], shared[0]["state_messages"])
        self.assertEqual(rows[1]["state_messages"], shared[1]["state_messages"])

    def test_toolrl_and_gad_receive_identical_tool_decision_state(self):
        record = {"id": "x", "messages": self.messages}
        gad_rows, _, _ = convert_records([record])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "react.jsonl"
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")
            output = root / "toolrl.jsonl"
            convert_react_to_toolrl_steps(source, output)
            toolrl_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(toolrl_rows[0]["prompt"], gad_rows[0]["state_messages"])


if __name__ == "__main__":
    unittest.main()
