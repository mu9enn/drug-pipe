import unittest

from drug_agent.gad.data import convert_records


TOOL = '<thought>inspect</thought><tool_call>{"tool_name":"is_valid_smiles","arguments":{"smiles_list":["CCO"]}}</tool_call>'
OBS = '<observation tool_name="is_valid_smiles">{"ok":true,"content":{}}</observation>'
FINAL = '<thought>finish</thought><final_answer>{"answer":{"summary":"ok","evidence":[],"result":{}}}</final_answer>'


class TestGADData(unittest.TestCase):
    def test_state_target_boundary_and_final_answer(self):
        records = [
            {
                "id": "x",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": TOOL},
                    {"role": "user", "content": OBS},
                    {"role": "assistant", "content": FINAL},
                ],
            }
        ]
        rows, skipped, report = convert_records(records)
        self.assertFalse(skipped)
        self.assertEqual(report["counts"]["kept_tool_call"], 1)
        self.assertEqual(report["counts"]["kept_final_answer"], 1)
        self.assertNotIn(TOOL, [m["content"] for m in rows[0]["prompt"]])
        self.assertIn(OBS, [m["content"] for m in rows[1]["prompt"]])
        self.assertNotIn(FINAL, [m["content"] for m in rows[1]["prompt"]])
        self.assertEqual(rows[1]["metadata"]["state_messages"], rows[1]["state_messages"])

    def test_non_molclaw_is_skipped(self):
        bad = '<thought>x</thought><tool_call>{"tool_name":"Bash","arguments":{"cmd":"pwd"}}</tool_call>'
        records = [{"id": "bad", "messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": bad}]}]
        rows, skipped, _ = convert_records(records)
        self.assertFalse(rows)
        self.assertEqual(skipped[0]["skip_reason"], "non_molclaw_tool")

    def test_bare_cleaned_molclaw_tool_does_not_require_mini_allowlist(self):
        tool = '<thought>x</thought><tool_call>{"tool_name":"extract_pdb_chains","arguments":{"path":"x.pdb"}}</tool_call>'
        records = [{"id": "bare", "messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": tool}]}]
        rows, skipped, _ = convert_records(records)
        self.assertFalse(skipped)
        self.assertEqual(rows[0]["label"]["target_tool_calls"][0]["tool_name"], "extract_pdb_chains")


if __name__ == "__main__":
    unittest.main()
