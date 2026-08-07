import unittest

from drug_agent.gad.data import _partition_supported_tool_calls, convert_records


TOOL = '<thought>inspect</thought><tool_call>{"tool_name":"is_valid_smiles","arguments":{"smiles_list":["CCO"]}}</tool_call>'
OBS = '<observation tool_name="is_valid_smiles">{"ok":true,"content":{}}</observation>'
FINAL = '<thought>finish</thought><final_answer>{"task_type":"kg","result":{},"evidence":[]}</final_answer>'


class TestGADData(unittest.TestCase):
    def test_state_target_boundary_and_final_answer(self):
        records = [
            {
                "id": "react_pf_x",
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
        self.assertEqual(rows[1]["metadata"]["task_type"], "pf")

    def test_supported_local_tool_is_kept(self):
        local = '<thought>x</thought><tool_call>{"tool_name":"Bash","arguments":{"command":"pwd"}}</tool_call>'
        records = [{"id": "local", "messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": local}]}]
        rows, skipped, _ = convert_records(records)
        self.assertFalse(skipped)
        self.assertEqual(rows[0]["label"]["target_tool_calls"][0]["tool_name"], "Bash")
        self.assertIn("Bash", rows[0]["metadata"]["allowed_tool_names"])

    def test_teacher_only_tool_is_skipped(self):
        bad = '<thought>x</thought><tool_call>{"tool_name":"WebSearch","arguments":{"query":"x"}}</tool_call>'
        records = [{"id": "bad", "messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": bad}]}]
        rows, skipped, _ = convert_records(records)
        self.assertFalse(rows)
        self.assertEqual(skipped[0]["skip_reason"], "unsupported_tool")

    def test_explicit_catalog_rejects_unknown_bare_tool(self):
        calls = {"tool_calls": [{"tool_name": "made_up_tool", "tool_name_raw": "made_up_tool", "arguments": {}}]}
        supported, rejected = _partition_supported_tool_calls(calls, {"is_valid_smiles"})
        self.assertFalse(supported)
        self.assertEqual(rejected[0]["tool_name"], "made_up_tool")

    def test_bare_cleaned_molclaw_tool_does_not_require_mini_allowlist(self):
        tool = '<thought>x</thought><tool_call>{"tool_name":"extract_pdb_chains","arguments":{"path":"x.pdb"}}</tool_call>'
        records = [{"id": "bare", "messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": tool}]}]
        rows, skipped, _ = convert_records(records)
        self.assertFalse(skipped)
        self.assertEqual(rows[0]["label"]["target_tool_calls"][0]["tool_name"], "extract_pdb_chains")


if __name__ == "__main__":
    unittest.main()
