from __future__ import annotations

import unittest

from drug_agent.data.migrate_terminal_turns import migrate_record


class TerminalMigrationTest(unittest.TestCase):
    def test_merges_boundary_and_preserves_structured_facts(self) -> None:
        record = {
            "schema_version": "drug_agent_sft_react_json_v1",
            "id": "x",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "<thought>final analysis</thought>", "step_loss_mask": 1},
                {
                    "role": "assistant",
                    "content": (
                        '<final_answer>{"task_type":"kg","result":"artifact","evidence":[],'
                        '"summary":"final analysis\\n\\nfinal analysis"}</final_answer>'
                    ),
                    "step_loss_mask": 1,
                },
            ],
        }
        migrated, audit = migrate_record(record)
        self.assertEqual([message["role"] for message in migrated["messages"]], ["system", "user", "assistant"])
        content = migrated["messages"][-1]["content"]
        self.assertIn("<thought>final analysis</thought>", content)
        self.assertIn('"result":"artifact"', content)
        self.assertNotIn('"summary"', content)
        self.assertTrue(audit["merged_terminal_assistant_turn"])


if __name__ == "__main__":
    unittest.main()
