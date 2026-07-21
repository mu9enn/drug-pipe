from __future__ import annotations

import json
import unittest

from drug_agent.protocol.react_protocol import parse_react_sequence


class CanonicalFinalAnswerTest(unittest.TestCase):
    def _parse(self, payload: dict) -> dict:
        content = f"<final_answer>{json.dumps(payload)}</final_answer>"
        return parse_react_sequence(content, role="assistant")

    def test_accepts_data_pipe_task_specific_finals(self) -> None:
        payloads = [
            {
                "task_type": "vs",
                "ranked_smiles": ["CCO"],
                "selected_smiles": "CCO",
                "summary": "ranked",
                "evidence": [],
            },
            {
                "task_type": "ac",
                "answer_smiles": "CCO",
                "summary": "selected",
                "evidence": [],
            },
            {
                "task_type": "pf",
                "selected_smiles": ["CCO"],
                "summary": "filtered",
                "evidence": [],
            },
            {
                "task_type": "kg",
                "result": "<artifact:structure/fixed.pdb>",
                "summary": "repaired",
                "evidence": [],
            },
            {
                "task_type": "e2e",
                "result": {"status": "done"},
                "summary": "completed",
                "evidence": [],
            },
        ]
        for payload in payloads:
            with self.subTest(task_type=payload["task_type"]):
                self.assertTrue(self._parse(payload)["ok"])

    def test_keeps_action_style_final_compatibility(self) -> None:
        parsed = self._parse(
            {
                "answer": {
                    "summary": "done",
                    "evidence": [],
                    "result": {},
                    "ranked_molecules": [],
                }
            }
        )
        self.assertTrue(parsed["ok"])

    def test_rejects_incomplete_task_specific_final(self) -> None:
        parsed = self._parse(
            {"task_type": "vs", "selected_smiles": "CCO", "summary": "ranked", "evidence": []}
        )
        self.assertFalse(parsed["ok"])
        self.assertIn("ranked_smiles", str(parsed["error_message"]))


if __name__ == "__main__":
    unittest.main()
