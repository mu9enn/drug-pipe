from __future__ import annotations

import json
import unittest

from drug_agent.protocol.react_protocol import parse_react_sequence, parse_runtime_decision, project_final_answer


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

    def test_rejects_legacy_action_style_final(self) -> None:
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
        self.assertFalse(parsed["ok"])

    def test_rejects_incomplete_task_specific_final(self) -> None:
        parsed = self._parse(
            {"task_type": "vs", "selected_smiles": "CCO", "summary": "ranked", "evidence": []}
        )
        self.assertFalse(parsed["ok"])
        self.assertIn("ranked_smiles", str(parsed["error_message"]))

    def test_runtime_accepts_multiple_calls_and_rejects_mixed_terminal_types(self) -> None:
        multiple = parse_runtime_decision(
            '<thought>inspect</thought>'
            '<tool_call>{"tool_name":"Read","arguments":{"file_path":"x"}}</tool_call>'
            '<tool_call>{"tool_name":"fix_pdb","arguments":{"input_path":"x"}}</tool_call>'
        )
        self.assertTrue(multiple["ok"])
        self.assertEqual([call["tool_name"] for call in multiple["tool_calls"]], ["Read", "fix_pdb"])
        mixed = parse_runtime_decision(
            '<tool_call>{"tool_name":"fix_pdb","arguments":{}}</tool_call>'
            '<final_answer>{"task_type":"kg","result":{},"evidence":[]}</final_answer>'
        )
        self.assertFalse(mixed["ok"])

    def test_projects_task_specific_final(self) -> None:
        self.assertEqual(project_final_answer({"task_type": "vs", "ranked_smiles": ["CCO"]}), ["CCO"])
        self.assertEqual(project_final_answer({"task_type": "kg", "result": "artifact"}), "artifact")


if __name__ == "__main__":
    unittest.main()
