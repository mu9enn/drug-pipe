#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import final_hard_clean as cleaner  # noqa: E402


def message(role: str, content: str) -> dict:
    return {"role": role, "content": content}


class FinalHardCleanTest(unittest.TestCase):
    def test_strips_metadata_and_sanitizes_relative_path(self) -> None:
        sample = {
            "schema_version": "x",
            "id": "mcp_sft_ac_demo",
            "messages": [
                message("system", "s"),
                message("user", "Molecule A: CCO\nMolecule B: CCC"),
                message(
                    "user",
                    '<observation tool_name="retrieve_protein_sequence">'
                    '{"ok":true,"tool_name":"retrieve_protein_sequence","status":"success",'
                    '"content":{"status":"success","msg":"saved to ./protein_seq/ACE_7789.json"},'
                    '"metadata":{"tool_use_id":"x","raw_event_index":1,"pointers":{"raw_pointer":"./protein_seq/ACE_7789.json"}}}'
                    "</observation>",
                ),
                message("assistant", '<final_answer>{"task_type":"ac","answer_smiles":"CCO","evidence":[]}</final_answer>'),
            ],
        }
        cleaned, report = cleaner.clean_sample(sample)
        self.assertIsNotNone(cleaned)
        text = json.dumps(cleaned)
        self.assertNotIn('"metadata"', text)
        self.assertNotIn("raw_pointer", text)
        self.assertIn("<artifact:protein_sequence/ACE_7789.json>", text)
        self.assertTrue(report["removed_observation_metadata"])

    def test_status_conflict_is_flagged_not_repaired(self) -> None:
        sample = {
            "schema_version": "x",
            "id": "mcp_sft_ac_demo",
            "messages": [
                message("system", "s"),
                message("user", "Molecule A: CCO\nMolecule B: CCC"),
                message(
                    "user",
                    '<observation tool_name="x">{"ok":true,"status":"success","content":{"error":"connection refused"},'
                    '"metadata":{"raw_is_error":true}}</observation>',
                ),
                message("assistant", '<final_answer>{"task_type":"ac","answer_smiles":"CCO","evidence":[]}</final_answer>'),
            ],
        }
        cleaned, report = cleaner.clean_sample(sample)
        self.assertIsNone(cleaned)
        self.assertIn("observation_status_conflict_after_llm_clean", report["errors"])
        self.assertIn('"ok":true', sample["messages"][2]["content"])

    def test_vs_inconsistent_ranking_is_flagged_not_reranked(self) -> None:
        sample = {
            "schema_version": "x",
            "id": "mcp_sft_vs_demo",
            "messages": [
                message("system", "s"),
                message("user", 'Question payload (MolBench-VS): {"candidates":["A","B","C","D"]}'),
                message("assistant", '<tool_call>{"tool_name":"quickvina","arguments":{"smiles":"A"}}</tool_call>'),
                message("user", '<observation tool_name="quickvina">{"ok":true,"status":"success","content":{"docking_affinity_value":-4.5}}</observation>'),
                message("assistant", '<tool_call>{"tool_name":"quickvina","arguments":{"smiles":"B"}}</tool_call>'),
                message("user", '<observation tool_name="quickvina">{"ok":true,"status":"success","content":{"docking_affinity_value":-5.4}}</observation>'),
                message("assistant", '<tool_call>{"tool_name":"quickvina","arguments":{"smiles":"C"}}</tool_call>'),
                message("user", '<observation tool_name="quickvina">{"ok":true,"status":"success","content":{"docking_affinity_value":-4.7}}</observation>'),
                message("assistant", '<final_answer>{"task_type":"vs","ranked_smiles":["A","C","B","D"],"evidence":[]}</final_answer>'),
            ],
        }
        cleaned, report = cleaner.clean_sample(sample)
        self.assertIsNone(cleaned)
        self.assertIn("vs_ranking_inconsistent_after_llm_clean", report["errors"])
        self.assertEqual(report["vs_ranking_detection"]["ranked_smiles"], ["A", "C", "B", "D"])
        self.assertEqual(
            json.loads(sample["messages"][-1]["content"].removeprefix("<final_answer>").removesuffix("</final_answer>"))[
                "ranked_smiles"
            ],
            ["A", "C", "B", "D"],
        )

    def test_smiles_are_not_sanitized(self) -> None:
        report: dict = {}
        for value in ("CCCC1CCCCC/C(N)=N\\1", "COC(=O)/C(O)=C/C(=O)c1cccn1", "[C@@H](N)"):
            self.assertEqual(cleaner.sanitize_relative_paths(value, report), value)

    def test_vs_candidates_with_brackets_are_parsed(self) -> None:
        user_text = 'Question payload (MolBench-VS): {"candidates":["C[C@@H](N)O","CCO"]}'
        self.assertEqual(cleaner._extract_candidates(user_text), ["C[C@@H](N)O", "CCO"])

    def test_relative_path_sanitizer_is_conservative(self) -> None:
        report: dict = {}
        value = (
            "saved ./protein_seq/ACE_7789.json; keep skills/tool kcal/mol C/C "
            "https://errors.pydantic.dev/2.0/v/missing_argument"
        )
        cleaned = cleaner.sanitize_relative_paths(value, report)
        self.assertIn("<artifact:protein_sequence/ACE_7789.json>", cleaned)
        for unchanged in ("skills/tool", "kcal/mol", "C/C", "https://errors.pydantic.dev/2.0/v/missing_argument"):
            self.assertIn(unchanged, cleaned)
        self.assertEqual(
            cleaner.sanitize_relative_paths("./fpocket_demo.json", {}),
            "<artifact:fpocket/fpocket_demo.json>",
        )


if __name__ == "__main__":
    unittest.main()
