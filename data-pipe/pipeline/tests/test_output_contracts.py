from __future__ import annotations

import json
import unittest

from pipeline.output_contracts import (
    CONTRACTS,
    LEGACY_CONTRACTS_V1,
    normalize_final_answer,
    normalize_task_prompt,
    parsed_answer_values,
)


class OutputContractsTest(unittest.TestCase):
    def test_teacher_prompt_normalization_is_idempotent(self) -> None:
        cases = {
            "ac": "Compare candidates. Only output the corresponding SMILES.",
            "pf": (
                "Filter candidates.\nIf none satisfy, output an empty line.\n"
                "Output format:\nPrint each satisfying SMILES on its own line, and nothing else."
            ),
            "kg": "Run the requested workflow.",
            "e2e": "Complete the drug-discovery task.",
        }
        for task_type, source in cases.items():
            with self.subTest(task_type=task_type):
                normalized = normalize_task_prompt(source, task_type)
                self.assertEqual(normalize_task_prompt(normalized, task_type), normalized)
                self.assertTrue(normalized.endswith(CONTRACTS[task_type]))
                self.assertNotIn("output an empty line", normalized.lower())

        vs = json.dumps({"task": "rank", "output_format": "Return an array."})
        normalized_vs = normalize_task_prompt(vs, "vs")
        self.assertEqual(normalize_task_prompt(normalized_vs, "vs"), normalized_vs)
        self.assertEqual(json.loads(normalized_vs)["output_format"], CONTRACTS["vs"])

    def test_upgrades_v1_prompt_without_stacking_output_blocks(self) -> None:
        old = "Compare candidates.\n\nOutput format:\n" + LEGACY_CONTRACTS_V1["ac"]
        upgraded = normalize_task_prompt(old, "ac")
        self.assertEqual(upgraded.count("Output format:"), 1)
        self.assertTrue(upgraded.endswith(CONTRACTS["ac"]))

    def test_final_answer_requires_exact_fields(self) -> None:
        answer = '{"answer_smiles":"CCO","evidence":[{"tool":"x"}]}'
        self.assertEqual(parsed_answer_values(answer, "ac"), ["CCO"])
        self.assertEqual(json.loads(normalize_final_answer(answer, "ac")), {
            "answer_smiles": "CCO",
            "evidence": [{"tool": "x"}],
        })
        with self.assertRaises(ValueError):
            normalize_final_answer('{"task_type":"ac","answer_smiles":"CCO","evidence":[]}', "ac")
        with self.assertRaises(ValueError):
            normalize_final_answer('["CCO"]', "vs")


if __name__ == "__main__":
    unittest.main()
