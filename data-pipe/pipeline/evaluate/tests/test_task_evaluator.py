from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pipeline.evaluate.eval_runner import eval_molbench_vs_file
from pipeline.evaluate.task_evaluator import RDKIT_REQUIRED_MESSAGE, evaluate_task_answer


class FakeChemistry:
    EQUIVALENTS = {
        "C(C)O": "CCO",
        "OCC": "CCO",
    }

    @classmethod
    def MolFromSmiles(cls, value):
        return value if value and value != "invalid" else None

    @classmethod
    def MolToSmiles(cls, value, canonical=True, isomericSmiles=True):
        return cls.EQUIVALENTS.get(value, value)


class TaskEvaluatorTest(unittest.TestCase):
    def test_rdkit_is_required_for_molbench(self) -> None:
        with self.assertRaisesRegex(RuntimeError, RDKIT_REQUIRED_MESSAGE):
            evaluate_task_answer("ac", prediction=["CCO"], ground_truth=["CCO"], chemistry=None)

    def test_equivalent_smiles_use_canonical_equality(self) -> None:
        result = evaluate_task_answer(
            "ac",
            prediction=["C(C)O"],
            ground_truth=["OCC"],
            chemistry=FakeChemistry(),
        )
        self.assertTrue(result["metrics"]["is_correct"])

    def test_kg_dispatch_never_calls_smiles_parser(self) -> None:
        class ExplodingChemistry:
            @staticmethod
            def MolFromSmiles(_value):
                raise AssertionError("KG must not call RDKit")

        result = evaluate_task_answer(
            "kg",
            prediction={"summary": "done", "result": {"artifact": "protein.pdb"}},
            ground_truth=[],
            chemistry=ExplodingChemistry(),
            task_contract={"required_final_fields": ["summary", "result"]},
        )
        self.assertTrue(result["task_answer_valid"])
        self.assertFalse(result["audit"]["chemistry_canonicalization"])

    def test_kg_process_only_answer_is_not_a_result(self) -> None:
        result = evaluate_task_answer(
            "kg",
            prediction="Analysis complete",
            ground_truth=None,
        )
        self.assertFalse(result["task_answer_valid"])
        self.assertIn("missing_result_content", result["invalid_reasons"])

    def test_invalid_vs_is_excluded_from_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "answer": ["CCO"],
                            "candidates": ["CCO", "CCC"],
                            "json_results": {"ranking": ["CCO", "CCC"]},
                        },
                        {
                            "answer": ["CCO"],
                            "candidates": ["CCO", "CCC"],
                            "json_results": {"ranking": ["CCO", "CCO"]},
                        },
                    ]
                ),
                encoding="utf-8",
            )
            with patch("pipeline.evaluate.eval_runner.load_chemistry_module", return_value=(FakeChemistry(), None)):
                summary = eval_molbench_vs_file(str(path))
            scores = summary["molbench_vs_molbench_vs"]
            self.assertEqual(scores["n_samples"], 2)
            self.assertEqual(scores["n_valid_scored"], 1)
            self.assertEqual(scores["top3_avg_hit_num"], 1.0)


if __name__ == "__main__":
    unittest.main()
