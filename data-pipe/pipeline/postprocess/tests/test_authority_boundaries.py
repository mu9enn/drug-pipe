#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[2]
POSTPROCESS_DIR = PIPELINE_DIR / "postprocess"
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(POSTPROCESS_DIR))

from evaluate.task_evaluator import evaluate_task_answer  # noqa: E402
from trace_curator import RolloutSample, curate_sample, reconstruct_react_messages  # noqa: E402


def assistant_event(*items: dict) -> dict:
    return {"type": "assistant", "message": {"content": list(items)}}


def user_event(*items: dict) -> dict:
    return {"type": "user", "message": {"content": list(items)}}


class EvaluatorAuthorityTest(unittest.TestCase):
    def test_evaluator_owns_metrics_and_answer_validity(self) -> None:
        result = evaluate_task_answer(
            "pf",
            prediction=["A", "B"],
            ground_truth=["B", "C"],
            chemistry=None,
        )
        self.assertTrue(result["task_answer_valid"])
        self.assertEqual(result["metrics"]["precision"], 0.5)
        self.assertEqual(result["metrics"]["recall"], 0.5)
        self.assertEqual(result["metrics"]["f1"], 0.5)

    def test_vs_shape_failure_is_not_scientific_correctness(self) -> None:
        result = evaluate_task_answer(
            "vs",
            prediction=["A", "A"],
            ground_truth=["A"],
            candidates=["A", "B"],
            chemistry=None,
        )
        self.assertFalse(result["task_answer_valid"])
        self.assertIn("duplicate_predictions", result["invalid_reasons"])
        self.assertEqual(result["metrics"]["top3_hit_num"], 2.0)

    def test_kg_narrative_answer_is_not_parsed_as_smiles(self) -> None:
        class RejectingChemistry:
            @staticmethod
            def MolFromSmiles(_value):
                return None

        result = evaluate_task_answer(
            "kg",
            prediction="The repaired structure was written to /tmp/egfr_fixed.pdb.",
            ground_truth=[],
            chemistry=RejectingChemistry(),
        )
        self.assertTrue(result["task_answer_valid"])
        self.assertEqual(result["metrics"], {"answer_present": True})
        self.assertFalse(result["audit"]["chemistry_canonicalization"])


class CuratorAuthorityTest(unittest.TestCase):
    def test_curator_counts_usage_once_and_builds_canonical_roles(self) -> None:
        events = [
            assistant_event(
                {"type": "thinking", "thinking": "Inspect the molecule."},
                {"type": "tool_use", "id": "workspace", "name": "Read", "input": {"path": "/tmp/a"}},
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "mcp__molclaw-scp__predict",
                    "input": {"smiles": "CCO"},
                },
            ),
            user_event(
                {"type": "tool_result", "tool_use_id": "workspace", "content": "debug"},
                {"type": "tool_result", "tool_use_id": "call-1", "content": {"score": 1.2}},
            ),
        ]
        messages, stats = reconstruct_react_messages(events, question_text="Analyze CCO", final_answer=["CCO"])
        self.assertEqual(stats["molclaw_usage_count"], 1)
        self.assertEqual(stats["molclaw_usage_computation_count"], 1)
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user", "assistant"])
        self.assertNotIn("Read", "".join(message["content"] for message in messages))
        self.assertEqual(messages[3]["step_loss_mask"], 0)
        self.assertIn("<final_answer>", messages[-1]["content"])

    def test_curator_separates_three_validity_states(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sample_dir = Path(td)
            (sample_dir / "question.json").write_text(
                '{"task":"kg","question":"Do the grounded task","answer":[]}',
                encoding="utf-8",
            )
            (sample_dir / "parsed_answer.json").write_text('{"answer":["done"]}', encoding="utf-8")
            (sample_dir / "run_meta.json").write_text('{"return_code":0}', encoding="utf-8")
            (sample_dir / "complete_session.jsonl").write_text(
                '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"c1",'
                '"name":"mcp__molclaw-scp__x","input":{}}]}}\n',
                encoding="utf-8",
            )
            record = curate_sample(
                RolloutSample(sample_dir, sample_dir, 1, "1", 1),
                default_task="kg",
                chemistry=None,
            )
            self.assertTrue(record["status"]["execution_valid"])
            self.assertTrue(record["status"]["task_answer_valid"])
            self.assertFalse(record["status"]["training_trace_valid"])
            self.assertFalse(record["status"]["accepted"])


if __name__ == "__main__":
    unittest.main()
