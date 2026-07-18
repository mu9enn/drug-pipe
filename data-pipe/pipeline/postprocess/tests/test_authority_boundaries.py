#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[2]
POSTPROCESS_DIR = PIPELINE_DIR / "postprocess"
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(POSTPROCESS_DIR))

from evaluate.task_evaluator import evaluate_task_answer  # noqa: E402
from trace_curator import RolloutSample, curate_results_dir, curate_sample, reconstruct_react_messages  # noqa: E402


def assistant_event(*items: dict) -> dict:
    return {"type": "assistant", "message": {"content": list(items)}}


def user_event(*items: dict) -> dict:
    return {"type": "user", "message": {"content": list(items)}}


class EvaluatorAuthorityTest(unittest.TestCase):
    class IdentityChemistry:
        @staticmethod
        def MolFromSmiles(value):
            return value if value else None

        @staticmethod
        def MolToSmiles(value, canonical=True, isomericSmiles=True):
            return value

    def test_evaluator_owns_metrics_and_answer_validity(self) -> None:
        result = evaluate_task_answer(
            "pf",
            prediction=["A", "B"],
            ground_truth=["B", "C"],
            chemistry=self.IdentityChemistry(),
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
            chemistry=self.IdentityChemistry(),
        )
        self.assertFalse(result["task_answer_valid"])
        self.assertFalse(result["aggregate_eligible"])
        self.assertTrue(any(reason.startswith("duplicate_predictions") for reason in result["invalid_reasons"]))
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
        self.assertTrue(result["metrics"]["answer_present"])
        self.assertTrue(result["metrics"]["result_content_present"])
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
            (sample_dir / "parsed_answer.json").write_text(
                '{"answer":["A grounded molecular result"]}',
                encoding="utf-8",
            )
            (sample_dir / "run_meta.json").write_text('{"return_code":0}', encoding="utf-8")
            (sample_dir / "complete_session.jsonl").write_text(
                '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"c1",'
                '"name":"mcp__molclaw-scp__x","input":{}},{"type":"tool_use","id":"c2",'
                '"name":"mcp__molclaw-scp__y","input":{}}]}}\n'
                '{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"c1",'
                '"content":{"status":"ok"}}]}}\n',
                encoding="utf-8",
            )
            record = curate_sample(
                RolloutSample(sample_dir, sample_dir, 1, "1", 1),
                default_task="kg",
                chemistry=None,
            )
            self.assertTrue(record["audit"]["execution_valid"])
            self.assertTrue(record["audit"]["task_answer_valid"])
            self.assertFalse(record["audit"]["training_trace_valid"])
            self.assertEqual(record["audit"]["final_status"], "rejected")
            self.assertEqual(
                set(record["training_record"]),
                {"schema_version", "id", "messages"},
            )

    def test_results_separate_canonical_training_from_audit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            results = Path(td)
            (results / "run_config.json").write_text('{"task":"kg"}', encoding="utf-8")
            sample_dir = results / "row0001_idx0"
            sample_dir.mkdir()
            (sample_dir / "question.json").write_text(
                '{"task":"kg","question":"Run the task","answer":[]}',
                encoding="utf-8",
            )
            (sample_dir / "parsed_answer.json").write_text(
                '{"answer":{"result":"A grounded molecular result"}}',
                encoding="utf-8",
            )
            (sample_dir / "run_meta.json").write_text('{"return_code":0}', encoding="utf-8")
            (sample_dir / "complete_session.jsonl").write_text(
                '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"c1",'
                '"name":"mcp__molclaw-scp__x","input":{}}]}}\n'
                '{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"c1",'
                '"content":{"status":"success","result":"done"}}]}}\n',
                encoding="utf-8",
            )
            summary = curate_results_dir(results)
            output_dir = results / "trajectories"
            training = json.loads((output_dir / "react_trajectories.jsonl").read_text(encoding="utf-8"))
            audit = json.loads((output_dir / "curation_audit.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(set(training), {"schema_version", "id", "messages"})
            self.assertNotIn("source_session", training)
            self.assertEqual(audit["final_status_authority"], "final_acceptance_gate")
            self.assertIn("source_session", audit)
            self.assertEqual(summary["output_count"], 1)
            self.assertEqual(
                set(summary["outputs"]),
                {"react_trajectories", "curation_audit", "rejected"},
            )


if __name__ == "__main__":
    unittest.main()
