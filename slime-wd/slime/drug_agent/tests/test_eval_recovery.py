from __future__ import annotations

import concurrent.futures
import asyncio
import json
import os
import sys
import tempfile
import types
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from drug_agent.evaluation.materialize_saved_results import materialize_records
from drug_agent.evaluation.task_store import bind_task_identity, checkpoint_sample, load_records, restore_sample


class Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


def _sample(task_id: str, *, prompt: str | None = None, index: int = 0, done_reason: str = "final_answer"):
    return SimpleNamespace(
        index=index,
        prompt=[{"role": "user", "content": prompt or f"question {task_id}"}],
        label={"answer": f"answer {task_id}"},
        response=f"response {task_id}",
        response_length=3,
        reward=None,
        status=Status.COMPLETED if done_reason == "final_answer" else Status.FAILED,
        metadata={
            "env_kwargs": {
                "task_id": task_id,
                "task_type": "pf",
                "data_source": "molbench_ms1",
            },
            "benchmark": {"suite": "molbench_ms1", "subtask": "filter"},
            "drug_agent_trace": {
                "done_reason": done_reason,
                "projected_final_answer": [f"answer {task_id}"],
                "actions": [{"raw_response": "<thought>done</thought>"}],
                "observations": [],
                "artifact_audit": {"path_map": {}},
            },
        },
    )


class EvaluationTaskStoreTest(unittest.TestCase):
    def _env(self, root: str, *, resume: str = "0", expected: str = "2") -> dict[str, str]:
        return {
            "DRUG_AGENT_EVAL_RUN_DIR": root,
            "DRUG_AGENT_EVAL_RUN_FINGERPRINT": "run-fingerprint",
            "DRUG_AGENT_EVAL_EXPECTED_TASK_COUNT": expected,
            "DRUG_AGENT_EVAL_RESUME": resume,
        }

    def test_each_task_is_atomic_and_can_be_restored_after_interruption(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, self._env(tmp), clear=False):
            first = _sample("task-1", index=0)
            checkpoint_sample(first, evaluation=True)

            task_files = list((Path(tmp) / "task_results").glob("*.json"))
            self.assertEqual(len(task_files), 1)
            self.assertFalse(list((Path(tmp) / "task_results").glob("*.tmp")))
            progress = json.loads((Path(tmp) / "progress.json").read_text())
            self.assertEqual(progress["checkpointed_count"], 1)
            self.assertEqual(progress["remaining_count"], 1)

            fresh = _sample("task-1", index=0)
            fresh.status = Status.PENDING
            fresh.metadata.pop("drug_agent_trace")
            with patch.dict(os.environ, {"DRUG_AGENT_EVAL_RESUME": "1"}, clear=False):
                restored = restore_sample(fresh, evaluation=True)
            self.assertIs(restored, fresh)
            self.assertEqual(restored.status, Status.COMPLETED)
            self.assertEqual(restored.metadata["drug_agent_trace"]["done_reason"], "final_answer")
            self.assertTrue(restored.metadata["drug_agent_eval_resume"]["restored"])

            checkpoint_sample(_sample("task-2", index=1, done_reason="task_timeout"), evaluation=True)
            progress = json.loads((Path(tmp) / "progress.json").read_text())
            self.assertEqual(progress["checkpointed_count"], 2)
            self.assertEqual(progress["remaining_count"], 0)
            self.assertEqual(progress["final_answer_count"], 1)
            self.assertEqual(progress["non_final_count"], 1)
            self.assertEqual(len((Path(tmp) / "partial_results.jsonl").read_text().splitlines()), 2)

    def test_resume_rejects_changed_task_or_run_identity(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, self._env(tmp), clear=False):
            checkpoint_sample(_sample("task-1", prompt="original"), evaluation=True)
            changed = _sample("task-1", prompt="changed")
            changed.status = Status.PENDING
            with patch.dict(os.environ, {"DRUG_AGENT_EVAL_RESUME": "1"}, clear=False):
                with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                    restore_sample(changed, evaluation=True)
                with patch.dict(
                    os.environ,
                    {"DRUG_AGENT_EVAL_RUN_FINGERPRINT": "different-run"},
                    clear=False,
                ):
                    with self.assertRaisesRegex(ValueError, "different run"):
                        restore_sample(_sample("task-1"), evaluation=True)

    def test_bound_identity_survives_rendered_prompt_mutation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, self._env(tmp), clear=False):
            completed = _sample("task-1", prompt="original")
            original_fingerprint = bind_task_identity(completed)
            completed.prompt = "rendered tokenizer prompt"
            checkpoint_sample(completed, evaluation=True)

            record = load_records(tmp, run_fingerprint="run-fingerprint")[0]
            self.assertEqual(record["task_fingerprint"], original_fingerprint)

            fresh = _sample("task-1", prompt="original")
            fresh.status = Status.PENDING
            fresh.metadata.pop("drug_agent_trace")
            bind_task_identity(fresh)
            with patch.dict(os.environ, {"DRUG_AGENT_EVAL_RESUME": "1"}, clear=False):
                restored = restore_sample(fresh, evaluation=True)
            self.assertIs(restored, fresh)

    def test_concurrent_completions_produce_complete_deduplicated_snapshot(self):
        count = 24
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            self._env(tmp, expected=str(count)),
            clear=False,
        ):
            samples = [_sample(f"task-{index:02d}", index=index) for index in range(count)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda sample: checkpoint_sample(sample, evaluation=True), samples))

            records = load_records(tmp, run_fingerprint="run-fingerprint")
            self.assertEqual(len(records), count)
            self.assertEqual(len({record["id"] for record in records}), count)
            progress = json.loads((Path(tmp) / "progress.json").read_text())
            self.assertEqual(progress["checkpointed_count"], count)
            self.assertEqual(progress["remaining_count"], 0)
            self.assertFalse(list(Path(tmp).rglob("*.tmp")))

    def test_training_path_does_not_write_evaluation_state(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, self._env(tmp), clear=False):
            checkpoint_sample(_sample("task-1"), evaluation=False)
            self.assertFalse((Path(tmp) / "task_results").exists())
            self.assertIsNone(restore_sample(_sample("task-1"), evaluation=False))

    def test_resume_can_retry_non_final_and_overwrite_it_atomically(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            self._env(tmp, resume="1", expected="1"),
            clear=False,
        ):
            failed = _sample("task-1", done_reason="fatal_error")
            checkpoint_sample(failed, evaluation=True)

            fresh = _sample("task-1")
            fresh.status = Status.PENDING
            fresh.metadata.pop("drug_agent_trace")
            bind_task_identity(fresh)
            with patch.dict(
                os.environ,
                {"DRUG_AGENT_EVAL_RETRY_NON_FINAL": "1"},
                clear=False,
            ):
                self.assertIsNone(restore_sample(fresh, evaluation=True))
                checkpoint_sample(_sample("task-1"), evaluation=True)

            records = load_records(tmp, run_fingerprint="run-fingerprint")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "completed")
            self.assertEqual(records[0]["trace"]["done_reason"], "final_answer")
            progress = json.loads((Path(tmp) / "progress.json").read_text())
            self.assertEqual(progress["successful_final_count"], 1)
            self.assertEqual(progress["retryable_non_final_count"], 0)
            self.assertEqual(progress["remaining_to_success_count"], 0)

    def test_resumed_generation_skips_per_task_executor_and_model(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            self._env(tmp, resume="1", expected="1"),
            clear=False,
        ):
            checkpoint_sample(_sample("task-1"), evaluation=True)
            fresh = _sample("task-1")
            fresh.status = Status.PENDING
            fresh.metadata.pop("drug_agent_trace")
            runtime: dict[str, object] = {}
            fake_sglang = types.ModuleType("slime.rollout.sglang_rollout")
            fake_sglang.GenerateState = object
            fake_types = types.ModuleType("slime.utils.types")
            fake_types.Sample = SimpleNamespace
            fake_http = types.ModuleType("slime.utils.http_utils")
            fake_http.post = lambda *_args, **_kwargs: None
            with patch.dict(
                sys.modules,
                {
                    "slime.rollout.sglang_rollout": fake_sglang,
                    "slime.utils.types": fake_types,
                    "slime.utils.http_utils": fake_http,
                },
            ):
                from drug_agent.rollout.generate_with_drug_agent import generate

                with patch(
                    "drug_agent.rollout.generate_with_drug_agent._get_runtime",
                    return_value=runtime,
                ), patch(
                    "drug_agent.rollout.generate_with_drug_agent.MCPToolExecutor",
                    side_effect=AssertionError("resumed task must not create an executor"),
                ):
                    restored = asyncio.run(generate(SimpleNamespace(), fresh, {}, evaluation=True))
            self.assertEqual(restored.status, Status.COMPLETED)
            self.assertTrue(restored.metadata["drug_agent_eval_resume"]["restored"])

    def test_saved_result_materialization_keeps_failures_in_metric_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "metrics"
            records = []
            for sample in (_sample("task-1"), _sample("task-2", done_reason="fatal_error")):
                records.append(
                    {
                        "id": sample.metadata["env_kwargs"]["task_id"],
                        "task_type": "pf",
                        "suite": "molbench_ms1",
                        "subtask": "filter",
                        "status": sample.status.value,
                        "label": sample.label,
                        "trace": sample.metadata["drug_agent_trace"],
                        "saved_at": "now",
                    }
                )
            with patch(
                "drug_agent.evaluation.materialize_saved_results.run_official_evaluation",
                return_value={"rdkit_bench_all": {"acc": 0.5}},
            ):
                summary = materialize_records(
                    records,
                    output_dir=output,
                    molbench_root=Path(tmp) / "molbench",
                    source_run_dir=Path(tmp) / "source",
                    selected_suites={"molbench_ms1"},
                )

            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(summary["final_answer_count"], 1)
            self.assertEqual(summary["failure_count"], 1)
            payloads = json.loads((output / "preds/rdkit_bench/all.json").read_text())
            self.assertEqual(len(payloads), 2)
            failed = next(item for item in payloads if item["id"] == "task-2")
            self.assertEqual(failed["json_results"]["output"], "")


if __name__ == "__main__":
    unittest.main()
