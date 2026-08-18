from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from drug_agent.evaluation.molbench_adapter import build_molbench_dataset
from drug_agent.evaluation.official_eval import (
    _load_eval_runner,
    _require_pytdc,
    _require_rdkit,
    run_official_evaluation,
)
from drug_agent.evaluation.preflight import _checkpoint_info, _validate_model_assets
from drug_agent.evaluation.prompt_adapter import build_prompt_suite_dataset, build_single_prompt_dataset
from drug_agent.evaluation.prompt_logger import log_eval_rollout_data
from drug_agent.protocol.react_protocol import parse_runtime_decision, project_final_answer


MOLBENCH_ROOT = Path(__file__).resolve().parents[3] / "molbench"


@unittest.skipUnless(MOLBENCH_ROOT.is_dir(), "local MolBench reference assets unavailable")
class MolBenchAdapterTest(unittest.TestCase):
    def test_builds_exact_held_out_fresh_task_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_molbench_dataset(MOLBENCH_ROOT, tmp)
            self.assertEqual(manifest["counts"], {
                "molbench_ms1": 50, "molbench_ms2": 33, "molbench_ms3": 25, "molbench_mo": 78,
            })
            rows = [json.loads(line) for line in (Path(tmp) / "molbench_eval.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 186)
            self.assertTrue(all([message["role"] for message in row["prompt"]] == ["system", "user"] for row in rows))
            self.assertTrue(all(row["metadata"]["env_kwargs"]["max_steps"] == 0 for row in rows))
            overlap_rows = (Path(tmp) / "overlap_audit.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(overlap_rows), 4)
            self.assertEqual(manifest["missing_mo_target_optimization"], 41)

    def test_adapter_labels_preserve_complete_source_smiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_molbench_dataset(MOLBENCH_ROOT, tmp)
            rows = [json.loads(line) for line in (Path(tmp) / "molbench_eval.jsonl").read_text().splitlines()]
            mo_rows = [row for row in rows if row["metadata"]["benchmark"]["suite"].startswith("molbench_mo_")]
            self.assertTrue(all(row["label"]["source_smiles"] for row in mo_rows))
            self.assertTrue(all(" " not in row["label"]["source_smiles"] for row in mo_rows))

    def test_selects_first_two_ms1_tasks_without_building_an_ad_hoc_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_molbench_dataset(
                MOLBENCH_ROOT,
                tmp,
                selected_suites=["molbench_ms1"],
                limit_per_suite=2,
            )
            rows = [json.loads(line) for line in (Path(tmp) / "molbench_eval.jsonl").read_text().splitlines()]
            self.assertEqual([row["id"] for row in rows], ["molbench_ms1_001", "molbench_ms1_002"])
            self.assertEqual(manifest["total"], 2)
            self.assertEqual(manifest["counts"], {
                "molbench_ms1": 2, "molbench_ms2": 0, "molbench_ms3": 0, "molbench_mo": 0,
            })
            self.assertEqual(manifest["source_counts"], {
                "molbench_ms1": 50, "molbench_ms2": 33, "molbench_ms3": 25, "molbench_mo": 78,
            })
            self.assertTrue(all([message["role"] for message in row["prompt"]] == ["system", "user"] for row in rows))

    def test_propagates_runtime_step_cap_into_every_molbench_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_molbench_dataset(
                MOLBENCH_ROOT,
                tmp,
                selected_suites=["molbench_ms1", "molbench_ms2"],
                limit_per_suite=2,
                max_steps=17,
            )
            rows = [json.loads(line) for line in (Path(tmp) / "molbench_eval.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["metadata"]["env_kwargs"]["max_steps"] == 17 for row in rows))
            manifest = json.loads((Path(tmp) / "benchmark_manifest.json").read_text())
            self.assertEqual(manifest["selection"]["max_steps"], 17)

    def test_selects_all_molecule_edit_tasks_without_molecule_optimization(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_molbench_dataset(
                MOLBENCH_ROOT,
                tmp,
                selected_suites=["molbench_mo_edit"],
            )
            rows = [json.loads(line) for line in (Path(tmp) / "molbench_eval.jsonl").read_text().splitlines()]
            self.assertEqual(manifest["total"], 39)
            self.assertEqual(manifest["counts"]["molbench_mo"], 39)
            self.assertTrue(all(row["metadata"]["benchmark"]["suite"] == "molbench_mo_edit" for row in rows))

    def test_official_wrapper_scores_a_ms1_only_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_dir = root / "preds/rdkit_bench"
            pred_dir.mkdir(parents=True)
            (pred_dir / "all.json").write_text(json.dumps([
                {"gt": "CCO", "json_results": {"output": "CCO"}},
                {"gt": "CCN", "json_results": {"output": "CCO"}},
            ]))
            fake_rdkit = types.ModuleType("rdkit")
            fake_rdkit.Chem = types.SimpleNamespace(
                MolFromSmiles=lambda value: object() if value in {"CCO", "CCN"} else None
            )
            with patch.dict(sys.modules, {"rdkit": fake_rdkit}):
                metrics = run_official_evaluation(root, MOLBENCH_ROOT)
            self.assertEqual(metrics["rdkit_bench_all"]["n_samples"], 2)
            self.assertEqual(metrics["rdkit_bench_all"]["acc"], 0.5)
            self.assertEqual(metrics["rdkit_bench_all"]["validity"], 1.0)
            self.assertEqual(set(metrics), {"rdkit_bench_all"})

    def test_ms1_evaluator_fails_explicitly_without_rdkit(self):
        with patch.dict(sys.modules, {"rdkit": None}):
            with self.assertRaisesRegex(
                RuntimeError, "RDKit is required for MolBench chemical evaluation"
            ):
                _require_rdkit()

    def test_mo_evaluator_fails_explicitly_without_pytdc(self):
        with patch.dict(sys.modules, {"tdc": None}):
            with self.assertRaisesRegex(
                RuntimeError, "PyTDC is required for MolBench molecule-optimization evaluation"
            ):
                _require_pytdc()

    def test_wrapper_uses_external_ac_and_vs_evaluator_classes(self):
        module = _load_eval_runner(MOLBENCH_ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ac = root / "ac"
            vs = root / "vs"
            ac.mkdir()
            vs.mkdir()
            (ac / "all.json").write_text(json.dumps([
                {"gt": "CCO", "s1": "CCO", "s2": "CCN", "json_results": {"output": "CCO"}},
                {"gt": "CCC", "s1": "CCC", "s2": "CCCl", "json_results": {"output": "CCCl"}},
            ]))
            (vs / "all.json").write_text(json.dumps([
                {"index": 7, "answer": ["B"], "candidates": ["A", "B", "C"],
                 "json_results": {"ranking": ["A", "B", "C"], "top3": ["A", "B", "C"]}}
            ]))
            ac_metrics = module.ACNetCuratedEval().run(str(ac), str(root), str(root))
            vs_metrics = module.MolbenchVsEval().run(str(vs), str(root), str(root))
            self.assertEqual(ac_metrics["acnet_curated_all"]["acc"], 0.5)
            self.assertEqual(vs_metrics["molbench_vs_all"]["hit_at_3"], 1.0)

    def test_molecule_optimization_final_projection(self):
        edit = parse_runtime_decision(
            '<thought>edit</thought><final_answer>{"task_type":"mol_edit","output_smiles":"CCO","evidence":[]}</final_answer>'
        )
        opt = parse_runtime_decision(
            '<thought>optimize</thought><final_answer>{"task_type":"mol_opt_physchem","optimized_smiles":"CCN","evidence":[]}</final_answer>'
        )
        self.assertTrue(edit["ok"])
        self.assertTrue(opt["ok"])
        self.assertEqual(project_final_answer(edit["final_answer"]), "CCO")
        self.assertEqual(project_final_answer(opt["final_answer"]), "CCN")


class CheckpointPreflightTest(unittest.TestCase):
    def test_requires_exact_slime_iteration_and_hf_tokenizer_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            (checkpoint / "iter_0000007").mkdir(parents=True)
            (checkpoint / "latest_checkpointed_iteration.txt").write_text("7\n")
            (checkpoint / "iter_0000007/common.pt").write_bytes(b"marker")
            info = _checkpoint_info(checkpoint)
            self.assertEqual(info["iteration"], 7)

            hf = root / "hf"
            hf.mkdir()
            (hf / "config.json").write_text("{}")
            (hf / "tokenizer_config.json").write_text("{}")
            model_args = root / "model.sh"
            model_args.write_text("MODEL_ARGS=()\n")
            assets = _validate_model_assets(hf, model_args)
            self.assertEqual(assets["hf_checkpoint"], str(hf))

            (checkpoint / "iter_0000007/common.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                _checkpoint_info(checkpoint)


class SinglePromptAdapterTest(unittest.TestCase):
    def test_builds_one_fresh_prompt_without_teacher_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "question.txt"
            prompt.write_text("Use a real tool, then report its observation.\n")

            manifest = build_single_prompt_dataset(
                prompt,
                root / "run",
                task_type="e2e",
                task_id="manual_test",
                max_steps=0,
            )

            self.assertEqual(manifest["sample_count"], 1)
            row = json.loads((root / "run/prompt_eval.jsonl").read_text())
            self.assertEqual([message["role"] for message in row["prompt"]], ["system", "user"])
            self.assertEqual(row["prompt"][1]["content"], "Use a real tool, then report its observation.")
            self.assertEqual(row["metadata"]["env_kwargs"]["task_type"], "e2e")
            self.assertEqual(row["metadata"]["env_kwargs"]["max_steps"], 0)
            self.assertNotIn("assistant", [message["role"] for message in row["prompt"]])

    def test_rejects_unknown_task_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "question.txt"
            prompt.write_text("question")
            with self.assertRaises(ValueError):
                build_single_prompt_dataset(prompt, Path(tmp) / "run", task_type="unknown")

    def test_builds_multi_prompt_suite_with_unique_fresh_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = root / "suite.json"
            suite.write_text(json.dumps([
                {"id": "one", "task_type": "e2e", "prompt": "first"},
                {"id": "two", "task_type": "kg", "prompt": "second"},
            ]))

            manifest = build_prompt_suite_dataset(suite, root / "run", max_steps=0)

            rows = [json.loads(line) for line in (root / "run/prompt_eval.jsonl").read_text().splitlines()]
            self.assertEqual(manifest["sample_count"], 2)
            self.assertEqual([row["id"] for row in rows], ["one", "two"])
            self.assertTrue(all([message["role"] for message in row["prompt"]] == ["system", "user"] for row in rows))
            self.assertTrue(all(row["metadata"]["env_kwargs"]["max_steps"] == 0 for row in rows))

    def test_prompt_logger_preserves_actions_observations_and_final(self):
        class Sample:
            status = "completed"
            metadata = {
                "manual_prompt": "test question",
                "env_kwargs": {"task_id": "manual_test", "task_type": "e2e"},
                "drug_agent_trace": {
                    "done_reason": "final_answer",
                    "actions": [{"raw_response": "<thought>inspect</thought>"}],
                    "observations": [{"tool_name": "fix_pdb", "status": "success"}],
                    "final_answer": {"task_type": "e2e", "result": "ok", "evidence": []},
                    "projected_final_answer": "ok",
                    "num_steps": 2,
                    "num_tool_success": 1,
                    "num_tool_error": 0,
                    "artifact_audit": {"path_map": {}},
                },
            }

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"DRUG_AGENT_EVAL_RUN_DIR": tmp}
        ):
            self.assertTrue(
                log_eval_rollout_data("rollout_1", None, {"single": {"samples": [Sample()]}}, {})
            )
            payload = json.loads((Path(tmp) / "prompt_result.json").read_text())
            self.assertEqual(payload["trace"]["actions"][0]["raw_response"], "<thought>inspect</thought>")
            self.assertEqual(payload["trace"]["observations"][0]["tool_name"], "fix_pdb")
            self.assertEqual(payload["result"]["projected_final_answer"], "ok")
