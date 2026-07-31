from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from drug_agent.evaluation.molbench_adapter import build_molbench_dataset
from drug_agent.evaluation.official_eval import _load_eval_runner
from drug_agent.evaluation.preflight import _checkpoint_info, _validate_model_assets
from drug_agent.protocol.react_protocol import parse_runtime_decision, project_final_answer


MOLBENCH_ROOT = Path(
    "/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug_wd/MolClaw/molbench"
)


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
