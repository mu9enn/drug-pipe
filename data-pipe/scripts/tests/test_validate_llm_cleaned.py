#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "validate_llm_cleaned.py"
SPEC = importlib.util.spec_from_file_location("validate_llm_cleaned", SCRIPT)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def message(role: str, content: str) -> dict:
    return {"role": role, "content": content}


class LlmCleanValidatorSmokeTest(unittest.TestCase):
    def _validate_text(self, text: str) -> dict:
        sample = {
            "schema_version": "drug_agent_sft_react_json_v1",
            "id": "mcp_sft_e2e_demo",
            "messages": [
                message("system", "s"),
                message("user", "Analyze the task."),
                message("assistant", f"<thought>{text}</thought>"),
                message("assistant", '<final_answer>{"task_type":"e2e","answer":"done","evidence":[]}</final_answer>'),
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            path.write_text(json.dumps(sample), encoding="utf-8")
            return validator._validate_file(path)

    def test_vs_ranking_and_exact_string_checks(self) -> None:
        sample = {
            "schema_version": "drug_agent_sft_react_json_v1",
            "id": "mcp_sft_vs_demo",
            "messages": [
                message("system", "s"),
                message("user", 'Question payload (MolBench-VS): {"candidates":["A","B"]}'),
                message("assistant", '<tool_call>{"tool_name":"quickvina","arguments":{"smiles":"A"}}</tool_call>'),
                message("user", '<observation tool_name="quickvina">{"ok":true,"status":"success","content":{"docking_affinity_value":-7.0}}</observation>'),
                message("assistant", '<tool_call>{"tool_name":"quickvina","arguments":{"smiles":"B"}}</tool_call>'),
                message("user", '<observation tool_name="quickvina">{"ok":true,"status":"success","content":{"docking_affinity_value":-7.2}}</observation>'),
                message("assistant", '<final_answer>{"task_type":"vs","ranked_smiles":["A","B"],"evidence":["scores"]}</final_answer>'),
            ],
        }
        path = Path(self._testMethodName + ".json")
        try:
            path.write_text(json.dumps(sample), encoding="utf-8")
            report = validator._validate_file(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertIn("vs_ranking_not_sorted_by_observed_docking_score", report["errors"])
        self.assertIn("vs_ranking_inconsistent_after_llm_clean", report["errors"])

    def test_fpocket_and_empty_evidence_checks(self) -> None:
        sample = {
            "schema_version": "drug_agent_sft_react_json_v1",
            "id": "mcp_sft_ac_demo",
            "messages": [
                message("system", "s"),
                message("user", "Molecule A: CCO\nMolecule B: CCC"),
                message("assistant", '<tool_call>{"tool_name":"fpocket_toolkit","arguments":{"smiles":"CCO"}}</tool_call>'),
                message("user", '<observation tool_name="fpocket_toolkit">{"ok":true,"status":"success","content":{"top_pocket":{"center":[1,2,3],"size":[1,2,3]}}}</observation>'),
                message("assistant", '<final_answer>{"task_type":"ac","answer_smiles":"CCO","evidence":[]}</final_answer>'),
            ],
        }
        path = Path(self._testMethodName + ".json")
        try:
            path.write_text(json.dumps(sample), encoding="utf-8")
            report = validator._validate_file(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertIn("fpocket_size_equals_center", report["errors"])
        self.assertIn("empty_evidence_with_success_observations", report["warnings"])

    def test_final_hard_clean_invariants(self) -> None:
        sample = {
            "schema_version": "drug_agent_sft_react_json_v1",
            "id": "mcp_sft_ac_demo",
            "messages": [
                message("system", "s"),
                message("user", "Molecule A: CCO\nMolecule B: CCC"),
                message(
                    "user",
                    '<observation tool_name="x">{"ok":true,"status":"success",'
                    '"content":{"error":"connection refused","path":"./protein_seq/ACE.json"},'
                    '"metadata":{"raw_is_error":true,"pointers":{"raw_pointer":"./protein_seq/ACE.json"}}}</observation>',
                ),
                message("assistant", '<final_answer>{"task_type":"ac","answer_smiles":"CCO","evidence":[]}</final_answer>'),
            ],
        }
        path = Path(self._testMethodName + ".json")
        try:
            path.write_text(json.dumps(sample), encoding="utf-8")
            report = validator._validate_file(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertIn("observation_debug_metadata_present", report["errors"])
        self.assertIn("local_relative_path", report["errors"])
        self.assertIn("observation_status_conflict_after_llm_clean", report["errors"])

    def test_pre_llm_mode_flags_instead_of_invalidating(self) -> None:
        sample = {
            "schema_version": "drug_agent_sft_react_json_v1",
            "id": "mcp_sft_ac_demo",
            "messages": [
                message("system", "s"),
                message("user", "Molecule A: CCO\nMolecule B: CCC"),
                message(
                    "user",
                    '<observation tool_name="x">{"ok":true,"status":"success",'
                    '"content":{"error":"connection refused"},"metadata":{"raw_is_error":true}}</observation>',
                ),
                message("assistant", '<final_answer>{"task_type":"ac","answer_smiles":"CCO","evidence":[]}</final_answer>'),
            ],
        }
        path = Path(self._testMethodName + ".json")
        try:
            path.write_text(json.dumps(sample), encoding="utf-8")
            report = validator._validate_file(path, mode="pre-llm")
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(report["status"], "flagged")
        self.assertEqual(report["errors"], [])
        self.assertTrue(report["needs_llm_semantic_repair"])
        self.assertIn("observation_status_conflict", report["repair_reasons"])
        self.assertNotIn("observation_status_conflict_after_llm_clean", report["warnings"])

    def test_post_llm_quarantine_moves_invalid_file(self) -> None:
        sample = {
            "schema_version": "drug_agent_sft_react_json_v1",
            "id": "mcp_sft_ac_demo",
            "messages": [
                message("system", "s"),
                message("user", "Molecule A: CCO\nMolecule B: CCC"),
                message("assistant", '<final_answer>{"task_type":"ac","answer_smiles":"CCO","evidence":[]}</final_answer>'),
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "bad.json"
            source.write_text(json.dumps({**sample, "audit": {}}), encoding="utf-8")
            item = validator._validate_file(source, mode="post-llm")
            self.assertEqual(item["status"], "invalid")
            quarantine = root / "quarantine"
            quarantine.mkdir()
            target = quarantine / source.name
            source.replace(target)
            self.assertFalse(source.exists())
            self.assertTrue(target.exists())

    def test_metric_interpretation_rules_are_narrow(self) -> None:
        for text in (
            "The ligand was converted to PDBQT.",
            "The molecule was converted to SDF.",
            "The molecule was converted to MOL2, PDB, and CIF.",
        ):
            with self.subTest(text=text):
                report = self._validate_text(text)
                self.assertNotIn("unsupported_metric_interpretation", report["warnings"])
        for text in (
            "affinity_pred_value is log10(IC50).",
            "The score was converted to IC50.",
            "The value corresponds to Ki.",
            "IC50 ≈ 10 nM.",
        ):
            with self.subTest(text=text):
                report = self._validate_text(text)
                self.assertIn("unsupported_metric_interpretation", report["warnings"])
                self.assertEqual(report["status"], "warning")
                self.assertTrue(report["is_training_candidate"])

    def test_engineering_level_requires_context(self) -> None:
        report = self._validate_text("The text happens to include L2.")
        self.assertNotIn("engineering_chatter:level_workflow", report["errors"])
        for text in ("I need to read the L2 workflow.", "Following the L2 methodology file..."):
            with self.subTest(text=text):
                report = self._validate_text(text)
                self.assertIn("engineering_chatter:level_workflow", report["errors"])
                self.assertEqual(report["status"], "invalid")
                self.assertFalse(report["is_training_candidate"])

    def test_report_candidate_policy(self) -> None:
        files = [
            {"status": "valid", "errors": [], "warnings": [], "is_training_candidate": True},
            {"status": "warning", "errors": [], "warnings": ["review"], "is_training_candidate": True},
            {"status": "invalid", "errors": ["bad"], "warnings": [], "is_training_candidate": False},
        ]
        report = validator._build_report(Path("examples"), files, "post-llm")
        self.assertEqual(report["valid"], 1)
        self.assertEqual(report["warning"], 1)
        self.assertEqual(report["invalid"], 1)
        self.assertEqual(report["candidate"], 2)
        self.assertEqual(report["excluded"], 1)
        self.assertEqual(report["candidate_policy"], "valid+warning included; invalid excluded")


if __name__ == "__main__":
    unittest.main()
