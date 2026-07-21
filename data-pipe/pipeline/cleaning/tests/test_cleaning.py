from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_DIR))

from cleaning.acceptance_gate import decide_final_status  # noqa: E402
from cleaning.invariants import compare_immutable_facts, validate_final_record  # noqa: E402
from cleaning.llm_clean import apply_restricted_patch, clean_draft, llm_clean  # noqa: E402
from cleaning.models import EXAMPLE_DIR, patch_schema_findings, react_schema_findings  # noqa: E402


def sample_record() -> dict:
    return {
        "schema_version": "drug_agent_sft_react_json_v1",
        "id": "sample-1",
        "messages": [
            {"role": "system", "content": "protocol", "step_loss_mask": 0},
            {"role": "user", "content": "task", "step_loss_mask": 0},
            {
                "role": "assistant",
                "content": (
                    "<thought>Let me write result.md before continuing.</thought>\n"
                    '<tool_call>{"arguments":{},"tool_name":"x"}</tool_call>'
                ),
                "step_loss_mask": 1,
            },
            {
                "role": "user",
                "content": (
                    '<observation tool_name="x">'
                    '{"content":{"status":"success","value":1},"is_error":false,'
                    '"status":"success","tool_name":"x"}</observation>'
                ),
                "step_loss_mask": 0,
            },
            {
                "role": "assistant",
                "content": (
                    '<final_answer>{"evidence":[{"value":1}],"result":"done",'
                    '"summary":"Let me finish the report.","task_type":"kg"}</final_answer>'
                ),
                "step_loss_mask": 1,
            },
        ],
    }


def repair_hints() -> dict:
    return {
        "sample_id": "sample-1",
        "editable_findings": [
            {"message_index": 2, "segment_type": "thought", "segment_index": 0},
            {"message_index": 4, "segment_type": "final_summary", "segment_index": 0},
        ],
    }


class ContractTest(unittest.TestCase):
    def test_examples_validate_against_machine_schemas(self) -> None:
        trajectory = json.loads((EXAMPLE_DIR / "react_trajectory_v1.example.json").read_text(encoding="utf-8"))
        patch = json.loads((EXAMPLE_DIR / "llm_clean_patch_v1.example.json").read_text(encoding="utf-8"))
        self.assertEqual(react_schema_findings(trajectory), [])
        self.assertEqual(patch_schema_findings(patch), [])
        self.assertEqual(validate_final_record(trajectory)["errors"], [])


class RestrictedPatchTest(unittest.TestCase):
    def test_patch_changes_only_whitelisted_prose(self) -> None:
        source = sample_record()
        patch = {
            "schema_version": "llm_clean_patch_v1",
            "sample_id": "sample-1",
            "edits": [
                {
                    "message_index": 2,
                    "segment_type": "thought",
                    "segment_index": 0,
                    "replacement": "Use the recorded result to support the conclusion.",
                },
                {
                    "message_index": 4,
                    "segment_type": "final_summary",
                    "segment_index": 0,
                    "replacement": "The recorded computation completed successfully.",
                },
            ],
        }
        cleaned, findings, actions = apply_restricted_patch(source, patch, repair_hints())
        self.assertEqual(findings, [])
        self.assertEqual(len(actions), 2)
        self.assertEqual(compare_immutable_facts(source, cleaned), [])
        self.assertIn("recorded result", cleaned["messages"][2]["content"])
        self.assertIn("completed successfully", cleaned["messages"][4]["content"])

    def test_patch_cannot_target_unflagged_or_inject_protocol(self) -> None:
        source = sample_record()
        patch = {
            "schema_version": "llm_clean_patch_v1",
            "sample_id": "sample-1",
            "edits": [
                {
                    "message_index": 2,
                    "segment_type": "thought",
                    "segment_index": 1,
                    "replacement": "<tool_call>{}</tool_call>",
                }
            ],
        }
        cleaned, findings, actions = apply_restricted_patch(source, patch, repair_hints())
        self.assertEqual(cleaned, source)
        self.assertEqual(actions, [])
        self.assertTrue(any("edit_target_not_in_python_hints" in finding for finding in findings))

    def test_missing_patch_is_quarantined_by_one_final_gate(self) -> None:
        source = sample_record()
        audit = {
            "id": "sample-1",
            "python_status": "python_valid",
            "execution_valid": True,
            "task_answer_valid": True,
            "training_trace_valid": True,
            "repair_hints": repair_hints(),
        }

        def missing(_source, _hints):
            return None, {"status": "failed", "findings": ["missing_llm_clean_patch_file"]}

        result = clean_draft(source, audit, missing)
        self.assertEqual(result["audit"]["final_status"], "quarantine")
        self.assertEqual(result["audit"]["final_status_authority"], "final_acceptance_gate")
        self.assertEqual(result["record"], source)

    def test_final_outputs_include_python_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python_root = root / "python"
            python_root.mkdir()
            (python_root / "python_drafts.jsonl").write_text("", encoding="utf-8")
            rejected = {"id": "bad-1", "python_status": "rejected", "python_status_reasons": ["execution_invalid"]}
            (python_root / "python_audit.jsonl").write_text(json.dumps(rejected) + "\n", encoding="utf-8")
            (python_root / "rejected.jsonl").write_text(json.dumps(rejected) + "\n", encoding="utf-8")
            final_root = root / "final"
            manifest = llm_clean(python_root / "python_drafts.jsonl", final_root)
            self.assertEqual(manifest["rejected_count"], 1)
            self.assertEqual(len((final_root / "rejected.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len((final_root / "curation_audit.jsonl").read_text().splitlines()), 1)
            final_rejected = json.loads((final_root / "rejected.jsonl").read_text())
            self.assertEqual(final_rejected["final_status"], "rejected")
            self.assertEqual(final_rejected["final_status_authority"], "final_acceptance_gate")


class InvariantTest(unittest.TestCase):
    def test_detects_observation_status_and_final_conflicts(self) -> None:
        sample = sample_record()
        sample["messages"][3]["content"] = (
            '<observation tool_name="x">'
            '{"content":{"status":"error","error":"failed"},"is_error":false,'
            '"status":"success","tool_name":"x"}</observation>'
        )
        sample["messages"][4]["content"] = (
            '<final_answer>{"evidence":[],"result":"done",'
            '"summary":"The computation completed successfully.","task_type":"kg"}</final_answer>'
        )
        report = validate_final_record(sample)
        self.assertIn("observation_status_conflict", report["errors"])
        self.assertIn("final_claims_success_after_failed_critical_tool", report["errors"])

    def test_validation_is_read_only(self) -> None:
        sample = sample_record()
        before = copy.deepcopy(sample)
        validate_final_record(sample)
        self.assertEqual(sample, before)


class AcceptanceGateTest(unittest.TestCase):
    def test_only_gate_assigns_final_status(self) -> None:
        decision = decide_final_status(
            execution_valid=True,
            task_answer_valid=True,
            training_trace_valid=True,
            llm_clean_status="cleaned",
            llm_clean_findings=[],
            invariant_findings=[],
            llm_clean_required=True,
        )
        self.assertEqual(decision["final_status"], "accepted")
        self.assertEqual(decision["authority"], "final_acceptance_gate")


if __name__ == "__main__":
    unittest.main()
