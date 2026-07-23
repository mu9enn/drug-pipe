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
from cleaning.invariants import collect_repair_hints, compare_immutable_facts, validate_final_record  # noqa: E402
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
    def test_empty_replacement_deletes_only_a_flagged_scaffolding_thought(self) -> None:
        source = sample_record()
        patch = {
            "schema_version": "llm_clean_patch_v1",
            "sample_id": "sample-1",
            "edits": [
                {
                    "message_index": 2,
                    "segment_type": "thought",
                    "segment_index": 0,
                    "replacement": "",
                }
            ],
        }
        cleaned, findings, actions = apply_restricted_patch(source, patch, repair_hints())
        self.assertEqual(findings, ["unresolved_repair_target:4:final_summary:0"])
        self.assertNotIn("<thought>", cleaned["messages"][2]["content"])
        self.assertIn("<tool_call>", cleaned["messages"][2]["content"])
        self.assertEqual(actions[0]["operation"], "delete")
        self.assertEqual(compare_immutable_facts(source, cleaned), [])

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

    def test_missing_patch_falls_back_and_is_accepted_by_python_gates(self) -> None:
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
        self.assertEqual(result["audit"]["final_status"], "accepted")
        self.assertEqual(result["audit"]["final_status_authority"], "final_acceptance_gate")
        self.assertEqual(result["audit"]["llm_clean"]["status"], "failed_fallback")
        self.assertEqual(result["record"], source)

    def test_remaining_premature_claim_is_audited_and_safe_rewrite_is_accepted(self) -> None:
        source = sample_record()
        source["messages"][2]["content"] = (
            "<thought>Transfer complete.</thought>\n"
            '<tool_call>{"arguments":{"file_path":"<artifact:structure/fixed.pdb>"},'
            '"tool_name":"server_file_to_base64"}</tool_call>'
        )
        source["messages"][3]["content"] = (
            '<observation tool_name="server_file_to_base64">'
            '{"content":{"status":"success","value":1},"is_error":false,'
            '"status":"success","tool_name":"server_file_to_base64"}</observation>'
        )
        source["messages"][4]["content"] = (
            '<final_answer>{"evidence":[{"value":1}],"result":"done",'
            '"summary":"The recorded transfer produced the requested artifact.",'
            '"task_type":"kg"}</final_answer>'
        )
        hints = collect_repair_hints(source)
        audit = {
            "id": "sample-1",
            "python_status": "python_valid",
            "execution_valid": True,
            "task_answer_valid": True,
            "training_trace_valid": True,
            "repair_hints": hints,
        }

        def no_rewrite(_source, _hints):
            return {"schema_version": "llm_clean_patch_v1", "sample_id": "sample-1", "edits": []}, {
                "status": "patch_received",
                "findings": [],
            }

        incomplete = clean_draft(source, audit, no_rewrite)
        self.assertEqual(incomplete["audit"]["final_status"], "accepted")
        self.assertEqual(incomplete["audit"]["llm_clean"]["status"], "incomplete_patch")
        self.assertTrue(incomplete["audit"]["final_invariants"]["prose_findings"])

        def safe_rewrite(_source, _hints):
            return {
                "schema_version": "llm_clean_patch_v1",
                "sample_id": "sample-1",
                "edits": [
                    {
                        "message_index": 2,
                        "segment_type": "thought",
                        "segment_index": 0,
                        "replacement": "Encode the repaired structure for transfer from the server.",
                    }
                ],
            }, {"status": "patch_received", "findings": []}

        accepted = clean_draft(source, audit, safe_rewrite)
        self.assertEqual(accepted["audit"]["final_status"], "accepted")
        self.assertEqual(compare_immutable_facts(source, accepted["record"]), [])

    def test_partial_safe_patch_is_applied_and_unresolved_targets_are_audited(self) -> None:
        source = sample_record()
        audit = {
            "id": "sample-1",
            "python_status": "python_valid",
            "execution_valid": True,
            "task_answer_valid": True,
            "training_trace_valid": True,
            "repair_hints": repair_hints(),
        }

        def partial(_source, _hints):
            return {
                "schema_version": "llm_clean_patch_v1",
                "sample_id": "sample-1",
                "edits": [
                    {
                        "message_index": 2,
                        "segment_type": "thought",
                        "segment_index": 0,
                        "replacement": "Use the scientific evidence to guide the next call.",
                    }
                ],
            }, {"status": "patch_received", "findings": []}

        result = clean_draft(source, audit, partial)
        self.assertEqual(result["audit"]["final_status"], "accepted")
        self.assertEqual(result["audit"]["llm_clean"]["status"], "partially_cleaned")
        self.assertIn("scientific evidence", result["record"]["messages"][2]["content"])
        self.assertIn(
            "unresolved_repair_target:4:final_summary:0",
            result["audit"]["llm_clean"]["findings"],
        )

    def test_no_hints_skips_provider_and_is_not_required(self) -> None:
        source = sample_record()
        called = False

        def should_not_run(_source, _hints):
            nonlocal called
            called = True
            raise AssertionError("provider must not run")

        result = clean_draft(
            source,
            {
                "id": "sample-1",
                "python_status": "python_valid",
                "execution_valid": True,
                "task_answer_valid": True,
                "training_trace_valid": True,
                "repair_hints": {"sample_id": "sample-1", "editable_findings": []},
            },
            should_not_run,
        )
        self.assertFalse(called)
        self.assertEqual(result["audit"]["llm_clean"]["status"], "not_required")
        self.assertEqual(result["audit"]["final_status"], "accepted")

    def test_unsafe_patch_falls_back_without_rejecting_sample(self) -> None:
        source = sample_record()

        def unsafe(_source, _hints):
            return {
                "schema_version": "llm_clean_patch_v1",
                "sample_id": "sample-1",
                "edits": [
                    {
                        "message_index": 2,
                        "segment_type": "thought",
                        "segment_index": 0,
                        "replacement": "<tool_call>{}</tool_call>",
                    }
                ],
            }, {"status": "patch_received", "findings": []}

        result = clean_draft(
            source,
            {
                "id": "sample-1",
                "python_status": "python_valid",
                "execution_valid": True,
                "task_answer_valid": True,
                "training_trace_valid": True,
                "repair_hints": repair_hints(),
            },
            unsafe,
        )
        self.assertEqual(result["record"], source)
        self.assertEqual(result["audit"]["llm_clean"]["status"], "unsafe_patch_fallback")
        self.assertEqual(result["audit"]["final_status"], "accepted")

    def test_invalid_python_draft_schema_fails_cleaning_command(self) -> None:
        source = sample_record()
        source["schema_version"] = "broken"
        with self.assertRaisesRegex(ValueError, "invalid Python-clean draft schema"):
            clean_draft(
                source,
                {
                    "id": "sample-1",
                    "python_status": "python_valid",
                    "execution_valid": True,
                    "task_answer_valid": True,
                    "training_trace_valid": True,
                    "repair_hints": {"editable_findings": []},
                },
                lambda _source, _hints: (None, {}),
            )

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
            self.assertNotIn("quarantine_count", manifest)
            self.assertNotIn("quarantine", manifest["outputs"])
            self.assertFalse((final_root / "quarantine.jsonl").exists())


class InvariantTest(unittest.TestCase):
    def test_hints_are_targeted_to_teacher_scaffolding_and_premature_claims(self) -> None:
        long_science = "ADRA2A receptor evidence remains relevant. " * 80
        sample = sample_record()
        sample["messages"][2]["content"] = (
            "<thought>Use the observed ADRA2A structure after consulting the L2 docking workflow.</thought>\n"
            f"<thought>{long_science}</thought>\n"
            "<thought>Transfer complete.</thought>\n"
            "<thought>Encode the repaired structure for transfer.</thought>\n"
            '<tool_call>{"arguments":{},"tool_name":"server_file_to_base64"}</tool_call>'
        )
        sample["messages"][4]["content"] = (
            '<final_answer>{"evidence":[{"value":1}],"result":"done",'
            '"summary":"ADRA2A was repaired after following the L3 methodology skill.",'
            '"task_type":"kg"}</final_answer>'
        )
        hints = collect_repair_hints(sample)["editable_findings"]
        targets = {(item["message_index"], item["segment_type"], item["segment_index"]): item for item in hints}
        self.assertIn((2, "thought", 0), targets)
        self.assertIn("l2_l3_teacher_orchestration", targets[(2, "thought", 0)]["reasons"])
        self.assertNotIn((2, "thought", 1), targets)
        self.assertIn((2, "thought", 2), targets)
        self.assertTrue(
            any(reason.startswith("premature_completion_before_tool") for reason in targets[(2, "thought", 2)]["reasons"])
        )
        self.assertNotIn((2, "thought", 3), targets)
        self.assertIn((4, "final_summary", 0), targets)

    def test_default_allows_l1_file_tools_and_only_molclaw_mode_hints_narration(self) -> None:
        sample = sample_record()
        sample["messages"][2]["content"] = (
            "<thought>Read the L1 tool skill, then append evidence to run_log.md "
            "and write result.md.</thought>\n"
            '<tool_call>{"arguments":{},"tool_name":"Read"}</tool_call>'
        )
        default_hints = collect_repair_hints(sample)
        self.assertEqual(default_hints["editable_findings"], [])

        only_hints = collect_repair_hints(sample, only_molclaw_tool=True)
        self.assertTrue(only_hints["only_molclaw_tool"])
        self.assertEqual(len(only_hints["editable_findings"]), 1)
        self.assertIn(
            "local_tool_narration_removed_in_only_molclaw_mode",
            only_hints["editable_findings"][0]["reasons"],
        )

    def test_teacher_sidecar_narration_is_cleaned_but_run_logs_are_allowed(self) -> None:
        sample = sample_record()
        sample["messages"][2]["content"] = (
            "<thought>Read question.json and inspect the skills directory before starting.</thought>\n"
            "<thought>Write the measured score to run_log.md.</thought>\n"
            '<tool_call>{"arguments":{},"tool_name":"x"}</tool_call>'
        )
        hints = collect_repair_hints(sample)["editable_findings"]
        self.assertEqual(
            [(item["message_index"], item["segment_index"]) for item in hints],
            [(2, 0)],
        )
        self.assertIn(
            "teacher_sidecar_or_skill_catalog_narration",
            hints[0]["reasons"],
        )

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
        )
        self.assertEqual(decision["final_status"], "accepted")
        self.assertEqual(decision["authority"], "final_acceptance_gate")

    def test_llm_and_invariant_findings_are_not_acceptance_inputs(self) -> None:
        decision = decide_final_status(
            execution_valid=True,
            task_answer_valid=True,
            training_trace_valid=True,
        )
        self.assertEqual(decision["final_status"], "accepted")
        self.assertEqual(decision["reasons"], [])
        with self.assertRaises(TypeError):
            decide_final_status(
                execution_valid=True,
                task_answer_valid=True,
                training_trace_valid=True,
                llm_clean_status="failed",
            )


if __name__ == "__main__":
    unittest.main()
