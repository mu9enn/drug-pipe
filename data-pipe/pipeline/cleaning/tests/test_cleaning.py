from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_DIR))

from cleaning.acceptance_gate import decide_final_status  # noqa: E402
from cleaning.invariants import compare_immutable_facts, validate_final_record  # noqa: E402
from cleaning.llm_clean import (  # noqa: E402
    apply_restricted_patch,
    build_claude_patch_provider,
    clean_draft,
    llm_clean,
)
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


class ContractTest(unittest.TestCase):
    def test_examples_validate_against_machine_schemas(self) -> None:
        trajectory = json.loads((EXAMPLE_DIR / "react_trajectory_v1.example.json").read_text(encoding="utf-8"))
        patch = json.loads((EXAMPLE_DIR / "llm_clean_patch_v1.example.json").read_text(encoding="utf-8"))
        self.assertEqual(react_schema_findings(trajectory), [])
        self.assertEqual(patch_schema_findings(patch), [])
        self.assertEqual(validate_final_record(trajectory)["errors"], [])


class RestrictedPatchTest(unittest.TestCase):
    def test_empty_replacement_deletes_only_the_targeted_thought(self) -> None:
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
        cleaned, findings, actions = apply_restricted_patch(source, patch)
        self.assertEqual(findings, [])
        self.assertNotIn("<thought>", cleaned["messages"][2]["content"])
        self.assertIn("<tool_call>", cleaned["messages"][2]["content"])
        self.assertEqual(actions[0]["operation"], "delete")
        self.assertEqual(compare_immutable_facts(source, cleaned), [])

    def test_patch_changes_only_existing_editable_prose(self) -> None:
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
        cleaned, findings, actions = apply_restricted_patch(source, patch)
        self.assertEqual(findings, [])
        self.assertEqual(len(actions), 2)
        self.assertEqual(compare_immutable_facts(source, cleaned), [])
        self.assertIn("recorded result", cleaned["messages"][2]["content"])
        self.assertIn("completed successfully", cleaned["messages"][4]["content"])

    def test_patch_cannot_inject_protocol_tags(self) -> None:
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
        cleaned, findings, actions = apply_restricted_patch(source, patch)
        self.assertEqual(cleaned, source)
        self.assertEqual(actions, [])
        self.assertTrue(any("replacement_contains_protocol_tag" in finding for finding in findings))

    def test_missing_patch_falls_back_and_is_accepted_by_python_gates(self) -> None:
        source = sample_record()
        audit = {
            "id": "sample-1",
            "python_status": "python_valid",
            "execution_valid": True,
            "task_answer_valid": True,
            "training_trace_valid": True,
        }

        def missing(_source, _hints):
            return None, {"status": "failed", "findings": ["missing_llm_clean_patch_file"]}

        result = clean_draft(source, audit, missing)
        self.assertEqual(result["audit"]["final_status"], "accepted")
        self.assertEqual(result["audit"]["final_status_authority"], "final_acceptance_gate")
        self.assertEqual(result["audit"]["llm_clean"]["status"], "failed_fallback")
        self.assertEqual(result["record"], source)

class ClaudePatchCaptureTest(unittest.TestCase):
    def _fake_claude(self, root: Path, *, emit_stream: bool = True) -> Path:
        executable = root / "fake-claude"
        stream_line = (
            "print('{\"type\":\"result\",\"result\":\"patch written\"}', flush=True)\n"
            if emit_stream else ""
        )
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "from pathlib import Path\n"
            "source = json.loads(Path('source_trajectory.json').read_text())\n"
            "Path('llm_clean_patch.json').write_text(json.dumps({"
            "'schema_version':'llm_clean_patch_v1','sample_id':source['id'],'edits':[]}))\n"
            + stream_line,
            encoding="utf-8",
        )
        executable.chmod(0o755)
        return executable

    def test_provider_keeps_raw_stream_and_reads_written_patch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            provider = build_claude_patch_provider(
                claude_bin=str(self._fake_claude(root)),
                debug_root=root / "debug",
                timeout_sec=5,
            )
            patch, metadata = provider(sample_record(), {"only_molclaw_tool": False})
            self.assertIsNotNone(patch)
            self.assertEqual(metadata["status"], "patch_received")
            self.assertTrue(metadata["raw_session_valid"])
            attempt = Path(metadata["attempt_session_file"])
            canonical = Path(metadata["session_file"])
            self.assertEqual(attempt.read_bytes(), canonical.read_bytes())
            self.assertFalse((canonical.parent / "claude_stdout.txt").exists())
            self.assertFalse((canonical.parent / "claude_stderr.txt").exists())
            self.assertTrue((canonical.parent / "cleaning_context.json").is_file())
            self.assertFalse((canonical.parent / "repair_hints.json").exists())

    def test_invalid_raw_stream_falls_back_even_if_patch_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            provider = build_claude_patch_provider(
                claude_bin=str(self._fake_claude(root, emit_stream=False)),
                debug_root=root / "debug",
                timeout_sec=5,
            )
            patch, metadata = provider(sample_record(), {"only_molclaw_tool": False})
            self.assertIsNone(patch)
            self.assertIn("raw_session_invalid", metadata["findings"])
            self.assertTrue(Path(metadata["attempt_session_file"]).is_file())

    def test_safe_partial_patch_is_applied_without_python_target_allowlist(self) -> None:
        source = sample_record()
        audit = {
            "id": "sample-1",
            "python_status": "python_valid",
            "execution_valid": True,
            "task_answer_valid": True,
            "training_trace_valid": True,
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
        self.assertEqual(result["audit"]["llm_clean"]["status"], "cleaned")
        self.assertIn("scientific evidence", result["record"]["messages"][2]["content"])
        self.assertEqual(result["audit"]["llm_clean"]["findings"], [])

    def test_every_python_valid_sample_calls_provider_and_empty_patch_is_not_required(self) -> None:
        source = sample_record()
        called = False

        def empty_patch(_source, context):
            nonlocal called
            called = True
            self.assertFalse(context["only_molclaw_tool"])
            return {
                "schema_version": "llm_clean_patch_v1",
                "sample_id": "sample-1",
                "edits": [],
            }, {"status": "patch_received", "findings": []}

        result = clean_draft(
            source,
            {
                "id": "sample-1",
                "python_status": "python_valid",
                "execution_valid": True,
                "task_answer_valid": True,
                "training_trace_valid": True,
            },
            empty_patch,
        )
        self.assertTrue(called)
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

    def test_llm_clean_uses_worker_pool_and_preserves_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python_root = root / "python"
            python_root.mkdir()
            records = []
            audits = []
            for index in range(3):
                record = sample_record()
                record["id"] = f"sample-{index}"
                records.append(record)
                audits.append({
                    "id": record["id"],
                    "python_status": "python_valid",
                    "execution_valid": True,
                    "task_answer_valid": True,
                    "training_trace_valid": True,
                })
            (python_root / "python_drafts.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
            )
            (python_root / "python_audit.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in audits), encoding="utf-8"
            )

            lock = threading.Lock()
            active = 0
            peak = 0

            def provider(source, _context):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.03)
                with lock:
                    active -= 1
                return {
                    "schema_version": "llm_clean_patch_v1",
                    "sample_id": source["id"],
                    "edits": [],
                }, {"status": "patch_received", "findings": []}

            output_root = root / "final"
            manifest = llm_clean(
                python_root / "python_drafts.jsonl",
                output_root,
                max_workers=2,
                patch_provider=provider,
            )
            output_ids = [
                json.loads(line)["id"]
                for line in (output_root / "react_trajectories.jsonl").read_text().splitlines()
            ]
            self.assertEqual(peak, 2)
            self.assertEqual(output_ids, ["sample-0", "sample-1", "sample-2"])
            self.assertEqual(manifest["max_workers"], 2)


class InvariantTest(unittest.TestCase):
    def test_allows_concurrent_tool_observations_to_return_out_of_call_order(self) -> None:
        sample = sample_record()
        sample["messages"][2]["content"] = (
            '<tool_call>{"arguments":{"index":1},"tool_name":"dock"}</tool_call>\n'
            '<tool_call>{"arguments":{"path":"notes.txt"},"tool_name":"Write"}</tool_call>'
        )
        sample["messages"][3]["content"] = (
            '<observation tool_name="Write">'
            '{"content":{"status":"success"},"is_error":false,"status":"success",'
            '"tool_name":"Write"}</observation>\n'
            '<observation tool_name="dock">'
            '{"content":{"status":"success","value":1},"is_error":false,'
            '"status":"success","tool_name":"dock"}</observation>'
        )
        report = validate_final_record(sample)
        self.assertNotIn("tool_observation_order_mismatch", report["errors"])
        self.assertNotIn("missing_observations:1", report["errors"])
        self.assertIn("observation_returned_out_of_call_order", report["warnings"])
        self.assertEqual(report["out_of_order_observations"][0]["tool_name"], "Write")

    def test_prose_audit_targets_teacher_orchestration_without_timeline_rules(self) -> None:
        sample = sample_record()
        sample["messages"][2]["content"] = (
            "<thought>Use the observed ADRA2A structure after consulting the L2 docking workflow.</thought>\n"
            "<thought>Transfer complete.</thought>\n"
            '<tool_call>{"arguments":{},"tool_name":"server_file_to_base64"}</tool_call>'
        )
        sample["messages"][4]["content"] = (
            '<final_answer>{"evidence":[{"value":1}],"result":"done",'
            '"summary":"ADRA2A was repaired after following the L3 methodology skill.",'
            '"task_type":"kg"}</final_answer>'
        )
        findings = validate_final_record(sample)["prose_findings"]
        targets = {
            (item["message_index"], item["segment_type"], item["segment_index"]): item
            for item in findings
        }
        self.assertIn((2, "thought", 0), targets)
        self.assertIn("l2_l3_teacher_orchestration", targets[(2, "thought", 0)]["reasons"])
        self.assertNotIn((2, "thought", 1), targets)
        self.assertIn((4, "final_summary", 0), targets)

    def test_default_allows_l1_file_tools_and_logs(self) -> None:
        sample = sample_record()
        sample["messages"][2]["content"] = (
            "<thought>Read the L1 tool skill, then append evidence to run_log.md "
            "and write result.md.</thought>\n"
            '<tool_call>{"arguments":{},"tool_name":"Read"}</tool_call>'
        )
        self.assertEqual(validate_final_record(sample)["prose_findings"], [])

    def test_teacher_sidecar_narration_is_cleaned_but_run_logs_are_allowed(self) -> None:
        sample = sample_record()
        sample["messages"][2]["content"] = (
            "<thought>Read question.json and inspect the skills directory before starting.</thought>\n"
            "<thought>Write the measured score to run_log.md.</thought>\n"
            '<tool_call>{"arguments":{},"tool_name":"x"}</tool_call>'
        )
        hints = validate_final_record(sample)["prose_findings"]
        self.assertEqual(
            [(item["message_index"], item["segment_index"]) for item in hints],
            [(2, 0)],
        )
        self.assertIn(
            "teacher_sidecar_or_skill_catalog_narration",
            hints[0]["reasons"],
        )

    def test_detects_observation_status_but_not_final_success_after_tool_error(self) -> None:
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
        self.assertNotIn("final_claims_success_after_failed_critical_tool", report["errors"])

    def test_validation_is_read_only(self) -> None:
        sample = sample_record()
        before = copy.deepcopy(sample)
        validate_final_record(sample)
        self.assertEqual(sample, before)

    def test_malformed_artifact_reference_is_reported(self) -> None:
        sample = sample_record()
        sample["messages"][2]["content"] = (
            "<thought>Inspect <artifact:structure/truncated.pdb before continuing.</thought>\n"
            '<tool_call>{"arguments":{},"tool_name":"x"}</tool_call>'
        )
        report = validate_final_record(sample)
        self.assertIn("message_2_malformed_artifact_reference", report["errors"])


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
