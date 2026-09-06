from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pipeline.cleaning.llm_clean import apply_reasoning_patch, llm_clean


def semantic() -> dict:
    return {
        "schema_version": "drug_agent_semantic_trajectory_v1",
        "id": "sample",
        "user_task": "task",
        "events": [
            {"type": "assistant_decision", "source_message_id": "d1", "reasoning": "read L2 then act", "tool_calls": [{"name": "tool", "arguments": {}, "source_tool_use_id": "c1"}], "final_answer": None},
            {"type": "tool_observation", "name": "tool", "source_tool_use_id": "c1", "status": "success", "is_error": False, "content": "ok"},
            {"type": "assistant_decision", "source_message_id": "d2", "reasoning": "done", "tool_calls": [], "final_answer": "answer"},
        ],
        "metadata": {"task_type": "kg", "source_session_sha256": "source-sha"},
    }


class LlmCleanTest(unittest.TestCase):
    def test_patch_changes_only_reasoning_and_prepends_plan(self) -> None:
        source = semantic()
        patch = {
            "schema_version": "semantic_reasoning_patch_v1",
            "sample_id": "sample",
            "high_level_plan": {"decision_id": "d1", "text": "Validate the input, run the required analysis, then report evidence."},
            "reasoning_replacements": [{"decision_id": "d1", "replacement": "Run the required analysis."}],
        }
        candidate, findings, actions = apply_reasoning_patch(source, patch, require_high_level_plan=True)
        self.assertEqual(findings, [])
        self.assertTrue(candidate["events"][0]["reasoning"].startswith("High-level plan:"))
        self.assertEqual(candidate["events"][0]["tool_calls"], source["events"][0]["tool_calls"])
        self.assertEqual(len(actions), 2)

    def test_existing_server_absolute_path_is_allowed(self) -> None:
        source = semantic()
        source["events"][0]["reasoning"] = "Reuse /server/run-123/result.pdb."
        patch = {
            "schema_version": "semantic_reasoning_patch_v1",
            "sample_id": "sample",
            "high_level_plan": {"decision_id": "d1", "text": "Inspect the input, run the tool, and verify its result."},
            "reasoning_replacements": [{"decision_id": "d1", "replacement": "Reuse /server/run-123/result.pdb."}],
        }
        candidate, findings, _ = apply_reasoning_patch(source, patch, require_high_level_plan=True)
        self.assertEqual(findings, [])
        self.assertIn("/server/run-123/result.pdb", candidate["events"][0]["reasoning"])

    def test_forbidden_runtime_narration_and_duplicate_paragraph_are_rejected(self) -> None:
        for replacement, expected in (
            ("Read CLAUDE.md before acting.", "forbidden_teacher_narration:d1"),
            ("Run analysis.\n\nRun analysis.", "duplicate_consecutive_reasoning_paragraph:d1"),
        ):
            with self.subTest(replacement=replacement):
                patch = {
                    "schema_version": "semantic_reasoning_patch_v1",
                    "sample_id": "sample",
                    "high_level_plan": {"decision_id": "d1", "text": "Inspect the input, run the tool, and verify its result."},
                    "reasoning_replacements": [{"decision_id": "d1", "replacement": replacement}],
                }
                _, findings, _ = apply_reasoning_patch(semantic(), patch, require_high_level_plan=True)
                self.assertIn(expected, findings)

    def test_provider_failure_keeps_mother_dataset_and_marks_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "semantic.jsonl"
            input_path.write_text(json.dumps(semantic()) + "\n")
            manifest = llm_clean(
                input_path,
                root / "cleaned",
                patch_provider=lambda _source, _context: (None, {"findings": ["network_failure"]}),
            )
            self.assertEqual(manifest["cleaned_count"], 0)
            self.assertEqual(manifest["pending_count"], 1)
            pending = json.loads((root / "cleaned/llm_pending.jsonl").read_text())
            self.assertEqual(pending["source"], semantic())
            self.assertEqual(json.loads(input_path.read_text()), semantic())
