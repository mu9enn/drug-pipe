from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline.cleaning.llm_clean import apply_reasoning_patch, build_claude_patch_provider, llm_clean


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
    def test_deepseek_harness_patch_provider_uses_prose_curation_scene(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node = root / "node"
            node.write_text("#!/bin/sh\nprintf 'v24.19.0\\n'\n", encoding="utf-8")
            node.chmod(0o755)
            fake = root / "fake-dsh"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os\n"
                "from pathlib import Path\n"
                "source=json.loads(Path('source_trajectory.json').read_text())\n"
                "patch={'schema_version':'semantic_reasoning_patch_v1','sample_id':source['id'],'high_level_plan':{'decision_id':'d1','text':'Validate the input, run the required analysis, then report evidence.'},'reasoning_replacements':[{'decision_id':'d1','replacement':'Run the required analysis.'}]}\n"
                "Path('semantic_reasoning_patch.json').write_text(json.dumps(patch))\n"
                "session=Path(os.environ['DSH_HOME'])/'sessions/project/s/session.jsonl'\n"
                "session.parent.mkdir(parents=True)\n"
                "rows=[{'type':'session','version':0},{'type':'assistant/message','data':{'message':{'id':'a','content':[{'type':'text','text':'done'}]}}},{'type':'turn/end','data':{'reason':{'kind':'completed'}}}]\n"
                "session.write_text(''.join(json.dumps(row)+'\\n' for row in rows))\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            provider = build_claude_patch_provider(
                claude_bin="unused", debug_root=root / "debug", timeout_sec=5,
                harness="deepseek", dsh_bin=str(fake), dsh_node_bin=str(node),
            )
            with patch.dict(os.environ, {
                "DEEPSEEK_BASE_URL": "https://example.invalid/v1",
                "DEEPSEEK_API_KEY": "secret",
            }):
                result, report = provider(semantic(), {"require_high_level_plan": True})
            self.assertIsInstance(result, dict)
            self.assertEqual(report["status"], "patch_received")
            self.assertTrue((root / "debug/sample/.agents/skills/clean-drug-trajectory/SKILL.md").is_file())

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

    def test_prose_quality_findings_are_warnings(self) -> None:
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
                _, findings, actions = apply_reasoning_patch(semantic(), patch, require_high_level_plan=True)
                self.assertEqual(findings, [])
                self.assertIn(expected, [a.get("finding") for a in actions])

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
