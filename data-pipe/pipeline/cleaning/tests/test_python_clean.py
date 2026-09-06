from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from pipeline.cleaning.python_clean import python_clean


SLIME = Path(__file__).resolve().parents[4] / "slime-wd/slime"
sys.path.insert(0, str(SLIME))


class PythonCleanTest(unittest.TestCase):
    def test_parsed_answer_failure_does_not_invalidate_raw_semantic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            sample = run / "row0001_idx1"
            sample.mkdir(parents=True)
            (sample / "question.json").write_text('{"task":"kg","question":"Do task"}')
            (sample / "prompt.txt").write_text("prompt")
            (sample / "parsed_answer.json").write_text('{"parse_error":"collector parse failed"}')
            events = [
                {"type": "assistant", "message": {"id": "d1", "content": [{"type": "tool_use", "id": "c1", "name": "tool", "input": {}}]}},
                {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "c1", "content": {"status": "success"}}]}},
                {"type": "assistant", "message": {"id": "d2", "content": [{"type": "text", "text": "answer"}]}},
                {"type": "result", "subtype": "success", "is_error": False},
            ]
            session = sample / "complete_session.jsonl"
            session.write_text("".join(json.dumps(event) + "\n" for event in events))
            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()
            binding = {
                "question_sha256": digest(sample / "question.json"),
                "user_prompt_sha256": digest(sample / "prompt.txt"),
                "system_prompt_sha256": "system",
                "selected_session_sha256": digest(session),
                "source_dataset_sha256": "dataset",
            }
            (run / "run_config.json").write_text(json.dumps({
                "task": "kg",
                "system_prompt_sha256": "system",
                "source_dataset_sha256": "dataset",
            }))
            (sample / "run_meta.json").write_text(json.dumps({"return_code": 0, **binding}))
            (sample / "selected_attempt_artifacts.json").write_text(json.dumps(binding))
            manifest = python_clean(
                run,
                root / "out",
            )
            self.assertEqual(manifest["semantic_valid_count"], 1)
            audit = json.loads((root / "out/python_audit.jsonl").read_text())
            self.assertEqual(audit["status"], "semantic_valid")
            self.assertEqual(audit["parsed_answer_audit"]["parse_error"], "collector parse failed")
            pretty = json.loads((root / "out/semantic_trajectories.pretty.json").read_text())
            self.assertEqual(len(pretty), 1)
            self.assertEqual(pretty[0]["schema_version"], "drug_agent_semantic_trajectory_v1")

    def test_capture_hash_mismatch_rejects_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            sample = run / "row0001_idx1"
            sample.mkdir(parents=True)
            (run / "run_config.json").write_text('{"task":"kg","system_prompt_sha256":"system","source_dataset_sha256":"dataset"}')
            (sample / "question.json").write_text('{"task":"kg","question":"Do task"}')
            (sample / "prompt.txt").write_text("prompt")
            events = [
                {"type": "assistant", "message": {"id": "d1", "content": [{"type": "tool_use", "id": "c1", "name": "tool", "input": {}}]}},
                {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "c1", "content": {"status": "success"}}]}},
                {"type": "assistant", "message": {"id": "d2", "content": [{"type": "text", "text": "answer"}]}},
                {"type": "result", "subtype": "success", "is_error": False},
            ]
            (sample / "complete_session.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
            (sample / "run_meta.json").write_text(json.dumps({
                "return_code": 0,
                "question_sha256": "wrong",
                "user_prompt_sha256": "wrong",
                "system_prompt_sha256": "system",
                "selected_session_sha256": "wrong",
                "source_dataset_sha256": "dataset",
            }))
            manifest = python_clean(run, root / "out")
            self.assertEqual(manifest["semantic_valid_count"], 0)
            audit = json.loads((root / "out/python_audit.jsonl").read_text())
            self.assertIn("capture_hash_binding_mismatch", audit["execution_invalid_reasons"])
