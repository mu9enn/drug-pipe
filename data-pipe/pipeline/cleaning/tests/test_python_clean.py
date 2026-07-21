from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_DIR))

from cleaning.python_clean import python_clean  # noqa: E402


class PythonCleanTest(unittest.TestCase):
    def test_python_step_emits_draft_without_final_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            sample = run / "row0001_idx0"
            sample.mkdir(parents=True)
            (run / "run_config.json").write_text('{"task":"kg"}', encoding="utf-8")
            (sample / "question.json").write_text(
                '{"task":"kg","question":"Run the grounded task","answer":[]}', encoding="utf-8"
            )
            (sample / "parsed_answer.json").write_text(
                '{"answer":{"result":"grounded result"}}', encoding="utf-8"
            )
            (sample / "run_meta.json").write_text('{"return_code":0}', encoding="utf-8")
            events = [
                {
                    "type": "assistant",
                    "message": {"content": [{
                        "type": "tool_use", "id": "c1",
                        "name": "mcp__molclaw-scp__x", "input": {},
                    }]},
                },
                {
                    "type": "user",
                    "message": {"content": [{
                        "type": "tool_result", "tool_use_id": "c1",
                        "content": {"status": "success", "result": "done"},
                    }]},
                },
            ]
            (sample / "complete_session.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            output = root / "python"
            manifest = python_clean(run, output)
            audit = json.loads((output / "python_audit.jsonl").read_text(encoding="utf-8"))
            draft = json.loads((output / "python_drafts.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(manifest["python_valid_count"], 1)
            self.assertEqual(audit["python_status"], "python_valid")
            self.assertNotIn("final_status", audit)
            self.assertEqual(set(draft), {"schema_version", "id", "messages"})
            self.assertNotIn("ground_truth", json.dumps(draft))

    def test_missing_parsed_answer_is_discovered_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            sample = run / "row0001_idx0"
            sample.mkdir(parents=True)
            (run / "run_config.json").write_text('{"task":"kg"}', encoding="utf-8")
            (sample / "question.json").write_text(
                '{"task":"kg","question":"Run the grounded task"}', encoding="utf-8"
            )
            (sample / "run_meta.json").write_text('{"return_code":0}', encoding="utf-8")
            (sample / "complete_session.jsonl").write_text('{}\n', encoding="utf-8")
            output = root / "python"
            manifest = python_clean(run, output)
            audit = json.loads((output / "python_audit.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(manifest["input_count"], 1)
            self.assertEqual(manifest["rejected_count"], 1)
            self.assertIn("missing_parsed_answer", audit["execution_invalid_reasons"])


if __name__ == "__main__":
    unittest.main()
