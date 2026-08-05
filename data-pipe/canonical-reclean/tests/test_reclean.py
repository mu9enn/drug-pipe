from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
from unittest import mock
from argparse import Namespace
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "reclean.py"
SPEC = importlib.util.spec_from_file_location("canonical_reclean_script", MODULE_PATH)
assert SPEC and SPEC.loader
reclean = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reclean
SPEC.loader.exec_module(reclean)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [abs(hash(token)) % 100000 for token in text.split()]


def record_with_thought(thought: str, *, record_id: str = "sample") -> dict:
    return {
        "schema_version": "drug_agent_sft_react_json_v1",
        "id": record_id,
        "messages": [
            {"role": "system", "content": "Use ReAct.", "step_loss_mask": 0},
            {"role": "user", "content": "Repair the protein.", "step_loss_mask": 0},
            {
                "role": "assistant",
                "content": (
                    f"<thought>{thought}</thought>"
                    '<tool_call>{"tool_name":"fix_pdb","arguments":{"input_path":"<artifact:protein_1>"}}</tool_call>'
                ),
                "step_loss_mask": 1,
            },
            {
                "role": "user",
                "content": '<observation tool_name="fix_pdb">{"status":"success","output_file":"<artifact:protein_2>"}</observation>',
                "step_loss_mask": 0,
            },
            {
                "role": "assistant",
                "content": '<thought>The repaired structure is available.</thought><final_answer>{"task_type":"kg","result":"<artifact:protein_2>","summary":"Repair completed."}</final_answer>',
                "step_loss_mask": 1,
            },
        ],
    }


def catalog(path: Path) -> None:
    path.write_text(
        json.dumps({"tools": [{"name": "fix_pdb"}]}, ensure_ascii=False),
        encoding="utf-8",
    )


def fake_claude(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json, pathlib, sys
root = pathlib.Path.cwd()
request = json.loads((root / 'request.json').read_text())
schema = request.get('schema_version')
if schema == 'canonical_reclean_chunk_request_v1':
    value = {
      'schema_version': 'canonical_reclean_chunk_notes_v1',
      'record_id': request['record_id'],
      'coordinate': request['coordinate'],
      'chunk_index': request['chunk_index'],
      'unique_content': 'Use the observed failure to repair the structure.'
    }
    (root / 'chunk_notes.json').write_text(json.dumps(value))
elif schema == 'canonical_reclean_reduce_request_v1':
    c = request['coordinate']
    value = {'schema_version':'canonical_reclean_review_v1','record_id':request['record_id'],
      'reviews':[dict(c, action='replace', replacement='Use the observed failure to repair the structure.', rationale='Deduplicated.')]
    }
    (root / 'review.json').write_text(json.dumps(value))
else:
    reviews=[]
    for segment in request['segments']:
      c=segment['coordinate']
      if c['segment_type']=='final_summary':
        reviews.append(dict(c, action='keep', rationale='Already concise.'))
      else:
        reviews.append(dict(c, action='replace', replacement='Use the observed evidence to perform the next action.', rationale='Deduplicated.'))
    value={'schema_version':'canonical_reclean_review_v1','record_id':request['record_id'],'reviews':reviews}
    (root / 'review.json').write_text(json.dumps(value))
print(json.dumps({'type':'result','subtype':'success','is_error':False}))
print(json.dumps({'type':'diagnostic','stream':'stderr'}), file=sys.stderr)
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def flaky_claude(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json, pathlib
root=pathlib.Path.cwd()
request=json.loads((root/'request.json').read_text())
if root.parent.name == 'pass_01':
    (root/'review.json').write_text('{bad json')
else:
    reviews=[]
    for segment in request['segments']:
      c=segment['coordinate']
      if c['segment_type']=='final_summary': reviews.append(dict(c,action='keep'))
      else: reviews.append(dict(c,action='replace',replacement='Use the evidence to repair the structure.'))
    (root/'review.json').write_text(json.dumps({'schema_version':'canonical_reclean_review_v1','record_id':request['record_id'],'reviews':reviews}))
print(json.dumps({'type':'result','is_error':False}))
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def sleeping_claude(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import time
time.sleep(5)
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def quota_failed_claude(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json, sys
print(json.dumps({'type':'result','subtype':'success','is_error':True,'api_error_status':403,
 'result':'Failed to authenticate. API Error: 403 quota exceeded'}))
sys.exit(1)
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class CoreTests(unittest.TestCase):
    def test_repeat_quality_gate_and_safe_rewrite(self):
        source = record_with_thought(("repeat block " * 40 + "\n\n") * 4)
        replacement = {
            (2, "thought", 0): {
                "message_index": 2,
                "segment_type": "thought",
                "segment_index": 0,
                "action": "replace",
                "replacement": "Repair the structure using the observed input artifact.",
            },
            (4, "thought", 0): {
                "message_index": 4,
                "segment_type": "thought",
                "segment_index": 0,
                "action": "keep",
            },
            (4, "final_summary", 0): {
                "message_index": 4,
                "segment_type": "final_summary",
                "segment_index": 0,
                "action": "keep",
            },
        }
        candidate = reclean.apply_reviews(source, replacement)
        self.assertEqual(reclean.compare_immutable_facts(source, candidate), [])
        report = reclean.quality_report(source, candidate, FakeTokenizer(), {"fix_pdb"})
        self.assertTrue(report["valid"], report["findings"])

    def test_review_requires_exact_coverage_and_rejects_protocol(self):
        source = record_with_thought("Inspect the structure.")
        segments = reclean.editable_segments(source)
        value = {
            "schema_version": reclean.REVIEW_SCHEMA_VERSION,
            "record_id": "sample",
            "reviews": [
                {
                    **reclean.asdict(segments[0].coordinate),
                    "action": "replace",
                    "replacement": "<tool_call>bad</tool_call>",
                }
            ],
        }
        _reviews, findings = reclean.validate_review(value, "sample", segments)
        self.assertTrue(any("protocol_tag" in item for item in findings))
        self.assertTrue(any("missing_review" in item for item in findings))

    def test_oversized_split_is_bounded_and_overlapping(self):
        text = "\n\n".join(f"paragraph-{index} " + "x" * 500 for index in range(20))
        chunks = reclean.split_paragraph_chunks(text, 2000, overlap_chars=100)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 2100 for chunk in chunks))
        self.assertEqual(chunks[1][:100], chunks[0][-100:])

    def test_context_excerpt_is_bounded_and_hashed(self):
        text = "a" * 20000
        excerpt = reclean.context_excerpt(text, 1000)
        self.assertTrue(excerpt["truncated"])
        self.assertLess(len(excerpt["text"]), 1200)
        self.assertEqual(excerpt["source_characters"], 20000)
        self.assertEqual(excerpt["source_sha256"], reclean.sha256_bytes(text.encode()))

    def test_final_summary_cannot_be_deleted(self):
        source = record_with_thought("Inspect the structure.")
        segment = [item for item in reclean.editable_segments(source) if item.coordinate.segment_type == "final_summary"]
        value = {
            "schema_version": reclean.REVIEW_SCHEMA_VERSION,
            "record_id": "sample",
            "reviews": [{**reclean.asdict(segment[0].coordinate), "action": "delete"}],
        }
        _reviews, findings = reclean.validate_review(value, "sample", segment)
        self.assertTrue(any("final_summary_delete_forbidden" in item for item in findings))

    def test_equivalent_nested_coordinate_is_normalized(self):
        source = record_with_thought("Inspect the structure.")
        segment = reclean.editable_segments(source)[0]
        value = {
            "schema_version": reclean.REVIEW_SCHEMA_VERSION,
            "record_id": "sample",
            "reviews": [
                {
                    "coordinate": reclean.asdict(segment.coordinate),
                    "action": "replace",
                    "replacement": "Repair the observed structure.",
                }
            ],
        }
        reviews, findings = reclean.validate_review(value, "sample", [segment])
        self.assertEqual(findings, [])
        self.assertIn(segment.coordinate.key, reviews)


class FakeClaudeTests(unittest.TestCase):
    def test_record_review_archives_raw_stream_and_preserves_facts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "fake-claude"
            fake_claude(fake)
            catalog_path = root / "catalog.json"
            catalog(catalog_path)
            provider = reclean.provider_snapshot()
            args = Namespace(
                run_root=root / "run",
                tool_catalog=catalog_path,
                claude_bin=str(fake),
                timeout_sec=30.0,
                max_attempts=2,
                batch_max_chars=10000,
                chunk_max_chars=5000,
                resume=False,
            )
            runner = reclean.RecleanRunner(args, FakeTokenizer(), provider)
            source = record_with_thought("Repeated. Repeated. Repeated.")
            result = runner.process_record(source, 0)
            self.assertEqual(result["status"], "ready")
            self.assertEqual(reclean.compare_immutable_facts(source, result["record"]), [])
            unit = root / "run/work/0000_sample/pass_01/batch_0000"
            selected = unit / "complete_session.jsonl"
            attempt = unit / "attempts/attempt_0001/complete_session.jsonl"
            self.assertTrue(selected.is_file())
            self.assertEqual(selected.read_bytes(), attempt.read_bytes())
            self.assertIn(b'"stream": "stderr"', selected.read_bytes())

    def test_oversized_thought_uses_map_reduce(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "fake-claude"
            fake_claude(fake)
            catalog_path = root / "catalog.json"
            catalog(catalog_path)
            args = Namespace(
                run_root=root / "run",
                tool_catalog=catalog_path,
                claude_bin=str(fake),
                timeout_sec=30.0,
                max_attempts=1,
                batch_max_chars=900,
                chunk_max_chars=500,
                resume=False,
            )
            runner = reclean.RecleanRunner(args, FakeTokenizer(), reclean.provider_snapshot())
            source = record_with_thought("\n\n".join("science " * 80 for _ in range(8)))
            result = runner.process_record(source, 0)
            self.assertEqual(result["status"], "ready", result.get("findings"))
            oversized = root / "run/work/0000_sample/pass_01/oversized_0000"
            self.assertTrue((oversized / "map_0000/complete_session.jsonl").is_file())
            self.assertTrue((oversized / "reduce/complete_session.jsonl").is_file())
            self.assertEqual(reclean.compare_immutable_facts(source, result["record"]), [])

    def test_bad_json_is_retried_and_ready_result_resumes_without_new_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "flaky-claude"
            flaky_claude(fake)
            catalog_path = root / "catalog.json"
            catalog(catalog_path)
            args = Namespace(
                run_root=root / "run",
                tool_catalog=catalog_path,
                claude_bin=str(fake),
                timeout_sec=30.0,
                max_attempts=2,
                batch_max_chars=10000,
                chunk_max_chars=5000,
                resume=False,
            )
            runner = reclean.RecleanRunner(args, FakeTokenizer(), reclean.provider_snapshot())
            source = record_with_thought("Repeated progress.")
            result = runner.process_record(source, 0)
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["attempts_used"], 2)
            args.resume = True
            fake.unlink()
            resumed = runner.process_record(source, 0)
            self.assertEqual(resumed["status"], "ready")

    def test_provider_change_fails_before_invocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "fake-claude"
            fake_claude(fake)
            expected = reclean.provider_snapshot()
            changed = {**expected, "provider_id": "changed", "fingerprint": "changed"}
            with mock.patch.object(reclean, "provider_snapshot", return_value=changed):
                with self.assertRaisesRegex(RuntimeError, "provider changed"):
                    reclean.invoke_claude(
                        workdir=root / "work",
                        request={"schema_version": "x"},
                        output_name="review.json",
                        prompt_name="review.md",
                        claude_bin=str(fake),
                        timeout_sec=1.0,
                        provider=expected,
                    )

    def test_timeout_preserves_attempt_without_runner_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "sleeping-claude"
            sleeping_claude(fake)
            value, metadata = reclean.invoke_claude(
                workdir=root / "work",
                request={"schema_version": "x"},
                output_name="review.json",
                prompt_name="review.md",
                claude_bin=str(fake),
                timeout_sec=0.05,
                provider=reclean.provider_snapshot(),
            )
            self.assertIsNone(value)
            self.assertEqual(metadata["failure"], "claude_timeout")
            raw = root / "work/attempts/attempt_0001/complete_session.jsonl"
            self.assertTrue(raw.is_file())
            self.assertNotIn(b"runner", raw.read_bytes())

    def test_quota_failure_is_global_fatal_after_raw_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "quota-claude"
            quota_failed_claude(fake)
            with self.assertRaises(reclean.FatalProviderError) as raised:
                reclean.invoke_claude(
                    workdir=root / "work",
                    request={"schema_version": "x"},
                    output_name="review.json",
                    prompt_name="review.md",
                    claude_bin=str(fake),
                    timeout_sec=1.0,
                    provider=reclean.provider_snapshot(),
                )
            self.assertEqual(
                raised.exception.metadata["fatal_provider_error"]["api_error_status"], 403
            )
            self.assertTrue((root / "work/attempts/attempt_0001/complete_session.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
