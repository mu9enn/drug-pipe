from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pipeline.cleaning.reasoning_shorten import reasoning_shorten, shorten_semantic


def semantic(reasoning: str) -> dict:
    return {
        "schema_version": "drug_agent_semantic_trajectory_v1", "id": "sample", "user_task": "task",
        "events": [
            {"type": "assistant_decision", "source_message_id": "d1", "reasoning": reasoning,
             "tool_calls": [{"name": "tool", "arguments": {}, "source_tool_use_id": "c1"}], "final_answer": None},
            {"type": "tool_observation", "name": "tool", "source_tool_use_id": "c1", "status": "success",
             "is_error": False, "content": "ok"},
            {"type": "assistant_decision", "source_message_id": "d2", "reasoning": "done",
             "tool_calls": [], "final_answer": "answer"}],
        "metadata": {"task_type": "kg", "source_session_sha256": "source-sha"},
    }


def words(text: str) -> int:
    return len(text.split())


def patch_for(source: dict, replacement: str) -> dict:
    return {"schema_version": "reasoning_shorten_patch_v1", "sample_id": source["id"],
            "reasoning_replacements": [{"decision_id": "d1", "replacement": replacement}]}


class ReasoningShortenTest(unittest.TestCase):
    def test_legacy_only_replaces_selected_assistant_thought_spans(self) -> None:
        from pipeline.cleaning.reasoning_shorten import apply_shorten_patch
        source = {'id': 'legacy', 'messages': [
            {'role': 'user', 'content': '<thought>do not edit</thought>'},
            {'role': 'assistant', 'content': '<thought>long first thought</thought><tool_call>{"name":"x"}</tool_call><thought>keep me</thought>', 'step_loss_mask': 1}]}
        patch = {'schema_version': 'reasoning_shorten_patch_v1', 'sample_id': 'legacy',
                 'reasoning_replacements': [{'decision_id': 'm1_thought0', 'replacement': 'short'}]}
        result, findings = apply_shorten_patch(source, patch, {'m1_thought0'})
        self.assertEqual(findings, [])
        self.assertEqual(result['messages'][0], source['messages'][0])
        self.assertEqual(result['messages'][1]['content'], '<thought>short</thought><tool_call>{"name":"x"}</tool_call><thought>keep me</thought>')
        self.assertEqual(result['messages'][1]['step_loss_mask'], 1)

    def test_short_record_passes_without_llm_call(self) -> None:
        calls = []
        result = shorten_semantic(semantic("one two"), lambda *args: calls.append(args), words, token_limit=4)
        self.assertEqual(result["audit"]["status"], "not_required")
        self.assertEqual(calls, [])
        self.assertEqual(result["record"], semantic("one two"))

    def test_exact_limit_is_not_selected(self) -> None:
        source = semantic("one two three four")
        calls = []
        provider = lambda *args: calls.append(args)
        result = shorten_semantic(source, provider, words, token_limit=4)
        self.assertEqual(result["audit"]["status"], "not_required")
        self.assertEqual(calls, [])

    def test_one_pass_is_accepted_even_if_still_over_limit(self) -> None:
        calls = []
        def provider(row, _targets, _context):
            calls.append(row)
            return patch_for(row, "a b c d e"), {"status": "patch_received"}
        result = shorten_semantic(semantic("a b c d e f"), provider, words, token_limit=4)
        self.assertEqual(result["audit"]["status"], "processed_once")
        self.assertEqual(words(result["record"]["events"][0]["reasoning"]), 5)
        self.assertEqual(len(calls), 1)

    def test_no_reduction_is_still_accepted_after_one_pass(self) -> None:
        source = semantic("a b c d")
        provider = lambda row, _targets, _context: (patch_for(row, "a b c d"), {"status": "patch_received"})
        result = shorten_semantic(source, provider, words, token_limit=3)
        self.assertEqual(result["audit"]["status"], "processed_once")
        self.assertIsNotNone(result["record"])

    def test_pipeline_writes_pending_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); input_path = root / "input.jsonl"
            input_path.write_text(json.dumps(semantic("a b c d")) + "\n")
            provider = lambda _row, _targets, _context: (None, {"status": "failed", "findings": ["api_error"]})
            manifest = reasoning_shorten(input_path, root / "out", token_counter=words,
                                         patch_provider=provider, token_limit=3, target_tokens=2)
            self.assertEqual(manifest["accepted_count"], 0)
            self.assertEqual(manifest["pending_count"], 1)
            self.assertEqual((root / "out/semantic_trajectories.jsonl").read_text(), "")


if __name__ == "__main__":
    unittest.main()
