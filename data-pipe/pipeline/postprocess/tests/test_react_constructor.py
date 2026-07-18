from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PIPELINE_DIR / "postprocess"))

from cleaning.hard_cleaner import hard_clean  # noqa: E402
from react_constructor import reconstruct_react_messages  # noqa: E402
from trace_curator import discover_rollout_samples  # noqa: E402


def assistant_event(*items: dict) -> dict:
    return {"type": "assistant", "message": {"content": list(items)}}


def user_event(*items: dict) -> dict:
    return {"type": "user", "message": {"content": list(items)}}


class ReactConstructorTest(unittest.TestCase):
    def test_discovers_complete_session_without_parsed_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            row = results / "row0001_idx0"
            row.mkdir()
            (row / "complete_session.jsonl").write_text("{}\n", encoding="utf-8")
            samples = discover_rollout_samples(results)
            self.assertEqual([sample.sample_dir for sample in samples], [row])

    def test_preserves_reasoning_uses_bare_names_and_sanitizes_paths(self) -> None:
        events = [
            assistant_event({"type": "text", "text": "First inspect /home/user/work/input.pdb."}),
            assistant_event(
                {"type": "thinking", "thinking": "Now validate the structure."},
                {"type": "tool_use", "id": "w1", "name": "Read", "input": {"path": "/tmp/debug"}},
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "mcp__molclaw-scp__is_valid_smiles",
                    "input": {"smiles": "CCO"},
                },
            ),
            user_event(
                {"type": "tool_result", "tool_use_id": "w1", "content": "workspace chatter"},
                {"type": "tool_result", "tool_use_id": "c1", "content": {"status": "success", "valid": True}},
            ),
            assistant_event({"type": "thinking", "thinking": "The recorded result supports the conclusion."}),
        ]
        messages, stats = reconstruct_react_messages(
            events,
            question_text="Validate CCO",
            final_answer={"answer": "valid"},
            task="kg",
        )
        rendered = "\n".join(message["content"] for message in messages)
        self.assertIn("First inspect", rendered)
        self.assertIn('"tool_name":"is_valid_smiles"', rendered)
        self.assertNotIn("mcp__molclaw-scp__", rendered)
        self.assertNotIn("workspace chatter", rendered)
        self.assertNotIn("/home/user/work", rendered)
        self.assertIn("<artifact:structure/input.pdb>", rendered)
        self.assertEqual(stats["raw_tool_name_map"]["mcp__molclaw-scp__is_valid_smiles"], "is_valid_smiles")

    def test_compacts_as_json_and_records_error_status_conflict(self) -> None:
        events = [
            assistant_event(
                {"type": "tool_use", "id": "c1", "name": "mcp__molclaw-scp__fpocket_toolkit", "input": {}}
            ),
            user_event(
                {
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "is_error": False,
                    "content": {
                        "status": "error",
                        "message": "pocket calculation failed",
                        "rows": [{"value": index} for index in range(200)],
                    },
                }
            ),
        ]
        messages, stats = reconstruct_react_messages(
            events,
            question_text="Find a pocket",
            final_answer="No pocket was produced.",
            task="e2e",
            max_observation_chars=200,
        )
        observation_text = next(message["content"] for message in messages if message["role"] == "user" and "<observation" in message["content"])
        payload_text = observation_text.split(">", 1)[1].rsplit("</observation>", 1)[0]
        payload = json.loads(payload_text)
        self.assertTrue(payload["is_error"])
        self.assertTrue(payload["content"]["compacted"])
        self.assertEqual(payload["compaction"]["method"], "structured_summary")
        self.assertEqual(len(stats["error_status_conflicts"]), 1)

    def test_final_answer_keeps_task_result_summary_and_evidence(self) -> None:
        events = [
            assistant_event(
                {"type": "tool_use", "id": "c1", "name": "mcp__molclaw-scp__dock", "input": {"smiles": "CCO"}}
            ),
            user_event(
                {
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "content": {"status": "success", "score": -7.1, "output_path": "/tmp/dock/result.sdf"},
                }
            ),
            assistant_event({"type": "text", "text": "Docking completed with the recorded score."}),
        ]
        messages, _ = reconstruct_react_messages(
            events,
            question_text="Dock CCO",
            final_answer={"result": "completed"},
            task="e2e",
        )
        final_text = messages[-1]["content"]
        payload = json.loads(final_text.removeprefix("<final_answer>").removesuffix("</final_answer>"))
        self.assertEqual(payload["result"], {"result": "completed"})
        self.assertIn("Docking completed", payload["summary"])
        self.assertEqual(payload["evidence"][0]["key_values"]["score"], -7.1)
        self.assertIn("<artifact:", json.dumps(payload, ensure_ascii=False))

    def test_hard_clean_does_not_reconstruct_or_recompact_canonical_react(self) -> None:
        events = [
            assistant_event(
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "mcp__molclaw-scp__fpocket_toolkit",
                    "input": {},
                }
            ),
            user_event(
                {
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "content": {
                        "status": "success",
                        "score": 3.5,
                        "rows": [{"value": index} for index in range(200)],
                    },
                }
            ),
        ]
        messages, stats = reconstruct_react_messages(
            events,
            question_text="Find a pocket",
            final_answer={"result": "Pocket 1"},
            task="e2e",
            max_observation_chars=200,
        )
        self.assertEqual(stats["compacted_observation_count"], 1)
        source_roles = [message["role"] for message in messages]
        self.assertEqual(
            sum(
                "<final_answer>" in message["content"]
                for message in messages
                if message["role"] == "assistant"
            ),
            1,
        )
        source_observation = next(
            message["content"] for message in messages if "<observation " in message["content"]
        )
        source_payload = json.loads(source_observation.split(">", 1)[1].split("</observation>", 1)[0])

        assistant_message = next(
            message for message in messages if message["role"] == "assistant" and "<tool_call>" in message["content"]
        )
        assistant_message["content"] = (
            "<thought>Review ../outputs/result.pdb.</thought>\n" + assistant_message["content"]
        )
        cleaned, report = hard_clean(
            {
                "schema_version": "drug_agent_sft_react_json_v1",
                "id": "react_test",
                "messages": messages,
            }
        )
        cleaned_text = "\n".join(message["content"] for message in cleaned["messages"])
        cleaned_observation = next(
            message["content"] for message in cleaned["messages"] if "<observation " in message["content"]
        )
        cleaned_payload = json.loads(cleaned_observation.split(">", 1)[1].split("</observation>", 1)[0])

        self.assertEqual([message["role"] for message in cleaned["messages"]], source_roles)
        self.assertEqual(
            sum(
                "<final_answer>" in message["content"]
                for message in cleaned["messages"]
                if message["role"] == "assistant"
            ),
            1,
        )
        self.assertEqual(cleaned_payload, source_payload)
        self.assertIn("<artifact:structure/result.pdb>", cleaned_text)
        self.assertEqual(report["counts"]["tool_calls"], 1)
        self.assertEqual(report["counts"]["observations"], 1)
        self.assertNotIn("final_status", report)
        self.assertNotIn("accepted", report)


if __name__ == "__main__":
    unittest.main()
