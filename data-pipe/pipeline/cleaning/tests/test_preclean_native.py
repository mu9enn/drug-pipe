from __future__ import annotations

import json
import unittest

from pipeline.cleaning.preclean_native import raw_events_to_qwen35_native


class PrecleanNativeTest(unittest.TestCase):
    def test_preserves_raw_decisions_paths_and_observations(self) -> None:
        long_observation = "x" * 7000
        events = [
            {
                "type": "assistant",
                "_line_no": 1,
                "message": {
                    "id": "decision-1",
                    "content": [
                        {"type": "thinking", "thinking": "Read L2 before acting"},
                        {
                            "type": "tool_use",
                            "id": "call-read",
                            "name": "Read",
                            "input": {"file_path": "/machine/.claude/skills/L2_workflows/a.md"},
                        },
                        {
                            "type": "tool_use",
                            "id": "call-tool",
                            "name": "mcp__server__tool",
                            "input": {"path": "/server/input.pdb"},
                        },
                    ],
                },
            },
            {
                "type": "user",
                "_line_no": 2,
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call-read", "content": "skill text"},
                        {"type": "tool_result", "tool_use_id": "call-tool", "content": long_observation},
                    ]
                },
            },
            {
                "type": "assistant",
                "_line_no": 3,
                "message": {
                    "id": "decision-2",
                    "content": [
                        {"type": "thinking", "thinking": "done"},
                        {"type": "text", "text": "<answer>final</answer>"},
                    ],
                },
            },
        ]

        record = raw_events_to_qwen35_native(
            events,
            record_id="raw-1",
            user_task="task",
            source_session="/machine/session.jsonl",
        )

        assistants = [message for message in record["messages"] if message["role"] == "assistant"]
        tools = [message for message in record["messages"] if message["role"] == "tool"]
        self.assertEqual(len(assistants), 2)
        self.assertEqual(len(assistants[0]["tool_calls"]), 2)
        self.assertEqual(assistants[0]["tool_calls"][0]["function"]["name"], "Read")
        self.assertEqual(
            assistants[0]["tool_calls"][0]["function"]["arguments"]["file_path"],
            "/machine/.claude/skills/L2_workflows/a.md",
        )
        self.assertEqual(tools[1]["content"], long_observation)
        self.assertEqual(assistants[-1]["content"], "final")
        self.assertEqual(record["cleaning_state"]["path_sanitization_applied"], False)
        self.assertEqual(
            [item["source_message_id"] for item in record["provenance"]["assistant_messages"]],
            ["decision-1", "decision-2"],
        )
        self.assertGreater(len(json.dumps(record)), 7000)
