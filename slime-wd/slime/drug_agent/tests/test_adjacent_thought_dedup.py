from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from drug_agent.data.deduplicate_adjacent_thoughts import (
    deduplicate_content,
    deduplicate_records,
)


class AdjacentThoughtDedupTest(unittest.TestCase):
    def test_exact_duplicate_deletes_later(self):
        content, actions = deduplicate_content(
            "<thought>Run docking now.</thought><thought>Run docking now.</thought>"
            '<tool_call>{"tool_name":"dock","arguments":{}}</tool_call>'
        )
        self.assertEqual(content.count("<thought>"), 1)
        self.assertEqual(actions[0]["strategy"], "exact_delete_later")
        self.assertIn("Run docking now.", content)
        self.assertIn("<tool_call>", content)

    def test_near_duplicate_keeps_more_informative_longer_text(self):
        content, actions = deduplicate_content(
            "<thought>Retrieve the TP53 protein structure now.</thought>"
            "<thought>Retrieve the human TP53 protein structure now.</thought>"
        )
        self.assertEqual(content, "<thought>Retrieve the human TP53 protein structure now.</thought>")
        self.assertEqual(actions[0]["strategy"], "near_duplicate_keep_more_informative")

    def test_tool_boundary_prevents_merge(self):
        source = (
            "<thought>Run docking now.</thought>"
            '<tool_call>{"tool_name":"dock","arguments":{}}</tool_call>'
            "<thought>Run docking now.</thought>"
        )
        content, actions = deduplicate_content(source)
        self.assertEqual(content, source)
        self.assertEqual(actions, [])

    def test_file_migration_writes_audit_without_changing_tool_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.jsonl"
            tool_call = '<tool_call>{"tool_name":"fix_pdb","arguments":{"input_path":"x"}}</tool_call>'
            source.write_text(json.dumps({
                "id": "sample",
                "messages": [{
                    "role": "assistant",
                    "content": "<thought>Fix the structure.</thought><thought>Fix the structure.</thought>" + tool_call,
                }],
            }) + "\n", encoding="utf-8")
            output = root / "output.jsonl"
            audit = root / "audit.jsonl"
            report_path = root / "report.json"
            report = deduplicate_records(source, output, audit, report_path)
            migrated = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(migrated["messages"][0]["content"].count("<thought>"), 1)
            self.assertIn(tool_call, migrated["messages"][0]["content"])
            self.assertEqual(report["counts"]["exact_delete_later"], 1)


if __name__ == "__main__":
    unittest.main()
