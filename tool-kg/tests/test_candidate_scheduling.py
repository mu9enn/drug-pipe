from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from molclaw_kg.candidate_generation import generate_candidates
from molclaw_kg.io_utils import read_json, read_jsonl, write_jsonl


class FakeTaxonomy:
    def is_transition_allowed(self, source_stage: str, target_stage: str) -> bool:
        return (source_stage, target_stage) == ("source", "target")

    def alternative_pairs(self):
        return [("alternative_a", "alternative_b", "alternative_family", "family_1")]


class CandidateSchedulingTest(unittest.TestCase):
    def test_taxonomy_is_the_only_candidate_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            cards = [
                {
                    "tool_id": "source_tool",
                    "primary_stage": "source",
                    "connectable_outputs": [],
                },
                {
                    "tool_id": "target_tool",
                    "primary_stage": "target",
                    "connectable_inputs": [],
                },
                {
                    "tool_id": "disallowed_tool",
                    "primary_stage": "disallowed",
                    "connectable_inputs": [{"name": "matching_name"}],
                    "connectable_outputs": [{"name": "matching_name"}],
                },
                {"tool_id": "alternative_a", "primary_stage": "alternative_a"},
                {"tool_id": "alternative_b", "primary_stage": "alternative_b"},
            ]
            write_jsonl(run_dir / "tool_cards.jsonl", cards)
            write_jsonl(
                run_dir / "pair_adjudications.jsonl",
                [{"pair_id": "pair::source_tool__to__target_tool"}],
            )
            config = SimpleNamespace(paths=SimpleNamespace(run_dir=run_dir), stage_taxonomy_path=run_dir / "taxonomy.json")

            with patch("molclaw_kg.candidate_generation.load_stage_taxonomy", return_value=FakeTaxonomy()):
                result = generate_candidates(config)

            rows = {row["pair_id"]: row for row in read_jsonl(run_dir / "candidate_pairs.jsonl")}
            self.assertIn("pair::source_tool__to__target_tool", rows)
            self.assertIn("pair::alternative_a__to__alternative_b", rows)
            self.assertNotIn("pair::disallowed_tool__to__target_tool", rows)
            self.assertNotIn("schema_score", json.dumps(list(rows.values())))
            self.assertNotIn("suggested_edge", json.dumps(list(rows.values())))

            meta = read_json(run_dir / "candidate_meta.json")
            self.assertEqual(meta["scheduling_authority"], "stage_taxonomy")
            self.assertFalse(meta["tool_interfaces_used_as_gate"])
            self.assertEqual(meta["cached_pairs"], 1)
            self.assertEqual(meta["new_claude_calls"], result["candidate_count"] - 1)


if __name__ == "__main__":
    unittest.main()
