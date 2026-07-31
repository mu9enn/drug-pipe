from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from molclaw_kg.canonical_outputs import project_graph, publish_canonical_outputs
from molclaw_kg.io_utils import read_json, read_jsonl, write_jsonl


class CanonicalOutputsTest(unittest.TestCase):
    def test_results_directory_contains_only_canonical_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "run_test"
            configs = root / "configs"
            run_dir.mkdir(parents=True)
            configs.mkdir()
            (configs / "edge_ontology.yaml").write_text("ontology", encoding="utf-8")
            taxonomy = configs / "stage_taxonomy.json"
            taxonomy.write_text("{}", encoding="utf-8")
            write_jsonl(run_dir / "tool_cards.jsonl", [{"tool_id": "a"}, {"tool_id": "b"}])
            write_jsonl(
                run_dir / "canonical_edges.jsonl",
                [
                    {
                        "pair_id": "pair::a__to__b",
                        "source_tool": "a",
                        "target_tool": "b",
                        "relation_status": "valid",
                        "direct_transition": True,
                        "edge_types": [{"type": "feeds_into", "evidence_ids": ["e1"]}],
                        "satisfied_mappings": [{"source_output_slot": "x", "target_input_slot": "y"}],
                        "confidence_raw": 0.9,
                        "confidence_calibrated": 0.9,
                        "source_authority": "claude_pair_adjudication",
                    }
                ],
            )
            config = SimpleNamespace(
                paths=SimpleNamespace(root=root, run_dir=run_dir, configs=configs),
                runtime=SimpleNamespace(skills_root=root / "skills", model_name="claude", server_url="test"),
            )

            publish_canonical_outputs(config)

            names = {path.name for path in (run_dir / "results").iterdir()}
            self.assertEqual(
                names,
                {"tool_catalog.jsonl", "edge_decisions.jsonl", "graph.jsonl", "run_manifest.json"},
            )
            decision = read_jsonl(run_dir / "results/edge_decisions.jsonl")[0]
            self.assertEqual(decision["edge_types"][0]["type"], "feeds_into")
            self.assertTrue(decision["eligible_for_sampling"])
            graph = read_jsonl(run_dir / "results/graph.jsonl")
            self.assertEqual(graph[0]["edge_type"], "feeds_into")
            self.assertEqual(read_json(run_dir / "results/run_manifest.json")["counts"]["graph_edges"], 1)

    def test_ineligible_decision_is_not_projected(self) -> None:
        decisions = [
            {
                "pair_id": "pair::a__to__b",
                "source_tool": "a",
                "target_tool": "b",
                "relation_status": "valid",
                "eligible_for_sampling": False,
                "edge_types": [{"type": "feeds_into"}],
            }
        ]
        self.assertEqual(project_graph(decisions, "run"), [])


if __name__ == "__main__":
    unittest.main()
