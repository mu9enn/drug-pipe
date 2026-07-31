from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from molclaw_kg.canonical_edges import build_canonical_edges
from molclaw_kg.canonical_outputs import canonical_edge_to_decision, project_graph
from molclaw_kg.io_utils import read_jsonl, write_jsonl


def adjudication(
    *,
    pair_id: str = "pair::source__to__target",
    relation_status: str = "negative",
    edge_types: list[dict] | None = None,
) -> dict:
    return {
        "pair_id": pair_id,
        "source_tool": "source",
        "target_tool": "target",
        "source_stage": "source_stage",
        "target_stage": "target_stage",
        "relation_status": relation_status,
        "direct_transition": False,
        "edge_types": [] if edge_types is None else edge_types,
        "negative_reason": "requires_intermediate" if relation_status == "negative" else None,
        "context": "original context",
        "satisfied_mappings": [],
        "unsatisfied_required_inputs": [],
        "evidence_refs": ["snapshot::source"],
        "rationale": "original rationale",
        "agent_confidence": 0.91,
        "agent_model": "claude-test",
        "response_schema_ok": True,
        "response_schema_error": None,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
    }


def config_for(root: Path) -> SimpleNamespace:
    run_dir = root / "runs" / "test"
    run_dir.mkdir(parents=True)
    return SimpleNamespace(
        paths=SimpleNamespace(run_dir=run_dir),
    )


class CanonicalEdgeAuthorityTest(unittest.TestCase):
    def test_graph_projection_does_not_invent_type_for_negative_edge(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = config_for(Path(td))
            write_jsonl(config.paths.run_dir / "pair_adjudications.jsonl", [adjudication()])
            write_jsonl(
                config.paths.run_dir / "tool_cards.jsonl",
                [
                    {"tool_id": "source", "primary_stage": "source_stage"},
                    {"tool_id": "target", "primary_stage": "target_stage"},
                ],
            )

            build_canonical_edges(config)
            canonical = read_jsonl(config.paths.run_dir / "canonical_edges.jsonl")
            decision = canonical_edge_to_decision(canonical[0])
            self.assertEqual(canonical[0]["edge_types"], [])
            self.assertNotIn("confidence_calibrated", canonical[0])
            self.assertEqual(decision["relation_status"], "negative")
            self.assertEqual(decision["confidence_raw"], 0.91)
            self.assertFalse(decision["eligible_for_sampling"])
            self.assertEqual(project_graph([decision], "test"), [])

    def test_graph_projection_preserves_claude_edge_type_exactly(self) -> None:
        edge_type = {
            "type": "generates_partial_input_for",
            "source_slot": "output.result",
            "target_slot_or_precondition": "input.structure",
            "confidence": 0.7,
            "evidence_ids": ["snapshot::source"],
        }
        with tempfile.TemporaryDirectory() as td:
            config = config_for(Path(td))
            write_jsonl(
                config.paths.run_dir / "pair_adjudications.jsonl",
                [
                    {
                        **adjudication(relation_status="valid", edge_types=[edge_type]),
                        "direct_transition": True,
                    }
                ],
            )
            write_jsonl(
                config.paths.run_dir / "tool_cards.jsonl",
                [
                    {"tool_id": "source", "primary_stage": "source_stage"},
                    {"tool_id": "target", "primary_stage": "target_stage"},
                ],
            )

            build_canonical_edges(config)
            canonical = read_jsonl(config.paths.run_dir / "canonical_edges.jsonl")
            graph = project_graph([canonical_edge_to_decision(canonical[0])], "test")
            self.assertEqual(graph[0]["edge_type"], "generates_partial_input_for")

    def test_non_claude_authority_is_rejected(self) -> None:
        row = {
            "pair_id": "pair::source__to__target",
            "source_authority": "legacy_scored_supplement",
            "relation_status": "valid",
            "direct_transition": True,
            "edge_types": [{"type": "feeds_into"}],
        }
        with self.assertRaisesRegex(ValueError, "require Claude adjudication authority"):
            canonical_edge_to_decision(row)


if __name__ == "__main__":
    unittest.main()
