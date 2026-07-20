from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from molclaw_kg.canonical_edges import build_canonical_edges
from molclaw_kg.graph_views import build_graph_views
from molclaw_kg.io_utils import read_jsonl, write_jsonl
from molclaw_kg.migration import migrate_historical_kg


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
            build_graph_views(config)

            canonical = read_jsonl(config.paths.run_dir / "canonical_edges.jsonl")
            graph = read_jsonl(config.paths.run_dir / "graph_all.jsonl")
            self.assertEqual(canonical[0]["edge_types"], [])
            self.assertNotIn("confidence_calibrated", canonical[0])
            self.assertIsNone(graph[0]["edge_type"])
            self.assertEqual(graph[0]["relation_status"], "negative")
            self.assertEqual(graph[0]["confidence_raw"], 0.91)

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
            build_graph_views(config)

            graph = read_jsonl(config.paths.run_dir / "graph_all.jsonl")
            self.assertEqual(graph[0]["edge_type"], "generates_partial_input_for")

    def test_historical_migration_prefers_adjudication_and_reports_graph_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_dir = Path(td) / "source"
            output_dir = Path(td) / "output"
            source_dir.mkdir()
            raw = adjudication()
            scored = {
                **{key: raw[key] for key in [
                    "pair_id",
                    "source_tool",
                    "target_tool",
                    "relation_status",
                    "direct_transition",
                    "edge_types",
                    "context",
                    "satisfied_mappings",
                    "unsatisfied_required_inputs",
                    "negative_reason",
                    "evidence_refs",
                ]},
                "confidence_raw": 0.91,
                "confidence_calibrated": 0.91,
            }
            graph = {
                "pair_id": raw["pair_id"],
                "source_tool": "source",
                "target_tool": "target",
                "edge_type": "generates_partial_input_for",
                "relation_status": "negative",
                "confidence_raw": 0.91,
            }
            write_jsonl(source_dir / "pair_adjudications.jsonl", [raw])
            scored_supplement = {
                **scored,
                "pair_id": "pair::legacy__to__target",
                "source_tool": "legacy",
                "target_tool": "target",
                "relation_status": "valid",
                "direct_transition": True,
                "edge_types": [{"type": "feeds_into"}],
            }
            write_jsonl(source_dir / "scored_edges.jsonl", [scored, scored_supplement])
            write_jsonl(source_dir / "graph_all.jsonl", [graph])

            report = migrate_historical_kg(source_dir, output_dir)

            migrated = read_jsonl(output_dir / "edge_decisions.jsonl")
            issues = read_jsonl(output_dir / "issues.jsonl")
            persisted_report = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
            by_pair = {row["pair_id"]: row for row in migrated}
            self.assertEqual(by_pair[raw["pair_id"]]["edge_types"], [])
            self.assertEqual(by_pair[raw["pair_id"]]["source_authority"], "claude_adjudication")
            self.assertFalse(by_pair[scored_supplement["pair_id"]]["eligible_for_sampling"])
            self.assertEqual(report, persisted_report)
            self.assertTrue(
                any(
                    row["kind"] == "historical_conflict"
                    and row["comparison"] == "canonical_edge_vs_graph_all"
                    for row in issues
                )
            )


if __name__ == "__main__":
    unittest.main()
