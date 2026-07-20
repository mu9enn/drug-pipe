from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from molclaw_kg.question_sampling.sampler import (
    _closure_analysis,
    _filtered_edges,
    _topological,
    _trajectory,
    _validate_edge_claims,
    _validate_grounding,
)
from molclaw_kg.science_kb import initialize_database, ScienceKB


class Stage3GroundingTests(unittest.TestCase):
    def test_partial_requires_mapping(self) -> None:
        rows = [{"source_tool": "a", "target_tool": "b", "edge_type": "generates_partial_input_for", "view": "core", "relation_status": "valid", "direct_transition": True}]
        cfg = {"edge_profiles": {"core_strict": {"allowed_views": ["core"]}}, "partial_edge_policy": {"require_satisfied_mapping": True}}
        edge_types = {"generates_partial_input_for"}
        self.assertEqual(
            _filtered_edges(rows, {}, "core_strict", "closure_required", cfg, edge_types),
            [],
        )
        dbg = {("a", "b", "generates_partial_input_for"): {"satisfied_mappings": [{"source_output_slot": "x", "target_input_slot": "y"}]}}
        self.assertEqual(
            len(
                _filtered_edges(
                    rows,
                    dbg,
                    "core_strict",
                    "closure_required",
                    cfg,
                    edge_types,
                )
            ),
            1,
        )

    def test_canonical_graph_edge_does_not_require_legacy_view(self) -> None:
        canonical_edge = {
            "schema_version": "tool_kg_graph_edge_v1",
            "pair_id": "pair::a__to__b",
            "edge_id": "pair::a__to__b::edge::01",
            "source_tool": "a",
            "target_tool": "b",
            "edge_type": "generates_full_input_for",
            "relation_status": "valid",
            "direct_transition": True,
            "eligible_for_sampling": True,
            "confidence": 0.83,
        }
        claim = {
            "source_tool": "a",
            "target_tool": "b",
            "support_source": "toolkg",
            "support_ref": "pair::a__to__b",
        }
        errors, edges, edge_types = _validate_edge_claims(
            {"a", "b"},
            [claim],
            {("a", "b"): [canonical_edge]},
        )
        self.assertEqual(errors, [])
        self.assertEqual(edges, [("a", "b")])
        self.assertEqual(edge_types[("a", "b")], "generates_full_input_for")

    def test_transitive_closure(self) -> None:
        cards = {
            "a": {"connectable_inputs": [], "connectable_outputs": [{"name": "seq", "semantic_type": "protein_sequence", "format": "text"}]},
            "b": {"connectable_inputs": [], "connectable_outputs": []},
            "c": {"connectable_inputs": [{"name": "seq", "semantic_type": "protein_sequence", "format": "text", "required": True}], "connectable_outputs": []},
        }
        report = _closure_analysis({"a", "b", "c"}, [("a", "b"), ("b", "c")], cards, [], {}, {}, {("a", "b"): "", ("b", "c"): ""})
        self.assertEqual(report["closure_status"], "closed")

    def test_topological_cycle(self) -> None:
        self.assertIsNone(_topological({"a", "b"}, [("a", "b"), ("b", "a")]))

    def test_python_trajectory(self) -> None:
        traj = _trajectory({"a", "b"}, [("a", "b")], [], "result", [])
        self.assertEqual(traj["schema_version"], "trajectory_v2_graph")
        self.assertEqual(traj["execution_plan"]["tool_order"], ["a", "b"])
        order = traj["execution_plan"]["topological_order"]
        self.assertLess(order.index("llm::interpret::a"), order.index("llm::parameterize::b"))

    def test_grounding_value_must_be_public_and_in_record(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "kb.sqlite"
            manifest = Path(td) / "manifest.json"
            conn = initialize_database(db)
            conn.execute(
                "INSERT INTO compounds VALUES (?,?,?,?,?,?,?)",
                ("compound::1", "test", "v1", "C1", "aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O", "{}"),
            )
            conn.commit()
            conn.close()
            manifest.write_text("{}", encoding="utf-8")
            kb = ScienceKB(db, manifest)
            inp = [{"name": "smiles", "value": "CC(=O)OC1=CC=CC=C1C(=O)O", "semantic_type": "ligand_smiles", "format": "smiles", "grounding_record_id": "compound::1"}]
            self.assertFalse(_validate_grounding(kb, inp, ["compound::1"], "Analyze CC(=O)OC1=CC=CC=C1C(=O)O and return results.", {}))
            self.assertFalse(_validate_grounding(kb, inp, ["compound::1"], "Analyze the supplied molecule.", {"smiles": inp[0]["value"]}))
            self.assertTrue(_validate_grounding(kb, inp, ["compound::1"], "Analyze aspirin and return results.", {}))
            kb.close()

    def test_noncanonical_edge_support_is_rejected(self) -> None:
        claim = {
            "source_tool": "a",
            "target_tool": "b",
            "support_source": "skills",
            "support_ref": "doc",
        }
        errors, edges, edge_types = _validate_edge_claims({"a", "b"}, [claim], {})
        self.assertIn("noncanonical_edge_support:a->b:skills", errors)
        self.assertFalse(edges)
        self.assertFalse(edge_types)


if __name__ == "__main__":
    unittest.main()
