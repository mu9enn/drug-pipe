from __future__ import annotations

import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from molclaw_kg.adjudicators.claude_code_runtime import extract_json_object
from molclaw_kg.io_utils import read_jsonl, write_json, write_jsonl
from molclaw_kg.question_sampling.simple_sampler import (
    _sequence_hint,
    _tool_leaks,
    contains_user_followup_request,
    sample_hidden_toolchain,
    sample_simple_questions,
    select_grounding_seed,
    validate_simple_output,
)
from molclaw_kg.science_kb import initialize_database


def edge(source: str, target: str, status: str = "valid") -> dict:
    return {
        "source_tool": source,
        "target_tool": target,
        "edge_type": "generates_partial_input_for",
        "relation_status": status,
        "pair_id": f"pair::{source}__to__{target}",
        "view": "core",
    }


class SimpleSamplingTests(unittest.TestCase):
    def test_hidden_toolchain_only_uses_valid_simple_path(self) -> None:
        cards = {name: {"tool_id": name} for name in ["a", "b", "c", "d"]}
        result = sample_hidden_toolchain(
            [edge("a", "b"), edge("b", "c"), edge("c", "a"), edge("a", "d", "negative")],
            cards,
            2,
            2,
            random.Random(4),
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result["hidden_toolchain_edges"]), 2)
        self.assertEqual(len(set(result["hidden_toolchain_nodes"])), 3)
        self.assertTrue(all(x["relation_status"] == "valid" for x in result["hidden_toolchain_edges"]))

    def test_hidden_toolchain_is_reproducible_for_same_seed_and_inputs(self) -> None:
        cards = {name: {"tool_id": name} for name in ["a", "b", "c", "d"]}
        edges = [edge("a", "b"), edge("a", "c"), edge("b", "d"), edge("c", "d")]
        first = sample_hidden_toolchain(edges, cards, 2, 2, random.Random(42))
        second = sample_hidden_toolchain(edges, cards, 2, 2, random.Random(42))
        self.assertEqual(first, second)

    def test_simple_output_parser_contract(self) -> None:
        success = {
            "status": "success",
            "public_question_text": "Analyze the supplied molecule and return results.",
            "question_payload": {"task": "analysis", "inputs": {"smiles": "CCO"}, "expected_output": "report"},
            "rationale": "Concrete and actionable.",
        }
        reject = {"status": "reject", "public_question_text": "", "question_payload": {}, "rationale": "Insufficient facts."}
        self.assertEqual(validate_simple_output(success), [])
        self.assertEqual(validate_simple_output(reject), [])
        self.assertTrue(validate_simple_output({"status": "success"}))

    def test_json_extraction_accepts_markdown_and_surrounding_text(self) -> None:
        payload = {
            "status": "reject",
            "public_question_text": "",
            "question_payload": {},
            "rationale": "Insufficient facts.",
        }
        encoded = json.dumps(payload)
        self.assertEqual(extract_json_object(f"```json\n{encoded}\n```"), payload)
        self.assertEqual(extract_json_object(f"Here is the result:\n{encoded}\nEnd."), payload)

    def test_hidden_brand_and_sequence_hint_detection(self) -> None:
        cards = {"foldx_tool": {"aliases": []}, "openmm_extract_frames": {"aliases": []}}
        self.assertEqual(_tool_leaks("Compute the result with FoldX.", ["foldx_tool"], cards), ["foldx_tool"])
        self.assertEqual(_tool_leaks("Analyze conformational stability.", ["foldx_tool"], cards), [])
        self.assertTrue(_sequence_hint("First extract frames, then score them, and finally return a report."))

    def test_grounding_seed_selection_respects_repeat_limits(self) -> None:
        records = [
            {"protein_id": f"target_{i}", "compound_id": f"compound_{i}"}
            for i in range(4)
        ]
        seen_targets: Counter[str] = Counter()
        seen_compounds: Counter[str] = Counter()
        selected = [
            select_grounding_seed(records, seen_targets, seen_compounds, 1, 1, random.Random(7))
            for _ in range(4)
        ]
        self.assertEqual(len({row["protein_id"] for row in selected}), 4)
        self.assertEqual(len({row["compound_id"] for row in selected}), 4)

    def test_user_followup_detection_checks_payload_and_rationale(self) -> None:
        self.assertTrue(contains_user_followup_request({
            "public_question_text": "Analyze this peptide.",
            "question_payload": {"inputs": {"seed": "to be requested from the user"}},
            "rationale": "",
        }))

    def test_prompt_keeps_hidden_toolchain_as_soft_constraint(self) -> None:
        prompt_path = Path(__file__).parents[1] / "configs/prompts/toolchain_question_simple_v1.md"
        prompt = prompt_path.read_text(encoding="utf-8")
        self.assertIn("hidden toolchain", prompt.lower())
        self.assertIn("never expose", prompt.lower())
        self.assertIn("no human available for follow-up", prompt.lower())

    def test_simple_output_writer_and_no_hidden_chain_in_question(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root, run_dir = Path(td), Path(td) / "runs/run_test"
            (root / "configs/prompts").mkdir(parents=True)
            (root / "configs/prompts/toolchain_question_simple_v1.md").write_text("Generate JSON.", encoding="utf-8")
            (root / "configs/prompts/toolchain_question_json_repair_v1.md").write_text("Repair JSON.", encoding="utf-8")
            (root / "science_kb/processed").mkdir(parents=True)
            (root / "science_kb/manifests").mkdir(parents=True)
            db = root / "science_kb/processed/science_kb.sqlite"
            conn = initialize_database(db)
            conn.execute(
                "INSERT INTO proteins VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("protein::1", "test", "1", "P1", "P1", "GENE1", "Protein 1", "Human", None, '["1ABC"]', "{}"),
            )
            conn.execute(
                "INSERT INTO compounds VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("compound::1", "test", "1", "C1", "Compound 1", "CCO", "{}"),
            )
            conn.execute(
                "INSERT INTO target_ligand_pairs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("pair::1", "test", "1", "P1", "C1", "Ki", 1.0, "nM", "{}"),
            )
            conn.commit()
            conn.close()
            (root / "science_kb/manifests/science_kb_manifest.json").write_text("{}", encoding="utf-8")
            results_dir = run_dir / "results"
            results_dir.mkdir(parents=True)
            canonical_edge = {
                **edge("foldx_tool", "tool_b"),
                "schema_version": "tool_kg_graph_edge_v1",
                "eligible_for_sampling": True,
                "direct_transition": True,
            }
            write_jsonl(results_dir / "graph.jsonl", [canonical_edge])
            cards = [
                {"tool_id": "foldx_tool", "description_summary": "A", "connectable_inputs": [], "connectable_outputs": []},
                {"tool_id": "tool_b", "description_summary": "B", "connectable_inputs": [], "connectable_outputs": []},
            ]
            write_jsonl(results_dir / "tool_catalog.jsonl", cards)
            write_jsonl(
                results_dir / "edge_decisions.jsonl",
                [{
                    "pair_id": "pair::foldx_tool__to__tool_b",
                    "source_tool": "foldx_tool",
                    "target_tool": "tool_b",
                    "edge_types": [{"type": "generates_partial_input_for"}],
                    "satisfied_inputs": [],
                    "unsatisfied_inputs": [],
                    "source_authority": "claude_adjudication",
                }],
            )
            write_json(results_dir / "run_manifest.json", {"counts": {}, "outputs": {}})
            config = SimpleNamespace(
                paths=SimpleNamespace(root=root, run_dir=run_dir, configs=root / "configs"),
                runtime=SimpleNamespace(server_url="", api_key=""),
            )
            followup = {
                "status": "success",
                "public_question_text": "Ask me for a seed peptide sequence before proceeding.",
                "question_payload": {"task": "analysis", "inputs": {"seed": "to be requested"}, "expected_output": "report"},
                "rationale": "Requires user input.",
            }
            output = {
                "status": "success",
                "public_question_text": "Use FoldX to analyze the supplied scientific input and return a concise report.",
                "question_payload": {"task": "analysis", "inputs": {}, "expected_output": "report"},
                "rationale": "Actionable.",
            }
            with patch(
                "molclaw_kg.question_sampling.simple_sampler._call_agent",
                side_effect=[(followup, json.dumps(followup)), (output, json.dumps(output))],
            ):
                meta = sample_simple_questions(config, target_successes=1, max_attempts=2, min_hops=1, max_hops=1, seed=1)
            self.assertEqual(meta["success_count"], 1)
            success_rows = read_jsonl(results_dir / "tasks.jsonl")
            self.assertEqual(len(success_rows), 1)
            self.assertIn("FoldX", success_rows[0]["public_question_text"])
            self.assertEqual(success_rows[0]["schema_version"], "tool_kg_task_v1")
            attempts_path = run_dir / "intermediate/stage3/sample_attempts.jsonl"
            attempts = read_jsonl(attempts_path)
            self.assertEqual(attempts[0]["failure_reason"], "non_rolloutable_user_followup")
            self.assertTrue(attempts_path.is_file())
            self.assertFalse((results_dir / "questions_simple.csv").exists())
            manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"]["tasks"], 1)


if __name__ == "__main__":
    unittest.main()
