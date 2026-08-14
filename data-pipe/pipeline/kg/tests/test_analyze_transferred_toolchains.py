from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_transferred_toolchains.py"
SPEC = importlib.util.spec_from_file_location("analyze_transferred_toolchains", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _tool_call(name: str) -> str:
    return f'<tool_call>{{"arguments":{{}},"tool_name":"{name}"}}</tool_call>'


def _trajectory(record_id: str, calls: list[str], *, local_first: bool = False) -> dict[str, object]:
    rendered = [_tool_call("Read")] if local_first else []
    rendered.extend(_tool_call(name) for name in calls)
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "system", "step_loss_mask": 0},
        {"role": "user", "content": "question", "step_loss_mask": 0},
    ]
    if rendered:
        split = max(1, len(rendered) // 2)
        messages.append(
            {"role": "assistant", "content": "\n".join(rendered[:split]), "step_loss_mask": 1}
        )
        if rendered[split:]:
            messages.append(
                {"role": "assistant", "content": "\n".join(rendered[split:]), "step_loss_mask": 1}
            )
    return {"schema_version": "drug_agent_sft_react_json_v1", "id": record_id, "messages": messages}


def _metadata(record_id: str, index: int, batch: str = "old_100") -> dict[str, object]:
    tools = ["a", "b"]
    edges = [
        {
            "source_tool": "a",
            "target_tool": "b",
            "edge_type": "generates_full_input_for",
        }
    ]
    return {
        "schema_version": "drug_pipe_toolkg_metadata_link_v1",
        "training_record_id": record_id,
        "training_record_index": index,
        "source_batch": batch,
        "question_match": {"exact": True, "whitespace_normalized": True},
        "source_task": {
            "task_id": f"task_{index}",
            "toolchain": {"tools": tools, "edges": edges, "hops": 1},
            "expected_trajectory": {
                "schema_version": "trajectory_v2_graph",
                "execution_plan": {"tool_order": tools},
            },
        },
    }


class AnalyzeTransferredToolchainsTest(unittest.TestCase):
    def _make_fixture(self, root: Path) -> tuple[Path, Path]:
        transfer = root / "transfer"
        (transfer / "data").mkdir(parents=True)
        (transfer / "metadata/toolkg").mkdir(parents=True)
        specifications = [
            ("r_exact", ["a", "b"], "old_100", True),
            ("r_subsequence", ["a", "x", "b"], "old_100", False),
            ("r_full_order", ["b", "a"], "historical_40", False),
            ("r_partial", ["a", "x"], "historical_40", False),
            ("r_none", ["x"], "new_100", False),
            ("r_empty", [], "new_100", False),
        ]
        trajectories = [
            _trajectory(record_id, calls, local_first=local_first)
            for record_id, calls, _, local_first in specifications
        ]
        metadata = [
            _metadata(record_id, index, batch)
            for index, (record_id, _, batch, _) in enumerate(specifications, 1)
        ]
        (transfer / "data/react_trajectories.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in trajectories), encoding="utf-8"
        )
        (transfer / "metadata/toolkg/toolkg_metadata_240.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in metadata), encoding="utf-8"
        )
        (transfer / "metadata/toolkg/toolkg_metadata_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "drug_pipe_toolkg_metadata_manifest_v1",
                    "record_count": len(metadata),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        catalog = root / "catalog.json"
        catalog.write_text(
            json.dumps({"schema_version": "test", "limits": {"4": ["a"], "30": ["b", "x"]}})
            + "\n",
            encoding="utf-8",
        )
        return transfer, catalog

    def _analyze(self, root: Path, *, skip_plots: bool = True) -> tuple[dict[str, object], Path]:
        transfer, catalog = self._make_fixture(root)
        output = root / "analysis"
        result = MODULE.analyze(
            transfer_root=transfer,
            output_dir=output,
            tool_catalog=catalog,
            expected_count=6,
            skip_plots=skip_plots,
        )
        return result, output

    def test_consistency_diversity_and_local_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, output = self._analyze(Path(tmp))
            self.assertEqual(result["record_count"], 6)
            self.assertEqual(result["pairwise_row_count"], 15)

            rows = {
                row["training_record_id"]: row
                for row in (
                    json.loads(line)
                    for line in (output / "per_trajectory.jsonl").read_text(encoding="utf-8").splitlines()
                )
            }
            self.assertEqual(rows["r_exact"]["alignment_class"], "exact")
            self.assertEqual(rows["r_exact"]["actual_chain"], ["a", "b"])
            self.assertEqual(rows["r_exact"]["ignored_local_tool_hist"], {"Read": 1})
            self.assertEqual(
                rows["r_subsequence"]["alignment_class"], "expected_subsequence_with_extras"
            )
            self.assertAlmostEqual(rows["r_subsequence"]["actual_precision"], 2 / 3)
            self.assertAlmostEqual(rows["r_subsequence"]["canonical_edit_similarity"], 2 / 3)
            self.assertEqual(rows["r_subsequence"]["edge_precedence_recall"], 1.0)
            self.assertEqual(rows["r_subsequence"]["edge_canonical_adjacency_recall"], 0.0)
            self.assertEqual(
                rows["r_full_order"]["alignment_class"], "full_coverage_order_deviation"
            )
            self.assertEqual(rows["r_full_order"]["ordered_expected_coverage"], 0.5)
            self.assertEqual(rows["r_full_order"]["canonical_edit_similarity"], 0.0)
            self.assertEqual(rows["r_partial"]["alignment_class"], "partial_overlap")
            self.assertEqual(rows["r_partial"]["expected_recall"], 0.5)
            self.assertEqual(rows["r_none"]["alignment_class"], "no_overlap")
            self.assertEqual(rows["r_empty"]["alignment_class"], "no_overlap")
            self.assertEqual(rows["r_empty"]["actual_chain"], [])
            self.assertEqual(rows["r_empty"]["actual_precision"], 0.0)

            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                summary["consistency"]["alignment_class_counts"],
                {
                    "exact": 1,
                    "expected_subsequence_with_extras": 1,
                    "full_coverage_order_deviation": 1,
                    "partial_overlap": 1,
                    "no_overlap": 2,
                },
            )
            self.assertEqual(summary["input_integrity"]["ignored_local_tool_call_count"], 1)
            self.assertEqual(
                summary["chain_signature_diversity"]["actual"]["raw"][
                    "unique_signature_count"
                ],
                6,
            )
            self.assertEqual(summary["pairwise_diversity"]["pair_count"], 15)
            self.assertGreater(
                summary["transition_diversity"]["actual_raw"]["effective_shannon"], 1.0
            )

            expected_files = {
                "batch_summary.csv",
                "chain_signatures.csv",
                "manifest.json",
                "pairwise_distances.csv",
                "per_trajectory.csv",
                "per_trajectory.jsonl",
                "report.md",
                "summary.json",
                "tool_usage.csv",
                "transition_usage.csv",
            }
            self.assertTrue(expected_files <= {path.name for path in output.iterdir()})
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn("manifest.json", manifest["output_sha256"])
            self.assertEqual(len(manifest["output_sha256"]), len(expected_files) - 1)

    def test_generates_four_nonempty_plots(self) -> None:
        os.environ.setdefault("MPLBACKEND", "Agg")
        with tempfile.TemporaryDirectory() as tmp:
            _, output = self._analyze(Path(tmp), skip_plots=False)
            figures = sorted((output / "figures").glob("*.png"))
            self.assertEqual(len(figures), 4)
            self.assertTrue(all(path.stat().st_size > 0 for path in figures))

    def test_rejects_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transfer, catalog = self._make_fixture(root)
            output = root / "analysis"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                MODULE.analyze(
                    transfer_root=transfer,
                    output_dir=output,
                    tool_catalog=catalog,
                    expected_count=6,
                    skip_plots=True,
                )

    def test_rejects_schema_and_manifest_count_mismatches(self) -> None:
        with self.subTest("trajectory schema"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                transfer, catalog = self._make_fixture(root)
                trajectory_path = transfer / "data/react_trajectories.jsonl"
                rows = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines()]
                rows[0]["schema_version"] = "wrong"
                trajectory_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "expected schema_version"):
                    MODULE.analyze(
                        transfer_root=transfer,
                        output_dir=root / "out",
                        tool_catalog=catalog,
                        expected_count=6,
                        skip_plots=True,
                    )

        with self.subTest("manifest count"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                transfer, catalog = self._make_fixture(root)
                manifest_path = transfer / "metadata/toolkg/toolkg_metadata_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["record_count"] = 7
                manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "record_count differs"):
                    MODULE.analyze(
                        transfer_root=transfer,
                        output_dir=root / "out",
                        tool_catalog=catalog,
                        expected_count=6,
                        skip_plots=True,
                    )

    def test_rejects_duplicate_ids_invalid_tool_json_and_chain_mismatch(self) -> None:
        with self.subTest("duplicate trajectory ID"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                transfer, catalog = self._make_fixture(root)
                trajectory_path = transfer / "data/react_trajectories.jsonl"
                first = trajectory_path.read_text(encoding="utf-8").splitlines()[0]
                trajectory_path.write_text(first + "\n" + first + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "duplicate id"):
                    MODULE.analyze(
                        transfer_root=transfer,
                        output_dir=root / "out",
                        tool_catalog=catalog,
                        expected_count=0,
                        skip_plots=True,
                    )

        with self.subTest("invalid tool-call JSON"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                transfer, catalog = self._make_fixture(root)
                trajectory_path = transfer / "data/react_trajectories.jsonl"
                rows = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines()]
                rows[0]["messages"][2]["content"] = "<tool_call>{bad}</tool_call>"
                trajectory_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "invalid JSON"):
                    MODULE.analyze(
                        transfer_root=transfer,
                        output_dir=root / "out",
                        tool_catalog=catalog,
                        expected_count=6,
                        skip_plots=True,
                    )

        with self.subTest("metadata chain mismatch"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                transfer, catalog = self._make_fixture(root)
                metadata_path = transfer / "metadata/toolkg/toolkg_metadata_240.jsonl"
                rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
                rows[0]["source_task"]["expected_trajectory"]["execution_plan"]["tool_order"] = [
                    "b",
                    "a",
                ]
                metadata_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "differs"):
                    MODULE.analyze(
                        transfer_root=transfer,
                        output_dir=root / "out",
                        tool_catalog=catalog,
                        expected_count=6,
                        skip_plots=True,
                    )


if __name__ == "__main__":
    unittest.main()
