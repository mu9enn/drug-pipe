from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_toolchain_groups.py"
SPEC = importlib.util.spec_from_file_location("analyze_toolchain_groups", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _trajectory(record_id: str, tools: list[str]) -> dict[str, object]:
    calls = "\n".join(
        f'<tool_call>{{"arguments":{{}},"tool_name":"{tool}"}}</tool_call>' for tool in tools
    )
    return {
        "schema_version": "drug_agent_sft_react_json_v1",
        "id": record_id,
        "messages": [
            {"role": "system", "content": "system", "step_loss_mask": 0},
            {"role": "user", "content": "question", "step_loss_mask": 0},
            {"role": "assistant", "content": calls, "step_loss_mask": 1},
        ],
    }


class AnalyzeToolchainGroupsTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        toolkg = root / "toolkg"
        (toolkg / "data").mkdir(parents=True)
        (toolkg / "metadata/toolkg").mkdir(parents=True)
        group_tools = {
            "A": ["a", "shared"],
            "B": ["b", "shared"],
            "C": ["c", "shared"],
            "D": ["d", "shared"],
            "E": ["e", "shared"],
            "F": ["f", "shared"],
            "G": ["g", "shared"],
            "H": ["h", "shared"],
        }
        group_indices = {group: [] for group in MODULE.TOOLKG_ORDER}
        trajectories = []
        metadata = []
        for index in range(1, 241):
            group = MODULE.TOOLKG_ORDER[(index - 1) % 8]
            group_indices[group].append(index)
            record_id = f"react_kg_{index:04d}"
            trajectories.append(_trajectory(record_id, group_tools[group]))
            metadata.append(
                {
                    "training_record_id": record_id,
                    "training_record_index": index,
                }
            )
        (toolkg / "data/react_trajectories.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in trajectories), encoding="utf-8"
        )
        (toolkg / "metadata/toolkg/toolkg_metadata_240.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8"
        )

        molbench = root / "molbench.jsonl"
        molbench_rows = [
            _trajectory("react_pf_1", ["f", "shared"]),
            _trajectory("react_pf_2", ["f", "shared"]),
            _trajectory("react_ac_1", ["b", "shared"]),
            _trajectory("react_ac_2", ["b", "shared"]),
            _trajectory("react_vs_1", ["b", "dock"]),
            _trajectory("react_vs_2", ["b", "dock"]),
            _trajectory("react_e2e_ignored", ["ignored"]),
        ]
        molbench.write_text(
            "".join(json.dumps(row) + "\n" for row in molbench_rows), encoding="utf-8"
        )
        mapping = root / "mapping.json"
        mapping.write_text(
            json.dumps(
                {
                    "schema_version": "drug_pipe_toolchain_group_mapping_v1",
                    "toolkg_primary_patterns": {
                        group: {"name": group, "indices": group_indices[group]}
                        for group in MODULE.TOOLKG_ORDER
                    },
                    "molbench_ms_groups": {
                        "MS1_PF": {
                            "name": "pf",
                            "trajectory_id_prefix": "react_pf_",
                            "toolkg_primary_pattern": "F",
                        },
                        "MS2_AC": {
                            "name": "ac",
                            "trajectory_id_prefix": "react_ac_",
                            "toolkg_primary_pattern": "B",
                        },
                        "MS3_VS": {
                            "name": "vs",
                            "trajectory_id_prefix": "react_vs_",
                            "toolkg_primary_pattern": "B",
                        },
                    },
                    "mapped_comparisons": [
                        {"toolkg_group": "F", "molbench_group": "MS1_PF"},
                        {"toolkg_group": "B", "molbench_group": "MS2_AC"},
                        {"toolkg_group": "B", "molbench_group": "MS3_VS"},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return toolkg, molbench, mapping

    def test_pair_metrics_and_js_similarity(self) -> None:
        left = {
            "tool_set_signature": ["a", "b"],
            "ordered_unique_chain": ["a", "b"],
            "canonical_chain": ["a", "b"],
        }
        right = {
            "tool_set_signature": ["b", "c"],
            "ordered_unique_chain": ["b", "c"],
            "canonical_chain": ["b", "c"],
        }
        metrics = MODULE._pair_metrics(left, right)
        self.assertAlmostEqual(metrics["tool_set_jaccard_similarity"], 1 / 3)
        self.assertAlmostEqual(metrics["tool_set_jaccard_distance"], 2 / 3)
        self.assertEqual(MODULE._js_similarity(MODULE.Counter({"a": 1}), MODULE.Counter({"a": 1})), 1.0)
        self.assertEqual(MODULE._js_similarity(MODULE.Counter({"a": 1}), MODULE.Counter({"b": 1})), 0.0)

    def test_end_to_end_group_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            toolkg, molbench, mapping = self._fixture(root)
            output = root / "output"
            result = MODULE.analyze(
                toolkg_root=toolkg,
                molbench_trajectories=molbench,
                mapping_path=mapping,
                output_dir=output,
                skip_plots=False,
            )
            self.assertEqual(result["record_count"], 246)
            self.assertEqual(result["group_counts"]["A"], 30)
            self.assertEqual(result["group_counts"]["MS1_PF"], 2)
            self.assertEqual(result["mapped_pair_count"], 180)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["within_group"]["F"]["mean_pairwise_tool_set_similarity"], 1.0)
            comparisons = {
                (row["group_a"], row["group_b"]): row for row in summary["mapped_comparisons"]
            }
            self.assertEqual(comparisons[("F", "MS1_PF")]["mean_cross_tool_set_similarity"], 1.0)
            self.assertEqual(comparisons[("B", "MS2_AC")]["tool_call_js_similarity"], 1.0)
            self.assertAlmostEqual(
                comparisons[("B", "MS3_VS")]["mean_cross_tool_set_similarity"], 1 / 3
            )
            self.assertEqual(len(list((output / "figures").glob("*.png"))), 4)
            with (output / "mapped_pairwise.csv").open() as handle:
                self.assertEqual(sum(1 for _ in handle) - 1, 180)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn("manifest.json", manifest["output_sha256"])

    def test_rejects_bad_mapping_and_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            toolkg, molbench, mapping = self._fixture(root)
            bad = json.loads(mapping.read_text(encoding="utf-8"))
            bad["toolkg_primary_patterns"]["A"]["indices"].append(2)
            mapping.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "multiple groups"):
                MODULE.analyze(
                    toolkg_root=toolkg,
                    molbench_trajectories=molbench,
                    mapping_path=mapping,
                    output_dir=root / "output",
                    skip_plots=True,
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            toolkg, molbench, mapping = self._fixture(root)
            output = root / "output"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                MODULE.analyze(
                    toolkg_root=toolkg,
                    molbench_trajectories=molbench,
                    mapping_path=mapping,
                    output_dir=output,
                    skip_plots=True,
                )


if __name__ == "__main__":
    unittest.main()
