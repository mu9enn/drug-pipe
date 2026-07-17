from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_kg_task_dataset.py"


class BuildKgTaskDatasetTest(unittest.TestCase):
    def test_default_simple_sampler_output_is_exported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "tool-kg" / "runs" / "run_simple"
            sample_dir = run_dir / "results"
            output_dir = root / "tasks"
            sample_dir.mkdir(parents=True)
            sample = {
                "schema_version": "tool_kg_task_v1",
                "id": "simple_0001",
                "public_question_text": "Retrieve and repair the structure for EGFR.",
                "question_payload": {
                    "task": "protein preparation",
                    "inputs": {"gene_name": "EGFR"},
                    "expected_output": "A repaired protein structure.",
                },
                "toolchain_nodes": [
                    "retrieve_protein_structure_by_gene_name",
                    "fix_pdb",
                ],
                "toolchain_edges": [
                    {
                        "source_tool": "retrieve_protein_structure_by_gene_name",
                        "target_tool": "fix_pdb",
                        "edge_type": "feeds_into",
                        "relation_status": "valid",
                        "pair_id": "pair::retrieve__to__fix",
                    }
                ],
                "walk_hops": 1,
                "expected_trajectory": {
                    "schema_version": "trajectory_v2_graph",
                    "workflow_graph": {
                        "nodes": [
                            {
                                "node_id": "tool::01::retrieve",
                                "type": "tool",
                                "tool_id": "retrieve_protein_structure_by_gene_name",
                            },
                            {"node_id": "tool::02::fix", "type": "tool", "tool_id": "fix_pdb"},
                        ],
                        "edges": [
                            {
                                "edge_id": "edge::01",
                                "source": "tool::01::retrieve",
                                "target": "tool::02::fix",
                                "relation": "feeds_into",
                            }
                        ],
                    },
                    "execution_plan": {
                        "topological_order": ["tool::01::retrieve", "tool::02::fix"],
                        "tool_order": ["retrieve_protein_structure_by_gene_name", "fix_pdb"],
                    },
                },
            }
            (sample_dir / "tasks.jsonl").write_text(
                json.dumps(sample) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--kg-run-dir",
                    str(run_dir),
                    "--output-dir",
                    str(output_dir),
                    "--max-samples",
                    "1",
                    "--no-include-raw-sample",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            rows = [
                json.loads(line)
                for line in (output_dir / "kg_sampled_tasks.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            task = rows[0]
            self.assertEqual(task["toolchain"]["hops"], 1)
            self.assertEqual(task["toolchain"]["start_tool"], "retrieve_protein_structure_by_gene_name")
            self.assertEqual(task["toolchain"]["end_tool"], "fix_pdb")
            self.assertEqual(task["expected_trajectory"]["schema_version"], "trajectory_v2_graph")
            self.assertEqual(
                task["expected_trajectory"]["execution_plan"]["tool_order"],
                sample["toolchain_nodes"],
            )


if __name__ == "__main__":
    unittest.main()
