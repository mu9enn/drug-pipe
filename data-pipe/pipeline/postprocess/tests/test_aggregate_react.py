from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_DIR))

from postprocess.aggregate_react import aggregate_react  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class AggregateReactTest(unittest.TestCase):
    def test_aggregates_minimal_training_and_audit_without_duplicate_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectories = root / "run-a" / "trajectories"
            training = {
                "schema_version": "drug_agent_sft_react_json_v1",
                "id": "one",
                "messages": [{"role": "user", "content": "task"}],
            }
            audit = {
                "id": "one",
                "task": "kg",
                "final_status": "accepted",
                "final_status_authority": "final_acceptance_gate",
                "task_metrics": {},
            }
            write_jsonl(trajectories / "react_trajectories.jsonl", [training])
            write_jsonl(trajectories / "curation_audit.jsonl", [audit])
            write_jsonl(trajectories / "rejected.jsonl", [])
            output = root / "aggregate"
            summary = aggregate_react(root, output)
            aggregated = json.loads((output / "react_trajectories.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(set(aggregated), {"schema_version", "id", "messages"})
            self.assertEqual(summary["output_count"], 1)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "react_trajectories.jsonl",
                    "curation_audit.jsonl",
                    "rejected.jsonl",
                    "curation_summary.json",
                },
            )


if __name__ == "__main__":
    unittest.main()
