import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from drug_agent.gad.trajectory_logger import log_rollout_data


class TestTrajectoryLogger(unittest.TestCase):
    def test_explicit_log_path_works_without_checkpoint_save_path(self):
        sample = SimpleNamespace(
            metadata={"sample_id": "probe"},
            label={"teacher_response": "teacher"},
            response="student",
            weight_versions=[1],
            reward={"score": 0.25},
            prompt=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "gad.jsonl")
            with patch.dict(os.environ, {"GAD_TRAJECTORY_LOG": output}):
                metrics = {}
                self.assertFalse(log_rollout_data(0, SimpleNamespace(save=None), [sample], metrics, 1.0))
            with open(output, encoding="utf-8") as handle:
                row = json.loads(handle.readline())
            self.assertEqual(row["sample_id"], "probe")
            self.assertEqual(metrics["gad/trajectory_path"], output)


if __name__ == "__main__":
    unittest.main()
