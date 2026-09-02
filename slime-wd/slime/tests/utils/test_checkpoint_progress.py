from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from slime.utils.checkpoint_progress import (
    progress_path,
    read_checkpoint_progress,
    write_checkpoint_progress,
)


class CheckpointProgressTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_round_trip_keeps_optimizer_step_and_rollout_id_separate(self) -> None:
        path = write_checkpoint_progress(
            self.root,
            checkpoint_iteration=183,
            optimizer_step=183,
            rollout_id=0,
        )

        self.assertEqual(path, progress_path(self.root, 183))
        self.assertEqual(
            read_checkpoint_progress(self.root, checkpoint_iteration=183),
            {
                "schema_version": "slime_checkpoint_progress_v1",
                "checkpoint_iteration": 183,
                "optimizer_step": 183,
                "rollout_id": 0,
            },
        )

    def test_direct_iteration_path_resolves_to_checkpoint_root(self) -> None:
        iteration_dir = self.root / "iter_0000552"
        iteration_dir.mkdir()
        write_checkpoint_progress(
            self.root,
            checkpoint_iteration=552,
            optimizer_step=552,
            rollout_id=551,
        )

        payload = read_checkpoint_progress(iteration_dir, checkpoint_iteration=552)

        self.assertEqual(payload["optimizer_step"], 552)
        self.assertEqual(payload["rollout_id"], 551)

    def test_rejects_iteration_that_does_not_name_optimizer_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "must equal optimizer_step"):
            write_checkpoint_progress(
                self.root,
                checkpoint_iteration=0,
                optimizer_step=183,
                rollout_id=0,
            )


if __name__ == "__main__":
    unittest.main()
