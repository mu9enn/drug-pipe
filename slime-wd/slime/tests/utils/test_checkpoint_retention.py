from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from slime.utils.checkpoint_retention import (
    CheckpointRetentionError,
    prune_checkpoint_root,
    validate_latest_checkpoint,
)


def _checkpoint(root: Path, iteration: int, *, complete: bool = True) -> Path:
    path = root / f"iter_{iteration:07d}"
    path.mkdir()
    if complete:
        for name in (".metadata", "common.pt", "metadata.json", "__0_0.distcp"):
            (path / name).write_bytes(b"checkpoint")
    return path


class CheckpointRetentionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_prune_keeps_only_newest_iterations(self) -> None:
        for iteration in range(5):
            _checkpoint(self.root, iteration)
        (self.root / "latest_checkpointed_iteration.txt").write_text("4\n")
        unrelated = self.root / "model_export"
        unrelated.mkdir()

        removed = prune_checkpoint_root(self.root, 2, expected_iteration=4)

        self.assertEqual([path.name for path in removed], ["iter_0000000", "iter_0000001", "iter_0000002"])
        self.assertEqual(sorted(path.name for path in self.root.glob("iter_*")), ["iter_0000003", "iter_0000004"])
        self.assertTrue(unrelated.is_dir())

    def test_final_retention_keeps_latest_only(self) -> None:
        for iteration in (199, 399, 599):
            _checkpoint(self.root, iteration)
        (self.root / "latest_checkpointed_iteration.txt").write_text("599\n")

        prune_checkpoint_root(self.root, 1)

        self.assertEqual([path.name for path in self.root.glob("iter_*")], ["iter_0000599"])

    def test_incomplete_latest_never_prunes_previous_checkpoint(self) -> None:
        previous = _checkpoint(self.root, 1)
        _checkpoint(self.root, 2, complete=False)
        (self.root / "latest_checkpointed_iteration.txt").write_text("2\n")

        with self.assertRaisesRegex(CheckpointRetentionError, "incomplete"):
            prune_checkpoint_root(self.root, 1, expected_iteration=2)

        self.assertTrue(previous.is_dir())

    def test_marker_mismatch_never_prunes(self) -> None:
        first = _checkpoint(self.root, 1)
        second = _checkpoint(self.root, 2)
        (self.root / "latest_checkpointed_iteration.txt").write_text("2\n")

        with self.assertRaisesRegex(CheckpointRetentionError, "marker mismatch"):
            prune_checkpoint_root(self.root, 1, expected_iteration=3)

        self.assertTrue(first.is_dir() and second.is_dir())

    def test_validate_rejects_empty_distcp_shards(self) -> None:
        checkpoint = _checkpoint(self.root, 0)
        (checkpoint / "__0_0.distcp").write_bytes(b"")
        (self.root / "latest_checkpointed_iteration.txt").write_text("0\n")

        with self.assertRaisesRegex(CheckpointRetentionError, "distcp"):
            validate_latest_checkpoint(self.root)


if __name__ == "__main__":
    unittest.main()
