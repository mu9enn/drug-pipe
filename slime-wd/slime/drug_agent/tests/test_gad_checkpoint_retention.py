from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from drug_agent.gad.checkpoint_retention import (
    DiscriminatorCheckpointError,
    prune_numbered_checkpoints,
)


def _checkpoint(root: Path, name: str, *, complete: bool = True) -> Path:
    path = root / name
    path.mkdir()
    if complete:
        (path / "backbone").mkdir()
        (path / "backbone" / "config.json").write_text("{}\n")
        (path / "gad_state.pt").write_bytes(b"state")
        (path / "metadata.json").write_text("{}\n")
    return path


class GADCheckpointRetentionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_discriminator_rotation_keeps_latest_two(self) -> None:
        checkpoints = [_checkpoint(self.root, f"version_{step:06d}") for step in (10, 20, 30)]
        unrelated = _checkpoint(self.root, "final")

        removed = prune_numbered_checkpoints(
            self.root,
            prefix="version_",
            keep_last=2,
            newest=checkpoints[-1],
        )

        self.assertEqual([path.name for path in removed], ["version_000010"])
        self.assertTrue(checkpoints[1].is_dir() and checkpoints[2].is_dir())
        self.assertTrue(unrelated.is_dir())

    def test_discriminator_final_removes_periodic_checkpoints(self) -> None:
        _checkpoint(self.root, "step_000010")
        _checkpoint(self.root, "step_000020")
        latest = _checkpoint(self.root, "latest")

        prune_numbered_checkpoints(self.root, prefix="step_", keep_last=0, newest=latest)

        self.assertFalse(list(self.root.glob("step_*")))
        self.assertTrue(latest.is_dir())

    def test_incomplete_discriminator_checkpoint_does_not_prune(self) -> None:
        previous = _checkpoint(self.root, "version_000010")
        incomplete = _checkpoint(self.root, "version_000020", complete=False)

        with self.assertRaisesRegex(DiscriminatorCheckpointError, "incomplete"):
            prune_numbered_checkpoints(
                self.root,
                prefix="version_",
                keep_last=1,
                newest=incomplete,
            )

        self.assertTrue(previous.is_dir())


if __name__ == "__main__":
    unittest.main()
