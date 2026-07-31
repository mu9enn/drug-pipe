"""Validated retention for Slime torch-distributed checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


ITERATION_DIR_RE = re.compile(r"^iter_(\d{7})$")


class CheckpointRetentionError(RuntimeError):
    """Raised when a checkpoint cannot be proven complete before pruning."""


def _read_latest_iteration(save_dir: Path) -> int:
    marker = save_dir / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise CheckpointRetentionError(f"checkpoint marker is missing: {marker}")
    try:
        return int(marker.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise CheckpointRetentionError(f"checkpoint marker is not an integer: {marker}") from exc


def validate_latest_checkpoint(save_dir: str | Path, expected_iteration: int | None = None) -> Path:
    """Return the latest iteration directory after checking its required files."""

    root = Path(save_dir).resolve()
    if not root.is_dir():
        raise CheckpointRetentionError(f"checkpoint save directory does not exist: {root}")

    latest = _read_latest_iteration(root)
    if expected_iteration is not None and latest != expected_iteration:
        raise CheckpointRetentionError(
            f"checkpoint marker mismatch in {root}: expected {expected_iteration}, found {latest}"
        )

    iteration_dir = root / f"iter_{latest:07d}"
    if iteration_dir.is_symlink() or not iteration_dir.is_dir():
        raise CheckpointRetentionError(f"latest checkpoint directory is missing or unsafe: {iteration_dir}")

    required_files = (".metadata", "common.pt", "metadata.json")
    missing = [name for name in required_files if not (iteration_dir / name).is_file()]
    if missing:
        raise CheckpointRetentionError(
            f"latest checkpoint is incomplete: {iteration_dir}; missing {', '.join(missing)}"
        )
    empty = [name for name in required_files if (iteration_dir / name).stat().st_size <= 0]
    if empty:
        raise CheckpointRetentionError(
            f"latest checkpoint has empty metadata: {iteration_dir}; empty {', '.join(empty)}"
        )
    shards = [path for path in iteration_dir.glob("*.distcp") if path.is_file()]
    if not shards or any(path.stat().st_size <= 0 for path in shards):
        raise CheckpointRetentionError(f"latest checkpoint has no non-empty distcp shards: {iteration_dir}")
    temporary = [path.name for path in iteration_dir.iterdir() if path.name.endswith(".tmp") or ".tmp." in path.name]
    if temporary:
        raise CheckpointRetentionError(
            f"latest checkpoint still contains temporary files: {iteration_dir}; {', '.join(temporary)}"
        )
    return iteration_dir


def prune_checkpoint_root(
    save_dir: str | Path,
    keep_last: int,
    *,
    expected_iteration: int | None = None,
) -> list[Path]:
    """Delete old iteration directories only after validating the latest save."""

    if keep_last < 1:
        raise ValueError(f"keep_last must be >= 1, got {keep_last}")

    root = Path(save_dir).resolve()
    latest_dir = validate_latest_checkpoint(root, expected_iteration=expected_iteration)
    iterations: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = ITERATION_DIR_RE.fullmatch(child.name)
        if match and child.is_dir() and not child.is_symlink():
            iterations.append((int(match.group(1)), child))
    iterations.sort()

    if not any(path == latest_dir for _, path in iterations):
        raise CheckpointRetentionError(f"validated latest checkpoint was not discovered under save root: {latest_dir}")

    remove = [path for _, path in iterations[:-keep_last]]
    for path in remove:
        if path.parent != root:
            raise CheckpointRetentionError(f"refusing to remove checkpoint outside save root: {path}")
        shutil.rmtree(path)
    return remove


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and prune Slime checkpoint iterations")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--keep-last", required=True, type=int)
    parser.add_argument("--expected-iteration", type=int, default=None)
    args = parser.parse_args()
    removed = prune_checkpoint_root(
        args.save_dir,
        args.keep_last,
        expected_iteration=args.expected_iteration,
    )
    print(
        json.dumps(
            {
                "event": "checkpoint_retention",
                "save_dir": str(Path(args.save_dir).resolve()),
                "keep_last": args.keep_last,
                "removed": [str(path) for path in removed],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
