"""Retention helpers for GAD discriminator checkpoints."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


class DiscriminatorCheckpointError(RuntimeError):
    pass


def validate_discriminator_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path).resolve()
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise DiscriminatorCheckpointError(f"discriminator checkpoint is missing or unsafe: {checkpoint}")
    required = (checkpoint / "backbone", checkpoint / "gad_state.pt", checkpoint / "metadata.json")
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise DiscriminatorCheckpointError(
            f"discriminator checkpoint is incomplete: {checkpoint}; missing {', '.join(missing)}"
        )
    if any(item.is_file() and item.stat().st_size <= 0 for item in required):
        raise DiscriminatorCheckpointError(f"discriminator checkpoint contains an empty required file: {checkpoint}")
    backbone = checkpoint / "backbone"
    if not backbone.is_dir() or not any(path.is_file() and path.stat().st_size > 0 for path in backbone.rglob("*")):
        raise DiscriminatorCheckpointError(f"discriminator backbone is empty: {checkpoint}")
    return checkpoint


def prune_numbered_checkpoints(
    output_dir: str | Path,
    *,
    prefix: str,
    keep_last: int,
    newest: str | Path | None = None,
) -> list[Path]:
    if keep_last < 0:
        raise ValueError(f"keep_last must be >= 0, got {keep_last}")
    root = Path(output_dir).resolve()
    if newest is not None:
        validate_discriminator_checkpoint(newest)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = pattern.fullmatch(child.name)
        if match and child.is_dir() and not child.is_symlink():
            candidates.append((int(match.group(1)), child))
    candidates.sort()
    remove = candidates if keep_last == 0 else candidates[:-keep_last]
    for _, path in remove:
        if path.parent != root:
            raise DiscriminatorCheckpointError(f"refusing to remove checkpoint outside output root: {path}")
        shutil.rmtree(path)
    return [path for _, path in remove]
