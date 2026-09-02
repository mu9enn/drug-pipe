"""Separate optimizer-step checkpoint names from Slime rollout cursors."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "slime_checkpoint_progress_v1"
PROGRESS_DIR_NAME = "slime_checkpoint_progress"
ITERATION_DIR_RE = re.compile(r"^iter_(\d{7})$")


def _checkpoint_root(path: str | Path) -> Path:
    candidate = Path(path).resolve()
    if ITERATION_DIR_RE.fullmatch(candidate.name):
        return candidate.parent
    return candidate


def progress_path(save_dir: str | Path, checkpoint_iteration: int) -> Path:
    return _checkpoint_root(save_dir) / PROGRESS_DIR_NAME / f"iter_{checkpoint_iteration:07d}.json"


def write_checkpoint_progress(
    save_dir: str | Path,
    *,
    checkpoint_iteration: int,
    optimizer_step: int,
    rollout_id: int,
) -> Path:
    """Atomically persist the two counters associated with a checkpoint."""

    if checkpoint_iteration != optimizer_step:
        raise ValueError(
            "checkpoint_iteration must equal optimizer_step so iter_* names report "
            f"completed optimizer updates, got {checkpoint_iteration=} {optimizer_step=}"
        )
    if optimizer_step < 0 or rollout_id < 0:
        raise ValueError(f"checkpoint counters must be non-negative, got {optimizer_step=} {rollout_id=}")

    path = progress_path(save_dir, checkpoint_iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_iteration": checkpoint_iteration,
        "optimizer_step": optimizer_step,
        "rollout_id": rollout_id,
    }
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def read_checkpoint_progress(
    load_dir: str | Path,
    *,
    checkpoint_iteration: int,
) -> dict[str, Any] | None:
    """Read and validate Slime progress metadata for one Megatron checkpoint."""

    path = progress_path(load_dir, checkpoint_iteration)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint progress schema in {path}: {payload.get('schema_version')!r}")
    expected = {
        "checkpoint_iteration": checkpoint_iteration,
        "optimizer_step": checkpoint_iteration,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"checkpoint progress mismatch in {path}: expected {key}={value}, found {payload.get(key)!r}")
    rollout_id = payload.get("rollout_id")
    if not isinstance(rollout_id, int) or rollout_id < 0:
        raise ValueError(f"invalid rollout_id in {path}: {rollout_id!r}")
    return payload
