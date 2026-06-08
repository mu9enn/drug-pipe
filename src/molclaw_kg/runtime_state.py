from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl


_ATTEMPT_RE = re.compile(r"^attempt_(\d{4})$")


def latest_jsonl_by_key(path: Path, key_field: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    for row in read_jsonl(path):
        key = str(row.get(key_field) or "").strip()
        if key:
            latest[key] = row
    return latest


def next_attempt_dir(unit_dir: Path) -> Path:
    unit_dir.mkdir(parents=True, exist_ok=True)
    max_idx = -1
    for child in unit_dir.iterdir():
        if not child.is_dir():
            continue
        m = _ATTEMPT_RE.match(child.name)
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except ValueError:
            continue
        max_idx = max(max_idx, idx)
    attempt_dir = unit_dir / f"attempt_{max_idx + 1:04d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    return attempt_dir
