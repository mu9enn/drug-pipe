from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical_edges import build_canonical_edges
from .settings import ProjectConfig


def score_edges(config: ProjectConfig, calibration_file: Path | None = None) -> dict[str, Any]:
    """Compatibility entry point; semantic scoring has been replaced by canonicalization."""
    result = build_canonical_edges(config, calibration_file=calibration_file)
    return {
        **result,
        "compatibility_entrypoint": "score",
        "note": "No semantic score or relation rewrite is performed.",
    }
