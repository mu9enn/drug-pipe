#!/usr/bin/env python3
"""Compatibility entry point for the canonical deterministic trace curator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .trace_curator import TASK_CHOICES, curate_results_dir
except ImportError:
    from trace_curator import TASK_CHOICES, curate_results_dir


def export_results_dir(results_dir: Path, task: str | None = None) -> dict[str, Any]:
    return curate_results_dir(Path(results_dir), task=task)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical ReAct trajectories from raw complete_session events.")
    parser.add_argument("results_dir")
    parser.add_argument("--task", choices=sorted(TASK_CHOICES), default="")
    args = parser.parse_args()
    print(json.dumps(export_results_dir(Path(args.results_dir), args.task or None), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
