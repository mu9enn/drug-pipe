#!/usr/bin/env python3
"""Compatibility entry point for the canonical deterministic trace curator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .trace_curator import TASK_CHOICES, curate_results_dir
    from ..cleaning.llm_cleaner import build_claude_rewriter
except ImportError:
    from trace_curator import TASK_CHOICES, curate_results_dir
    from cleaning.llm_cleaner import build_claude_rewriter


def export_results_dir(results_dir: Path, task: str | None = None) -> dict[str, Any]:
    return curate_results_dir(Path(results_dir), task=task)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical ReAct trajectories from raw complete_session events.")
    parser.add_argument("results_dir")
    parser.add_argument("--task", choices=sorted(TASK_CHOICES), default="")
    parser.add_argument("--llm-clean", action="store_true")
    parser.add_argument("--claude-bin", default="claude")
    args = parser.parse_args()
    rewrite = None
    if args.llm_clean:
        prompt = (Path(__file__).resolve().parents[2] / "llm_clean" / "prompt.md").read_text(encoding="utf-8")
        rewrite = build_claude_rewriter(args.claude_bin, prompt)
    result = curate_results_dir(
        Path(args.results_dir),
        task=args.task or None,
        llm_rewrite=rewrite,
        llm_clean_required=args.llm_clean,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
