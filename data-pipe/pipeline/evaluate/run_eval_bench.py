#!/usr/bin/env python3
"""Evaluate MolBench run outputs."""
from __future__ import annotations

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from evaluate.eval_runner import eval_results_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MolBench evaluation on results_dir.")
    parser.add_argument("results_dir", help="Run directory produced by claude_agent/run_claude.py")
    parser.add_argument("--task", choices=["vs", "ac", "pf"], default="", help="Task type. Auto infer if omitted.")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_dir):
        raise NotADirectoryError(results_dir)

    scores = eval_results_dir(results_dir, task=args.task or None)
    out_path = os.path.join(results_dir, "bench_scores.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    print(json.dumps(scores, ensure_ascii=False, indent=2))
    print(f"SCORES_FILE={out_path}")


if __name__ == "__main__":
    main()
