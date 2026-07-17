#!/usr/bin/env python3
"""Compatibility report for usage already computed by the canonical curator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .aggregate_react import aggregate_react
except ImportError:
    from aggregate_react import aggregate_react


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate curator records; MolClaw usage and metrics are not recomputed."
    )
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--csv-name", default="molclaw_usage_summary.csv")
    parser.add_argument("--use-accepted-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    result = aggregate_react(Path(args.results_root), Path(args.output_root))
    result["compatibility_entrypoint"] = "scan_molclaw_usage"
    result["note"] = "Usage and metrics were consumed from canonical curator records."
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
