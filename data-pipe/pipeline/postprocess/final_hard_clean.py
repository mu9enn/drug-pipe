#!/usr/bin/env python3
"""Compatibility CLI for deterministic hard-clean findings.

This stage never decides accepted/rejected/quarantine; the final acceptance gate
owns that status.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_PIPE_DIR = Path(__file__).resolve().parents[2]
if str(DATA_PIPE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPE_DIR))

from pipeline.cleaning.hard_cleaner import hard_clean  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard-clean canonical ReAct JSONL and write findings.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    audit = Path(args.audit).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open("r", encoding="utf-8") as input_handle, output.open(
        "w", encoding="utf-8"
    ) as output_handle, audit.open("w", encoding="utf-8") as audit_handle:
        for line in input_handle:
            if not line.strip():
                continue
            if args.limit > 0 and count >= args.limit:
                break
            sample = json.loads(line)
            cleaned, report = hard_clean(sample)
            output_handle.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            audit_handle.write(json.dumps({"id": sample.get("id"), **report}, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"processed": count, "output": str(output), "audit": str(audit)}, indent=2))


if __name__ == "__main__":
    main()
