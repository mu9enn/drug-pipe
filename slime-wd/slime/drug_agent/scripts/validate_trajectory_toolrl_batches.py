#!/usr/bin/env python3
"""Validate that ToolRL rows are packed as complete trajectory batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from drug_agent.toolrl.trajectory_batching import validate_packed_decision_batches


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def validate_file(input_path: Path, rollout_batch_size: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{input_path}:{line_number}: row is not an object")
            records.append(record)
    batches = validate_packed_decision_batches(
        records,
        metadata_of=_metadata,
        rollout_batch_size=rollout_batch_size,
    )
    source_ids = {str(_metadata(record).get("source_id") or "") for record in records}
    return {
        "status": "pass",
        "records": len(records),
        "trajectories": len(source_ids),
        "trajectory_batches": len(batches),
        "rollout_batch_size": rollout_batch_size,
        "sampling_unit": "complete_trajectory",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--rollout-batch-size", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(validate_file(args.input, args.rollout_batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
