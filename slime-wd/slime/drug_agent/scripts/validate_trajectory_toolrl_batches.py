#!/usr/bin/env python3
"""Validate canonical, non-shuffled v8 ToolRL decision traversal."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from drug_agent.toolrl.trajectory_batching import validate_trajectory_order


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def validate_file(input_path: Path, rollout_batch_size: int) -> dict[str, Any]:
    metadata_rows: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{input_path}:{line_number}: row is not an object")
            metadata_rows.append(_metadata(record))
    validate_trajectory_order(metadata_rows, metadata_of=lambda value: value)
    source_ids = {str(metadata.get("source_id") or "") for metadata in metadata_rows}
    batch_count = math.ceil(len(metadata_rows) / rollout_batch_size) if metadata_rows else 0
    tail_size = len(metadata_rows) % rollout_batch_size
    return {
        "status": "pass",
        "records": len(metadata_rows),
        "trajectories": len(source_ids),
        "rollout_batches_including_tail": batch_count,
        "tail_batch_size": tail_size or (rollout_batch_size if metadata_rows else 0),
        "rollout_batch_size": rollout_batch_size,
        "sampling_order": "canonical_trajectory_then_decision_ordinal",
        "trajectory_may_cross_batch_boundary": True,
        "selection_holes_allowed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--rollout-batch-size", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(validate_file(args.input, args.rollout_batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
