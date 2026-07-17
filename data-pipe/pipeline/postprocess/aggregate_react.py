#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _answer_hit(record: dict[str, Any]) -> bool:
    task = str((record.get("metadata") or {}).get("task") or "")
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    if task == "vs":
        return float(metrics.get("top3_hit_num") or 0.0) >= 1.0
    if task == "ac":
        return metrics.get("is_correct") is True
    if task == "pf":
        return bool(metrics.get("exact_set_match"))
    return True


def aggregate_react(
    results_root: Path,
    output_root: Path,
    *,
    answer_hit_only: bool = False,
) -> dict[str, Any]:
    results_root = results_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    accepted_by_id: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    duplicate_ids: list[dict[str, Any]] = []
    filtered = 0

    for source in sorted(results_root.rglob("trajectories/react_trajectories.jsonl")):
        for record in _read_jsonl(source):
            record_id = str(record.get("id") or "")
            if not record_id:
                rejected.append({"source": str(source), "reason": "missing_id", "record": record})
                continue
            if answer_hit_only and not _answer_hit(record):
                filtered += 1
                continue
            existing = accepted_by_id.get(record_id)
            if existing is not None and existing != record:
                duplicate_ids.append({"id": record_id, "source": str(source), "reason": "nonidentical_duplicate"})
                continue
            accepted_by_id[record_id] = record

    for source in sorted(results_root.rglob("trajectories/react_rejected.jsonl")):
        rejected.extend(_read_jsonl(source))

    accepted = [accepted_by_id[key] for key in sorted(accepted_by_id)]
    accepted_path = output_root / "react_trajectories.jsonl"
    rejected_path = output_root / "react_rejected.jsonl"
    duplicate_path = output_root / "duplicate_id_conflicts.jsonl"
    for path, rows in [
        (accepted_path, accepted),
        (rejected_path, rejected),
        (duplicate_path, duplicate_ids),
    ]:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    task_counts = Counter(str((row.get("metadata") or {}).get("task") or "unknown") for row in accepted)
    report = {
        "target_contract": "react_trajectory_v2",
        "source": str(results_root),
        "input_file_count": len(list(results_root.rglob("trajectories/react_trajectories.jsonl"))),
        "output_count": len(accepted),
        "rejected_count": len(rejected),
        "duplicate_conflict_count": len(duplicate_ids),
        "answer_hit_filtered_count": filtered,
        "task_counts": dict(sorted(task_counts.items())),
        "output": str(accepted_path),
        "rejected_output": str(rejected_path),
    }
    (output_root / "curation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate canonical ReAct trajectories without re-curating them.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--answer-hit-only", action="store_true")
    args = parser.parse_args()
    result = aggregate_react(
        Path(args.results_root),
        Path(args.output_root),
        answer_hit_only=args.answer_hit_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
