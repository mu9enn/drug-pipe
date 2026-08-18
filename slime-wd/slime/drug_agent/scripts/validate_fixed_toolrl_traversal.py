"""Fail closed unless a fixed ToolRL view was consumed exactly once."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _decision_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return (
        f"{metadata.get('source_id') or metadata.get('task_id')}:"
        f"{metadata.get('assistant_index')}:{metadata.get('assistant_subturn_index', 0)}:"
        f"{metadata.get('decision_type')}"
    )


def validate(dataset: Path, audit: Path) -> dict[str, Any]:
    expected: list[str] = []
    with dataset.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                expected.append(_decision_key(json.loads(line)))
    expected_counts = Counter(expected)
    if any(count != 1 for count in expected_counts.values()):
        raise ValueError("fixed ToolRL dataset contains duplicate decision keys")

    consumed: list[str] = []
    group_indices: list[int] = []
    cursor_pairs: list[tuple[int, int]] = []
    with audit.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("accepted_for_update", True):
                continue
            consumed.append(str(row["decision_key"]))
            group_indices.append(int(row["group_index"]))
            if row.get("dataset_cursor") is not None:
                cursor_pairs.append((int(row.get("dataset_epoch") or 0), int(row["dataset_cursor"])))

    consumed_counts = Counter(consumed)
    omitted = sorted(set(expected_counts) - set(consumed_counts))
    unexpected = sorted(set(consumed_counts) - set(expected_counts))
    duplicates = {key: count for key, count in consumed_counts.items() if count != 1}
    group_duplicates = len(group_indices) - len(set(group_indices))
    ok = not omitted and not unexpected and not duplicates and not group_duplicates and len(consumed) == len(expected)
    report = {
        "schema_version": "toolrl_fixed_traversal_audit_v1",
        "ok": ok,
        "expected_decisions": len(expected),
        "consumed_decisions": len(consumed),
        "unique_consumed_decisions": len(consumed_counts),
        "unique_group_indices": len(set(group_indices)),
        "group_index_duplicates": group_duplicates,
        "omitted_decision_keys": omitted,
        "unexpected_decision_keys": unexpected,
        "non_unit_consumption_counts": duplicates,
        "cursor_start": list(min(cursor_pairs)) if cursor_pairs else None,
        "cursor_end": list(max(cursor_pairs)) if cursor_pairs else None,
    }
    if not ok:
        raise ValueError(json.dumps(report, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = validate(args.dataset, args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
