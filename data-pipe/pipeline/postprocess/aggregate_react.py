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


def _answer_hit(audit: dict[str, Any]) -> bool:
    task = str(audit.get("task") or "")
    metrics = audit.get("task_metrics") if isinstance(audit.get("task_metrics"), dict) else {}
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
    audit_by_id: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    filtered = 0
    source_files = [
        source
        for source in sorted(results_root.rglob("trajectories/react_trajectories.jsonl"))
        if output_root not in source.parents
    ]

    for source in source_files:
        audit_path = source.with_name("curation_audit.jsonl")
        source_audits = {
            str(row.get("id") or ""): row
            for row in _read_jsonl(audit_path)
        } if audit_path.is_file() else {}
        audit_by_id.update(source_audits)
        for record in _read_jsonl(source):
            record_id = str(record.get("id") or "")
            if not record_id:
                rejected.append({"source": str(source), "reason": "missing_id", "record": record})
                continue
            if set(record) != {"schema_version", "id", "messages"}:
                rejected.append({"id": record_id, "source": str(source), "reason": "noncanonical_training_shape"})
                continue
            audit = source_audits.get(record_id)
            if audit is None:
                rejected.append({"id": record_id, "source": str(source), "reason": "missing_curation_audit"})
                continue
            if answer_hit_only and not _answer_hit(audit):
                filtered += 1
                continue
            existing = accepted_by_id.get(record_id)
            if existing is not None and existing != record:
                rejected.append({"id": record_id, "source": str(source), "reason": "nonidentical_duplicate"})
                continue
            accepted_by_id[record_id] = record

    for source in sorted(results_root.rglob("trajectories/rejected.jsonl")):
        if output_root in source.parents:
            continue
        rejected.extend(_read_jsonl(source))
    for source in sorted(results_root.rglob("trajectories/quarantine.jsonl")):
        if output_root in source.parents:
            continue
        quarantine.extend(_read_jsonl(source))

    accepted = [accepted_by_id[key] for key in sorted(accepted_by_id)]
    audit_rows = [audit_by_id[key] for key in sorted(audit_by_id)]
    accepted_path = output_root / "react_trajectories.jsonl"
    audit_path = output_root / "curation_audit.jsonl"
    rejected_path = output_root / "rejected.jsonl"
    for path, rows in [
        (accepted_path, accepted),
        (audit_path, audit_rows),
        (rejected_path, rejected),
    ]:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    quarantine_path = output_root / "quarantine.jsonl"
    if quarantine:
        with quarantine_path.open("w", encoding="utf-8") as handle:
            for row in quarantine:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif quarantine_path.exists():
        quarantine_path.unlink()

    task_counts = Counter(
        str(audit_by_id.get(str(row.get("id") or ""), {}).get("task") or "unknown")
        for row in accepted
    )
    report = {
        "target_contract": "drug_agent_sft_react_json_v1",
        "source": str(results_root),
        "input_file_count": len(source_files),
        "output_count": len(accepted),
        "rejected_count": len(rejected),
        "quarantine_count": len(quarantine),
        "answer_hit_filtered_count": filtered,
        "task_counts": dict(sorted(task_counts.items())),
        "outputs": {
            "react_trajectories": str(accepted_path),
            "curation_audit": str(audit_path),
            "rejected": str(rejected_path),
            **({"quarantine": str(quarantine_path)} if quarantine else {}),
        },
    }
    (output_root / "curation_summary.json").write_text(
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
