#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from .trace_curator import SCHEMA_VERSION, curate_sample, discover_rollout_samples, infer_task
except ImportError:
    from trace_curator import SCHEMA_VERSION, curate_sample, discover_rollout_samples, infer_task
try:
    from ..cleaning.acceptance_gate import decide_final_status
    from ..evaluate.task_evaluator import load_chemistry_module
except ImportError:
    from cleaning.acceptance_gate import decide_final_status
    from evaluate.task_evaluator import load_chemistry_module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hash(paths: list[Path]) -> str:
    payload = {str(path.resolve()): _sha256_file(path) for path in sorted(paths)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _read_jsonl_tolerant(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected.append({"line_number": line_number, "reason": f"JSONDecodeError: {exc}"})
                continue
            if not isinstance(value, dict):
                rejected.append({"line_number": line_number, "reason": "row is not an object"})
                continue
            rows.append(value)
    return rows, rejected


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def migrate_legacy_sft(source: Path, output_dir: Path) -> dict[str, Any]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    rows, rejected = _read_jsonl_tolerant(source)
    migrated: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    message_hash_changes = 0
    for line_number, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            rejected.append({"line_number": line_number, "id": row.get("id"), "reason": "missing_messages"})
            continue
        before_hash = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        copied_messages = deepcopy(messages)
        after_hash = hashlib.sha256(json.dumps(copied_messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        message_hash_changes += int(before_hash != after_hash)
        sample_id = str(row.get("id") or f"legacy_{line_number:06d}")
        task = next((name for name in ("vs", "ac", "pf", "kg", "e2e") if f"_{name}_" in sample_id), "unknown")
        migrated.append({"schema_version": SCHEMA_VERSION, "id": sample_id, "messages": copied_messages})
        decision = decide_final_status(
            execution_valid=True,
            task_answer_valid=True,
            training_trace_valid=True,
            llm_clean_status="not_run",
            llm_clean_findings=[],
            hard_clean_findings=[],
        )
        audits.append(
            {
                "id": sample_id,
                "task": task,
                "final_status": decision["final_status"],
                "final_status_authority": decision["authority"],
                "final_status_reasons": decision["reasons"],
                "migration_source_schema": row.get("schema_version"),
                "status_source": "legacy_accepted_sft",
                "message_sha256": before_hash,
            }
        )

    _write_jsonl(output_dir / "react_trajectories.jsonl", migrated)
    _write_jsonl(output_dir / "curation_audit.jsonl", audits)
    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    report = {
        "source": str(source),
        "source_hash": _source_hash([source]),
        "target_contract": SCHEMA_VERSION,
        "input_count": len(rows),
        "output_count": len(migrated),
        "rejected_count": len(rejected),
        "conflict_count": message_hash_changes,
        "semantic_changes": [],
        "notes": [
            "messages were copied byte-semantically without role/content/order changes",
            "legacy accepted status was preserved; benchmark metrics were not recomputed",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "migration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def migrate_raw_reference(source_root: Path, output_dir: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    source_files = sorted(source_root.rglob("complete_session.jsonl"))
    records: list[dict[str, Any]] = []
    chemistry, chemistry_error = load_chemistry_module()
    for run_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        task = infer_task(run_dir)
        for sample in discover_rollout_samples(run_dir):
            record = curate_sample(sample, default_task=task, chemistry=chemistry)
            records.append(record)
    accepted = [record for record in records if record["audit"]["final_status"] == "accepted"]
    rejected = [record["audit"] for record in records if record["audit"]["final_status"] == "rejected"]
    quarantine = [record["audit"] for record in records if record["audit"]["final_status"] == "quarantine"]
    _write_jsonl(
        output_dir / "react_trajectories.jsonl",
        [record["training_record"] for record in accepted],
    )
    _write_jsonl(output_dir / "curation_audit.jsonl", [record["audit"] for record in records])
    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    if quarantine:
        _write_jsonl(output_dir / "quarantine.jsonl", quarantine)
    report = {
        "source": str(source_root),
        "source_hash": _source_hash(source_files),
        "target_contract": SCHEMA_VERSION,
        "input_count": len(source_files),
        "output_count": len(accepted),
        "rejected_count": len(rejected),
        "quarantine_count": len(quarantine),
        "conflict_count": 0,
        "chemistry_error": chemistry_error,
        "semantic_changes": [
            "workspace/debug calls removed from training messages",
            "MolClaw usage and three validity states materialized once",
            "final answer wrapped in canonical final_answer tag",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "migration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically migrate historical trace assets.")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    sft = subparsers.add_parser("legacy-sft")
    sft.add_argument("--source", required=True)
    sft.add_argument("--output-dir", required=True)
    raw = subparsers.add_parser("raw-reference")
    raw.add_argument("--source-root", required=True)
    raw.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.mode == "legacy-sft":
        result = migrate_legacy_sft(Path(args.source), Path(args.output_dir))
    else:
        result = migrate_raw_reference(Path(args.source_root), Path(args.output_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
