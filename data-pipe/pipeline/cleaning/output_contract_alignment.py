"""Historical schema migration only; evidence-based repairs use release_aligned."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from pipeline.cleaning.invariants import validate_semantic_record
from pipeline.cleaning.io import base_manifest, read_jsonl, write_json, write_jsonl, write_pretty_json
from pipeline.output_contracts import ANSWER_KEYS, CONTRACTS, normalize_final_answer, normalize_task_prompt, task_constraints, strict_json_loads


def align_record(record: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(record)
    task_type = str((output.get("metadata") or {}).get("task_type") or "").lower()
    if task_type not in CONTRACTS:
        raise ValueError(f"{record.get('id')}: unsupported task type {task_type!r}")
    output["user_task"] = normalize_task_prompt(str(output.get("user_task") or ""), task_type)
    finals = [event for event in output.get("events") or [] if event.get("final_answer") is not None]
    if len(finals) != 1:
        raise ValueError(f"{record.get('id')}: expected one terminal answer, found {len(finals)}")
    try:
        original_answer = str(finals[0]["final_answer"])
        legacy_payload = strict_json_loads(original_answer)
        answer_key = ANSWER_KEYS[task_type]
        if not isinstance(legacy_payload, dict):
            raise ValueError("terminal answer is not a JSON object")
        legacy_payload = {
            answer_key: legacy_payload.get(answer_key),
            "evidence": legacy_payload.get("evidence"),
        }
        finals[0]["final_answer"] = normalize_final_answer(
            json.dumps(legacy_payload, ensure_ascii=False), task_type,
            constraints=task_constraints(output['user_task'], task_type),
        )
        output.setdefault('metadata', {})['historical_schema_alignment'] = {
            'original_answer_sha256': hashlib.sha256(original_answer.encode()).hexdigest(),
            'method': 'remove_legacy_envelope_fields_without_changing_answer',
        }
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{record.get('id')}: {exc}") from exc
    validation = validate_semantic_record(output)
    if not validation["ok"]:
        raise ValueError(f"{record.get('id')}: aligned semantic validation failed: {validation['errors']}")
    return output


def align_file(input_path: Path, output_root: Path) -> dict[str, Any]:
    records, parse_errors = read_jsonl(input_path)
    if parse_errors:
        raise ValueError(f"invalid semantic JSONL: {parse_errors}")
    aligned = [align_record(record) for record in records]
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "semantic_trajectories.jsonl"
    write_jsonl(output_path, aligned)
    write_pretty_json(output_root / "semantic_trajectories.pretty.json", aligned)
    manifest = {
        **base_manifest(
            step="output_contract_alignment",
            source=input_path.resolve(),
            repo_root=Path(__file__).resolve().parents[3],
        ),
        "input_count": len(records),
        "output_count": len(aligned),
        "contracts": {key: [value] for key, value in CONTRACTS.items()},
        "output": str(output_path.resolve()),
    }
    write_json(output_root / "alignment_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Align terminal targets and user output contracts.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(align_file(args.input.resolve(), args.output_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
