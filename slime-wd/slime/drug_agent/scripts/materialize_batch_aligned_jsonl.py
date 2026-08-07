#!/usr/bin/env python3
"""Create an auditable JSONL derivative aligned to an arbitrary batch multiple."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--multiple", required=True, type=int)
    args = parser.parse_args()

    if args.multiple <= 0:
        raise SystemExit("--multiple must be positive")
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("--output must not overwrite the canonical input")

    records: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise SystemExit(f"record {line_number} is not a JSON object")
            records.append(record)
    if not records:
        raise SystemExit("input dataset is empty")

    pad_count = (-len(records)) % args.multiple
    duplicates = [(index, records[index]) for index in range(pad_count)]
    output_records = records + [record for _, record in duplicates]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "schema_version": "batch_aligned_jsonl_v1",
        "source_path": str(args.input.resolve()),
        "source_sha256": _sha256(args.input),
        "source_records": len(records),
        "output_path": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "output_records": len(output_records),
        "required_multiple": args.multiple,
        "padding_records": pad_count,
        "padding_policy": "append_first_unchanged_source_records",
        "duplicates": [
            {
                "source_index": index,
                "record_id": record.get("id")
                or (record.get("metadata") or {}).get("sample_id")
                or (record.get("metadata") or {}).get("source_id"),
            }
            for index, record in duplicates
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
