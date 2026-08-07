#!/usr/bin/env python3
"""Materialize a lossless, batch-aligned SFT JSONL with an audit manifest.

Slime's epoch-mode SFT consumes complete rollout batches.  When a dataset size
is not divisible by the required global/data-parallel batch multiple, this
utility appends the shortest unchanged source records.  It never mutates or
drops a canonical record, and records every duplicate in a hash-bound manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rendered_length(tokenizer: Any, record: dict[str, Any], line_number: int) -> int:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise SystemExit(f"record {line_number} has no messages list")
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--multiple", required=True, type=int)
    args = parser.parse_args()

    if args.multiple <= 0:
        raise SystemExit("--multiple must be positive")
    if args.output.resolve() == args.input.resolve():
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
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ranked = sorted(
        (
            (_rendered_length(tokenizer, record, index + 1), index, record)
            for index, record in enumerate(records)
        ),
        key=lambda item: (item[0], item[1]),
    )
    duplicates = ranked[:pad_count]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        for _, _, record in duplicates:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "schema_version": "batch_aligned_sft_v1",
        "source_path": str(args.input.resolve()),
        "source_sha256": _sha256(args.input),
        "source_records": len(records),
        "output_path": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "output_records": len(records) + pad_count,
        "required_multiple": args.multiple,
        "padding_records": pad_count,
        "padding_policy": "append_shortest_unchanged_source_records",
        "enable_thinking": False,
        "model_path": str(args.model.resolve()),
        "duplicates": [
            {
                "source_index": index,
                "record_id": record.get("id"),
                "rendered_tokens": length,
            }
            for length, index, record in duplicates
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
