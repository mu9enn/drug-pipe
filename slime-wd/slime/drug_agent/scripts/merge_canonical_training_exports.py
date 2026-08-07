#!/usr/bin/env python3
"""Merge two canonical ReAct JSONL exports with fail-closed overlap checks."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise SystemExit(f"{path}:{line_number}: record is not an object")
            record_id = record.get("id")
            messages = record.get("messages")
            if not isinstance(record_id, str) or not record_id.strip():
                raise SystemExit(f"{path}:{line_number}: missing non-empty id")
            if not isinstance(messages, list) or len(messages) < 3:
                raise SystemExit(f"{path}:{line_number}: missing canonical messages")
            if any(not isinstance(item, dict) for item in messages):
                raise SystemExit(f"{path}:{line_number}: message is not an object")
            if messages[0].get("role") != "system" or messages[1].get("role") != "user":
                raise SystemExit(f"{path}:{line_number}: expected system then user")
            records.append(record)
    if not records:
        raise SystemExit(f"{path}: input is empty")
    return records


def _canonical_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prompt_hash(record: dict[str, Any]) -> str:
    user = next(
        (item.get("content", "") for item in record["messages"] if item.get("role") == "user"),
        "",
    )
    normalized = re.sub(r"\s+", " ", str(user)).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _summarize(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [record["id"] for record in records]
    duplicates = sorted(item for item, count in collections.Counter(ids).items() if count > 1)
    if duplicates:
        raise SystemExit(f"{path}: duplicate ids: {duplicates[:10]}")
    system_hashes = collections.Counter(
        hashlib.sha256(str(record["messages"][0].get("content", "")).encode("utf-8")).hexdigest()
        for record in records
    )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "records": len(records),
        "unique_ids": len(ids),
        "unique_normalized_prompts": len({_prompt_hash(record) for record in records}),
        "schema_versions": dict(collections.Counter(str(record.get("schema_version")) for record in records)),
        "system_prompt_sha256_counts": dict(sorted(system_hashes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--additional", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    for source in (args.base, args.additional):
        if source.resolve() in {args.output.resolve(), args.manifest.resolve()}:
            raise SystemExit("output/manifest must not overwrite either source")

    base = _load(args.base)
    additional = _load(args.additional)
    base_ids = {record["id"] for record in base}
    additional_ids = {record["id"] for record in additional}
    id_overlap = sorted(base_ids & additional_ids)
    prompt_overlap = sorted({_prompt_hash(record) for record in base} & {_prompt_hash(record) for record in additional})
    exact_overlap = sorted({_canonical_hash(record) for record in base} & {_canonical_hash(record) for record in additional})
    if id_overlap or prompt_overlap or exact_overlap:
        raise SystemExit(
            "source exports overlap: "
            f"ids={len(id_overlap)} prompts={len(prompt_overlap)} exact={len(exact_overlap)}"
        )

    merged = base + additional
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in merged:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "schema_version": "canonical_react_merge_v1",
        "merge_policy": "base_then_additional_fail_on_id_prompt_or_exact_overlap",
        "sources": {
            "base": _summarize(args.base, base),
            "additional": _summarize(args.additional, additional),
        },
        "overlap": {"ids": 0, "normalized_prompts": 0, "exact_records": 0},
        "output": {
            "path": str(args.output.resolve()),
            "sha256": _sha256(args.output),
            "records": len(merged),
            "unique_ids": len({record["id"] for record in merged}),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
