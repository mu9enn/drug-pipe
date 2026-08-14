#!/usr/bin/env python3
"""Build deterministic SFT length buckets for Qwen3.5-9B profiling.

The generated JSONL files contain unchanged source records.  A manifest records
the source hash, tokenizer, rendered token lengths, and selected record ids so
throughput comparisons can use exactly the same samples across topologies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rendered_length(tokenizer: Any, messages: list[dict[str, Any]]) -> int:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def _centered_slice(
    ranked: list[tuple[int, int, dict[str, Any]]], center: int, count: int
) -> list[tuple[int, int, dict[str, Any]]]:
    start = max(0, min(len(ranked) - count, center - count // 2))
    return ranked[start : start + count]


def _write_jsonl(path: Path, selected: list[tuple[int, int, dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for _, _, record in selected:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit("--count must be positive")

    records: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            if not isinstance(record.get("messages"), list):
                raise SystemExit(f"record {line_number} has no messages list")
            records.append(record)
    if len(records) < args.count:
        raise SystemExit(f"need at least {args.count} records, found {len(records)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ranked = sorted(
        (
            (_rendered_length(tokenizer, record["messages"]), index, record)
            for index, record in enumerate(records)
        ),
        key=lambda item: (item[0], item[1]),
    )
    lengths = [item[0] for item in ranked]

    def percentile(fraction: float) -> int:
        return lengths[round((len(lengths) - 1) * fraction)]

    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    model_max_length = int(config.get("max_position_embeddings") or tokenizer.model_max_length)

    last = len(ranked) - 1
    buckets = {
        "short": ranked[: args.count],
        "p50": _centered_slice(ranked, round(last * 0.50), args.count),
        "p95": _centered_slice(ranked, round(last * 0.95), args.count),
        "max": ranked[-args.count :],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": "qwen35_9b_sft_probe_sets_v2",
        "source_path": str(args.input.resolve()),
        "source_sha256": _sha256(args.input),
        "source_records": len(records),
        "model_path": str(args.model.resolve()),
        "enable_thinking": False,
        "records_per_bucket": args.count,
        "length_distribution": {
            "records": len(lengths),
            "total_tokens": sum(lengths),
            "minimum": lengths[0],
            "mean": statistics.fmean(lengths),
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "maximum": lengths[-1],
            "model_max_length": model_max_length,
            "over_131072": sum(length > 131072 for length in lengths),
            "over_245760": sum(length > 245760 for length in lengths),
            "over_262144": sum(length > 262144 for length in lengths),
            "over_model_max_length": sum(length > model_max_length for length in lengths),
        },
        "buckets": {},
    }
    manifest["over_limit_records"] = {
        str(threshold): [
            {
                "record_id": record.get("id"),
                "source_index": index,
                "token_length": length,
            }
            for length, index, record in ranked
            if length > threshold
        ]
        for threshold in sorted({131072, 245760, 262144, model_max_length})
    }
    for name, selected in buckets.items():
        output = args.output_dir / f"sft_{name}_{args.count}.jsonl"
        _write_jsonl(output, selected)
        manifest["buckets"][name] = {
            "path": str(output.resolve()),
            "sha256": _sha256(output),
            "token_lengths": [length for length, _, _ in selected],
            "source_indices": [index for _, index, _ in selected],
            "record_ids": [record.get("id") for _, _, record in selected],
            "total_tokens": sum(length for length, _, _ in selected),
        }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
