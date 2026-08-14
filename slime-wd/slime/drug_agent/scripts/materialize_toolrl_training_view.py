#!/usr/bin/env python3
"""Build an audited, capacity-safe, batch-aligned ToolRL training view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decision_key(record: dict[str, Any]) -> tuple[str, int, str]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return (
        str(metadata.get("source_id") or metadata.get("task_id") or ""),
        int(metadata.get("assistant_index", -1)),
        str(metadata.get("decision_type") or ""),
    )


def _target_text(record: dict[str, Any]) -> str:
    label = record.get("label") if isinstance(record.get("label"), dict) else {}
    target = record.get("target_assistant") if isinstance(record.get("target_assistant"), dict) else {}
    value = label.get("assistant_content", target.get("content"))
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing assistant target for decision {_decision_key(record)!r}")
    return value


def materialize_toolrl_training_view(
    *,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    tokenizer: Any,
    model_name: str,
    max_prompt_tokens: int,
    max_target_tokens: int,
    multiple: int,
    batch_size: int = 8,
) -> dict[str, Any]:
    if min(max_prompt_tokens, max_target_tokens, multiple, batch_size) < 1:
        raise ValueError("token limits, multiple, and batch size must be positive")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("capacity view must not overwrite its source")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    source_sha256 = _sha256(input_path)
    source_records = 0
    accepted_records = 0
    rejected: list[dict[str, Any]] = []
    shortest: list[tuple[int, int, str, tuple[str, int, str]]] = []

    def consume(batch: list[tuple[int, str, dict[str, Any]]], output: Any) -> None:
        nonlocal accepted_records
        rendered_prompts = [
            tokenizer.apply_chat_template(
                record["prompt"],
                tools=record.get("tools"),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for _, _, record in batch
        ]
        targets = [_target_text(record) for _, _, record in batch]
        prompt_ids = tokenizer(rendered_prompts, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(targets, add_special_tokens=False)["input_ids"]
        for (line_number, line, record), prompt_tokens, target_tokens in zip(
            batch, prompt_ids, target_ids, strict=True
        ):
            prompt_length = len(prompt_tokens)
            target_length = len(target_tokens)
            reasons = []
            if prompt_length > max_prompt_tokens:
                reasons.append("prompt_exceeds_max_tokens")
            if target_length > max_target_tokens:
                reasons.append("target_exceeds_max_tokens")
            key = _decision_key(record)
            if reasons:
                metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
                rejected.append(
                    {
                        "line_number": line_number,
                        "source_id": key[0],
                        "assistant_index": key[1],
                        "decision_type": key[2],
                        "task_type": metadata.get("task_type"),
                        "tool_names": metadata.get("tool_names") or [],
                        "prompt_tokens": prompt_length,
                        "target_tokens": target_length,
                        "reasons": reasons,
                    }
                )
                continue
            output.write(line if line.endswith("\n") else line + "\n")
            accepted_records += 1
            shortest.append((prompt_length + target_length, line_number, line, key))
            shortest.sort(key=lambda item: (item[0], item[1]))
            del shortest[multiple - 1 :]

    try:
        with input_path.open(encoding="utf-8") as source, tmp_path.open("w", encoding="utf-8") as output:
            batch: list[tuple[int, str, dict[str, Any]]] = []
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                source_records += 1
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{input_path}:{line_number}: row is not an object")
                batch.append((line_number, line, record))
                if len(batch) == batch_size:
                    consume(batch, output)
                    batch = []
            if batch:
                consume(batch, output)

            padding_records = (-accepted_records) % multiple
            if padding_records > len(shortest):
                raise RuntimeError("not enough accepted records to align output")
            duplicates = shortest[:padding_records]
            for _, _, line, _ in duplicates:
                output.write(line if line.endswith("\n") else line + "\n")
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    reason_counts: dict[str, int] = {}
    for row in rejected:
        key = "+".join(row["reasons"])
        reason_counts[key] = reason_counts.get(key, 0) + 1
    output_records = accepted_records + padding_records
    manifest = {
        "schema_version": "toolrl_capacity_aligned_view_v1",
        "contract": {
            "model": model_name,
            "apply_chat_template": True,
            "apply_chat_template_kwargs": {"enable_thinking": False},
            "add_generation_prompt": True,
            "max_prompt_tokens": max_prompt_tokens,
            "max_target_tokens": max_target_tokens,
            "policy": "exclude whole decisions; never truncate prompts or targets",
        },
        "source": {"path": str(input_path.resolve()), "sha256": source_sha256, "records": source_records},
        "accepted_records": accepted_records,
        "rejected_records": len(rejected),
        "rejection_reason_counts": reason_counts,
        "output": {
            "path": str(output_path.resolve()),
            "sha256": _sha256(output_path),
            "records": output_records,
            "required_multiple": multiple,
        },
        "padding_records": padding_records,
        "padding_policy": "append_shortest_unchanged_accepted_decisions",
        "duplicates": [
            {"source_line_number": line_number, "decision_key": list(key), "combined_tokens": length}
            for length, line_number, _, key in duplicates
        ],
        "rejected": rejected,
    }
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    return manifest


def main() -> None:
    # Keep the reusable materializer importable in lightweight validation
    # environments that do not have the worker's full torch/vision stack.
    from slime.utils.processing_utils import load_tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-prompt-tokens", required=True, type=int)
    parser.add_argument("--max-target-tokens", required=True, type=int)
    parser.add_argument("--multiple", required=True, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model, trust_remote_code=True)
    manifest = materialize_toolrl_training_view(
        input_path=args.input.resolve(),
        output_path=args.output.resolve(),
        manifest_path=args.manifest.resolve(),
        tokenizer=tokenizer,
        model_name=args.model,
        max_prompt_tokens=args.max_prompt_tokens,
        max_target_tokens=args.max_target_tokens,
        multiple=args.multiple,
        batch_size=args.batch_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
