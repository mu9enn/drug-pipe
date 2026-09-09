#!/usr/bin/env python3
"""Validate and copy a selected v8 ToolRL stream without packing or deletion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from drug_agent.toolrl.trajectory_batching import validate_trajectory_order


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(entry: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    value = entry[1].get("metadata")
    return value if isinstance(value, dict) else {}


def materialize_toolrl_training_view(
    *,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    tokenizer: Any,
    model_name: str,
    max_prompt_tokens: int,
    max_target_tokens: int,
    rollout_batch_size: int,
) -> dict[str, Any]:
    if min(max_prompt_tokens, max_target_tokens, rollout_batch_size) < 1:
        raise ValueError("limits must be positive")
    entries: list[tuple[int, dict[str, Any]]] = []
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                entries.append((line_number, json.loads(line)))
    validate_trajectory_order(entries, metadata_of=_metadata)

    findings: list[dict[str, Any]] = []
    for line_number, record in entries:
        prompt_text = tokenizer.apply_chat_template(
            record["prompt"], tools=record.get("tools"), tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        target = record.get("label", {}).get("target_assistant")
        full_text = tokenizer.apply_chat_template(
            record["prompt"] + [target], tools=record.get("tools"), tokenize=False,
            add_generation_prompt=False, enable_thinking=True,
        )
        if not full_text.startswith(prompt_text):
            raise ValueError(f"native target is not a prompt continuation at line {line_number}")
        prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        target_tokens = len(tokenizer.encode(full_text[len(prompt_text):], add_special_tokens=False))
        if prompt_tokens > max_prompt_tokens or target_tokens > max_target_tokens:
            findings.append(
                {
                    "line_number": line_number,
                    "decision_id": record.get("id"),
                    "prompt_tokens": prompt_tokens,
                    "target_tokens": target_tokens,
                    "prompt_over_configured_limit": prompt_tokens > max_prompt_tokens,
                    "target_over_configured_limit": target_tokens > max_target_tokens,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for _, record in entries:
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    pretty_path = output_path.with_suffix(".pretty.json")
    pretty_path.write_text(json.dumps([record for _, record in entries], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "toolrl_v8_training_stream_v1",
        "contract": {
            "model": model_name,
            "max_prompt_tokens_check": max_prompt_tokens,
            "max_target_tokens_check": max_target_tokens,
            "rollout_batch_size_decisions": rollout_batch_size,
            "ordering": "canonical trajectory then decision ordinal",
            "trajectory_may_cross_batch_boundary": True,
            "selection_holes_allowed": True,
            "tail_policy": "retained; fixed-size final rollout continues with the deterministic next-epoch prefix",
            "length_policy": "report only; do not truncate or delete teacher actions",
        },
        "source": {"path": str(input_path.resolve()), "sha256": _sha256(input_path), "records": len(entries)},
        "length_findings": findings,
        "output": {"path": str(output_path.resolve()), "pretty_path": str(pretty_path.resolve()), "sha256": _sha256(output_path), "records": len(entries)},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-prompt-tokens", required=True, type=int)
    parser.add_argument("--max-target-tokens", required=True, type=int)
    parser.add_argument("--rollout-batch-size", required=True, type=int)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    result = materialize_toolrl_training_view(
        input_path=args.input.resolve(), output_path=args.output.resolve(), manifest_path=args.manifest.resolve(),
        tokenizer=tokenizer, model_name=args.model, max_prompt_tokens=args.max_prompt_tokens,
        max_target_tokens=args.max_target_tokens, rollout_batch_size=args.rollout_batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
