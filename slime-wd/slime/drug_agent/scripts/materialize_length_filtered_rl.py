#!/usr/bin/env python3
"""Materialize an audited RL view whose prompts fit the rollout contract.

ToolRL and GAD are derived from the same assistant decisions.  This utility
walks the two releases in lockstep, verifies that their decision identities
and prompts agree, measures the exact Qwen chat-template token length, and
writes both accepted views atomically.  Source releases are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any

from slime.utils.processing_utils import load_tokenizer


def _decision_key(record: dict[str, Any]) -> tuple[str, int, str]:
    metadata = record.get("metadata") or {}
    return (
        str(metadata.get("source_id") or metadata.get("task_id") or ""),
        int(metadata.get("assistant_index", -1)),
        str(metadata.get("decision_type") or ""),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolrl-input", type=Path, required=True)
    parser.add_argument("--gad-input", type=Path, required=True)
    parser.add_argument("--toolrl-output", type=Path, required=True)
    parser.add_argument("--gad-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-prompt-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_prompt_tokens < 1 or args.batch_size < 1:
        raise ValueError("max-prompt-tokens and batch-size must be positive")
    for path in (args.toolrl_input, args.gad_input):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.toolrl_output == args.toolrl_input or args.gad_output == args.gad_input:
        raise ValueError("length-filtered outputs must not overwrite source releases")

    tokenizer = load_tokenizer(args.model, trust_remote_code=True)
    args.toolrl_output.parent.mkdir(parents=True, exist_ok=True)
    args.gad_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    toolrl_tmp = args.toolrl_output.with_name(f".{args.toolrl_output.name}.tmp.{os.getpid()}")
    gad_tmp = args.gad_output.with_name(f".{args.gad_output.name}.tmp.{os.getpid()}")

    source_records = 0
    kept_records = 0
    rejected: list[dict[str, Any]] = []
    toolrl_source_digest = hashlib.sha256()
    gad_source_digest = hashlib.sha256()

    try:
        with (
            args.toolrl_input.open(encoding="utf-8") as toolrl_in,
            args.gad_input.open(encoding="utf-8") as gad_in,
            toolrl_tmp.open("w", encoding="utf-8") as toolrl_out,
            gad_tmp.open("w", encoding="utf-8") as gad_out,
        ):
            paired = itertools.zip_longest(toolrl_in, gad_in)
            while True:
                batch: list[tuple[int, str, str, dict[str, Any], dict[str, Any]]] = []
                for _ in range(args.batch_size):
                    pair = next(paired, None)
                    if pair is None:
                        break
                    toolrl_line, gad_line = pair
                    source_records += 1
                    if toolrl_line is None or gad_line is None:
                        raise ValueError("ToolRL and GAD releases have different record counts")
                    toolrl_source_digest.update(toolrl_line.encode("utf-8"))
                    gad_source_digest.update(gad_line.encode("utf-8"))
                    toolrl_record = json.loads(toolrl_line)
                    gad_record = json.loads(gad_line)
                    toolrl_key = _decision_key(toolrl_record)
                    gad_key = _decision_key(gad_record)
                    if toolrl_key != gad_key:
                        raise ValueError(
                            f"decision mismatch at line {source_records}: {toolrl_key!r} != {gad_key!r}"
                        )
                    if toolrl_record.get("prompt") != gad_record.get("prompt"):
                        raise ValueError(f"prompt mismatch at line {source_records}: {toolrl_key!r}")
                    batch.append((source_records, toolrl_line, gad_line, toolrl_record, gad_record))
                if not batch:
                    break

                rendered = [
                    tokenizer.apply_chat_template(
                        item[3]["prompt"],
                        tools=item[3].get("tools"),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for item in batch
                ]
                input_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
                for item, ids in zip(batch, input_ids, strict=True):
                    line_number, toolrl_line, gad_line, toolrl_record, _ = item
                    prompt_tokens = len(ids)
                    if prompt_tokens <= args.max_prompt_tokens:
                        toolrl_out.write(toolrl_line)
                        gad_out.write(gad_line)
                        kept_records += 1
                        continue
                    metadata = toolrl_record.get("metadata") or {}
                    rejected.append(
                        {
                            "line_number": line_number,
                            "source_id": _decision_key(toolrl_record)[0],
                            "assistant_index": _decision_key(toolrl_record)[1],
                            "decision_type": _decision_key(toolrl_record)[2],
                            "task_type": metadata.get("task_type"),
                            "tool_names": metadata.get("tool_names") or [],
                            "prompt_tokens": prompt_tokens,
                            "reason": "prompt_exceeds_rollout_max_prompt_len",
                        }
                    )

        toolrl_source_sha256 = toolrl_source_digest.hexdigest()
        gad_source_sha256 = gad_source_digest.hexdigest()
        current_toolrl_sha256 = _sha256(args.toolrl_input)
        current_gad_sha256 = _sha256(args.gad_input)
        if toolrl_source_sha256 != current_toolrl_sha256 or gad_source_sha256 != current_gad_sha256:
            raise RuntimeError(
                "source release changed while the length view was being materialized: "
                f"ToolRL {toolrl_source_sha256} -> {current_toolrl_sha256}; "
                f"GAD {gad_source_sha256} -> {current_gad_sha256}"
            )
        os.replace(toolrl_tmp, args.toolrl_output)
        os.replace(gad_tmp, args.gad_output)
    finally:
        toolrl_tmp.unlink(missing_ok=True)
        gad_tmp.unlink(missing_ok=True)

    manifest = {
        "schema_version": "drug_agent_rl_prompt_length_view_v1",
        "contract": {
            "model": args.model,
            "apply_chat_template": True,
            "apply_chat_template_kwargs": {"enable_thinking": False},
            "add_generation_prompt": True,
            "max_prompt_tokens": args.max_prompt_tokens,
            "policy": "exclude whole decisions; never truncate prompts or labels",
        },
        "source_records": source_records,
        "kept_records": kept_records,
        "rejected_records": len(rejected),
        "artifacts": {
            "toolrl_source": {"path": str(args.toolrl_input), "sha256": toolrl_source_sha256},
            "gad_source": {"path": str(args.gad_input), "sha256": gad_source_sha256},
            "toolrl_view": {"path": str(args.toolrl_output), "sha256": _sha256(args.toolrl_output)},
            "gad_view": {"path": str(args.gad_output), "sha256": _sha256(args.gad_output)},
        },
        "rejected": rejected,
    }
    manifest_tmp = args.manifest.with_name(f".{args.manifest.name}.tmp.{os.getpid()}")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
