#!/usr/bin/env python3
"""Build a capacity-safe ToolRL view packed by complete trajectories."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from drug_agent.toolrl.trajectory_batching import (
    flatten_batch,
    group_complete_trajectories,
    pack_complete_trajectories,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(entry: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    record = entry[1]
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _decision_key(record: dict[str, Any]) -> tuple[str, int, str]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return (
        str(metadata.get("source_id") or ""),
        int(metadata.get("decision_ordinal", -1)),
        str(metadata.get("decision_type") or ""),
    )


def _target_assistant(record: dict[str, Any]) -> dict[str, Any]:
    label = record.get("label") if isinstance(record.get("label"), dict) else {}
    target = label.get("target_assistant")
    if not isinstance(target, dict) or target.get("role") != "assistant":
        raise ValueError(f"missing structured assistant target for decision {_decision_key(record)!r}")
    return target


def _trajectory_capacity_findings(
    trajectory: list[tuple[int, dict[str, Any]]],
    *,
    tokenizer: Any,
    max_prompt_tokens: int,
    max_target_tokens: int,
) -> list[dict[str, Any]]:
    records = [record for _, record in trajectory]
    rendered_prompts = [
        tokenizer.apply_chat_template(
            record["prompt"],
            tools=record.get("tools"),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for record in records
    ]
    rendered_full = [
        tokenizer.apply_chat_template(
            record["prompt"] + [_target_assistant(record)],
            tools=record.get("tools"),
            tokenize=False,
        )
        for record in records
    ]
    targets: list[str] = []
    for prompt, full in zip(rendered_prompts, rendered_full, strict=True):
        if not full.startswith(prompt):
            raise ValueError("structured assistant target is not a continuation of the generation prompt")
        targets.append(full[len(prompt) :])
    prompt_ids = tokenizer(rendered_prompts, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(targets, add_special_tokens=False)["input_ids"]

    findings: list[dict[str, Any]] = []
    for (line_number, record), prompt_tokens, target_tokens in zip(
        trajectory, prompt_ids, target_ids, strict=True
    ):
        reasons: list[str] = []
        if len(prompt_tokens) > max_prompt_tokens:
            reasons.append("prompt_exceeds_max_tokens")
        if len(target_tokens) > max_target_tokens:
            reasons.append("target_exceeds_max_tokens")
        if reasons:
            findings.append(
                {
                    "line_number": line_number,
                    "decision_key": list(_decision_key(record)),
                    "prompt_tokens": len(prompt_tokens),
                    "target_tokens": len(target_tokens),
                    "reasons": reasons,
                }
            )
    return findings


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
        raise ValueError("token limits and rollout_batch_size must be positive")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("training view must not overwrite its source")

    entries: list[tuple[int, dict[str, Any]]] = []
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{input_path}:{line_number}: row is not an object")
            entries.append((line_number, record))

    trajectories = group_complete_trajectories(entries, metadata_of=_metadata)
    accepted: list[tuple[str, list[tuple[int, dict[str, Any]]]]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for source_id, trajectory in trajectories:
        findings = _trajectory_capacity_findings(
            trajectory,
            tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            max_target_tokens=max_target_tokens,
        )
        if findings:
            reasons = sorted({reason for finding in findings for reason in finding["reasons"]})
            reason_counts.update(reasons)
            rejected.append(
                {
                    "source_id": source_id,
                    "decision_count": len(trajectory),
                    "reasons": reasons,
                    "decision_findings": findings,
                }
            )
            continue
        accepted.append((source_id, trajectory))

    if not accepted:
        raise ValueError("no complete trajectory passed ToolRL capacity checks")
    batches = pack_complete_trajectories(accepted, rollout_batch_size)

    output_records: list[dict[str, Any]] = []
    batch_plan: list[dict[str, Any]] = []
    for batch_id, batch in enumerate(batches):
        position = 0
        batch_plan.append(
            {
                "trajectory_batch_id": batch_id,
                "source_ids": [source_id for source_id, _ in batch],
                "trajectory_decision_counts": [len(items) for _, items in batch],
                "decision_count": rollout_batch_size,
            }
        )
        for _, record in flatten_batch(batch):
            output = copy.deepcopy(record)
            metadata = copy.deepcopy(_metadata((0, output)))
            metadata.update(
                {
                    "trajectory_batch_id": batch_id,
                    "trajectory_batch_position": position,
                    "trajectory_batch_decision_count": rollout_batch_size,
                }
            )
            output["metadata"] = metadata
            output_records.append(output)
            position += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    pretty_path = output_path.with_suffix(".pretty.json")
    pretty_tmp = pretty_path.with_name(f".{pretty_path.name}.tmp.{os.getpid()}")
    try:
        with output_tmp.open("w", encoding="utf-8") as output:
            for record in output_records:
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(output_tmp, output_path)
        pretty_tmp.write_text(
            json.dumps(output_records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(pretty_tmp, pretty_path)
    finally:
        output_tmp.unlink(missing_ok=True)
        pretty_tmp.unlink(missing_ok=True)

    manifest = {
        "schema_version": "toolrl_trajectory_batched_view_v1",
        "contract": {
            "model": model_name,
            "apply_chat_template": True,
            "apply_chat_template_kwargs": {"enable_thinking": True},
            "add_generation_prompt": True,
            "max_prompt_tokens": max_prompt_tokens,
            "max_target_tokens": max_target_tokens,
            "rollout_batch_size_decisions": rollout_batch_size,
            "sampling_unit": "complete_trajectory",
            "policy": "admit or reject whole trajectories; never split or decision-pad",
        },
        "source": {
            "path": str(input_path.resolve()),
            "sha256": _sha256(input_path),
            "records": len(entries),
            "trajectories": len(trajectories),
        },
        "accepted_trajectories": len(accepted),
        "accepted_records": len(output_records),
        "rejected_trajectories": len(rejected),
        "rejected_records": sum(item["decision_count"] for item in rejected),
        "rejection_reason_counts": dict(reason_counts),
        "output": {
            "path": str(output_path.resolve()),
            "pretty_path": str(pretty_path.resolve()),
            "sha256": _sha256(output_path),
            "records": len(output_records),
            "trajectory_batches": len(batches),
        },
        "batch_plan": batch_plan,
        "rejected": rejected,
    }
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
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
    manifest = materialize_toolrl_training_view(
        input_path=args.input.resolve(),
        output_path=args.output.resolve(),
        manifest_path=args.manifest.resolve(),
        tokenizer=tokenizer,
        model_name=args.model,
        max_prompt_tokens=args.max_prompt_tokens,
        max_target_tokens=args.max_target_tokens,
        rollout_batch_size=args.rollout_batch_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
