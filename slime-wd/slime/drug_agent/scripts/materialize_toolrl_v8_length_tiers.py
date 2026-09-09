#!/usr/bin/env python3
"""Partition selected v8 decisions by measured native-token runtime gates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from drug_agent.toolrl.v8_dataset import percentiles, sha256_file


def materialize(
    selected_path: Path,
    length_details_path: Path,
    output_root: Path,
    *,
    context_limit: int = 262144,
    standard_generation: int = 32768,
    long_generation: int = 65536,
    emit_pretty: bool = True,
) -> dict[str, Any]:
    details = {
        row["decision_id"]: row
        for row in (
            json.loads(line) for line in length_details_path.open(encoding="utf-8") if line.strip()
        )
    }
    tier_names = ("standard_32k", "long_64k_candidate", "oversize_review")
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    selected_details: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_paths = {tier: output_root / f"{tier}.jsonl" for tier in tier_names}
    pretty_paths = {tier: output_root / f"{tier}.pretty.json" for tier in tier_names}
    jsonl_outputs = {tier: path.open("w", encoding="utf-8") for tier, path in jsonl_paths.items()}
    pretty_outputs = {tier: path.open("w", encoding="utf-8") for tier, path in pretty_paths.items()} if emit_pretty else {}
    for output in pretty_outputs.values():
        output.write("[\n")
    try:
        for line_number, line in enumerate(selected_path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            decision_id = str(row.get("id") or "")
            detail = details.pop(decision_id, None)
            if detail is None:
                raise ValueError(f"missing length detail for selected row {line_number}: {decision_id}")
            selected_details.append(detail)
            prompt_tokens = int(detail["prompt_tokens"])
            reference_tokens = int(detail["full_reference_tokens"])
            if prompt_tokens <= context_limit - standard_generation and reference_tokens <= standard_generation:
                tier = "standard_32k"
            elif prompt_tokens <= context_limit - long_generation and reference_tokens <= long_generation:
                tier = "long_64k_candidate"
            else:
                tier = "oversize_review"
                if prompt_tokens > context_limit - standard_generation:
                    reasons["prompt_exceeds_32k_generation_context_gate"] += 1
                if reference_tokens > long_generation:
                    reasons["teacher_reference_exceeds_64k"] += 1
                elif reference_tokens > standard_generation and prompt_tokens > context_limit - long_generation:
                    reasons["32k_to_64k_reference_cannot_fit_64k_context_gate"] += 1
            jsonl_outputs[tier].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            if emit_pretty:
                if counts[tier]:
                    pretty_outputs[tier].write(",\n")
                pretty_outputs[tier].write(
                    "\n".join("  " + value for value in json.dumps(row, ensure_ascii=False, indent=2).splitlines())
                )
            counts[tier] += 1
    finally:
        for output in jsonl_outputs.values():
            output.close()
        for output in pretty_outputs.values():
            output.write("\n]\n")
            output.close()

    outputs = {}
    for tier in tier_names:
        jsonl_path = jsonl_paths[tier]
        pretty_path = pretty_paths[tier]
        outputs[tier] = {
            "records": counts[tier],
            "jsonl": str(jsonl_path.resolve()),
            "jsonl_sha256": sha256_file(jsonl_path),
        }
        if emit_pretty:
            outputs[tier]["pretty_json"] = str(pretty_path.resolve())
    manifest = {
        "schema_version": "toolrl_v8_length_tiers_v1",
        "source": {"path": str(selected_path.resolve()), "sha256": sha256_file(selected_path)},
        "length_details": {"path": str(length_details_path.resolve()), "sha256": sha256_file(length_details_path)},
        "gates": {
            "context_limit": context_limit,
            "standard_generation": standard_generation,
            "standard_max_prompt": context_limit - standard_generation,
            "long_generation": long_generation,
            "long_max_prompt": context_limit - long_generation,
        },
        "policy": "partition_only; no truncation, mutation, or deletion from selected mother data",
        "oversize_reasons": dict(reasons),
        "outputs": outputs,
    }
    length_fields = (
        "prompt_tokens",
        "reasoning_tokens",
        "structured_action_or_final_tokens",
        "native_action_or_final_tokens",
        "full_reference_tokens",
        "prompt_plus_reference_tokens",
    )
    selected_summary = {
        "schema_version": "toolrl_v8_selected_length_summary_v1",
        "records": len(selected_details),
        "distributions": {
            field: percentiles(detail[field] for detail in selected_details)
            for field in length_fields
        },
        "reference_threshold_counts": {
            str(threshold): sum(
                int(detail["full_reference_tokens"]) > threshold
                for detail in selected_details
            )
            for threshold in (16384, standard_generation, long_generation)
        },
    }
    selected_summary_path = output_root / "selected_length_summary.json"
    selected_summary_path.write_text(
        json.dumps(selected_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["selected_length_summary"] = {
        "path": str(selected_summary_path.resolve()),
        "sha256": sha256_file(selected_summary_path),
    }
    (output_root / "length_tier_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--length-details", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--context-limit", type=int, default=262144)
    parser.add_argument("--standard-generation", type=int, default=32768)
    parser.add_argument("--long-generation", type=int, default=65536)
    parser.add_argument("--no-pretty", action="store_true")
    args = parser.parse_args()
    report = materialize(
        args.selected.resolve(), args.length_details.resolve(), args.output_root.resolve(),
        context_limit=args.context_limit,
        standard_generation=args.standard_generation,
        long_generation=args.long_generation,
        emit_pretty=not args.no_pretty,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
