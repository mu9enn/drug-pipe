#!/usr/bin/env python3
"""Optional model-trial selector for v8 ToolRL decisions.

Generation and native parsing stay in the active serving environment. This
utility consumes exactly two content-correctness scores per decision and emits
the same ordered structured decision rows as the lightweight selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from drug_agent.toolrl.v8_dataset import sha256_file


def _rows(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield row


def prepare(input_path: Path, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for row in _rows(input_path):
                output.write(json.dumps({
                    "decision_id": row["id"],
                    "prompt": row["prompt"],
                    "tools": row["tools"],
                    "num_candidates": 2,
                    "metadata": {
                        "source_id": row["metadata"]["source_id"],
                        "trajectory_index": row["metadata"]["trajectory_index"],
                        "decision_ordinal": row["metadata"]["decision_ordinal"],
                    },
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema_version": "toolrl_v8_model_trial_requests_v1",
        "records": count,
        "num_candidates_per_decision": 2,
        "output": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
    }


def _load_scores(path: Path) -> dict[str, list[float]]:
    result = {}
    for row in _rows(path):
        decision_id = str(row.get("decision_id") or "")
        scores = row.get("content_scores")
        if not decision_id or decision_id in result:
            raise ValueError("trial scores contain a missing or duplicate decision_id")
        if not isinstance(scores, list) or len(scores) != 2:
            raise ValueError(f"{decision_id}: expected exactly two content_scores")
        values = [float(value) for value in scores]
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{decision_id}: content scores must be in [0, 1]")
        result[decision_id] = values
    return result


def select(
    input_path: Path,
    scores_path: Path,
    output_root: Path,
    *,
    budget: int,
    priority_fraction: float,
    seed: int,
) -> dict[str, Any]:
    scores = _load_scores(scores_path)
    projections = []
    for index, row in enumerate(_rows(input_path)):
        decision_id = str(row["id"])
        if decision_id not in scores:
            raise ValueError(f"missing trial scores for {decision_id}")
        values = scores.pop(decision_id)
        projections.append({
            "index": index,
            "decision_id": decision_id,
            "content_scores": values,
            "content_gap": 1.0 - sum(values) / len(values),
        })
    if scores:
        raise ValueError(f"trial scores contain unknown decisions: {sorted(scores)[:5]}")
    if not 0 < budget <= len(projections):
        raise ValueError("budget must be positive and no larger than the decision count")
    if not 0.0 <= priority_fraction <= 1.0:
        raise ValueError("priority_fraction must be in [0, 1]")

    priority_count = min(budget, math.floor(budget * priority_fraction))
    ranked = sorted(projections, key=lambda row: (-row["content_gap"], row["decision_id"]))
    priority_indices = {row["index"] for row in ranked[:priority_count]}
    chosen = set(priority_indices)
    remainder = [row for row in projections if row["index"] not in chosen]
    random_ranked = sorted(
        remainder,
        key=lambda row: hashlib.sha256(f"{seed}:{row['decision_id']}".encode()).hexdigest(),
    )
    chosen.update(row["index"] for row in random_ranked[: budget - priority_count])
    projection_by_index = {row["index"]: row for row in projections}

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "selected_decisions.jsonl"
    pretty_path = output_root / "selected_decisions.pretty.json"
    audit_path = output_root / "selection_audit.jsonl"
    with output_path.open("w", encoding="utf-8") as output, pretty_path.open("w", encoding="utf-8") as pretty:
        pretty.write("[\n")
        pretty_count = 0
        for index, row in enumerate(_rows(input_path)):
            if index not in chosen:
                continue
            trial = projection_by_index[index]
            row["metadata"]["selector"] = {
                "selector": "two_trial_content_gap",
                "content_scores": trial["content_scores"],
                "content_gap": trial["content_gap"],
                "selection_reason": "priority" if index in priority_indices else "seeded_random",
            }
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            if pretty_count:
                pretty.write(",\n")
            pretty.write("\n".join("  " + line for line in json.dumps(row, ensure_ascii=False, indent=2).splitlines()))
            pretty_count += 1
        pretty.write("\n]\n")
    with audit_path.open("w", encoding="utf-8") as output:
        for row in projections:
            output.write(json.dumps({
                **row,
                "selected": row["index"] in chosen,
                "reason": (
                    "model_trial_priority"
                    if row["index"] in priority_indices
                    else "seeded_random_supplement"
                    if row["index"] in chosen
                    else "model_trial_not_selected"
                ),
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "schema_version": "toolrl_v8_model_trial_selector_v1",
        "input": {"path": str(input_path.resolve()), "sha256": sha256_file(input_path), "records": len(projections)},
        "scores": {"path": str(scores_path.resolve()), "sha256": sha256_file(scores_path)},
        "selection": {
            "budget": budget,
            "priority_fraction": priority_fraction,
            "priority_records": priority_count,
            "seeded_random_records": budget - priority_count,
            "seed": seed,
        },
        "output": {"path": str(output_path.resolve()), "sha256": sha256_file(output_path), "records": budget},
        "ordering": "canonical trajectory_index then original decision_ordinal",
        "note": "Trial generations are screening evidence only and must not be reused as formal RL rollouts.",
    }
    (output_root / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--input", type=Path, required=True)
    select_parser.add_argument("--scores", type=Path, required=True)
    select_parser.add_argument("--output-root", type=Path, required=True)
    select_parser.add_argument("--budget", type=int, required=True)
    select_parser.add_argument("--priority-fraction", type=float, default=0.8)
    select_parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.input.resolve(), args.output.resolve())
    else:
        result = select(
            args.input.resolve(), args.scores.resolve(), args.output_root.resolve(),
            budget=args.budget, priority_fraction=args.priority_fraction, seed=args.seed,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
