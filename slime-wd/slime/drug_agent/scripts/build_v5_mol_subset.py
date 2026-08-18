#!/usr/bin/env python3
"""Atomically build the MolBench-only subset of a published v5 SFT+ToolRL view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", required=True, type=Path)
    parser.add_argument("--membership", required=True, type=Path,
                        help="JSONL containing the authoritative 365 trajectory IDs")
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    parent = args.parent_root.resolve()
    output = args.output_root.resolve()
    staging = output.with_name(output.name + f".staging.{os.getpid()}")
    if output.exists() or staging.exists():
        raise SystemExit(f"output or staging path already exists: {output} / {staging}")

    required = [
        parent / "RELEASE_COMPLETE",
        parent / "dataset_manifest.json",
        parent / "react_trajectories.jsonl",
        parent / "planning_annotations.jsonl",
        parent / "tool_catalog.json",
        parent / "toolrl/toolrl_steps.jsonl",
        parent / "toolrl/context_manifest.json",
        args.membership,
    ]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    parent_manifest = read_json(parent / "dataset_manifest.json")
    if parent_manifest.get("dataset_version") != "live_tool_catalog_v5-sftnrl":
        raise SystemExit("parent is not the published live_tool_catalog_v5-sftnrl release")

    membership_rows = list(iter_jsonl(args.membership))
    ids = [str(row.get("id", row.get("source_id", ""))) for row in membership_rows]
    if len(ids) != 365 or len(set(ids)) != 365 or any(not item for item in ids):
        raise SystemExit("membership authority must contain exactly 365 unique IDs")
    selected_ids = set(ids)

    sft_rows = [row for row in iter_jsonl(parent / "react_trajectories.jsonl") if str(row.get("id")) in selected_ids]
    annotations = [row for row in iter_jsonl(parent / "planning_annotations.jsonl")
                   if str(row.get("source_id")) in selected_ids]
    if len(sft_rows) != 365 or {str(row["id"]) for row in sft_rows} != selected_ids:
        raise SystemExit("v5 SFT does not contain the complete MolBench membership")
    if len(annotations) != 365 or {str(row["source_id"]) for row in annotations} != selected_ids:
        raise SystemExit("v5 planning sidecar does not align with the MolBench membership")

    toolrl_rows = []
    roles: Counter[str] = Counter()
    task_types: set[str] = set()
    tool_names: set[str] = set()
    prompt_max = target_max = 0
    for row in iter_jsonl(parent / "toolrl/toolrl_steps.jsonl"):
        metadata = row.get("metadata") or {}
        if str(metadata.get("source_id")) not in selected_ids:
            continue
        toolrl_rows.append(row)
        roles[str(metadata.get("decision_role", "unknown"))] += 1
        if metadata.get("task_type"):
            task_types.add(str(metadata["task_type"]))
        tool_names.update(str(name) for name in metadata.get("tool_names", []))
        prompt_max = max(prompt_max, int(metadata.get("prompt_tokens_final", 0)))
        target_max = max(target_max, int(metadata.get("canonical_target_tokens", 0)))

    if not toolrl_rows or roles.get("final", 0) == 0:
        raise SystemExit("MolBench ToolRL subset is empty or contains no final decisions")

    staging.mkdir(parents=True)
    (staging / "toolrl").mkdir()
    sft_path = staging / "react_trajectories.jsonl"
    planning_path = staging / "planning_annotations.jsonl"
    toolrl_path = staging / "toolrl/toolrl_steps.jsonl"
    write_jsonl(sft_path, sft_rows)
    write_jsonl(planning_path, annotations)
    write_jsonl(toolrl_path, toolrl_rows)
    shutil.copy2(parent / "tool_catalog.json", staging / "tool_catalog.json")

    parent_context = read_json(parent / "toolrl/context_manifest.json")
    excluded = [entry for entry in parent_context.get("excluded", [])
                if str(entry.get("decision_key", "")).split(":", 1)[0] in selected_ids]
    excluded_reasons = Counter(str(entry.get("reason", "unknown")) for entry in excluded)
    context_manifest = {
        "schema_version": "toolrl_policy_boundary_subset_view_v1",
        "parent_path": str(parent / "toolrl/toolrl_steps.jsonl"),
        "parent_sha256": sha256(parent / "toolrl/toolrl_steps.jsonl"),
        "membership_path": str(args.membership.resolve()),
        "membership_sha256": sha256(args.membership.resolve()),
        "candidate_records": len(toolrl_rows),
        "copies_added": 0,
        "role_counts": dict(sorted(roles.items())),
        "coverage": {
            "task_types": sorted(task_types),
            "tool_names": sorted(tool_names),
            "task_type_count": len(task_types),
            "tool_count": len(tool_names),
        },
        "selection": parent_context.get("selection", {}),
        "context": {
            **parent_context.get("context", {}),
            "observed_max_prompt_tokens": prompt_max,
            "observed_max_target_tokens": target_max,
        },
        "excluded_records_in_membership": len(excluded),
        "excluded_reason_counts_in_membership": dict(sorted(excluded_reasons.items())),
        "excluded": excluded,
    }
    context_path = staging / "toolrl/context_manifest.json"
    context_path.write_text(json.dumps(context_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    compatibility = {
        "schema_version": "toolrl_materialized_release_compat_v1",
        "source_sha256": sha256(sft_path),
        "limits": {"context": 262144, "prompt": 245760, "response": 16384},
        "records": len(toolrl_rows),
    }
    (staging / "manifest.json").write_text(json.dumps(compatibility, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "drug_agent_subset_manifest_v2",
        "dataset_version": "live_tool_catalog_v5-mol-sftnrl",
        "description": "MolBench-only 365-trajectory subset of the published v5 SFT+ToolRL release.",
        "parent": {
            "path": str(parent),
            "dataset_manifest_sha256": sha256(parent / "dataset_manifest.json"),
            "sft_records": 605,
            "toolrl_records": int(parent_manifest["toolrl"]["records"]),
        },
        "membership_authority": {
            "path": str(args.membership.resolve()),
            "records": 365,
            "sha256": sha256(args.membership.resolve()),
            "selection": "exact source-ID join against the audited v4-mol membership",
        },
        "training_flow": "SFT -> ToolRL",
        "sft": {"path": "react_trajectories.jsonl", "records": 365, "sha256": sha256(sft_path)},
        "planning_sidecar": {"path": "planning_annotations.jsonl", "records": 365,
                             "sha256": sha256(planning_path), "used_as_separate_rl_role": False},
        "tool_catalog": {"path": "tool_catalog.json", "sha256": sha256(staging / "tool_catalog.json")},
        "toolrl": {
            "path": "toolrl/toolrl_steps.jsonl", "records": len(toolrl_rows),
            "sha256": sha256(toolrl_path), "role_counts": dict(sorted(roles.items())),
            "context_limits": {"max_context_tokens": 262144, "max_prompt_tokens": 245760,
                               "max_response_tokens": 16384, "observed_max_prompt_tokens": prompt_max,
                               "observed_max_target_tokens": target_max},
            "context_manifest": {"path": "toolrl/context_manifest.json", "sha256": sha256(context_path)},
        },
        "excluded_parent_trajectories": 240,
        "source_files_unchanged": True,
    }
    (staging / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (staging / "README.md").write_text(
        "# Live Tool Catalog v5-mol-sftnrl\n\n"
        "This is the exact 365-trajectory MolBench source-ID subset of `live_tool_catalog_v5-sftnrl`. "
        "It reuses the parent's LLM-cleaned SFT rows and already selected/compacted ToolRL decisions; "
        "planning annotations remain provenance only and are not a separate RL role.\n",
        encoding="utf-8",
    )
    (staging / "materialize.complete").touch()
    (staging / "RELEASE_COMPLETE").write_text("atomic v5-mol SFT+ToolRL subset release\n", encoding="utf-8")
    os.rename(staging, output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
