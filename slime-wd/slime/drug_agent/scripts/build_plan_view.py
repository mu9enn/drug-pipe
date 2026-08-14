#!/usr/bin/env python3
"""Build one explicit high-level Plan-SFT example per successful trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.decision_extractor import iter_react_decisions


PLAN_SCHEMA = "drug_agent_plan_view_v1"
PLANNER_SYSTEM = """You are the planning component of a scientific agent.
Given a scientific task, return a concise ordered plan of scientific subgoals inside one <plan> JSON block.
Describe what must be established, not implementation details. Do not emit tool names, API arguments, file paths, artifact IDs, or a final scientific answer."""


FAMILY_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("retrieve", re.compile(r"retrieve_|compound"), "Retrieve and verify the molecular or protein inputs required by the task."),
    ("validate", re.compile(r"is_valid_|fix_pdb|pulchura|pack_sidechains"), "Validate and, where needed, repair the supplied molecular or structural inputs."),
    ("characterize", re.compile(r"calculate_(mol|protein|pdb)|admet|fingerprint|common_fragments"), "Characterize the relevant molecular or structural properties for comparison and filtering."),
    ("design", re.compile(r"reinvent|linkinvent|libinvent|pepinvent|proteinmpnn|evobind|chroma|goca"), "Generate task-constrained candidate molecules or protein designs."),
    ("structure", re.compile(r"esmfold|chai1|extract_.*chain|extract_pdb|convert_complex|prepare_complex"), "Obtain and prepare the three-dimensional structures needed for downstream analysis."),
    ("pocket", re.compile(r"pocket|fpocket"), "Identify and prioritize plausible binding regions on the prepared target structure."),
    ("binding", re.compile(r"docking|affinity|boltz|equiscore|hdock|karmadock|dleps"), "Evaluate binding or functional compatibility and rank the viable candidates."),
    ("simulation", re.compile(r"openmm|bioemu|openawsem|mmpbsa|foldx|traj|frames"), "Assess structural stability and energetic behavior under the requested simulation conditions."),
    ("interaction", re.compile(r"prolif|interaction|residue_mapper"), "Analyze the decisive molecular interactions and residues supporting the ranking."),
    ("visualize", re.compile(r"visualize"), "Produce an interpretable structural view of the key result and supporting interactions."),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _family(tool_name: str) -> tuple[str, str] | None:
    normalized = tool_name.strip().lower()
    if normalized in {"read", "write", "edit", "grep", "glob", "bash", "server_file_to_base64", "base64_to_server_file"}:
        return None
    if normalized.startswith("convert_") or normalized.startswith("prepare_"):
        return "prepare", "Prepare compatible molecular and structural representations for the scientific workflow."
    for name, pattern, sentence in FAMILY_RULES:
        if pattern.search(normalized):
            return name, sentence
    return "analyze", "Perform the next scientific analysis required to resolve the task."


def _original_task(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = str(message.get("content") or "").strip()
            if content and "<observation" not in content:
                return content
    raise ValueError("trajectory has no original user task")


def _plan_from_trajectory(messages: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    tools: list[str] = []
    for decision in iter_react_decisions(messages):
        if decision.get("decision_type") != "tool_call":
            continue
        tools.extend(str(call.get("tool_name") or "") for call in decision.get("tool_calls") or [])
    families: list[str] = []
    subgoals: list[str] = []
    for tool in tools:
        classified = _family(tool)
        if classified is None:
            continue
        family, sentence = classified
        if families and families[-1] == family:
            continue
        families.append(family)
        subgoals.append(sentence)
    if not subgoals:
        subgoals = ["Establish the scientific inputs and evidence needed to answer the task.", "Synthesize the evidence into a supported result."]
        families = ["analyze", "synthesize"]
    if len(subgoals) == 1:
        subgoals.append("Synthesize the resulting evidence into a concise, scientifically supported conclusion.")
        families.append("synthesize")
    return subgoals, families, tools


def build_plan_view(input_path: Path, output_path: Path, manifest_path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in input_path.open(encoding="utf-8") if line.strip()]
    rows: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    for index, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"record {index + 1} has no messages")
        subgoals, families, tools = _plan_from_trajectory(messages)
        family_counts.update(families)
        target = {"subgoals": [{"step": i + 1, "objective": text} for i, text in enumerate(subgoals)]}
        rows.append(
            {
                "schema_version": PLAN_SCHEMA,
                "id": f"plan_{record.get('id') or index}",
                "messages": [
                    {"role": "system", "content": PLANNER_SYSTEM},
                    {"role": "user", "content": _original_task(messages)},
                    {"role": "assistant", "content": "<plan>" + json.dumps(target, ensure_ascii=False, separators=(",", ":")) + "</plan>"},
                ],
                "metadata": {
                    "view": "plan_sft",
                    "source_id": record.get("id"),
                    "derivation": "full_teacher_trajectory_tool_sequence_abstraction",
                    "subgoal_families": families,
                    "source_tool_sequence": tools,
                },
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": PLAN_SCHEMA,
        "source_path": str(input_path.resolve()),
        "source_sha256": _sha256(input_path),
        "source_trajectories": len(records),
        "output_path": str(output_path.resolve()),
        "plan_samples": len(rows),
        "plans_per_trajectory": 1,
        "contains_tool_names_in_targets": False,
        "family_counts": dict(sorted(family_counts.items())),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_plan_view(args.input, args.output, args.manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
