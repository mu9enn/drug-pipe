#!/usr/bin/env python3
"""Build an immutable ToolRL turn-semantics SFT/RL release from cleaned ReAct data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.protocol.toolrl_turn import TOOLRL_TURN_PROTOCOL, normalize_trajectory, split_assistant_segments
from drug_agent.scripts.audit_runtime_parser_compatibility import audit as audit_runtime_parser
from drug_agent.scripts.audit_sft_toolrl_serializer_parity import audit as audit_serializer_parity
from drug_agent.scripts.audit_toolrl_turn_release import audit as audit_segmentation
from drug_agent.scripts.build_toolrl_serializer_examples import build as build_serializer_examples
from drug_agent.scripts.select_toolrl_decisions import select_decisions
from drug_agent.toolrl.convert_react_to_toolrl_steps import convert_react_to_toolrl_steps
from drug_agent.toolrl.prompt_strategy import apply_prompt_strategy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rl_accounting(decision_count: int) -> dict[str, int]:
    if decision_count % 4:
        raise ValueError("v6 decision count must be divisible by RBS=4")
    return {
        "decision_count": decision_count,
        "grpo_group_count": decision_count,
        "n_samples_per_decision": 4,
        "sampled_response_count": decision_count * 4,
        "rollout_batch_size_decisions": 4,
        "rollout_batch_count": decision_count // 4,
    }


def _sft_records_for_trajectory(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep the trajectory once, plus exact-prefix targets for multi-segment turns.

    Multi-segment messages are loss-masked in the base trajectory by
    ``normalize_trajectory``.  Each action segment then gets one supplemental
    record whose final assistant message contains the exact preceding assistant
    prefix but masks that prefix with ``loss_char_start``.
    """
    records = [copy.deepcopy(record)]
    messages = record.get("messages") if isinstance(record.get("messages"), list) else []
    source_id = str(record.get("id") or "")
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        segments = split_assistant_segments(str(message.get("content") or ""))
        if len(segments) <= 1:
            continue
        prefix_parts: list[str] = []
        action_index = 0
        for segment in segments:
            if not segment["is_action"]:
                prefix_parts.append(segment["content"])
                continue
            prefix = "\n".join(prefix_parts)
            if prefix:
                prefix += "\n"
            supplement = copy.deepcopy(record)
            supplement["id"] = f"{source_id}::sft_segment::{message_index}::{action_index}"
            supplement["source_id"] = source_id
            supplement["sft_segment_target"] = {
                "source_id": source_id,
                "assistant_index": message_index,
                "assistant_subturn_index": action_index,
                "assistant_prefix": prefix,
                "target_action": segment["content"],
            }
            supplement["messages"] = supplement["messages"][: message_index + 1]
            for earlier in supplement["messages"]:
                if isinstance(earlier, dict) and earlier.get("role") == "assistant":
                    earlier["step_loss_mask"] = 0
            final_message = supplement["messages"][-1]
            final_message["content"] = prefix + segment["content"]
            final_message["step_loss_mask"] = 1
            final_message["loss_char_start"] = len(prefix)
            records.append(supplement)
            prefix_parts.append(segment["content"])
            action_index += 1
    return records


def materialize_release(
    *,
    input_react: Path,
    output_root: Path,
    model_path: Path,
    dataset_version: str,
    tool_catalog: Path | None = None,
    planning_annotations: Path | None = None,
    parent_dataset_version: str = "live_tool_catalog_v5-sftnrl",
    excluded_parent_trajectories: int = 0,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output release directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    toolrl_root = output_root / "toolrl"
    toolrl_root.mkdir()

    normalized_path = output_root / "canonical_trajectories.jsonl"
    sft_path = output_root / "react_trajectories.jsonl"
    source_records = normalized_records = sft_records = 0
    audit_totals: Counter[str] = Counter()
    source_ids: set[str] = set()
    with (
        input_react.open(encoding="utf-8") as source,
        normalized_path.open("w", encoding="utf-8") as canonical_target,
        sft_path.open("w", encoding="utf-8") as sft_target,
    ):
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            source_records += 1
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"source line {line_number} is not an object")
            source_id = str(record.get("id") or "")
            if not source_id or source_id in source_ids:
                raise ValueError(f"missing or duplicate source id at line {line_number}: {source_id!r}")
            source_ids.add(source_id)
            normalized, audit = normalize_trajectory(record)
            canonical_target.write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n")
            for sft_record in _sft_records_for_trajectory(normalized):
                sft_target.write(json.dumps(sft_record, ensure_ascii=False, separators=(",", ":")) + "\n")
                sft_records += 1
            normalized_records += 1
            audit_totals.update({key: int(value) for key, value in audit.items() if isinstance(value, int)})
    if source_records == 0 or normalized_records != source_records or sft_records < normalized_records:
        raise ValueError("empty or incomplete normalized SFT view")

    raw_steps = toolrl_root / "raw_toolrl_steps.jsonl"
    conversion_report_path = toolrl_root / "toolrl_steps.report.json"
    skipped_path = toolrl_root / "toolrl_steps.skipped.jsonl"
    conversion_report = convert_react_to_toolrl_steps(
        normalized_path,
        raw_steps,
        skipped_report_path=skipped_path,
        report_path=conversion_report_path,
    )
    catalog_payload = None
    if tool_catalog is not None:
        catalog_payload = json.loads(tool_catalog.read_text(encoding="utf-8"))
    baseline_source = toolrl_root / "raw_toolrl_steps.official_catalog.jsonl"
    with raw_steps.open(encoding="utf-8") as source, baseline_source.open("w", encoding="utf-8") as target:
        for line in source:
            if line.strip():
                target.write(
                    json.dumps(
                        apply_prompt_strategy(
                            json.loads(line), strategy="official_catalog", catalog=catalog_payload
                        ),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    baseline_steps = toolrl_root / "toolrl_steps.official_baseline.jsonl"
    baseline_context_manifest = toolrl_root / "context_manifest.official_baseline.json"
    baseline_view = select_decisions(
        baseline_source,
        baseline_steps,
        baseline_context_manifest,
        model_path,
        245760,
        16384,
        32768,
        max_context_tokens=262144,
        batch_multiple=4,
        selection_mode="all_static",
    )
    final_steps = toolrl_root / "toolrl_steps.jsonl"
    context_manifest_path = toolrl_root / "context_manifest.json"
    view = select_decisions(
        raw_steps,
        final_steps,
        context_manifest_path,
        model_path,
        245760,
        16384,
        32768,
        max_context_tokens=262144,
        batch_multiple=4,
        selection_mode="curated_static",
    )

    if tool_catalog is not None:
        shutil.copyfile(tool_catalog, output_root / "tool_catalog.json")
    if planning_annotations is not None:
        shutil.copyfile(planning_annotations, output_root / "planning_annotations.jsonl")

    audit_root = output_root / "audit"
    segmentation_report = audit_segmentation(input_react, normalized_path)
    serializer_parity_report = audit_serializer_parity(sft_path, raw_steps)
    runtime_parser_report = audit_runtime_parser()
    serializer_examples = build_serializer_examples(input_react)
    _write_json(audit_root / "reasoning_action_segmentation.json", segmentation_report)
    _write_json(audit_root / "sft_toolrl_serializer_parity.json", serializer_parity_report)
    _write_json(audit_root / "runtime_parser_compatibility.json", runtime_parser_report)
    _write_json(audit_root / "serializer_examples.json", serializer_examples)

    dataset_manifest = {
        "schema_version": "drug_agent_toolrl_turn_release_v1",
        "dataset_version": dataset_version,
        "created_at": date.today().isoformat(),
        "source_version": parent_dataset_version,
        # Kept for existing production launchers; the profile map below is the
        # authoritative v6 statement because baseline intentionally has no SFT.
        "training_flow": "SFT -> ToolRL",
        "training_flows": {
            "official_baseline": "base checkpoint -> ToolRL",
            "drug_pipe_production": "SFT -> ToolRL",
        },
        "protocol": TOOLRL_TURN_PROTOCOL,
        "source": {
            "path": str(input_react.resolve()),
            "records": source_records,
            "sha256": _sha256(input_react),
            "authority": "v5_cleaned_trajectory_content_and_ordered_reasoning_action_boundaries",
        },
        "sft": {
            "path": "react_trajectories.jsonl",
            "records": sft_records,
            "sha256": _sha256(sft_path),
            "canonical_trajectory_path": "canonical_trajectories.jsonl",
            "canonical_trajectory_records": normalized_records,
            "canonical_trajectory_sha256": _sha256(normalized_path),
            "target_semantics": "one serialized reasoning/action segment per supervised target",
            "normalization": dict(audit_totals),
        },
        "toolrl": {
            "path": "toolrl/toolrl_steps.jsonl",
            "records": int(view["candidate_records"]),
            "sha256": _sha256(final_steps),
            "raw_decisions": int(conversion_report["kept_rows"]),
            "role_counts": view["role_counts"],
            "selection": view["selection"],
            "rbs_alignment_records": view["rbs_alignment_records"],
            "context_limits": {
                "max_context_tokens": 262144,
                "max_prompt_tokens": 245760,
                "max_response_tokens": 16384,
                "observed_max_prompt_tokens": view["context"]["observed_max_prompt_tokens"],
                "observed_max_target_tokens": view["context"]["observed_max_target_tokens"],
            },
            "excluded_records": view["excluded_records"],
            "excluded_reason_counts": view["excluded_reason_counts"],
            "accounting": _rl_accounting(int(view["candidate_records"])),
        },
        "toolrl_official_baseline": {
            "canonical_commit": "8cee13ec0ca72f0461da372a93a6fd8140dbb840",
            "path": "toolrl/toolrl_steps.official_baseline.jsonl",
            "records": int(baseline_view["candidate_records"]),
            "sha256": _sha256(baseline_steps),
            "prompt_strategy": "official_catalog",
            "role_counts": baseline_view["role_counts"],
            "selection": baseline_view["selection"],
            "excluded_records": baseline_view["excluded_records"],
            "excluded_reason_counts": baseline_view["excluded_reason_counts"],
            "accounting": _rl_accounting(int(baseline_view["candidate_records"])),
        },
        "extensions": {
            "frozen_sft_reference": {"default": False, "switch": "TOOLRL_REF_LOAD/BASE_SFT_DIR"},
            "hierarchical_reward": {"default": False, "switch": "TOOLRL_REWARD_MODE=hierarchical"},
            "structured_final_exact": {
                "default": False,
                "switch": "TOOLRL_STRUCTURED_FINAL_EXACT=1",
            },
            "drug_pipe_skill_discovery": {"default": False, "switch": "TOOLRL_PROMPT_STRATEGY=drug_pipe_skill_discovery"},
            "static_curated_selector": {"default": False, "switch": "TOOLRL_VIEW=production"},
            "sft_warm_start": {"default": False, "switch": "V6_PROFILE=drug_pipe_production"},
            "dynamic_policy_boundary_filter": {"default": False, "supported_for_ablation_only": True},
        },
        "excluded_parent_trajectories": excluded_parent_trajectories,
        "gad_included": False,
        "tool_catalog_injected_into_prompts": False,
        "prompt_strategies": {
            "toolrl": {"mode": "drug_pipe_skill_discovery", "tool_catalog_injected": False},
            "toolrl_official_baseline": {"mode": "official_catalog", "tool_catalog_injected": True},
        },
        "audits": {
            "reasoning_action_segmentation": "audit/reasoning_action_segmentation.json",
            "sft_toolrl_serializer_parity": "audit/sft_toolrl_serializer_parity.json",
            "runtime_parser_compatibility": "audit/runtime_parser_compatibility.json",
            "serializer_examples": "audit/serializer_examples.json",
        },
    }
    if tool_catalog is not None:
        dataset_manifest["tool_catalog"] = {
            "path": "tool_catalog.json",
            "sha256": _sha256(output_root / "tool_catalog.json"),
            "uses": {
                "official_baseline": "prompt + reward + runtime",
                "drug_pipe_production": "reward + runtime only",
            },
        }
    _write_json(output_root / "dataset_manifest.json", dataset_manifest)
    _write_json(
        output_root / "manifest.json",
        {
            "schema_version": "toolrl_turn_materialized_release_v1",
            "source_sha256": _sha256(sft_path),
            "protocol": TOOLRL_TURN_PROTOCOL,
            "limits": {"context": 262144, "prompt": 245760, "response": 16384},
            "records": int(view["candidate_records"]),
            "batch_multiple": 4,
        },
    )
    (output_root / "materialize.complete").write_text("complete\n", encoding="utf-8")
    (output_root / "RELEASE_COMPLETE").write_text("complete\n", encoding="utf-8")
    return dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-react", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--tool-catalog", type=Path)
    parser.add_argument("--planning-annotations", type=Path)
    parser.add_argument("--parent-dataset-version", default="live_tool_catalog_v5-sftnrl")
    parser.add_argument("--excluded-parent-trajectories", type=int, default=0)
    args = parser.parse_args()
    report = materialize_release(
        input_react=args.input_react,
        output_root=args.output_root,
        model_path=args.model,
        dataset_version=args.dataset_version,
        tool_catalog=args.tool_catalog,
        planning_annotations=args.planning_annotations,
        parent_dataset_version=args.parent_dataset_version,
        excluded_parent_trajectories=args.excluded_parent_trajectories,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
