#!/usr/bin/env python3
"""Build the audited, curated ToolRL view used by decision-aware GRPO."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.scripts.compact_rl_context import COMPACTION_SCHEMA, compact_prompt_with_audit, render_length


VIEW_SCHEMA = "toolrl_decision_aware_view_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    if not isinstance(value, dict):
        raise ValueError("ToolRL row is missing metadata")
    return value


def _decision_key(row: dict[str, Any]) -> tuple[str, int, str]:
    metadata = _metadata(row)
    return (
        str(metadata.get("source_id") or metadata.get("task_id") or ""),
        int(metadata.get("assistant_index", -1)),
        str(metadata.get("decision_type") or ""),
    )


def _without_summary(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_summary(item) for key, item in value.items() if key != "summary"}
    if isinstance(value, list):
        return [_without_summary(item) for item in value]
    return value


def _canonical_target_text(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    label = row.get("label") if isinstance(row.get("label"), dict) else {}
    decision_type = str(label.get("decision_type") or metadata.get("decision_type") or "")
    if decision_type == "final_answer":
        value = label.get("target_final_answer", metadata.get("target_final_answer"))
        if not isinstance(value, dict):
            raise ValueError(f"missing target_final_answer for {_decision_key(row)!r}")
        return "<final_answer>" + json.dumps(
            _without_summary(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "</final_answer>"

    calls = label.get("target_tool_calls", metadata.get("target_tool_calls"))
    if not isinstance(calls, list) or not calls:
        raise ValueError(f"missing target_tool_calls for {_decision_key(row)!r}")
    return "".join(
        "<tool_call>"
        + json.dumps(
            {
                "tool_name": str(call.get("tool_name") or call.get("name") or ""),
                "arguments": call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "</tool_call>"
        for call in calls
        if isinstance(call, dict)
    )


def _text_tokens(tokenizer, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    return len(encoded)


def _length_bin(tokens: int, max_prompt_tokens: int) -> str:
    if tokens <= 8192:
        return "le_8k"
    if tokens <= 32768:
        return "8k_32k"
    if tokens <= 131072:
        return "32k_128k"
    if tokens <= max_prompt_tokens:
        return "128k_limit"
    return "over_limit"


def _prompt_characters(prompt: list[dict[str, Any]]) -> int:
    return sum(len(str(message.get("content") or "")) for message in prompt if isinstance(message, dict))


def _depth_bin(metadata: dict[str, Any]) -> str:
    ordinal = max(0, int(metadata.get("decision_ordinal", 0)))
    total = max(1, int(metadata.get("trajectory_decision_count", 1)))
    ratio = ordinal / max(1, total - 1)
    if ratio < 0.25:
        return "early"
    if ratio < 0.75:
        return "middle"
    return "late"


def _observation_outcome(row: dict[str, Any]) -> str:
    prompt = row.get("prompt") if isinstance(row.get("prompt"), list) else []
    for message in reversed(prompt):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = str(message.get("content") or "").lower()
        if "<observation" not in content:
            continue
        if '"ok":false' in content or '"status":"error"' in content or '"status":"failed"' in content:
            return "error"
        if '"ok":true' in content or '"status":"success"' in content:
            return "success"
        return "unknown"
    return "none"


def _stable_jitter(seed: int, key: tuple[str, int, str]) -> float:
    raw = f"{seed}:{key[0]}:{key[1]}:{key[2]}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / 2**64


def _copy_for_output(row: dict[str, Any], reasons: list[str], copy_index: int) -> dict[str, Any]:
    out = copy.deepcopy(row)
    metadata = copy.deepcopy(_metadata(out))
    metadata["sampling_reason"] = sorted(set(reasons))
    metadata["sampling_copy_index"] = copy_index
    out["metadata"] = metadata
    return out


def materialize_decision_aware_view(
    *,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    tokenizer: Any,
    model_name: str,
    max_prompt_tokens: int = 245760,
    max_response_tokens: int = 16384,
    summary_max_tokens: int = 32768,
    intermediate_budget: int = 2500,
    min_per_tool: int = 8,
    max_per_trajectory: int = 8,
    multiple: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    if min(
        max_prompt_tokens,
        max_response_tokens,
        summary_max_tokens,
        intermediate_budget,
        min_per_tool,
        max_per_trajectory,
        multiple,
    ) < 1:
        raise ValueError("all limits and budgets must be positive")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("curated view must not overwrite its source")

    source_sha256 = _sha256(input_path)
    prepared: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    original_role_counts: Counter[str] = Counter()
    original_tool_names: set[str] = set()

    with input_path.open(encoding="utf-8") as source:
        for source_line, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{input_path}:{source_line}: row is not an object")
            metadata = _metadata(row)
            role = str(metadata.get("decision_role") or "")
            if role not in {"planning", "initial_tool_step", "tool_step", "final"}:
                raise ValueError(f"missing/invalid decision_role for {_decision_key(row)!r}: {role!r}")
            original_role_counts[role] += 1
            tool_names = tuple(str(name) for name in (metadata.get("tool_names") or []) if str(name))
            original_tool_names.update(tool_names)
            canonical_target = _canonical_target_text(row)
            target_tokens = _text_tokens(tokenizer, canonical_target)
            if target_tokens > max_response_tokens:
                rejected.append(
                    {
                        "source_line": source_line,
                        "decision_key": list(_decision_key(row)),
                        "decision_role": role,
                        "canonical_target_tokens": target_tokens,
                        "reason": "canonical_target_exceeds_max_response_tokens",
                    }
                )
                continue

            prompt = row.get("prompt")
            if not isinstance(prompt, list) or not prompt:
                raise ValueError(f"missing prompt for {_decision_key(row)!r}")
            # Curation only needs a coarse deterministic length stratum.  Exact
            # chat-template tokenization is deliberately deferred until after
            # selection: tokenizing every overlapping trajectory prefix makes
            # preprocessing quadratic in trajectory depth at long context.
            prompt_characters = _prompt_characters(prompt)
            approximate_tokens = max(1, prompt_characters // 4)
            metadata = copy.deepcopy(metadata)
            metadata["canonical_target_tokens"] = target_tokens
            metadata["prompt_characters"] = prompt_characters
            metadata["prompt_length_bin"] = _length_bin(approximate_tokens, max_prompt_tokens)
            metadata["decision_depth_bin"] = _depth_bin(metadata)
            metadata["previous_observation_outcome"] = _observation_outcome(row)
            row["metadata"] = metadata
            prepared.append(
                {
                    "source_line": source_line,
                    "row": row,
                    "key": _decision_key(row),
                    "source_id": _decision_key(row)[0],
                    "role": role,
                    "tool_names": tool_names,
                    "repeated": bool(metadata.get("is_repeated_tool_call")),
                    "trajectory_repeated": bool(metadata.get("trajectory_has_repeated_tool_call")),
                    "prompt_tokens": approximate_tokens,
                    "features": (
                        str(metadata.get("task_type") or "unknown"),
                        str(metadata["decision_depth_bin"]),
                        str(metadata["prompt_length_bin"]),
                        str(metadata["previous_observation_outcome"]),
                        "multi_call" if len(tool_names) > 1 else "single_call",
                    ),
                }
            )

    selected: dict[int, list[str]] = {}
    anchors = [item for item in prepared if item["role"] in {"planning", "initial_tool_step", "final"}]
    for item in anchors:
        selected[item["source_line"]] = ["mandatory_initial" if item["role"] != "final" else "mandatory_final"]

    intermediate = [item for item in prepared if item["role"] == "tool_step"]
    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in prepared:
        for tool_name in item["tool_names"]:
            by_tool[tool_name].append(item)

    tool_selected = Counter(
        tool_name
        for item in prepared
        if item["source_line"] in selected
        for tool_name in item["tool_names"]
    )
    trajectory_intermediate = Counter()
    intermediate_selected = 0

    def add_intermediate(item: dict[str, Any], reason: str, *, allow_cap_override: bool = False) -> bool:
        nonlocal intermediate_selected
        line_number = item["source_line"]
        if line_number in selected:
            selected[line_number].append(reason)
            return False
        if intermediate_selected >= intermediate_budget:
            return False
        if not allow_cap_override and trajectory_intermediate[item["source_id"]] >= max_per_trajectory:
            return False
        selected[line_number] = [reason]
        trajectory_intermediate[item["source_id"]] += 1
        intermediate_selected += 1
        tool_selected.update(item["tool_names"])
        return True

    # Cover rare tools first; exact repeated calls are only admitted if no
    # non-repeated candidate can satisfy that tool's minimum.
    for tool_name in sorted(original_tool_names, key=lambda name: (len(by_tool[name]), name.casefold())):
        target = min(min_per_tool, len(by_tool[tool_name]))
        candidates = sorted(
            by_tool[tool_name],
            key=lambda item: (
                item["role"] != "tool_step",
                item["repeated"],
                trajectory_intermediate[item["source_id"]],
                item["prompt_tokens"],
                _stable_jitter(seed, item["key"]),
            ),
        )
        for item in candidates:
            if tool_selected[tool_name] >= target:
                break
            if item["role"] != "tool_step":
                continue
            add_intermediate(item, f"tool_coverage:{tool_name}")
        if tool_selected[tool_name] < target:
            for item in candidates:
                if tool_selected[tool_name] >= target:
                    break
                if item["role"] != "tool_step":
                    continue
                add_intermediate(item, f"rare_tool_override:{tool_name}", allow_cap_override=True)

    feature_frequency: Counter[tuple[str, str]] = Counter()
    for item in intermediate:
        for name, value in zip(("task", "depth", "length", "outcome", "calls"), item["features"], strict=True):
            feature_frequency[(name, value)] += 1
        for tool_name in item["tool_names"]:
            feature_frequency[("tool", tool_name)] += 1

    def priority(item: dict[str, Any]) -> tuple[float, float, int]:
        score = 0.0
        for name, value in zip(("task", "depth", "length", "outcome", "calls"), item["features"], strict=True):
            score += 1.0 / math.sqrt(max(1, feature_frequency[(name, value)]))
        for tool_name in item["tool_names"]:
            score += 2.0 / math.sqrt(max(1, feature_frequency[("tool", tool_name)]))
        if item["repeated"]:
            score -= 1000.0
        return (-score, _stable_jitter(seed, item["key"]), item["source_line"])

    for item in sorted(intermediate, key=priority):
        if intermediate_selected >= intermediate_budget:
            break
        add_intermediate(item, "stratified_fill")

    # The cap is a diversity guard, not a reason to miss the explicit budget.
    if intermediate_selected < intermediate_budget:
        for item in sorted(intermediate, key=priority):
            if intermediate_selected >= intermediate_budget:
                break
            add_intermediate(item, "budget_fill_cap_override", allow_cap_override=True)
    if intermediate_selected != intermediate_budget:
        raise ValueError(
            f"could only select {intermediate_selected}/{intermediate_budget} intermediate decisions"
        )

    unique_items = sorted(
        (item for item in prepared if item["source_line"] in selected),
        key=lambda item: item["source_line"],
    )

    # Apply the real Qwen chat template and deterministic compactor only to the
    # unique selected decisions.  Weighted copies share the audited prompt.
    for item in unique_items:
        row = item["row"]
        prompt = row["prompt"]
        original_tokens = render_length(tokenizer, prompt)
        compacted_prompt, compaction = compact_prompt_with_audit(
            tokenizer,
            prompt,
            max_prompt_tokens,
            summary_max_tokens=summary_max_tokens,
        )
        final_tokens = int(compaction["output_tokens"])
        if final_tokens > max_prompt_tokens:
            raise AssertionError(f"prompt over limit after compaction: {final_tokens}")
        metadata = copy.deepcopy(_metadata(row))
        metadata["prompt_tokens_original"] = original_tokens
        metadata["prompt_tokens_final"] = final_tokens
        metadata["prompt_length_bin"] = _length_bin(original_tokens, max_prompt_tokens)
        if compaction["compacted"]:
            metadata["context_compaction"] = {**compaction, "source_line": item["source_line"]}
        else:
            metadata.pop("context_compaction", None)
        row["prompt"] = compacted_prompt
        row["metadata"] = metadata
        item["prompt_tokens"] = final_tokens

    output_rows: list[dict[str, Any]] = []
    copy_count_by_line: Counter[int] = Counter()

    def append_copy(item: dict[str, Any], extra_reason: str | None = None) -> None:
        reasons = list(selected[item["source_line"]])
        if extra_reason:
            reasons.append(extra_reason)
        copy_index = copy_count_by_line[item["source_line"]]
        output_rows.append(_copy_for_output(item["row"], reasons, copy_index))
        copy_count_by_line[item["source_line"]] += 1

    for item in unique_items:
        append_copy(item)
    for item in unique_items:
        if item["role"] == "planning":
            append_copy(item, "planning_2x")
        if item["role"] == "final":
            append_copy(item, "final_2x")
            if item["trajectory_repeated"]:
                append_copy(item, "repeated_trajectory_final_3x")

    padding_records = (-len(output_rows)) % multiple
    shortest = sorted(unique_items, key=lambda item: (item["prompt_tokens"], item["source_line"]))
    for item in shortest[:padding_records]:
        append_copy(item, "batch_alignment_padding")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
    try:
        with output_tmp.open("w", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(output_tmp, output_path)

        unique_role_counts = Counter(item["role"] for item in unique_items)
        effective_role_counts = Counter(_metadata(row)["decision_role"] for row in output_rows)
        selected_tool_names = {
            str(name)
            for row in output_rows
            for name in (_metadata(row).get("tool_names") or [])
            if str(name)
        }
        selected_task_types = {
            str(_metadata(row).get("task_type") or "unknown") for row in output_rows
        }
        selection_audit = []
        for item in prepared:
            reasons = selected.get(item["source_line"], [])
            selection_audit.append(
                {
                    "source_line": item["source_line"],
                    "decision_key": list(item["key"]),
                    "decision_role": item["role"],
                    "selected": bool(reasons),
                    "sampling_reasons": sorted(set(reasons)),
                    "sampling_copies": copy_count_by_line[item["source_line"]],
                    "exclusion_reason": None
                    if reasons
                    else ("exact_repeated_call_not_selected" if item["repeated"] else "outside_stratified_budget"),
                }
            )
        compaction_audit = []
        for item in unique_items:
            compaction = _metadata(item["row"]).get("context_compaction")
            if isinstance(compaction, dict):
                compaction_audit.append(
                    {
                        "source_line": item["source_line"],
                        "decision_key": list(item["key"]),
                        **compaction,
                    }
                )
        manifest = {
            "schema_version": VIEW_SCHEMA,
            "contract": {
                "model": model_name,
                "max_context_tokens": max_prompt_tokens + max_response_tokens,
                "max_prompt_tokens": max_prompt_tokens,
                "max_response_tokens": max_response_tokens,
                "summary_max_tokens": summary_max_tokens,
                "compaction_schema": COMPACTION_SCHEMA,
                "intermediate_budget": intermediate_budget,
                "min_per_tool": min_per_tool,
                "max_per_trajectory": max_per_trajectory,
                "multiple": multiple,
                "seed": seed,
            },
            "source": {"path": str(input_path.resolve()), "sha256": source_sha256, "records": sum(original_role_counts.values())},
            "original_role_counts": dict(original_role_counts),
            "prepared_records": len(prepared),
            "rejected_records": len(rejected),
            "rejected": rejected,
            "selection": selection_audit,
            "unique_records": len(unique_items),
            "unique_role_counts": dict(unique_role_counts),
            "intermediate_selected": intermediate_selected,
            "effective_records": len(output_rows),
            "effective_role_counts": dict(effective_role_counts),
            "padding_records": padding_records,
            "trajectories_with_repeated_tool_call": len(
                {item["source_id"] for item in prepared if item["trajectory_repeated"]}
            ),
            "compacted_records": sum(
                1 for item in unique_items if "context_compaction" in _metadata(item["row"])
            ),
            "compactions": compaction_audit,
            "coverage": {
                "original_tool_count": len(original_tool_names),
                "selected_tool_count": len(selected_tool_names),
                "missing_tools": sorted(original_tool_names - selected_tool_names),
                "selected_task_types": sorted(selected_task_types),
            },
            "output": {"path": str(output_path.resolve()), "sha256": _sha256(output_path), "records": len(output_rows)},
        }
        manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(manifest_tmp, manifest_path)
    finally:
        output_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
    return manifest


def main() -> None:
    from slime.utils.processing_utils import load_tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-prompt-tokens", default=245760, type=int)
    parser.add_argument("--max-response-tokens", default=16384, type=int)
    parser.add_argument("--summary-max-tokens", default=32768, type=int)
    parser.add_argument("--intermediate-budget", default=2500, type=int)
    parser.add_argument("--min-per-tool", default=8, type=int)
    parser.add_argument("--max-per-trajectory", default=8, type=int)
    parser.add_argument("--multiple", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model, trust_remote_code=True)
    manifest = materialize_decision_aware_view(
        input_path=args.input.resolve(),
        output_path=args.output.resolve(),
        manifest_path=args.manifest.resolve(),
        tokenizer=tokenizer,
        model_name=args.model,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
        summary_max_tokens=args.summary_max_tokens,
        intermediate_budget=args.intermediate_budget,
        min_per_tool=args.min_per_tool,
        max_per_trajectory=args.max_per_trajectory,
        multiple=args.multiple,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
