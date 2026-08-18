#!/usr/bin/env python3
"""Build the auditable candidate pool for online policy-boundary sampling.

The static pass enforces validity, no-progress removal, context limits, and
coverage.  Learnability is intentionally evaluated by the current policy at
rollout time (n=4), not guessed from teacher-data heuristics here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.scripts.compact_rl_context import COMPACTION_SCHEMA, compact_prompt_with_audit
from drug_agent.context_summary import ClaudeContextSummarizer


VIEW_SCHEMA = "toolrl_static_curated_view_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without_summary(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_summary(item) for key, item in value.items() if key != "summary"}
    if isinstance(value, list):
        return [_without_summary(item) for item in value]
    return value


def _canonical_target(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    label = row.get("label") if isinstance(row.get("label"), dict) else {}
    protocol = str(label.get("protocol") or metadata.get("protocol") or "")
    if protocol == "toolrl_turn_v1":
        assistant_content = label.get("assistant_content")
        if not isinstance(assistant_content, str) or not assistant_content.strip():
            target = label.get("target_assistant", metadata.get("target_assistant"))
            assistant_content = target.get("content") if isinstance(target, dict) else None
        if not isinstance(assistant_content, str) or not assistant_content.strip():
            raise ValueError("toolrl_turn_v1 row has no canonical assistant content")
        return assistant_content
    decision_type = str(label.get("decision_type") or metadata.get("decision_type") or "")
    if decision_type == "final_answer":
        target = _without_summary(label.get("target_final_answer", metadata.get("target_final_answer")))
        return "<final_answer>" + json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "</final_answer>"
    calls = label.get("target_tool_calls", metadata.get("target_tool_calls"))
    if not isinstance(calls, list) or not calls:
        raise ValueError("tool row has no target calls")
    return "".join(
        "<tool_call>"
        + json.dumps(
            {"tool_name": str(call.get("tool_name") or call.get("name") or ""), "arguments": call.get("arguments") or {}},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "</tool_call>"
        for call in calls
        if isinstance(call, dict)
    )


def _decision_key(metadata: dict[str, Any]) -> str:
    return (
        f"{metadata.get('source_id') or metadata.get('task_id')}:"
        f"{metadata.get('assistant_index')}:{metadata.get('assistant_subturn_index', 0)}:"
        f"{metadata.get('decision_type')}"
    )


def _render_prompt(tokenizer: Any, prompt: list[dict[str, Any]], assistant_prefix: str) -> str:
    rendered = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return rendered + assistant_prefix


def _approximate_length_bin(row: dict[str, Any]) -> str:
    chars = sum(len(str(message.get("content") or "")) for message in row.get("prompt") or [] if isinstance(message, dict))
    if chars <= 32_768:
        return "short"
    if chars <= 131_072:
        return "medium"
    if chars <= 524_288:
        return "long"
    return "very_long"


def _depth_bin(metadata: dict[str, Any]) -> int:
    ordinal = max(0, int(metadata.get("decision_ordinal") or 0))
    total = max(1, int(metadata.get("trajectory_decision_count") or 1) - 1)
    return min(9, int(10 * ordinal / total))


def _previous_observation_status(row: dict[str, Any]) -> str:
    for message in reversed(row.get("prompt") or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = str(message.get("content") or "").lower()
        if "<observation" not in content:
            continue
        if any(token in content for token in ('"ok":false', '"status":"error"', '"status":"failed"', '"status":"timeout"')):
            return "failure"
        if any(token in content for token in ('"ok":true', '"status":"success"', '"status":"completed"')):
            return "success"
        return "unknown"
    return "none"


def _middle_stratum(row: dict[str, Any], metadata: dict[str, Any]) -> tuple[Any, ...]:
    calls = metadata.get("target_tool_calls") if isinstance(metadata.get("target_tool_calls"), list) else []
    call_shapes = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        call_shapes.append((str(call.get("tool_name") or ""), tuple(sorted(str(key) for key in arguments))))
    return (
        str(metadata.get("task_type") or "unknown"),
        tuple(sorted(call_shapes)),
        _depth_bin(metadata),
        _approximate_length_bin(row),
        _previous_observation_status(row),
        "multi" if len(calls) > 1 else "single",
    )


def select_decisions(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    model_path: Path,
    max_prompt_tokens: int,
    max_response_tokens: int,
    summary_max_tokens: int,
    *,
    semantic_summarizer: str = "none",
    summary_cache_root: Path | None = None,
    claude_bin: str = "claude",
    llm_timeout_sec: float = 600.0,
    llm_max_attempts: int = 3,
    max_context_tokens: int = 262144,
    batch_multiple: int = 4,
    selection_mode: str = "curated_static",
) -> dict[str, Any]:
    if batch_multiple <= 0:
        raise ValueError("batch_multiple must be positive")
    if selection_mode not in {"curated_static", "all_static"}:
        raise ValueError(f"unsupported selection_mode: {selection_mode}")
    # First pass: create a diversity-constrained candidate pool without a
    # fixed record budget.  Initial/final actions remain candidates, while
    # equivalent middle decisions contribute one deterministic representative.
    selected_lines: set[int] = set()
    middle_entries: list[tuple[int, str, tuple[Any, ...], str]] = []
    stratum_winner: dict[tuple[Any, ...], tuple[str, int]] = {}
    pre_excluded: list[dict[str, Any]] = []
    rows_by_line: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(input_path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        rows_by_line[line_number] = row
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        key = _decision_key(metadata)
        if selection_mode == "curated_static" and metadata.get("is_no_progress_repeat"):
            pre_excluded.append({"line": line_number, "decision_key": key, "reason": "no_progress_repeat"})
            continue
        role = str(metadata.get("decision_role") or "")
        if role not in {"tool_step", "final"}:
            raise ValueError(f"invalid v2 ToolRL role at line {line_number}: {role!r}")
        if selection_mode == "all_static":
            selected_lines.add(line_number)
            continue
        if role == "final" or metadata.get("is_initial_step"):
            selected_lines.add(line_number)
            continue
        stratum = _middle_stratum(row, metadata)
        rank = hashlib.sha256((key + json.dumps(stratum, ensure_ascii=False, sort_keys=True)).encode()).hexdigest()
        middle_entries.append((line_number, key, stratum, rank))
        current = stratum_winner.get(stratum)
        if current is None or rank < current[0]:
            stratum_winner[stratum] = (rank, line_number)
    selected_lines.update(line_number for _, line_number in stratum_winner.values())
    alignment_pool = sorted(
        (entry for entry in middle_entries if entry[0] not in selected_lines),
        key=lambda entry: (entry[3], entry[1], entry[0]),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    summarizer = None
    if semantic_summarizer == "claude":
        if summary_cache_root is None:
            raise ValueError("summary_cache_root is required for Claude semantic summaries")
        summarizer = ClaudeContextSummarizer(
            cache_root=summary_cache_root,
            claude_bin=claude_bin,
            timeout_sec=llm_timeout_sec,
            max_attempts=llm_max_attempts,
        )
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = pre_excluded
    roles: Counter[str] = Counter()
    task_types: set[str] = set()
    tools: set[str] = set()
    compacted = 0
    max_prompt = max_target = 0
    seen_keys: set[str] = set()
    considered_lines: set[int] = set()
    alignment_added: list[dict[str, Any]] = []

    def materialize_candidate(line_number: int, selection_reason: str) -> bool:
        nonlocal compacted, max_prompt, max_target
        considered_lines.add(line_number)
        row = rows_by_line[line_number]
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        key = _decision_key(metadata)
        if key in seen_keys:
            excluded.append({"line": line_number, "decision_key": key, "reason": "duplicate_decision_key"})
            return False
        seen_keys.add(key)
        if selection_mode == "curated_static" and metadata.get("is_no_progress_repeat"):
            excluded.append({"line": line_number, "decision_key": key, "reason": "no_progress_repeat"})
            return False
        role = str(metadata.get("decision_role") or "")
        if role not in {"tool_step", "final"}:
            raise ValueError(f"invalid v2 ToolRL role at line {line_number}: {role!r}")
        target_tokens = len(tokenizer.encode(_canonical_target(row), add_special_tokens=False))
        if target_tokens > max_response_tokens:
            excluded.append(
                {
                    "line": line_number,
                    "decision_key": key,
                    "reason": "target_action_exceeds_runtime_response_limit",
                    "target_tokens": target_tokens,
                    "runtime_response_limit": max_response_tokens,
                }
            )
            return False
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"missing prompt at line {line_number}")
        assistant_prefix = str(metadata.get("assistant_prefix") or "")
        prefix_tokens = len(tokenizer.encode(assistant_prefix, add_special_tokens=False))
        compaction_budget = max(1, max_prompt_tokens - prefix_tokens)
        try:
            for _ in range(3):
                final_prompt, audit = compact_prompt_with_audit(
                    tokenizer,
                    prompt,
                    compaction_budget,
                    summary_max_tokens=summary_max_tokens,
                    semantic_summarizer=(
                        lambda messages: summarizer.summarize(
                            messages, tokenizer=tokenizer, max_tokens=max(1, summary_max_tokens - 128)
                        )
                    ) if summarizer is not None else None,
                )
                prompt_tokens = len(
                    tokenizer.encode(
                        _render_prompt(tokenizer, final_prompt, assistant_prefix),
                        add_special_tokens=False,
                    )
                )
                if prompt_tokens <= max_prompt_tokens:
                    break
                compaction_budget -= prompt_tokens - max_prompt_tokens + 16
                if compaction_budget < 1:
                    raise ValueError("assistant prefix alone exceeds prompt budget")
            else:
                raise ValueError("prefix-conditioned prompt did not fit after compaction retries")
        except Exception as exc:
            excluded.append(
                {
                    "line": line_number,
                    "decision_key": key,
                    "reason": "context_compaction_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            return False
        if prompt_tokens > max_prompt_tokens:
            raise ValueError(f"compactor exceeded prompt contract for {key}: {prompt_tokens}")
        if prompt_tokens + target_tokens > max_context_tokens:
            excluded.append(
                {
                    "line": line_number,
                    "decision_key": key,
                    "reason": "context_exceeds_limit",
                    "prompt_tokens": prompt_tokens,
                    "target_tokens": target_tokens,
                }
            )
            return False
        out = copy.deepcopy(row)
        out["prompt"] = final_prompt
        out_metadata = copy.deepcopy(metadata)
        out_metadata.update(
            {
                "selection_schema": VIEW_SCHEMA,
                "selection_stage": selection_reason,
                "learnability_selector": "none_static_fixed_view",
                "sampling_copy_index": 0,
                "prompt_tokens_original": int(audit["original_tokens"]),
                "prompt_tokens_final": prompt_tokens,
                "canonical_target_tokens": target_tokens,
                "assistant_prefix_tokens": prefix_tokens,
                "prompt_rendering": "qwen_chat_template_plus_exact_assistant_prefix",
                "context_compaction": audit,
            }
        )
        out["metadata"] = out_metadata
        kept.append(out)
        roles[role] += 1
        task_types.add(str(metadata.get("task_type") or "unknown"))
        tools.update(str(name) for name in metadata.get("tool_names") or [] if str(name))
        compacted += bool(audit.get("compacted"))
        max_prompt = max(max_prompt, prompt_tokens)
        max_target = max(max_target, target_tokens)
        return True

    for line_number in sorted(selected_lines):
        materialize_candidate(line_number, "static_coverage")

    for line_number, key, stratum, rank in alignment_pool:
        if kept and len(kept) % batch_multiple == 0:
            break
        if materialize_candidate(line_number, "rbs_alignment_coverage_fill"):
            alignment_added.append(
                {
                    "line": line_number,
                    "decision_key": key,
                    "rank": rank,
                    "reason": "rbs_alignment_coverage_fill",
                }
            )
    if selection_mode == "all_static" and kept:
        drop_count = len(kept) % batch_multiple
        for _ in range(drop_count):
            dropped = kept.pop()
            metadata = dropped.get("metadata") if isinstance(dropped.get("metadata"), dict) else {}
            excluded.append(
                {
                    "decision_key": _decision_key(metadata),
                    "reason": "rbs_alignment_deterministic_drop",
                }
            )
            roles[str(metadata.get("decision_role") or "")] -= 1
    if not kept or len(kept) % batch_multiple:
        raise ValueError(
            f"could not align {len(kept)} eligible decisions to batch multiple {batch_multiple}"
        )

    alignment_lines = {item["line"] for item in alignment_added}
    excluded.extend(
        {"line": line_number, "decision_key": key, "reason": "diversity_equivalent_middle_decision"}
        for line_number, key, _, _ in middle_entries
        if line_number not in selected_lines
        and line_number not in alignment_lines
        and line_number not in considered_lines
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": VIEW_SCHEMA,
        "source_path": str(input_path.resolve()),
        "source_sha256": _sha256(input_path),
        "output_path": str(output_path.resolve()),
        "candidate_records": len(kept),
        "eligible_before_diversity_constraint": len(rows_by_line) - len(pre_excluded),
        "middle_diversity_strata": len(stratum_winner),
        "copies_added": 0,
        "batch_multiple": batch_multiple,
        "rbs_alignment_records": alignment_added,
        "role_counts": dict(roles),
        "coverage": {"task_types": sorted(task_types), "tool_names": sorted(tools), "task_type_count": len(task_types), "tool_count": len(tools)},
        "selection": {
            "primary": "all_eligible_fixed_traversal" if selection_mode == "all_static" else "deterministic_static_coverage",
            "n_samples_per_decision": 4,
            "runtime_filter": None,
            "diversity_role": "none" if selection_mode == "all_static" else "fixed_view_constraint_and_audit",
            "zero_variance_groups_are_trained": True,
            "epoch_semantics": "one_unique_decision_group_per_epoch",
        },
        "context": {
            "compaction_schema": COMPACTION_SCHEMA,
            "compacted_records": compacted,
            "max_prompt_tokens": max_prompt_tokens,
            "max_response_tokens": max_response_tokens,
            "max_context_tokens": max_context_tokens,
            "semantic_summarizer": semantic_summarizer,
            "summary_cache_root": str(summary_cache_root.resolve()) if summary_cache_root else None,
            "observed_max_prompt_tokens": max_prompt,
            "observed_max_target_tokens": max_target,
        },
        "excluded_records": len(excluded),
        "excluded_reason_counts": dict(Counter(item["reason"] for item in excluded)),
        "excluded": excluded,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--max-prompt-tokens", type=int, default=245760)
    parser.add_argument("--max-response-tokens", type=int, default=16384)
    parser.add_argument("--summary-max-tokens", type=int, default=32768)
    parser.add_argument("--max-context-tokens", type=int, default=262144)
    parser.add_argument("--batch-multiple", type=int, default=4)
    parser.add_argument("--selection-mode", choices=("curated_static", "all_static"), default="curated_static")
    parser.add_argument("--semantic-summarizer", choices=("none", "claude"), default="none")
    parser.add_argument("--summary-cache-root", type=Path)
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--llm-timeout-sec", type=float, default=600.0)
    parser.add_argument("--llm-max-attempts", type=int, default=3)
    args = parser.parse_args()
    report = select_decisions(
        args.input, args.output, args.manifest, args.model,
        args.max_prompt_tokens, args.max_response_tokens, args.summary_max_tokens,
        semantic_summarizer=args.semantic_summarizer,
        summary_cache_root=args.summary_cache_root,
        claude_bin=args.claude_bin,
        llm_timeout_sec=args.llm_timeout_sec,
        llm_max_attempts=args.llm_max_attempts,
        max_context_tokens=args.max_context_tokens,
        batch_multiple=args.batch_multiple,
        selection_mode=args.selection_mode,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
