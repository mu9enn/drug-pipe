#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from drug_agent.toolrl.v8_dataset import percentiles, sha256_file, stable_json


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _tokenizer_identity(model_name: str) -> dict[str, Any]:
    path = Path(model_name)
    if not path.is_dir():
        return {"huggingface_id": model_name}
    files = {}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        candidate = path / name
        if candidate.is_file():
            files[name] = {"bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)}
    return {"path": str(path.resolve()), "files": files}


def audit_lengths(
    input_path: Path,
    tokenizer: Any,
    *,
    model_name: str,
    context_limit: int = 262144,
    generation_budgets: tuple[int, ...] = (16384, 32768, 65536),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row["prompt"]
            tools = row.get("tools")
            target = row["label"]["target_assistant"]
            rendered_prompt = tokenizer.apply_chat_template(
                prompt, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            rendered_full = tokenizer.apply_chat_template(
                prompt + [target], tools=tools, tokenize=False, add_generation_prompt=False, enable_thinking=True
            )
            if not rendered_full.startswith(rendered_prompt):
                raise ValueError(f"target is not a continuation of the native generation prompt: {row['id']}")
            rendered_target = rendered_full[len(rendered_prompt) :]
            reasoning = str(row["label"].get("target_reasoning") or "")
            calls = row["label"].get("target_tool_calls") or []
            action = stable_json(calls) if calls else stable_json(row["label"].get("target_final_answer"))
            target_without_reasoning = copy.deepcopy(target)
            target_without_reasoning["reasoning_content"] = ""
            rendered_action = tokenizer.apply_chat_template(
                prompt + [target_without_reasoning],
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True,
            )
            if not rendered_action.startswith(rendered_prompt):
                raise ValueError(f"action is not a continuation of the native generation prompt: {row['id']}")
            prompt_tokens = _token_count(tokenizer, rendered_prompt)
            target_tokens = _token_count(tokenizer, rendered_target)
            detail = {
                "decision_id": row["id"],
                "source_id": row["metadata"]["source_id"],
                "trajectory_index": row["metadata"]["trajectory_index"],
                "decision_ordinal": row["metadata"]["decision_ordinal"],
                "decision_type": row["metadata"]["decision_type"],
                "tool_names": row["metadata"].get("tool_names", []),
                "prompt_tokens": prompt_tokens,
                "reasoning_tokens": _token_count(tokenizer, reasoning),
                "structured_action_or_final_tokens": _token_count(tokenizer, action),
                "native_action_or_final_tokens": _token_count(
                    tokenizer, rendered_action[len(rendered_prompt) :]
                ),
                "full_reference_tokens": target_tokens,
                "prompt_plus_reference_tokens": prompt_tokens + target_tokens,
            }
            detail["budget_gates"] = {
                str(budget): {
                    "max_prompt_tokens": context_limit - budget,
                    "prompt_fits": prompt_tokens <= context_limit - budget,
                    "teacher_reference_fits_generation_budget": target_tokens <= budget,
                }
                for budget in generation_budgets
            }
            details.append(detail)

    fields = (
        "prompt_tokens",
        "reasoning_tokens",
        "structured_action_or_final_tokens",
        "native_action_or_final_tokens",
        "full_reference_tokens",
        "prompt_plus_reference_tokens",
    )
    report = {
        "schema_version": "toolrl_v8_length_audit_v1",
        "input": {"path": str(input_path.resolve()), "sha256": sha256_file(input_path), "records": len(details)},
        "tokenizer": model_name,
        "tokenizer_identity": _tokenizer_identity(model_name),
        "native_template": {"add_generation_prompt": True, "enable_thinking": True, "tools_included": True},
        "context_limit": context_limit,
        "distributions": {field: percentiles(item[field] for item in details) for field in fields},
        "reference_threshold_counts": {
            str(threshold): sum(item["full_reference_tokens"] > threshold for item in details)
            for threshold in generation_budgets
        },
        "budget_gates": {
            str(budget): {
                "generation_budget": budget,
                "max_prompt_tokens": context_limit - budget,
                "prompt_over_limit": sum(item["prompt_tokens"] > context_limit - budget for item in details),
                "teacher_reference_over_generation_budget": sum(item["full_reference_tokens"] > budget for item in details),
                "prompt_plus_teacher_reference_over_context": sum(item["prompt_plus_reference_tokens"] > context_limit for item in details),
            }
            for budget in generation_budgets
        },
        "over_limit_by_tool": {
            str(budget): dict(
                Counter(
                    name
                    for item in details
                    if item["full_reference_tokens"] > budget
                    for name in (item["tool_names"] or ["final_answer"])
                )
            )
            for budget in generation_budgets
        },
        "longest_references": sorted(details, key=lambda item: -item["full_reference_tokens"])[:100],
        "policy": "audit_only; no target truncation and no automatic deletion",
    }
    return report, details


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Audit native Qwen token lengths for v8 ToolRL decisions.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--context-limit", type=int, default=262144)
    parser.add_argument("--generation-budgets", type=int, nargs="+", default=[16384, 32768, 65536])
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    report, details = audit_lengths(
        args.input.resolve(), tokenizer, model_name=args.model, context_limit=args.context_limit,
        generation_budgets=tuple(args.generation_budgets),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "length_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_root / "length_details.jsonl").open("w", encoding="utf-8") as output:
        for detail in details:
            output.write(json.dumps(detail, ensure_ascii=False, separators=(",", ":")) + "\n")
    (args.output_root / "length_details.pretty.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
