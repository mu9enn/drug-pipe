#!/usr/bin/env python3
"""Fail-closed schema, uniqueness, protocol, and token gate for a v6 release."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.protocol.react_protocol import parse_runtime_decision
from drug_agent.scripts.select_toolrl_decisions import _canonical_target, _render_prompt


def validate(root: Path, model: Path, *, view: str = "production") -> dict:
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != "toolrl_turn_v1":
        raise ValueError("release protocol is not toolrl_turn_v1")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    keys: set[str] = set()
    roles: Counter[str] = Counter()
    thought_rows = multi_call_rows = 0
    max_prompt = max_target = max_context = 0
    rows = 0
    if view not in {"production", "official_baseline"}:
        raise ValueError(f"unsupported view: {view}")
    manifest_section = manifest["toolrl"] if view == "production" else manifest["toolrl_official_baseline"]
    relative_path = "toolrl/toolrl_steps.jsonl" if view == "production" else "toolrl/toolrl_steps.official_baseline.jsonl"
    with (root / relative_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            if metadata.get("protocol") != "toolrl_turn_v1":
                raise ValueError(f"wrong protocol at line {line_number}")
            key = (
                f"{metadata.get('source_id')}:{metadata.get('assistant_index')}:"
                f"{metadata.get('assistant_subturn_index', 0)}:{metadata.get('decision_type')}"
            )
            if key in keys:
                raise ValueError(f"duplicate decision key: {key}")
            keys.add(key)
            target = _canonical_target(row)
            parsed = parse_runtime_decision(target, strict_toolrl_turn=True)
            if not parsed.get("ok"):
                raise ValueError(f"invalid target at line {line_number}: {parsed.get('error_message')}")
            if "<observation" in target:
                raise ValueError(f"observation leaked into target at line {line_number}")
            prompts = row.get("prompt")
            if not isinstance(prompts, list) or not prompts:
                raise ValueError(f"missing prompt at line {line_number}")
            if any(set(message) - {"role", "content", "name"} for message in prompts if isinstance(message, dict)):
                raise ValueError(f"unexpected prompt-side tool/schema fields at line {line_number}")
            system_text = str(prompts[0].get("content") or "") if prompts and isinstance(prompts[0], dict) else ""
            has_catalog = "Available Tools" in system_text and "Parameters:" in system_text
            if has_catalog != (view == "official_baseline"):
                raise ValueError(f"prompt strategy mismatch at line {line_number}: view={view}")
            assistant_prefix = str(metadata.get("assistant_prefix") or "")
            rendered = _render_prompt(tokenizer, prompts, assistant_prefix)
            prompt_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
            target_tokens = len(tokenizer.encode(target, add_special_tokens=False))
            target_limit_failed = target_tokens > 16384
            context_limit_failed = prompt_tokens + 16384 > 262144
            if prompt_tokens > 245760 or target_limit_failed or context_limit_failed:
                raise ValueError(
                    f"token contract failed at line {line_number}: {prompt_tokens}+{target_tokens}"
                )
            stored_prompt = int(metadata.get("prompt_tokens_final") or -1)
            stored_target = int(metadata.get("canonical_target_tokens") or -1)
            if (prompt_tokens, target_tokens) != (stored_prompt, stored_target):
                raise ValueError(
                    f"stored token audit mismatch at line {line_number}: "
                    f"actual={(prompt_tokens, target_tokens)} stored={(stored_prompt, stored_target)}"
                )
            rows += 1
            roles[str(metadata.get("decision_role") or "unknown")] += 1
            thought_rows += bool(parsed.get("thoughts"))
            multi_call_rows += len(parsed.get("tool_calls") or []) > 1
            max_prompt = max(max_prompt, prompt_tokens)
            max_target = max(max_target, target_tokens)
            # Production trains against the materialized teacher target, while the
            # official-RL view only uses that target as out-of-band reward data.
            # Its runtime allocation is therefore prompt + rollout response cap.
            observed_context = (
                prompt_tokens + target_tokens
                if view == "production"
                else prompt_tokens + 16384
            )
            max_context = max(max_context, observed_context)
    if rows == 0 or rows % 4 or rows != int(manifest_section["records"]):
        raise ValueError(f"invalid fixed-view size: {rows}")
    return {
        "schema_version": "toolrl_turn_release_validation_v1",
        "ok": True,
        "view": view,
        "records": rows,
        "unique_decision_keys": len(keys),
        "decision_count": rows,
        "grpo_group_count": rows,
        "n_samples_per_decision": 4,
        "sampled_response_count": rows * 4,
        "rollout_batch_size_decisions": 4,
        "rollout_batch_count": rows // 4,
        "role_counts": dict(roles),
        "thought_rows": thought_rows,
        "multi_call_rows": multi_call_rows,
        "max_prompt_tokens": max_prompt,
        "max_target_tokens": max_target,
        "max_context_tokens": max_context,
        "tool_catalog_injected_into_prompts": view == "official_baseline",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--view", choices=("production", "official_baseline"), default="production")
    args = parser.parse_args()
    report = validate(args.root, args.model, view=args.view)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
