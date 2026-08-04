#!/usr/bin/env python3
"""Create an audited, message-boundary-preserving RL context dataset.

The source ToolRL/GAD records are decision prefixes: the initial system/task
messages and the most recent execution turns are the useful context for the
next decision.  Slime otherwise drops every prompt above its maximum length.
This utility retains all records by keeping the first two messages and the
largest contiguous suffix that fits, without changing labels or rewards.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from transformers import AutoTokenizer


OMISSION_NOTE = "[Context note: earlier recorded trajectory turns were omitted to fit the training context window.]"
CONTENT_OMISSION = "\n[... earlier content omitted ...]\n"


def render_length(tokenizer, messages: list[dict]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # transformers 5 returns BatchEncoding here (whose len is the number of
    # fields, usually 2), while transformers 4 returned the token-id list.
    token_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    return len(token_ids)


def compact_prompt(tokenizer, prompt: list[dict], max_tokens: int) -> tuple[list[dict], int, int]:
    original_tokens = render_length(tokenizer, prompt)
    if original_tokens <= max_tokens:
        return prompt, original_tokens, original_tokens
    if len(prompt) < 3:
        raise ValueError(f"Oversized prompt has only {len(prompt)} messages")

    prefix = copy.deepcopy(prompt[:2])
    prefix[0]["content"] = prefix[0].get("content", "").rstrip() + "\n\n" + OMISSION_NOTE

    # Find the earliest suffix start that fits.  Candidate length decreases
    # monotonically as complete early messages are removed.
    low, high = 2, len(prompt) - 1
    best = None
    while low <= high:
        middle = (low + high) // 2
        candidate = prefix + copy.deepcopy(prompt[middle:])
        if render_length(tokenizer, candidate) <= max_tokens:
            best = candidate
            high = middle - 1
        else:
            low = middle + 1

    if best is None:
        # A single final observation can itself exceed the budget.  Preserve
        # its role, a short identifying head, and as much recent tail as fits.
        last = copy.deepcopy(prompt[-1])
        content = last.get("content", "")
        content_tokens = tokenizer.encode(content, add_special_tokens=False)
        last["content"] = ""
        overhead = render_length(tokenizer, prefix + [last])
        marker_tokens = tokenizer.encode(CONTENT_OMISSION, add_special_tokens=False)
        available = max_tokens - overhead - len(marker_tokens)
        if available <= 0:
            raise ValueError("System/task prefix leaves no room for the final trajectory message")
        head_count = min(256, max(0, available // 4))
        tail_count = max(0, available - head_count)
        kept = content_tokens[:head_count] + marker_tokens + content_tokens[-tail_count:]
        last["content"] = tokenizer.decode(kept, skip_special_tokens=False)
        best = prefix + [last]

    # Tokenizer decoding and template delimiters can differ by a few tokens;
    # trim only the final content tail until the hard contract is satisfied.
    while render_length(tokenizer, best) > max_tokens:
        last = best[-1]
        tokens = tokenizer.encode(last.get("content", ""), add_special_tokens=False)
        if len(tokens) <= 64:
            raise ValueError("Unable to compact prompt within the requested token budget")
        last["content"] = tokenizer.decode(tokens[:-64], skip_special_tokens=False)

    return best, original_tokens, render_length(tokenizer, best)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-tokens", type=int, default=12032)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    records = 0
    compacted = 0
    original_max = 0
    output_max = 0
    removed_messages = 0
    with args.input.open() as source, args.output.open("w") as destination:
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)
            prompt = record["prompt"]
            compact, original_tokens, output_tokens = compact_prompt(tokenizer, prompt, args.max_tokens)
            if original_tokens > args.max_tokens:
                compacted += 1
                removed_messages += max(0, len(prompt) - len(compact))
                metadata = copy.deepcopy(record.get("metadata") or {})
                metadata["context_compaction"] = {
                    "schema_version": "react_head_suffix_v1",
                    "source_line": line_number,
                    "original_tokens": original_tokens,
                    "output_tokens": output_tokens,
                    "original_messages": len(prompt),
                    "output_messages": len(compact),
                }
                # GAD carries the same state in three places.  Keep the
                # discriminator's teacher/student comparison conditioned on
                # exactly the context shown to the policy, not the discarded
                # 80K-token history that its own tokenizer would truncate in
                # a different way.
                if isinstance(record.get("state_messages"), list):
                    record["state_messages"] = copy.deepcopy(compact)
                if isinstance(metadata.get("state_messages"), list):
                    metadata["state_messages"] = copy.deepcopy(compact)
                record["metadata"] = metadata
                record["prompt"] = compact
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            records += 1
            original_max = max(original_max, original_tokens)
            output_max = max(output_max, output_tokens)

    audit = {
        "schema_version": "react_head_suffix_v1",
        "source": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "tokenizer": args.tokenizer,
        "max_tokens": args.max_tokens,
        "records": records,
        "compacted_records": compacted,
        "unchanged_records": records - compacted,
        "removed_messages": removed_messages,
        "original_max_tokens": original_max,
        "output_max_tokens": output_max,
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
