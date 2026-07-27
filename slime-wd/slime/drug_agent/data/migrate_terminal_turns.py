from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from drug_agent.utils import read_jsonl, write_json, write_jsonl


FINAL_RE = re.compile(r"<final_answer>([\s\S]*?)</final_answer>")
THOUGHT_RE = re.compile(r"<thought>([\s\S]*?)</thought>")


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _paragraphs(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]


def _clean_summary(summary: str, thought_texts: list[str]) -> tuple[str | None, dict[str, int]]:
    thought_paragraphs = {
        _normalized(paragraph)
        for thought in thought_texts
        for paragraph in _paragraphs(thought)
        if _normalized(paragraph)
    }
    kept: list[str] = []
    seen: set[str] = set()
    removed_against_thought = 0
    removed_internal = 0
    for paragraph in _paragraphs(summary):
        key = _normalized(paragraph)
        if key in thought_paragraphs:
            removed_against_thought += 1
            continue
        if key in seen:
            removed_internal += 1
            continue
        seen.add(key)
        kept.append(paragraph)
    value = "\n\n".join(kept).strip() or None
    return value, {
        "summary_paragraphs_removed_against_thought": removed_against_thought,
        "summary_duplicate_paragraphs_removed": removed_internal,
    }


def _immutable_digest(record: dict[str, Any]) -> str:
    value = deepcopy(record)
    messages = value.get("messages") if isinstance(value.get("messages"), list) else []
    tool_calls: list[str] = []
    observations: list[str] = []
    finals: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content") or "")
        if message.get("role") == "assistant":
            for match in FINAL_RE.finditer(content):
                payload = json.loads(match.group(1))
                if isinstance(payload, dict):
                    payload.pop("summary", None)
                finals.append(payload)
            tool_calls.extend(re.findall(r"<tool_call>([\s\S]*?)</tool_call>", content))
        elif message.get("role") == "user":
            observations.extend(re.findall(r"<observation[^>]*>([\s\S]*?)</observation>", content))
    raw = json.dumps(
        {
            "schema_version": value.get("schema_version"),
            "id": value.get("id"),
            "tool_calls": tool_calls,
            "observations": observations,
            "finals": finals,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def migrate_record(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    migrated = deepcopy(record)
    messages = migrated.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record messages must be a list")
    before_digest = _immutable_digest(migrated)
    merged = False
    summary_stats = {
        "summary_paragraphs_removed_against_thought": 0,
        "summary_duplicate_paragraphs_removed": 0,
    }
    final_indices = [
        index for index, message in enumerate(messages)
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and "<final_answer>" in str(message.get("content") or "")
    ]
    if len(final_indices) != 1:
        raise ValueError(f"expected exactly one final_answer message, got {len(final_indices)}")
    final_index = final_indices[0]
    final_message = messages[final_index]
    final_content = str(final_message.get("content") or "")
    final_match = FINAL_RE.search(final_content)
    if final_match is None:
        raise ValueError("final_answer block is malformed")

    prior_thoughts: list[str] = []
    if final_index > 0 and isinstance(messages[final_index - 1], dict) and messages[final_index - 1].get("role") == "assistant":
        prior = messages[final_index - 1]
        prior_content = str(prior.get("content") or "")
        prior_thoughts = THOUGHT_RE.findall(prior_content)
        final_content = prior_content.rstrip() + "\n" + final_content.lstrip()
        prior["content"] = final_content
        messages.pop(final_index)
        final_message = prior
        merged = True
    else:
        prior_thoughts = THOUGHT_RE.findall(final_content)

    final_match = FINAL_RE.search(str(final_message.get("content") or ""))
    payload = json.loads(final_match.group(1))
    if not isinstance(payload, dict):
        raise ValueError("final_answer payload must be an object")
    summary = payload.get("summary")
    if isinstance(summary, str) and summary.strip():
        cleaned_summary, summary_stats = _clean_summary(summary, prior_thoughts)
        if cleaned_summary is None:
            payload.pop("summary", None)
        else:
            payload["summary"] = cleaned_summary
        replacement = "<final_answer>" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "</final_answer>"
        content = str(final_message.get("content") or "")
        final_message["content"] = content[: final_match.start()] + replacement + content[final_match.end() :]

    after_digest = _immutable_digest(migrated)
    if before_digest != after_digest:
        raise ValueError("migration changed immutable tool/observation/final-result facts")
    adjacent = sum(
        1 for left, right in zip(messages, messages[1:])
        if isinstance(left, dict) and isinstance(right, dict)
        and left.get("role") == right.get("role") == "assistant"
    )
    if adjacent:
        raise ValueError(f"record still contains {adjacent} consecutive assistant boundary/boundaries")
    return migrated, {
        "id": migrated.get("id"),
        "merged_terminal_assistant_turn": merged,
        **summary_stats,
        "immutable_sha256": after_digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate canonical ReAct terminal turns without rerunning LLM clean")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    rows = read_jsonl(Path(args.input))
    migrated = []
    audit = []
    for row in rows:
        updated, item = migrate_record(row)
        migrated.append(updated)
        audit.append(item)
    write_jsonl(Path(args.output), migrated)
    write_jsonl(Path(args.audit), audit)
    report = {
        "ok": True,
        "input": args.input,
        "output": args.output,
        "rows": len(rows),
        "merged_terminal_assistant_turns": sum(int(item["merged_terminal_assistant_turn"]) for item in audit),
        "summary_paragraphs_removed_against_thought": sum(item["summary_paragraphs_removed_against_thought"] for item in audit),
        "summary_duplicate_paragraphs_removed": sum(item["summary_duplicate_paragraphs_removed"] for item in audit),
    }
    write_json(Path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
