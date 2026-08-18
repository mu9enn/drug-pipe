"""Deterministic token budgeting for live ReAct evaluation histories."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any


SCHEMA_VERSION = "react_live_context_budget_v1"
SUMMARY_HEADER = "[Compressed earlier ReAct history; deterministic and grounded in prior turns.]"


class ContextBudgetError(ValueError):
    """Raised when a live ReAct history cannot fit the configured prompt budget."""


def bounded_step_limit(sample_limit: int, runtime_limit: int) -> int:
    """Return the strictest active positive limit; zero means no limit."""
    if sample_limit < 0 or runtime_limit < 0:
        raise ValueError("step limits must be non-negative (0 means unlimited)")
    active = [value for value in (sample_limit, runtime_limit) if value > 0]
    return min(active) if active else 0


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def compact_value(value: Any, *, string_limit: int = 512, depth: int = 0) -> Any:
    """Retain identity/state scalars while replacing bulky leaves with descriptors."""
    if depth >= 5:
        text = _canonical_json(value)
        return {"type": type(value).__name__, "chars": len(text), "sha256": _sha256_text(text)}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= string_limit:
            return value
        return {
            "type": "string",
            "chars": len(value),
            "sha256": _sha256_text(value),
            "head": value[:160],
            "tail": value[-160:],
        }
    if isinstance(value, list):
        if len(value) <= 12:
            return [compact_value(item, string_limit=string_limit, depth=depth + 1) for item in value]
        text = _canonical_json(value)
        return {
            "type": "list",
            "items": len(value),
            "sha256": _sha256_text(text),
            "head": [compact_value(item, string_limit=160, depth=depth + 1) for item in value[:4]],
            "tail": [compact_value(item, string_limit=160, depth=depth + 1) for item in value[-4:]],
        }
    if isinstance(value, dict):
        preferred = (
            "ok", "status", "error", "error_message", "message", "tool_name", "path", "file_path",
            "input_path", "output_path", "output_file", "artifact", "artifact_id", "id", "content", "metadata",
        )
        keys = list(value)
        ordered = [key for key in preferred if key in value]
        ordered.extend(sorted((key for key in keys if key not in ordered), key=str)[: max(0, 32 - len(ordered))])
        out = {
            str(key): compact_value(value[key], string_limit=string_limit, depth=depth + 1)
            for key in ordered[:32]
        }
        if len(keys) > len(out):
            text = _canonical_json(value)
            out["_omitted_keys"] = len(keys) - len(out)
            out["_sha256"] = _sha256_text(text)
        return out
    return compact_value(str(value), string_limit=string_limit, depth=depth + 1)


def token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def compact_observation_text(payloads: list[dict[str, Any]]) -> str:
    blocks = []
    for payload in payloads:
        tool_name = str(payload.get("tool_name") or "runtime")
        blocks.append(
            f'<observation tool_name={json.dumps(tool_name, ensure_ascii=False)}>'
            f'{_canonical_json(compact_value(payload))}</observation>'
        )
    return "\n" + "\n".join(blocks) + "\n"


def make_event(step: int, parsed: dict[str, Any], observations: list[dict[str, Any]], raw_response: str) -> dict[str, Any]:
    event: dict[str, Any] = {
        "step": step,
        "decision_type": parsed.get("decision_type") if parsed.get("ok") else "invalid",
        "tool_calls": [],
        "observations": [],
    }
    if parsed.get("ok"):
        for call in parsed.get("tool_calls") or []:
            event["tool_calls"].append(
                {
                    "tool_name": str(call.get("tool_name") or ""),
                    "arguments": compact_value(call.get("arguments") or {}),
                }
            )
    else:
        event["invalid_response"] = compact_value(raw_response, string_limit=256)
    for observation in observations:
        event["observations"].append(
            {
                "tool_name": str(observation.get("tool_name") or "runtime"),
                "status": observation.get("status") or "unknown",
                "content": compact_value(observation.get("content") or {}),
                "metadata": compact_value(observation.get("metadata") or {}),
            }
        )
    return event


def make_turn(
    tokenizer: Any,
    *,
    step: int,
    assistant_ids: list[int],
    observation_ids: list[int],
    parsed: dict[str, Any],
    observations: list[dict[str, Any]],
    raw_response: str,
) -> dict[str, Any]:
    return {
        "step": step,
        "assistant_ids": list(assistant_ids),
        "observation_ids": list(observation_ids),
        "compacted_observation_ids": token_ids(tokenizer, compact_observation_text(observations)),
        "event": make_event(step, parsed, observations, raw_response),
    }


def _summary_ids(tokenizer: Any, turns: list[dict[str, Any]], max_tokens: int) -> tuple[list[int], dict[str, Any]]:
    events = [turn["event"] for turn in turns]
    collapsed = 0
    counts: Counter[str] = Counter()
    while True:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "collapsed_older_turns": collapsed,
            "collapsed_tool_counts": dict(sorted(counts.items())),
            "recent_events": events[collapsed:],
        }
        text = (
            f'<observation tool_name="runtime">{SUMMARY_HEADER}\n'
            f'{_canonical_json(payload)}</observation>\n'
        )
        ids = token_ids(tokenizer, text)
        if len(ids) <= max_tokens or collapsed >= len(events):
            return ids, {
                "summary_tokens": len(ids),
                "summary_sha256": _sha256_text(text),
                "collapsed_older_turns": collapsed,
            }
        for call in events[collapsed].get("tool_calls") or []:
            name = str(call.get("tool_name") or "")
            if name:
                counts[name] += 1
        collapsed += 1


def fit_context(
    tokenizer: Any,
    *,
    prefix_ids: list[int],
    turns: list[dict[str, Any]],
    max_prompt_tokens: int,
    summary_max_tokens: int = 32768,
    keep_recent_turns: int = 4,
) -> tuple[list[int], dict[str, Any]]:
    """Fit a live prompt without cutting the task prefix or a token mid-turn."""
    if min(max_prompt_tokens, summary_max_tokens) < 1:
        raise ValueError("context budgets must be positive")
    if len(prefix_ids) > max_prompt_tokens:
        raise ContextBudgetError(
            f"immutable task prefix exceeds prompt budget: {len(prefix_ids)} > {max_prompt_tokens}"
        )
    original = list(prefix_ids)
    for turn in turns:
        original.extend(turn["assistant_ids"])
        original.extend(turn["observation_ids"])
    if len(original) <= max_prompt_tokens:
        return original, {
            "schema_version": SCHEMA_VERSION,
            "compacted": False,
            "original_tokens": len(original),
            "output_tokens": len(original),
        }

    protected_start = max(0, len(turns) - keep_recent_turns)
    microcompact = list(prefix_ids)
    microcompacted_turns = 0
    for index, turn in enumerate(turns):
        microcompact.extend(turn["assistant_ids"])
        if index < protected_start and len(turn["compacted_observation_ids"]) < len(turn["observation_ids"]):
            microcompact.extend(turn["compacted_observation_ids"])
            microcompacted_turns += 1
        else:
            microcompact.extend(turn["observation_ids"])
    if len(microcompact) <= max_prompt_tokens:
        return microcompact, {
            "schema_version": SCHEMA_VERSION,
            "compacted": True,
            "strategy": "typed_observation_compaction",
            "original_tokens": len(original),
            "output_tokens": len(microcompact),
            "microcompacted_turns": microcompacted_turns,
            "summarized_turns": 0,
        }

    summary_budget = min(summary_max_tokens, max_prompt_tokens - len(prefix_ids))
    for suffix_start in range(1, len(turns) + 1):
        summary, summary_audit = _summary_ids(tokenizer, turns[:suffix_start], summary_budget)
        candidate = list(prefix_ids) + summary
        for index in range(suffix_start, len(turns)):
            turn = turns[index]
            candidate.extend(turn["assistant_ids"])
            if index < protected_start and len(turn["compacted_observation_ids"]) < len(turn["observation_ids"]):
                candidate.extend(turn["compacted_observation_ids"])
            else:
                candidate.extend(turn["observation_ids"])
        if len(candidate) <= max_prompt_tokens:
            return candidate, {
                "schema_version": SCHEMA_VERSION,
                "compacted": True,
                "strategy": "typed_observation_then_structured_summary",
                "original_tokens": len(original),
                "output_tokens": len(candidate),
                "microcompacted_turns": microcompacted_turns,
                "summarized_turns": suffix_start,
                "preserved_recent_turns": len(turns) - suffix_start,
                **summary_audit,
            }
    raise ContextBudgetError("unable to fit live ReAct history into prompt budget")
