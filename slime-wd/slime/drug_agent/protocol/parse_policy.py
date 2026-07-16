from __future__ import annotations

import json
import os
from typing import Any

from drug_agent.protocol.action_parser import ParseResult, parse_action

ROLLOUT_MODE_TRAIN_STRICT = "train_strict"
ROLLOUT_MODE_DEBUG_PERMISSIVE = "debug_permissive"
SUPPORTED_ROLLOUT_MODES = {
    ROLLOUT_MODE_TRAIN_STRICT,
    ROLLOUT_MODE_DEBUG_PERMISSIVE,
}


def bool_from_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_rollout_controls(
    rollout_mode: str | None = None,
    allow_parse_recovery_override: bool | None = None,
) -> dict[str, Any]:
    requested_mode = rollout_mode if isinstance(rollout_mode, str) else os.environ.get(
        "DRUG_AGENT_ROLLOUT_MODE", ROLLOUT_MODE_TRAIN_STRICT
    )
    mode = requested_mode.strip().lower() if isinstance(requested_mode, str) else ROLLOUT_MODE_TRAIN_STRICT
    if mode not in SUPPORTED_ROLLOUT_MODES:
        mode = ROLLOUT_MODE_TRAIN_STRICT

    if allow_parse_recovery_override is None:
        allow_parse_recovery_override = bool_from_env("DRUG_AGENT_ALLOW_PARSE_RECOVERY", default=False)

    parse_recovery_enabled = mode == ROLLOUT_MODE_DEBUG_PERMISSIVE or bool(allow_parse_recovery_override)
    return {
        "rollout_mode": mode,
        "allow_parse_recovery_override": bool(allow_parse_recovery_override),
        "parse_recovery_enabled": bool(parse_recovery_enabled),
    }


def extract_json_object_candidate(text: str) -> str | None:
    if not isinstance(text, str):
        return None

    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, str]] = []
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        snippet = text[start:]
        try:
            obj, end = decoder.raw_decode(snippet)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        candidate = snippet[:end]
        tail = snippet[end:].strip()
        score = 0
        if "type" in obj:
            score += 10
        if tail == "":
            score += 3
        if isinstance(obj.get("type"), str):
            score += 2

        candidates.append((score, -start, candidate))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def parse_action_with_policy(
    raw_response: str,
    *,
    parse_recovery_enabled: bool,
) -> tuple[ParseResult, dict[str, Any] | None, str, str]:
    parsed = parse_action(raw_response)
    if parsed.ok:
        return parsed, None, raw_response, "strict"

    if parse_recovery_enabled:
        candidate = extract_json_object_candidate(raw_response)
        if candidate and candidate.strip() != raw_response.strip():
            repaired = parse_action(candidate)
            if repaired.ok:
                return repaired, {"recovered": True, "strategy": "extract_embedded_json_object"}, candidate, "recovered"

    return parsed, None, raw_response, "strict"

