from __future__ import annotations

import json
from pathlib import Path
from typing import Any


POLICY_PATH = Path(__file__).with_name("molclaw_tool_concurrency_v1.json")


def load_tool_limits(path: Path = POLICY_PATH) -> dict[str, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "molclaw_tool_concurrency_v1":
        raise ValueError(f"unsupported MolClaw tool concurrency policy: {path}")
    grouped = raw.get("limits")
    if not isinstance(grouped, dict) or set(grouped) != {"4", "30"}:
        raise ValueError("MolClaw tool concurrency policy must define limits 4 and 30")
    limits: dict[str, int] = {}
    for limit_text, names in grouped.items():
        if not isinstance(names, list):
            raise ValueError(f"MolClaw limit {limit_text} tool list must be an array")
        limit = int(limit_text)
        for value in names:
            name = str(value or "").strip()
            if not name:
                raise ValueError(f"MolClaw limit {limit} contains an empty tool name")
            if name in limits:
                raise ValueError(f"duplicate MolClaw concurrency entry: {name}")
            limits[name] = limit
    if len(limits) != 81:
        raise ValueError(f"expected 81 MolClaw concurrency entries, found {len(limits)}")
    return limits


def expected_tools_from_task_spec(task_spec: dict[str, Any]) -> tuple[str, ...]:
    toolchain = task_spec.get("toolchain")
    tools = toolchain.get("tools") if isinstance(toolchain, dict) else None
    if not isinstance(tools, list) or not tools:
        expected = task_spec.get("expected_trajectory")
        execution_plan = expected.get("execution_plan") if isinstance(expected, dict) else None
        tools = execution_plan.get("tool_order") if isinstance(execution_plan, dict) else None
    if not isinstance(tools, list):
        return ()
    return tuple(dict.fromkeys(str(tool).strip() for tool in tools if str(tool).strip()))


def task_spec_from_raw_question_json(raw_question_json: str) -> dict[str, Any]:
    try:
        parsed = json.loads((raw_question_json or "").strip())
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def serial_tool_claims(
    expected_tools: tuple[str, ...],
    limits: dict[str, int],
) -> frozenset[str]:
    unknown = sorted(set(expected_tools) - set(limits))
    if unknown:
        raise ValueError(f"unregistered MolClaw tools in expected toolchain: {unknown}")
    return frozenset(tool for tool in expected_tools if limits[tool] == 4)


def first_admissible_index(
    pending_claims: list[frozenset[str]],
    active_claims: list[frozenset[str]],
) -> int | None:
    occupied = set().union(*active_claims) if active_claims else set()
    for index, claims in enumerate(pending_claims):
        if claims.isdisjoint(occupied):
            return index
    return None
