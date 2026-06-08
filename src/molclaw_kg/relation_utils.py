from __future__ import annotations

from typing import Any


VALID_RELATION_STATUSES = {"valid", "negative", "uncertain", "alternative"}


def normalize_relation_status(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in VALID_RELATION_STATUSES:
        return raw
    if raw in {"valid_full", "valid_partial", "positive"}:
        return "valid"
    if raw in {"invalid", "negative"}:
        return "negative"
    if raw == "alternative":
        return "alternative"
    return "uncertain"


def normalize_edge_type_name(value: Any) -> str | None:
    t = str(value or "").strip()
    if not t:
        return None
    if t == "generates_input_for":
        return "generates_partial_input_for"
    if t == "requires_intermediate":
        return None
    return t


def normalize_edge_types(edge_types: Any) -> list[dict[str, Any]]:
    items = edge_types if isinstance(edge_types, list) else []
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        et = normalize_edge_type_name(it.get("type"))
        if not et:
            continue
        fixed = dict(it)
        fixed["type"] = et
        fixed.setdefault("source_slot", None)
        fixed.setdefault("target_slot_or_precondition", None)
        fixed.setdefault("confidence", 0.5)
        fixed.setdefault("evidence_ids", [])
        out.append(fixed)
    return out


def context_from_legacy_fields(row: dict[str, Any]) -> str:
    context = str(row.get("context") or "").strip()
    if context:
        return context
    reason = str(row.get("negative_reason") or "").strip()
    if reason:
        return reason
    unsat = row.get("unsatisfied_required_inputs") or []
    if isinstance(unsat, list) and unsat:
        first = unsat[0]
        if isinstance(first, dict):
            msg = str(first.get("reason") or "").strip()
            if msg:
                return msg
    rationale = str(row.get("rationale") or "").strip()
    if rationale:
        return rationale
    return ""


def compact_context_with_fail_reasons(context: str, fail_messages: list[str]) -> str:
    base = (context or "").strip()
    if not fail_messages:
        return base
    detail = "; ".join(x.strip() for x in fail_messages if str(x).strip())
    if not detail:
        return base
    suffix = f"validator_fail: {detail}"
    return f"{base} | {suffix}" if base else suffix
