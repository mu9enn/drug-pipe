from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.protocol.action_parser import parse_action
from drug_agent.protocol.action_schema import ACTION_FINAL_ANSWER, ACTION_TOOL_CALL
from drug_agent.protocol.parse_policy import extract_json_object_candidate, parse_action_with_policy
from drug_agent.protocol.react_protocol import (
    PROTOCOL_ACTION_JSON,
    PROTOCOL_AUTO,
    PROTOCOL_REACT_JSON,
    detect_sft_protocol,
    parse_react_sequence,
)


def _is_tool_response_user_text(text: str) -> bool:
    s = text.strip()
    return s.startswith("<tool_response>") and s.endswith("</tool_response>")


def _preview_messages(messages: Any, limit: int = 6) -> Any:
    if not isinstance(messages, list):
        return messages
    return messages[:limit]


def _roles(messages: Any) -> list[str]:
    if not isinstance(messages, list):
        return []
    out: list[str] = []
    for item in messages:
        if isinstance(item, dict) and isinstance(item.get("role"), str):
            out.append(item["role"])
    return out


def validate_structure(messages: Any) -> list[str]:
    reasons: list[str] = []

    if not isinstance(messages, list) or not messages:
        return ["messages_empty_or_not_list"]

    roles = []
    for msg in messages:
        if not isinstance(msg, dict):
            reasons.append("message_not_object")
            continue
        role = msg.get("role")
        content = msg.get("content")
        roles.append(role)

        if role not in {"system", "user", "assistant"}:
            reasons.append("unsupported_role")
        if not isinstance(content, str):
            reasons.append("content_not_string")
        elif not content.strip():
            reasons.append("content_empty")

    if roles and roles[0] != "system" and roles[0] != "user":
        reasons.append("bad_first_role")

    first_non_system = next((r for r in roles if r != "system"), None)
    if first_non_system is not None and first_non_system != "user":
        reasons.append("first_non_system_not_user")

    has_user = any(
        isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str) and m.get("content").strip()
        for m in messages
    )
    has_assistant = any(
        isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str) and m.get("content").strip()
        for m in messages
    )

    if not has_user:
        reasons.append("no_nonempty_user_turn")
    if not has_assistant:
        reasons.append("no_nonempty_assistant_turn")

    users = [
        m
        for m in messages
        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str) and m.get("content").strip()
    ]
    if users and not any(not _is_tool_response_user_text(m["content"]) for m in users):
        reasons.append("no_user_query_only_tool_response_users")

    return sorted(set(reasons))


def audit_assistant_actions(messages: Any) -> tuple[Counter, list[dict[str, Any]]]:
    counts = Counter()
    samples: list[dict[str, Any]] = []
    if not isinstance(messages, list):
        return counts, samples

    for turn_idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            continue

        counts["assistant_total"] += 1
        if "```" in content:
            counts["assistant_contains_code_fence"] += 1
        if "<think>" in content or "</think>" in content:
            counts["assistant_contains_think"] += 1

        strict = parse_action(content)
        if not strict.ok:
            counts["assistant_strict_action_failed"] += 1
        elif strict.action_type == ACTION_TOOL_CALL:
            counts["assistant_tool_call_total"] += 1
        elif strict.action_type == ACTION_FINAL_ANSWER:
            counts["assistant_final_answer_total"] += 1

        permissive, parse_recovery, normalized, parse_source = parse_action_with_policy(
            content, parse_recovery_enabled=True
        )
        recovered = isinstance(parse_recovery, dict) and parse_recovery.get("recovered") is True
        if recovered:
            counts["assistant_uses_recovery"] += 1

        has_embedded_json = False
        candidate = extract_json_object_candidate(content)
        if candidate and candidate.strip() != content.strip():
            has_embedded_json = True
            counts["assistant_contains_prose_plus_json"] += 1

        if (not strict.ok) or recovered or has_embedded_json:
            samples.append(
                {
                    "turn_index": turn_idx,
                    "strict_ok": strict.ok,
                    "strict_error_type": strict.error_type,
                    "strict_error_message": strict.error_message,
                    "permissive_ok": permissive.ok,
                    "parse_source": parse_source,
                    "parse_recovery": parse_recovery,
                    "has_embedded_json": has_embedded_json,
                    "normalized_preview": normalized[:300],
                    "content_preview": content[:300],
                }
            )
    return counts, samples


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return int(float(text))
        except Exception:
            return 0
    return 0


def _collect_count_aliases(value: Any, alias_map: dict[str, str]) -> Counter:
    counts = Counter()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, item in obj.items():
                if key in alias_map:
                    counts[alias_map[key]] += _coerce_int(item)
                walk(item)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(value)
    return counts


def _load_cleaning_report(metadata: dict[str, Any]) -> Any:
    report = metadata.get("cleaning_report")
    if isinstance(report, dict):
        return report

    report = metadata.get("cleaning_report_summary")
    if isinstance(report, dict):
        return report

    report_path = metadata.get("cleaning_report_path")
    if isinstance(report_path, str) and report_path.strip():
        path = Path(report_path)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {"cleaning_report_path": report_path}

    return {}


def _merge_record_counts(*counts_dicts: Counter) -> Counter:
    merged = Counter()
    for counts in counts_dicts:
        for key, value in counts.items():
            merged[key] = max(int(merged.get(key) or 0), int(value))
    return merged


def audit_react_actions(messages: Any) -> tuple[Counter, list[dict[str, Any]]]:
    counts = Counter()
    samples: list[dict[str, Any]] = []
    if not isinstance(messages, list):
        return counts, samples

    pending_tool_calls: list[str] = []
    seen_plain_user_prompt = False
    seen_react_turn = False
    seen_final_answer = False

    for turn_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            continue

        if role == "system":
            continue

        if seen_final_answer:
            counts["react_json_parse_failed"] += 1
            samples.append(
                {
                    "turn_index": turn_idx,
                    "role": role,
                    "strict_ok": False,
                    "strict_error_type": "ReactSequenceError",
                    "strict_error_message": "content found after final_answer",
                    "parse_source": "sequence",
                    "content_preview": content[:300],
                }
            )
            continue

        parsed = parse_react_sequence(content, role=role if isinstance(role, str) else None)
        if not bool(parsed.get("ok")):
            counts["react_json_parse_failed"] += 1
            samples.append(
                {
                    "turn_index": turn_idx,
                    "role": role,
                    "strict_ok": False,
                    "strict_error_type": parsed.get("error_type"),
                    "strict_error_message": parsed.get("error_message"),
                    "parse_source": parsed.get("mode") or "react",
                    "content_preview": content[:300],
                }
            )
            continue

        counts["fence_wrappers_stripped"] += int(parsed.get("fence_wrappers_stripped") or 0)
        counts["fence_inner_content_preserved"] += int(parsed.get("fence_inner_content_preserved") or 0)

        blocks = parsed.get("blocks")
        if not isinstance(blocks, list):
            continue

        if role == "assistant":
            counts["assistant_total"] += 1
            saw_block_kind = False
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                kind = block.get("kind")
                saw_block_kind = True
                if kind == "thought":
                    counts["assistant_thought_total"] += 1
                elif kind == "tool_call":
                    counts["assistant_tool_call_total"] += 1
                    counts["retained_mcp_tool_calls"] += 1
                    tool_name = block.get("tool_name")
                    if isinstance(tool_name, str):
                        pending_tool_calls.append(tool_name)
                    else:
                        pending_tool_calls.append("")
                    seen_react_turn = True
                elif kind == "final_answer":
                    counts["assistant_final_answer_total"] += 1
                    seen_final_answer = True
                    seen_react_turn = True
                else:
                    counts["react_json_parse_failed"] += 1
                    samples.append(
                        {
                            "turn_index": turn_idx,
                            "role": role,
                            "strict_ok": False,
                            "strict_error_type": "ReactSchemaError",
                            "strict_error_message": f"unsupported assistant block: {kind}",
                            "parse_source": "react",
                            "content_preview": content[:300],
                        }
                    )
            if not saw_block_kind:
                counts["react_json_parse_failed"] += 1
                samples.append(
                    {
                        "turn_index": turn_idx,
                        "role": role,
                        "strict_ok": False,
                        "strict_error_type": "ReactFormatError",
                        "strict_error_message": "assistant message did not contain any supported ReAct blocks",
                        "parse_source": "react",
                        "content_preview": content[:300],
                    }
                )
            continue

        if role == "user":
            if parsed.get("mode") == "plain_user_prompt":
                counts["user_prompt_total"] += 1
                if seen_react_turn or seen_plain_user_prompt:
                    counts["react_json_parse_failed"] += 1
                    samples.append(
                        {
                            "turn_index": turn_idx,
                            "role": role,
                            "strict_ok": False,
                            "strict_error_type": "ReactSequenceError",
                            "strict_error_message": "plain user prompt is only allowed for the initial seed turn",
                            "parse_source": "react",
                            "content_preview": content[:300],
                        }
                    )
                else:
                    seen_plain_user_prompt = True
                continue

            observation_blocks = [block for block in blocks if isinstance(block, dict) and block.get("kind") == "observation"]
            if not observation_blocks:
                counts["react_json_parse_failed"] += 1
                samples.append(
                    {
                        "turn_index": turn_idx,
                        "role": role,
                        "strict_ok": False,
                        "strict_error_type": "ReactSchemaError",
                        "strict_error_message": "user message must be plain prompt or observation blocks",
                        "parse_source": "react",
                        "content_preview": content[:300],
                    }
                )
                continue

            counts["user_observation_total"] += len(observation_blocks)
            for block in observation_blocks:
                tool_name = block.get("tool_name")
                if pending_tool_calls:
                    expected_tool = pending_tool_calls.pop(0)
                    if isinstance(tool_name, str) and expected_tool and tool_name != expected_tool:
                        counts["react_json_parse_failed"] += 1
                        samples.append(
                            {
                                "turn_index": turn_idx,
                                "role": role,
                                "strict_ok": False,
                                "strict_error_type": "ReactSequenceError",
                                "strict_error_message": f"observation tool_name mismatch: expected {expected_tool}, got {tool_name}",
                                "parse_source": "react",
                                "content_preview": content[:300],
                            }
                        )
                else:
                    counts["orphan_tool_results"] += 1
                    counts["react_json_parse_failed"] += 1
                    samples.append(
                        {
                            "turn_index": turn_idx,
                            "role": role,
                            "strict_ok": False,
                            "strict_error_type": "ReactSequenceError",
                            "strict_error_message": "observation encountered without a pending tool_call",
                            "parse_source": "react",
                            "content_preview": content[:300],
                        }
                    )
                seen_react_turn = True
            continue

        counts["react_json_parse_failed"] += 1
        samples.append(
            {
                "turn_index": turn_idx,
                "role": role,
                "strict_ok": False,
                "strict_error_type": "ReactSchemaError",
                "strict_error_message": f"unsupported role: {role}",
                "parse_source": "react",
                "content_preview": content[:300],
            }
        )

    if pending_tool_calls:
        counts["orphan_tool_calls"] += len(pending_tool_calls)
        counts["react_json_parse_failed"] += len(pending_tool_calls)
        samples.append(
            {
                "turn_index": len(messages),
                "role": "assistant",
                "strict_ok": False,
                "strict_error_type": "ReactSequenceError",
                "strict_error_message": f"{len(pending_tool_calls)} tool_call(s) were not followed by observation blocks",
                "parse_source": "react",
                "content_preview": None,
            }
        )

    return counts, samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SFT messages for slime/Qwen chat template compatibility")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument(
        "--protocol",
        type=str,
        default=PROTOCOL_AUTO,
        choices=[PROTOCOL_AUTO, PROTOCOL_ACTION_JSON, PROTOCOL_REACT_JSON],
        help="SFT message protocol to validate. auto detects per record.",
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional HF tokenizer path for apply_chat_template validation")
    parser.add_argument("--preview", type=int, default=20)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            records.append((i, obj))

    bad = []
    reason_counter = Counter()
    assistant_counter = Counter()
    assistant_issue_samples: list[dict[str, Any]] = []
    protocol_counter = Counter()
    sft_counts = Counter()
    cleaning_counts = Counter()
    legacy_protocol_used = False
    react_protocol_used = False

    cleaning_alias_map = {
        "retained_mcp_tool_count": "retained_mcp_tool_calls",
        "retained_mcp_tool_calls": "retained_mcp_tool_calls",
        "dropped_non_mcp_tool_count": "dropped_non_mcp_tool_calls",
        "dropped_non_mcp_tool_calls": "dropped_non_mcp_tool_calls",
        "orphan_tool_results": "orphan_tool_results",
        "orphan_tool_calls": "orphan_tool_calls",
        "fence_wrappers_stripped": "fence_wrappers_stripped",
        "fence_inner_content_preserved": "fence_inner_content_preserved",
        "react_json_parse_failed": "react_json_parse_failed",
    }

    for i, obj in records:
        messages = obj.get("messages")
        metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
        record_protocol = args.protocol
        if record_protocol == PROTOCOL_AUTO:
            record_protocol = detect_sft_protocol(obj)

        protocol_counter[record_protocol] += 1
        if record_protocol == PROTOCOL_REACT_JSON:
            react_protocol_used = True
            assistant_counts, assistant_issues = audit_react_actions(messages)
        else:
            legacy_protocol_used = True
            assistant_counts, assistant_issues = audit_assistant_actions(messages)

        if record_protocol == PROTOCOL_ACTION_JSON:
            reasons = validate_structure(messages)
            for r in reasons:
                reason_counter[r] += 1
        else:
            reasons = []
            if not isinstance(messages, list) or not messages:
                reasons.append("messages_empty_or_not_list")
            else:
                roles = []
                for msg in messages:
                    if not isinstance(msg, dict):
                        reasons.append("message_not_object")
                        continue
                    role = msg.get("role")
                    content = msg.get("content")
                    roles.append(role)
                    if role not in {"system", "user", "assistant"}:
                        reasons.append("unsupported_role")
                    if not isinstance(content, str):
                        reasons.append("content_not_string")
                    elif not content.strip():
                        reasons.append("content_empty")

                if roles and roles[0] != "system" and roles[0] != "user":
                    reasons.append("bad_first_role")
                first_non_system = next((r for r in roles if r != "system"), None)
                if first_non_system is not None and first_non_system != "user":
                    reasons.append("first_non_system_not_user")

                has_user = any(
                    isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str) and m.get("content").strip()
                    for m in messages
                )
                has_assistant = any(
                    isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str) and m.get("content").strip()
                    for m in messages
                )
                if not has_user:
                    reasons.append("no_nonempty_user_turn")
                if not has_assistant:
                    reasons.append("no_nonempty_assistant_turn")

        assistant_counter.update(assistant_counts)

        if record_protocol == PROTOCOL_REACT_JSON:
            report = _load_cleaning_report(metadata)
            report_counts = _collect_count_aliases(report, cleaning_alias_map)
            record_counts = _merge_record_counts(assistant_counts, report_counts)
            cleaning_counts.update(report_counts)
            for key, value in record_counts.items():
                sft_counts[key] += int(value)
        else:
            for key, value in assistant_counts.items():
                sft_counts[key] += int(value)

        if assistant_issues:
            for issue in assistant_issues:
                if len(assistant_issue_samples) >= args.preview:
                    break
                assistant_issue_samples.append(
                    {
                        "line": i,
                        "index": i,
                        "task_id": metadata.get("task_id"),
                        "roles": _roles(messages),
                        "issue": issue,
                    }
                )

        if record_protocol == PROTOCOL_ACTION_JSON:
            strict_failed = int(assistant_counts.get("assistant_strict_action_failed") or 0) > 0
            think_found = int(assistant_counts.get("assistant_contains_think") or 0) > 0
            code_fence_found = int(assistant_counts.get("assistant_contains_code_fence") or 0) > 0
            prose_plus_json = int(assistant_counts.get("assistant_contains_prose_plus_json") or 0) > 0
            uses_recovery = int(assistant_counts.get("assistant_uses_recovery") or 0) > 0

            if reasons or strict_failed or think_found or code_fence_found or prose_plus_json or uses_recovery:
                bad_reasons = list(reasons)
                if strict_failed:
                    bad_reasons.append("assistant_strict_action_failed")
                if think_found:
                    bad_reasons.append("assistant_contains_think")
                if code_fence_found:
                    bad_reasons.append("assistant_contains_code_fence")
                if prose_plus_json:
                    bad_reasons.append("assistant_contains_prose_plus_json")
                if uses_recovery:
                    bad_reasons.append("assistant_uses_recovery")

                bad.append(
                    {
                        "line": i,
                        "index": i,
                        "task_id": metadata.get("task_id"),
                        "roles": _roles(messages),
                        "reasons": sorted(set(bad_reasons)),
                        "preview": _preview_messages(messages),
                    }
                )
        else:
            report_counts = _collect_count_aliases(_load_cleaning_report(metadata), cleaning_alias_map)
            merged_counts = _merge_record_counts(assistant_counts, report_counts)
            react_reasons = list(reasons)
            parse_failed = int(merged_counts.get("react_json_parse_failed") or 0) > 0
            orphan_tool_results = int(merged_counts.get("orphan_tool_results") or 0) > 0
            if parse_failed or orphan_tool_results:
                if parse_failed:
                    react_reasons.append("react_json_parse_failed")
                if orphan_tool_results:
                    react_reasons.append("orphan_tool_results")
                for reason in react_reasons:
                    reason_counter[reason] += 1
                bad.append(
                    {
                        "line": i,
                        "index": i,
                        "task_id": metadata.get("task_id"),
                        "roles": _roles(messages),
                        "reasons": sorted(set(react_reasons)),
                        "preview": _preview_messages(messages),
                    }
                )

    apply_template_failed = []
    chat_template_import_failed: dict[str, Any] | None = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
            for i, obj in records:
                messages = obj.get("messages")
                try:
                    tok.apply_chat_template(messages, tokenize=False, return_dict=False)
                except Exception as exc:
                    apply_template_failed.append(
                        {
                            "line": i,
                            "index": i,
                            "task_id": (obj.get("metadata") or {}).get("task_id")
                            if isinstance(obj.get("metadata"), dict)
                            else None,
                            "roles": _roles(messages),
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "preview": _preview_messages(messages),
                        }
                    )
        except Exception as exc:
            chat_template_import_failed = {
                "line": -1,
                "index": -1,
                "task_id": None,
                "roles": [],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "preview": None,
            }

    protocol_counts = dict(protocol_counter)
    detected_protocol = PROTOCOL_AUTO
    if len(protocol_counter) == 1:
        detected_protocol = next(iter(protocol_counter))
    elif protocol_counter:
        detected_protocol = "mixed"

    assistant_total = int(assistant_counter.get("assistant_total") or 0)
    assistant_tool_call_total = int(assistant_counter.get("assistant_tool_call_total") or 0)
    assistant_final_answer_total = int(assistant_counter.get("assistant_final_answer_total") or 0)
    assistant_thought_total = int(assistant_counter.get("assistant_thought_total") or 0)
    user_observation_total = int(assistant_counter.get("user_observation_total") or 0)
    retained_mcp_tool_calls = assistant_tool_call_total
    dropped_non_mcp_tool_calls = int(cleaning_counts.get("dropped_non_mcp_tool_calls") or 0)
    orphan_tool_results = max(
        int(assistant_counter.get("orphan_tool_results") or 0),
        int(cleaning_counts.get("orphan_tool_results") or 0),
    )
    orphan_tool_calls = max(
        int(assistant_counter.get("orphan_tool_calls") or 0),
        int(cleaning_counts.get("orphan_tool_calls") or 0),
    )
    fence_wrappers_stripped = max(
        int(assistant_counter.get("fence_wrappers_stripped") or 0),
        int(cleaning_counts.get("fence_wrappers_stripped") or 0),
    )
    fence_inner_content_preserved = max(
        int(assistant_counter.get("fence_inner_content_preserved") or 0),
        int(cleaning_counts.get("fence_inner_content_preserved") or 0),
    )
    react_json_parse_failed = max(
        int(assistant_counter.get("react_json_parse_failed") or 0),
        int(cleaning_counts.get("react_json_parse_failed") or 0),
    )
    chat_template_failed = len(apply_template_failed)
    chat_template_checked = bool(args.tokenizer) and chat_template_import_failed is None

    summary = {
        "ok": len(bad) == 0 and chat_template_failed == 0,
        "input": str(input_path),
        "protocol_mode": args.protocol,
        "protocol_counts": protocol_counts,
        "detected_protocol": detected_protocol,
        "deprecated_legacy_protocol": bool(legacy_protocol_used and not react_protocol_used),
        "total_sessions": len(records),
        "total_sft_samples": len(records),
        "assistant_total": assistant_total,
        "assistant_thought_total": assistant_thought_total,
        "assistant_tool_call_total": assistant_tool_call_total,
        "assistant_final_answer_total": assistant_final_answer_total,
        "user_observation_total": user_observation_total,
        "retained_mcp_tool_calls": retained_mcp_tool_calls,
        "dropped_non_mcp_tool_calls": dropped_non_mcp_tool_calls,
        "orphan_tool_results": orphan_tool_results,
        "orphan_tool_calls": orphan_tool_calls,
        "fence_wrappers_stripped": fence_wrappers_stripped,
        "fence_inner_content_preserved": fence_inner_content_preserved,
        "react_json_parse_failed": react_json_parse_failed,
        "chat_template_failed": chat_template_failed,
        "chat_template_checked": chat_template_checked,
        "bad": len(bad),
        "bad_reason_counts": dict(reason_counter),
    }

    bad_preview: list[dict[str, Any]] = []
    for item in bad[: args.preview]:
        bad_preview.append(
            {
                "line": item.get("line"),
                "index": item.get("index"),
                "task_id": item.get("task_id"),
                "roles": item.get("roles"),
                "reason": item.get("reasons"),
                "preview": item.get("preview"),
            }
        )

    apply_failed_preview: list[dict[str, Any]] = []
    for item in apply_template_failed[: args.preview]:
        apply_failed_preview.append(
            {
                "line": item.get("line"),
                "index": item.get("index"),
                "task_id": item.get("task_id"),
                "roles": item.get("roles"),
                "reason": f"{item.get('error_type')}: {item.get('error_message')}",
                "preview": item.get("preview"),
            }
        )

    if bad_preview:
        summary["bad_samples_preview"] = bad_preview
    if apply_failed_preview:
        summary["apply_chat_template_failed_preview"] = apply_failed_preview
    if chat_template_import_failed is not None:
        summary["chat_template_import_failed"] = chat_template_import_failed
    if assistant_issue_samples:
        summary["assistant_issue_samples_preview"] = assistant_issue_samples
    if len(protocol_counter) > 1:
        summary["protocol_warning"] = "mixed_protocols_detected"

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
