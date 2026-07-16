from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import re
from typing import Any
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.constants import RAW_TASK_TYPES, SFT_OUT_ROOT, SFT_OUTPUTS_ANSWER_HIT
from drug_agent.data.common import load_usage_summary_by_basename
from drug_agent.protocol.action_schema import ACTION_FINAL_ANSWER, ACTION_TOOL_CALL
from drug_agent.utils import normalize_tool_name, write_json, write_jsonl


_FENCED_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", flags=re.DOTALL)


def _strip_markdown_fence_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    out = text
    while True:
        replaced = _FENCED_BLOCK_RE.sub(lambda m: (m.group(1) or "").strip(), out)
        if replaced == out:
            break
        out = replaced
    out = out.replace("```", "")
    out = out.replace("<think>", "").replace("</think>", "")
    return out


def _sanitize_json_strings(value: Any) -> Any:
    if isinstance(value, str):
        return _strip_markdown_fence_text(value)
    if isinstance(value, list):
        return [_sanitize_json_strings(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_json_strings(v) for k, v in value.items()}
    return value


def load_allowlist(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        values = obj.get("allowed_tools")
        if isinstance(values, list):
            return {normalize_tool_name(x) for x in values if isinstance(x, str) and x.strip()}
    if isinstance(obj, list):
        return {normalize_tool_name(x) for x in obj if isinstance(x, str) and x.strip()}
    return set()


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    return str(content)


def _is_tool_response_user_text(text: str) -> bool:
    s = text.strip()
    return s.startswith("<tool_response>") and s.endswith("</tool_response>")


def _derive_seed_user_prompt(sample_id: str | None, task_type: str | None, metadata: dict[str, Any]) -> str:
    candidates = []

    for key in ["instruction", "question", "query", "prompt", "user_query", "task_prompt"]:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    task_meta = metadata.get("task")
    if isinstance(task_meta, dict):
        for key in ["instruction", "question", "query", "prompt"]:
            value = task_meta.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())

    if candidates:
        return candidates[0]

    sid = sample_id or "unknown"
    ttype = task_type or "unknown"
    return (
        "Task context missing in source messages. "
        f"Continue task_id={sid}, task_type={ttype}. "
        "Use tools if needed and provide final answer in strict JSON."
    )


def normalize_tool_call(
    payload: dict[str, Any], allowlist: set[str], allow_all: bool
) -> tuple[dict[str, Any] | None, str | None]:
    tool_name = payload.get("tool_name")
    if not isinstance(tool_name, str):
        return None, "tool_name_not_string"

    raw = tool_name.strip()
    if not raw.startswith("mcp__"):
        return None, "non_mcp_tool"

    bare = normalize_tool_name(raw)
    if not bare:
        return None, "empty_bare_tool"

    if (not allow_all) and allowlist and bare not in allowlist:
        return None, "tool_not_in_allowlist"

    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        return None, "arguments_not_object"

    normalized = {
        "type": ACTION_TOOL_CALL,
        "tool_name": bare,
        "arguments": arguments,
    }
    return normalized, None


def normalize_final_answer(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    answer = payload.get("answer")
    if not isinstance(answer, dict):
        return None, "final_answer_missing_object"

    # Keep action schema while removing markdown-think artifacts inside nested strings.
    answer = _sanitize_json_strings(answer)

    if "summary" not in answer or not isinstance(answer.get("summary"), str):
        return None, "final_answer_summary_invalid"
    if "evidence" not in answer or not isinstance(answer.get("evidence"), list):
        return None, "final_answer_evidence_invalid"
    if "result" not in answer or not isinstance(answer.get("result"), dict):
        return None, "final_answer_result_invalid"

    if "ranked_molecules" in answer and not isinstance(answer.get("ranked_molecules"), list):
        return None, "final_answer_ranked_molecules_invalid"

    return {"type": ACTION_FINAL_ANSWER, "answer": answer}, None


def _fix_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    system_kept = False
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            if not system_kept and isinstance(content, str) and content.strip():
                out.append(msg)
                system_kept = True
            else:
                demoted = dict(msg)
                demoted["role"] = "user"
                out.append(demoted)
            continue

        out.append(msg)
    return out


def _first_non_system_index(messages: list[dict[str, Any]]) -> int | None:
    for i, msg in enumerate(messages):
        if msg.get("role") != "system":
            return i
    return None


def _ensure_valid_message_structure(
    messages: list[dict[str, Any]],
    *,
    sample_id: str | None,
    task_type: str | None,
    metadata: dict[str, Any],
    counters: Counter,
) -> list[dict[str, Any]]:
    msgs = _fix_system_messages(messages)

    first_idx = _first_non_system_index(msgs)
    if first_idx is None:
        first_idx = len(msgs)

    seed_prompt = _derive_seed_user_prompt(sample_id=sample_id, task_type=task_type, metadata=metadata)

    # Ensure first business turn is user for Qwen3.5 chat template robustness.
    if first_idx < len(msgs) and msgs[first_idx].get("role") != "user":
        msgs.insert(first_idx, {"role": "user", "content": seed_prompt, "step_loss_mask": 0})
        counters["insert_seed_user_for_first_non_system"] += 1

    users = [m for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), str) and m.get("content").strip()]
    if not users:
        insert_idx = 1 if msgs and msgs[0].get("role") == "system" else 0
        msgs.insert(insert_idx, {"role": "user", "content": seed_prompt, "step_loss_mask": 0})
        counters["insert_seed_user_for_missing_user"] += 1
        users = [m for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), str) and m.get("content").strip()]

    # Qwen3.5 template raises "No user query found" when all user turns are tool_response wrappers.
    has_non_tool_query_user = any(not _is_tool_response_user_text(m["content"]) for m in users)
    if not has_non_tool_query_user:
        insert_idx = 1 if msgs and msgs[0].get("role") == "system" else 0
        msgs.insert(insert_idx, {"role": "user", "content": seed_prompt, "step_loss_mask": 0})
        counters["insert_seed_user_for_no_user_query"] += 1

    return msgs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Legacy compatibility converter for action-json SFT messages. ReAct SFT comes from upstream postprocess."
    )
    parser.add_argument("--input-jsonl", type=str, default=str(SFT_OUTPUTS_ANSWER_HIT / "mcp_sft_all.jsonl"))
    parser.add_argument("--output-root", type=str, default=str(SFT_OUT_ROOT))
    parser.add_argument("--usage-summary-csv", type=str, default=None)
    parser.add_argument(
        "--allowlist", type=str, default=str(Path(__file__).resolve().parents[1] / "tools/allowlist_v0.json")
    )
    parser.add_argument("--allow-all", action="store_true")
    parser.add_argument("--max-samples-per-task-type", type=int, default=None)
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        "[DEPRECATED] convert_pipelined_to_slime_sft.py emits legacy action_json SFT only. "
        "Canonical ReAct SFT is produced upstream by pipeline/postprocess.",
        file=sys.stderr,
    )

    allowlist = load_allowlist(Path(args.allowlist) if args.allowlist else None)
    usage_summary_csv = (
        Path(args.usage_summary_csv) if args.usage_summary_csv else input_jsonl.parents[1] / "molclaw_usage_summary.csv"
    )
    usage_by_basename = load_usage_summary_by_basename(usage_summary_csv)

    out_rows: dict[str, list[dict[str, Any]]] = {k: [] for k in RAW_TASK_TYPES}
    skipped: list[dict[str, Any]] = []
    counters = Counter()

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            counters["input_rows"] += 1

            sample_id = obj.get("id")
            metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
            task_type = metadata.get("task_type")
            if not isinstance(task_type, str) or task_type not in RAW_TASK_TYPES:
                skipped.append(
                    {
                        "source": str(input_jsonl),
                        "task_id": sample_id,
                        "skip_reason": "unknown_task_type",
                        "details": {"line_no": line_no, "task_type": task_type},
                    }
                )
                counters["skip_unknown_task_type"] += 1
                continue

            messages = obj.get("messages")
            if not isinstance(messages, list):
                skipped.append(
                    {
                        "source": str(input_jsonl),
                        "task_id": sample_id,
                        "skip_reason": "messages_not_list",
                        "details": {"line_no": line_no},
                    }
                )
                counters["skip_messages_not_list"] += 1
                continue

            converted_messages: list[dict[str, Any]] = []
            tool_call_count = 0
            final_answer_count = 0

            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                role = msg.get("role")
                content = msg.get("content")
                step_loss_mask = msg.get("step_loss_mask", 1)
                if step_loss_mask not in (0, 1):
                    step_loss_mask = 1

                if role == "assistant":
                    if not isinstance(content, str):
                        counters["drop_assistant_non_string"] += 1
                        continue

                    payload = None
                    try:
                        payload = json.loads(content)
                    except Exception:
                        payload = None

                    if not isinstance(payload, dict):
                        counters["drop_assistant_non_json"] += 1
                        continue

                    action_type = payload.get("type")
                    if action_type == ACTION_TOOL_CALL:
                        normalized, reason = normalize_tool_call(payload, allowlist=allowlist, allow_all=args.allow_all)
                        if normalized is None:
                            counters[f"drop_tool_call_{reason}"] += 1
                            continue
                        converted_messages.append(
                            {
                                "role": "assistant",
                                "content": json.dumps(normalized, ensure_ascii=False, separators=(",", ":")),
                                "step_loss_mask": step_loss_mask,
                            }
                        )
                        tool_call_count += 1
                    elif action_type == ACTION_FINAL_ANSWER:
                        normalized, reason = normalize_final_answer(payload)
                        if normalized is None:
                            counters[f"drop_final_answer_{reason}"] += 1
                            continue
                        converted_messages.append(
                            {
                                "role": "assistant",
                                "content": json.dumps(normalized, ensure_ascii=False, separators=(",", ":")),
                                "step_loss_mask": step_loss_mask,
                            }
                        )
                        final_answer_count += 1
                    else:
                        counters["drop_assistant_unknown_action_type"] += 1

                    continue

                # Non-assistant messages are converted to system/user only.
                converted_role = role if role in {"system", "user"} else "user"
                text = _stringify_content(content).strip()
                if not text:
                    counters["drop_empty_non_assistant_content"] += 1
                    continue

                converted_messages.append(
                    {
                        "role": converted_role,
                        "content": text,
                        "step_loss_mask": 0 if converted_role == "system" else step_loss_mask,
                    }
                )

            if tool_call_count == 0:
                skipped.append(
                    {
                        "source": str(input_jsonl),
                        "task_id": sample_id,
                        "skip_reason": "no_tool_call_after_filter",
                        "details": {"line_no": line_no},
                    }
                )
                counters["skip_no_tool_call"] += 1
                continue

            if final_answer_count == 0:
                skipped.append(
                    {
                        "source": str(input_jsonl),
                        "task_id": sample_id,
                        "skip_reason": "no_final_answer_after_filter",
                        "details": {"line_no": line_no},
                    }
                )
                counters["skip_no_final_answer"] += 1
                continue

            converted_messages = _ensure_valid_message_structure(
                converted_messages,
                sample_id=sample_id if isinstance(sample_id, str) else None,
                task_type=task_type,
                metadata=metadata,
                counters=counters,
            )

            user_turn_count = sum(
                1
                for m in converted_messages
                if m.get("role") == "user" and isinstance(m.get("content"), str) and m.get("content").strip()
            )
            assistant_turn_count = sum(
                1
                for m in converted_messages
                if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m.get("content").strip()
            )

            if user_turn_count == 0:
                skipped.append(
                    {
                        "source": str(input_jsonl),
                        "task_id": sample_id,
                        "skip_reason": "no_user_after_normalize",
                        "details": {"line_no": line_no},
                    }
                )
                counters["skip_no_user_after_normalize"] += 1
                continue

            if assistant_turn_count == 0:
                skipped.append(
                    {
                        "source": str(input_jsonl),
                        "task_id": sample_id,
                        "skip_reason": "no_assistant_after_normalize",
                        "details": {"line_no": line_no},
                    }
                )
                counters["skip_no_assistant_after_normalize"] += 1
                continue

            unsupported_roles = [
                m.get("role") for m in converted_messages if m.get("role") not in {"system", "user", "assistant"}
            ]
            if unsupported_roles:
                skipped.append(
                    {
                        "source": str(input_jsonl),
                        "task_id": sample_id,
                        "skip_reason": "unsupported_roles_after_normalize",
                        "details": {"line_no": line_no, "roles": sorted(set(unsupported_roles))},
                    }
                )
                counters["skip_unsupported_roles_after_normalize"] += 1
                continue

            trajectory_path = metadata.get("trajectory_path") if isinstance(metadata.get("trajectory_path"), str) else None
            basename = Path(trajectory_path).name if trajectory_path else ""
            usage = usage_by_basename.get(basename)

            row = {
                "messages": converted_messages,
                "metadata": {
                    "task_id": sample_id,
                    "task_type": task_type,
                    "source_path": trajectory_path,
                    "usage_summary": usage,
                    "tool_call_count": tool_call_count,
                    "final_answer_count": final_answer_count,
                    "schema_version": "drug_agent_sft_action_json_legacy_v1",
                    "protocol": "action_json",
                    "source_protocol": "legacy_action_json_converter",
                },
            }
            out_rows[task_type].append(row)
            counters[f"kept_{task_type}"] += 1
            counters["kept_total"] += 1

    if args.max_samples_per_task_type is not None and args.max_samples_per_task_type >= 0:
        for task_type in RAW_TASK_TYPES:
            out_rows[task_type] = out_rows[task_type][: args.max_samples_per_task_type]

    mixed = []
    for task_type in RAW_TASK_TYPES:
        mixed.extend(out_rows[task_type])

    for task_type in RAW_TASK_TYPES:
        write_jsonl(output_root / f"{task_type}.jsonl", out_rows[task_type])
    write_jsonl(output_root / "mixed.jsonl", mixed)
    write_jsonl(output_root / "skipped_report.jsonl", skipped)

    manifest = {
        "input_jsonl": str(input_jsonl),
        "output_root": str(output_root),
        "usage_summary_csv": str(usage_summary_csv),
        "allowlist": str(args.allowlist),
        "allow_all": bool(args.allow_all),
        "output_protocol": "action_json",
        "schema_version": "drug_agent_sft_action_json_legacy_v1",
        "max_samples_per_task_type": args.max_samples_per_task_type,
        "counts": {
            **dict(counters),
            "out_ac": len(out_rows["ac"]),
            "out_pf": len(out_rows["pf"]),
            "out_vs": len(out_rows["vs"]),
            "out_mixed": len(mixed),
            "skipped": len(skipped),
        },
    }
    write_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
