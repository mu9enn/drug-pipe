from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


LEGACY_MARKERS = ("<thought>", "<tool_call>", "<observation>", "<final_answer>")


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != "drug_agent_qwen35_sft_v1":
        errors.append("unexpected_schema_version")
    messages = record.get("messages")
    tools = record.get("tools")
    if not isinstance(messages, list) or len(messages) < 3:
        return errors + ["messages_missing_or_short"]
    if not isinstance(tools, list) or not tools:
        return errors + ["tools_missing"]
    tool_names = {
        str((tool.get("function") or {}).get("name") or "")
        for tool in tools if isinstance(tool, dict)
    }
    if any(not name for name in tool_names):
        errors.append("tool_without_name")
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        errors.append("system_user_prefix_required")
    pending: list[tuple[str, str]] = []
    final_positions: list[int] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"message_not_object:{index}")
            continue
        role = message.get("role")
        if role == "assistant":
            if pending:
                errors.append(f"assistant_before_tool_results:{index}")
            if message.get("step_loss_mask") != 1:
                errors.append(f"assistant_mask:{index}")
            calls = message.get("tool_calls") or []
            content = message.get("content")
            if calls:
                if content not in (None, ""):
                    errors.append(f"tool_decision_has_content:{index}")
                for call in calls:
                    function = call.get("function") if isinstance(call, dict) else None
                    call_id = str(call.get("id") or "") if isinstance(call, dict) else ""
                    name = str((function or {}).get("name") or "")
                    arguments = (function or {}).get("arguments")
                    if not call_id or name not in tool_names or not isinstance(arguments, dict):
                        errors.append(f"invalid_tool_call:{index}")
                    pending.append((call_id, name))
            elif not isinstance(content, str) or not content.strip():
                errors.append(f"empty_assistant_decision:{index}")
            else:
                final_positions.append(index)
        elif role == "tool":
            if message.get("step_loss_mask") != 0:
                errors.append(f"tool_mask:{index}")
            call_id = str(message.get("tool_call_id") or "")
            name = str(message.get("name") or "")
            if not pending or pending[0] != (call_id, name):
                errors.append(f"tool_result_order:{index}:{call_id}:{name}")
            else:
                pending.pop(0)
        elif role in {"system", "user"}:
            if message.get("step_loss_mask") != 0:
                errors.append(f"context_mask:{index}")
        else:
            errors.append(f"unexpected_role:{index}:{role}")
    if pending:
        errors.append(f"missing_tool_results:{pending}")
    if final_positions != [len(messages) - 1]:
        errors.append(f"single_terminal_final_required:{final_positions}")
    serialized = json.dumps(record, ensure_ascii=False)
    for marker in LEGACY_MARKERS:
        if marker in serialized:
            errors.append(f"legacy_marker:{marker}")
    return errors


def validate_file(path: Path, *, model: str | None = None) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    tokenizer = None
    mask_generator = None
    if model:
        from transformers import AutoTokenizer
        from slime.utils.mask_utils import MultiTurnLossMaskGenerator

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            counts["records"] += 1
            record = json.loads(line)
            findings = validate_record(record)
            if not findings and tokenizer is not None:
                tokenizer.apply_chat_template(record["messages"], tools=record["tools"], tokenize=False)
                token_ids, mask = mask_generator.get_loss_mask(record["messages"], tools=record["tools"])
                if len(token_ids) != len(mask) or not any(mask):
                    findings.append("invalid_qwen35_loss_mask")
            if findings:
                errors.append({"line": line_number, "id": record.get("id"), "errors": findings})
    return {"ok": not errors and counts["records"] > 0, "counts": dict(counts), "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate structured Qwen3.5 SFT messages.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model")
    args = parser.parse_args()
    report = validate_file(args.input, model=args.model)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
