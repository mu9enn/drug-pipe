from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.toolrl.parse_tool_calls import default_molclaw_allowlist
from drug_agent.utils import read_jsonl, write_json, write_jsonl


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"unsupported payload in {path}")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _target_tool_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        row.get("target_tool_calls"),
        (row.get("label") or {}).get("target_tool_calls") if isinstance(row.get("label"), dict) else None,
        (row.get("metadata") or {}).get("target_tool_calls") if isinstance(row.get("metadata"), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except json.JSONDecodeError:
                continue
        if isinstance(candidate, list):
            out = [item for item in candidate if isinstance(item, dict)]
            if out:
                return out
    return []


def _validate_tool_call(call: dict[str, Any], allowlist: set[str]) -> list[str]:
    errors: list[str] = []
    tool_name = call.get("tool_name")
    arguments = call.get("arguments")
    if not isinstance(tool_name, str) or not tool_name.strip():
        errors.append("missing_tool_name")
    elif tool_name not in allowlist:
        errors.append("tool_not_in_molclaw_allowlist")
    if not isinstance(arguments, dict):
        errors.append("arguments_not_object")
    return errors


def validate_toolrl_offline_data(
    input_path: Path,
    *,
    max_error_rows: int = 200,
    report_path: Path | None = None,
    errors_path: Path | None = None,
) -> dict[str, Any]:
    rows = _load_rows(input_path)
    allowlist = default_molclaw_allowlist()
    counts = Counter()
    errors: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        row_errors: list[str] = []
        prompt = row.get("prompt")
        label = row.get("label")
        metadata = row.get("metadata")

        if not isinstance(prompt, list) or not prompt:
            row_errors.append("missing_prompt_messages")
        else:
            roles = [item.get("role") for item in prompt if isinstance(item, dict)]
            if "assistant" in roles:
                counts["prompt_contains_history_assistant"] += 1
            if not any(role == "user" for role in roles):
                row_errors.append("prompt_missing_user_turn")

        if not isinstance(label, dict):
            row_errors.append("missing_label_object")
        if not isinstance(metadata, dict):
            row_errors.append("missing_metadata_object")

        target_tool_calls = _target_tool_calls(row)
        if not target_tool_calls:
            row_errors.append("missing_target_tool_calls")
        else:
            counts["target_tool_call_total"] += len(target_tool_calls)
            for call in target_tool_calls:
                row_errors.extend(_validate_tool_call(call, allowlist))

        allowed_names = []
        if isinstance(metadata, dict):
            allowed_names = _as_list(metadata.get("allowed_tool_names") or metadata.get("allowed_tools"))
        if not allowed_names:
            counts["missing_allowed_tool_names"] += 1

        if isinstance(metadata, dict):
            for key in ("task_id", "task_type", "source_id", "assistant_index"):
                if key not in metadata:
                    counts[f"metadata_missing_{key}"] += 1

        if row_errors:
            counts["invalid_rows"] += 1
            if len(errors) < max_error_rows:
                errors.append(
                    {
                        "index": index,
                        "source_id": metadata.get("source_id") if isinstance(metadata, dict) else None,
                        "errors": sorted(set(row_errors)),
                    }
                )
        else:
            counts["valid_rows"] += 1

    report = {
        "ok": counts["invalid_rows"] == 0 and len(rows) > 0,
        "input_path": str(input_path),
        "total_rows": len(rows),
        "valid_rows": counts["valid_rows"],
        "invalid_rows": counts["invalid_rows"],
        "target_tool_call_total": counts["target_tool_call_total"],
        "counts": dict(counts),
        "error_preview": errors[:10],
        "required_runtime": {
            "calls_mcp": False,
            "requires_molclaw_env": False,
        },
    }
    if report_path is not None:
        write_json(report_path, report)
    if errors_path is not None:
        write_jsonl(errors_path, errors)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate offline ToolRL step-level JSONL")
    parser.add_argument("--input", required=True, help="ToolRL step JSONL path")
    parser.add_argument("--report", default=None, help="Optional report JSON path")
    parser.add_argument("--errors", default=None, help="Optional row-level error JSONL path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    report_path = Path(args.report).expanduser().resolve() if args.report else None
    errors_path = Path(args.errors).expanduser().resolve() if args.errors else None
    report = validate_toolrl_offline_data(input_path, report_path=report_path, errors_path=errors_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
