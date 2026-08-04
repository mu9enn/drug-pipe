#!/usr/bin/env python3
"""Audit canonical Drug-Pipe raw sessions without modifying them."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


MOLCLAW_PREFIX = "mcp__molclaw-"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit canonical row*/complete_session.jsonl files, strict run metadata, "
            "MolClaw success/failure ratios, pairing, and terminal results."
        )
    )
    parser.add_argument("roots", nargs="+", type=Path, help="Raw run/root directories")
    parser.add_argument(
        "--expected",
        type=int,
        help="Expected total canonical session count across all roots",
    )
    parser.add_argument(
        "--review-failure-ratio",
        type=float,
        default=0.5,
        help="Flag samples at or above this returned MolClaw failure ratio (default: 0.5)",
    )
    parser.add_argument("--json-output", type=Path, help="Write the full audit JSON")
    parser.add_argument(
        "--fail-on-hard",
        action="store_true",
        help="Exit 1 when a hard failure or expected-count mismatch exists",
    )
    return parser.parse_args()


def structured_failure(content: Any) -> bool:
    value = content
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped.startswith(("{", "[")):
            return False
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return False

    def walk(item: Any) -> bool:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = str(key).lower().replace("_", "")
                if normalized == "status" and str(child).strip().lower() in {
                    "error",
                    "failed",
                    "failure",
                }:
                    return True
                if normalized == "iserror" and child is True:
                    return True
                if normalized == "success" and child is False:
                    return True
            return any(walk(child) for child in item.values())
        if isinstance(item, list):
            return any(walk(child) for child in item)
        return False

    return walk(value)


def discover_sessions(root: Path) -> list[Path]:
    root = root.resolve()
    if root.is_file():
        return [root] if root.name == "complete_session.jsonl" else []
    direct = sorted(root.glob("row*/complete_session.jsonl"))
    if direct:
        return direct
    return sorted(
        path
        for path in root.glob("**/row*/complete_session.jsonl")
        if "attempts" not in path.parts and "debug" not in path.parts
    )


def strict_meta_findings(row_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    path = row_dir / "run_meta.json"
    if not path.is_file():
        return None, []
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"invalid_run_meta:{exc}"]
    findings = []
    if meta.get("return_code") != 0:
        findings.append(f"return_code:{meta.get('return_code')!r}")
    if meta.get("timed_out") is not False:
        findings.append(f"timed_out:{meta.get('timed_out')!r}")
    if meta.get("raw_session_valid") is not True:
        findings.append(f"raw_session_valid:{meta.get('raw_session_valid')!r}")
    return meta, findings


def inspect_session(path: Path, review_ratio: float) -> dict[str, Any]:
    calls: dict[str, str] = {}
    results: dict[str, str] = {}
    malformed_lines = 0
    terminal: dict[str, Any] | None = None
    init_molclaw_statuses: list[str] = []

    try:
        handle = path.open("r", encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {
            "session": str(path),
            "hard_findings": [f"unreadable_session:{exc}"],
            "review_findings": [],
        }

    with handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "result":
                terminal = event
            if event.get("type") == "system" and event.get("subtype") == "init":
                for server in event.get("mcp_servers") or []:
                    if not isinstance(server, dict):
                        continue
                    if str(server.get("name") or "").startswith("molclaw-"):
                        init_molclaw_statuses.append(str(server.get("status") or ""))

            message = event.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), list):
                continue
            for item in message["content"]:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "tool_use":
                    tool_id = str(item.get("id") or "").strip()
                    tool_name = str(item.get("name") or "").strip()
                    if tool_id and tool_name.startswith(MOLCLAW_PREFIX):
                        calls[tool_id] = tool_name
                elif item.get("type") == "tool_result":
                    tool_id = str(item.get("tool_use_id") or "").strip()
                    if tool_id:
                        failed = item.get("is_error") is True or structured_failure(
                            item.get("content")
                        )
                        results[tool_id] = "failed" if failed else "success"

    successful_ids = [tool_id for tool_id in calls if results.get(tool_id) == "success"]
    failed_ids = [tool_id for tool_id in calls if results.get(tool_id) == "failed"]
    unresolved_ids = [tool_id for tool_id in calls if tool_id not in results]
    returned = len(successful_ids) + len(failed_ids)
    failure_ratio = len(failed_ids) / returned if returned else None
    meta, meta_findings = strict_meta_findings(path.parent)

    hard = list(meta_findings)
    if path.stat().st_size == 0:
        hard.append("empty_session")
    if malformed_lines:
        hard.append(f"malformed_json_lines:{malformed_lines}")
    if terminal is None:
        hard.append("missing_terminal_result")
    elif terminal.get("is_error") is True:
        hard.append("terminal_result_error")
    if unresolved_ids:
        hard.append(f"unresolved_molclaw_calls:{len(unresolved_ids)}")
    if not successful_ids:
        hard.append("zero_successful_molclaw_results")

    review = []
    notes = []
    if failure_ratio is not None and failure_ratio >= review_ratio:
        review.append(f"molclaw_failure_ratio:{failure_ratio:.6f}")
    if init_molclaw_statuses and not any(
        status.lower() == "connected" for status in init_molclaw_statuses
    ):
        notes.append("init_molclaw_not_connected_but_results_may_exist")
    if meta and int(meta.get("selected_claude_attempt") or 1) > 1:
        notes.append(f"selected_attempt:{meta.get('selected_claude_attempt')}")

    successful_tools = Counter(calls[tool_id] for tool_id in successful_ids)
    failed_tools = Counter(calls[tool_id] for tool_id in failed_ids)
    return {
        "session": str(path),
        "row": path.parent.name,
        "bytes": path.stat().st_size,
        "run_meta_present": meta is not None,
        "selected_claude_attempt": meta.get("selected_claude_attempt") if meta else None,
        "molclaw_calls": len(calls),
        "molclaw_successes": len(successful_ids),
        "molclaw_failures": len(failed_ids),
        "molclaw_unresolved": len(unresolved_ids),
        "molclaw_failure_ratio": failure_ratio,
        "successful_tools": dict(successful_tools.most_common()),
        "failed_tools": dict(failed_tools.most_common()),
        "malformed_json_lines": malformed_lines,
        "terminal_present": terminal is not None,
        "terminal_error": bool(terminal and terminal.get("is_error") is True),
        "hard_findings": hard,
        "review_findings": review,
        "notes": notes,
    }


def main() -> int:
    args = parse_args()
    if not 0 <= args.review_failure_ratio <= 1:
        raise SystemExit("--review-failure-ratio must be between 0 and 1")

    sessions: list[Path] = []
    roots_report = []
    for root in args.roots:
        found = discover_sessions(root)
        roots_report.append({"root": str(root.resolve()), "sessions": len(found)})
        sessions.extend(found)
    sessions = sorted(set(sessions))

    rows = [inspect_session(path, args.review_failure_ratio) for path in sessions]
    hard_rows = [row for row in rows if row["hard_findings"]]
    review_rows = [row for row in rows if row["review_findings"]]
    aggregate = {
        "sessions": len(rows),
        "molclaw_calls": sum(int(row.get("molclaw_calls", 0)) for row in rows),
        "molclaw_successes": sum(int(row.get("molclaw_successes", 0)) for row in rows),
        "molclaw_failures": sum(int(row.get("molclaw_failures", 0)) for row in rows),
        "molclaw_unresolved": sum(int(row.get("molclaw_unresolved", 0)) for row in rows),
        "hard_failure_rows": len(hard_rows),
        "review_rows": len(review_rows),
    }
    expected_mismatch = args.expected is not None and len(rows) != args.expected
    report = {
        "schema_version": "drug_pipe_raw_audit_v1",
        "roots": roots_report,
        "expected_sessions": args.expected,
        "expected_count_mismatch": expected_mismatch,
        "review_failure_ratio": args.review_failure_ratio,
        "aggregate": aggregate,
        "hard_failure_samples": hard_rows,
        "review_samples": review_rows,
        "samples": rows,
    }

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps({
        "roots": roots_report,
        "expected_sessions": args.expected,
        "expected_count_mismatch": expected_mismatch,
        "aggregate": aggregate,
    }, ensure_ascii=False, indent=2))
    for row in hard_rows:
        print(
            "HARD",
            row.get("molclaw_successes"),
            row.get("molclaw_failures"),
            ",".join(row["hard_findings"]),
            row["session"],
        )
    for row in review_rows:
        print(
            "REVIEW",
            row.get("molclaw_successes"),
            row.get("molclaw_failures"),
            ",".join(row["review_findings"]),
            row["session"],
        )

    if args.fail_on_hard and (hard_rows or expected_mismatch):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
