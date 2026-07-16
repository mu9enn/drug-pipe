#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_RE = re.compile(r"^molbench_kg_.+_run_\d{8}_\d{6}(?:_.+)?$")
ROLLOUT_RE = re.compile(r"^rollout\d+$")
PROMPT_LEAK_PATTERNS = [
    "expected_trajectory",
    "toolchain_nodes",
    "toolchain_edges",
    "pair::",
    "expected toolchain",
    "expected_tools",
]
QUESTION_TOOL_USE_KEYWORDS = [
    "retrieve",
    "predict",
    "dock",
    "calculate",
    "analyze",
    "generate",
    "extract",
    "validate",
    "simulate",
    "screen",
    "structure",
]


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _normalize_tool_name(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return ""
    if "__" in s:
        return s.split("__")[-1]
    return s


def _extract_expected_tools(kg_task_spec: dict[str, Any]) -> list[str]:
    expected: list[str] = []

    traj = kg_task_spec.get("expected_trajectory")
    if isinstance(traj, dict):
        wf = traj.get("workflow_graph")
        if isinstance(wf, dict):
            nodes = wf.get("nodes")
            if isinstance(nodes, list):
                for node in nodes:
                    if not isinstance(node, dict):
                        continue
                    if str(node.get("type") or "") != "tool":
                        continue
                    tid = str(node.get("tool_id") or "").strip()
                    if tid:
                        expected.append(_normalize_tool_name(tid))
            if expected:
                return expected
        steps = traj.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                tid = str(step.get("tool_id") or step.get("tool") or "").strip()
                if tid:
                    expected.append(_normalize_tool_name(tid))
    elif isinstance(traj, list):
        for step in traj:
            if not isinstance(step, dict):
                continue
            tid = str(step.get("tool_id") or step.get("tool") or "").strip()
            if tid:
                expected.append(_normalize_tool_name(tid))

    if expected:
        return expected

    toolchain = kg_task_spec.get("toolchain") if isinstance(kg_task_spec.get("toolchain"), dict) else {}
    tools = toolchain.get("tools")
    if isinstance(tools, list):
        return [_normalize_tool_name(str(t)) for t in tools if str(t).strip()]

    nodes = kg_task_spec.get("toolchain_nodes")
    if isinstance(nodes, list):
        return [_normalize_tool_name(str(t)) for t in nodes if str(t).strip()]

    return []


def _extract_actual_tool_sequence(session_path: Path) -> tuple[list[str], int]:
    seq: list[str] = []
    molclaw_calls = 0
    if not session_path.is_file():
        return seq, molclaw_calls

    with session_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(obj.get("type") or "") != "assistant":
                continue
            msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if str(item.get("type") or "") != "tool_use":
                    continue
                raw_name = str(item.get("name") or "")
                if "mcp__molclaw" in raw_name:
                    molclaw_calls += 1
                norm = _normalize_tool_name(raw_name)
                if norm:
                    seq.append(norm)

    return seq, molclaw_calls


def _extract_tool_failure_info(session_path: Path) -> tuple[list[str], str | None]:
    if not session_path.is_file():
        return [], None

    tool_use_name_by_id: dict[str, str] = {}
    failed_tools_in_order: list[str] = []

    with session_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            ev_type = str(obj.get("type") or "")
            if ev_type == "assistant":
                msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if str(item.get("type") or "") != "tool_use":
                        continue
                    tool_use_id = str(item.get("id") or "")
                    tool_name = _normalize_tool_name(str(item.get("name") or ""))
                    if tool_use_id and tool_name:
                        tool_use_name_by_id[tool_use_id] = tool_name
                continue

            if ev_type == "user":
                msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if str(item.get("type") or "") != "tool_result":
                        continue
                    if not bool(item.get("is_error")):
                        continue
                    tool_use_id = str(item.get("tool_use_id") or "")
                    tool_name = _normalize_tool_name(tool_use_name_by_id.get(tool_use_id, ""))
                    if tool_name:
                        failed_tools_in_order.append(tool_name)

    failed_unique: list[str] = []
    seen: set[str] = set()
    for name in failed_tools_in_order:
        if name in seen:
            continue
        seen.add(name)
        failed_unique.append(name)

    first_failed = failed_tools_in_order[0] if failed_tools_in_order else None
    return failed_unique, first_failed


def _prompt_leak_detected(prompt_text: str) -> bool:
    low = (prompt_text or "").lower()
    if not low:
        return False
    return any(pat in low for pat in PROMPT_LEAK_PATTERNS)


def _question_requires_tool_use(question_text: str, expected_tools: list[str]) -> bool:
    text = (question_text or "").strip().lower()
    if not text:
        return bool(expected_tools)
    keyword_hit = any(k in text for k in QUESTION_TOOL_USE_KEYWORDS)
    # Conservative: if expected tools exist, assume tool-use is required unless question is empty.
    return bool(expected_tools) and keyword_hit


def _classify_failure_type(*, completed: bool, used_any_molclaw: bool, failed_tool_count: int) -> str:
    if completed and used_any_molclaw and failed_tool_count == 0:
        return "success"
    if not completed:
        return "execution_failure"
    if failed_tool_count > 0:
        return "tool_call_failure"
    if not used_any_molclaw:
        return "no_tool_use"
    return "execution_failure"


def _recommendation(*, failure_type: str, expected_tool_coverage: float | None) -> str:
    if failure_type == "success":
        if expected_tool_coverage is not None and expected_tool_coverage < 0.5:
            return "downweight"
        return "keep"
    if failure_type == "no_tool_use":
        return "reject"
    if failure_type == "tool_call_failure":
        return "downweight"
    return "needs_manual_review"


def _write_feedback_files(
    *,
    detailed_rows: list[dict[str, Any]],
    feedback_root: Path,
) -> list[str]:
    out_paths: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in detailed_rows:
        kg_run_id = str(row.get("kg_run_id") or "").strip()
        if not kg_run_id:
            continue
        grouped.setdefault(kg_run_id, []).append(row)

    for kg_run_id, rows in grouped.items():
        out_dir = feedback_root / kg_run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "kg_execution_feedback.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                feedback = {
                    "kg_task_id": row.get("kg_task_id"),
                    "expected_tools": row.get("expected_tools", []),
                    "actual_tools": row.get("actual_tools", []),
                    "used_any_molclaw": bool(row.get("used_any_molclaw")),
                    "expected_tool_coverage": row.get("expected_tool_coverage"),
                    "failed_tools": row.get("failed_tools", []),
                    "failure_type": row.get("failure_type"),
                    "recommendation": row.get("recommendation"),
                }
                f.write(json.dumps(feedback, ensure_ascii=False) + "\n")
        out_paths.append(str(out_path))
    return sorted(out_paths)


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = dp[j] if dp[j] >= dp[j - 1] else dp[j - 1]
            prev = cur
    return dp[-1]


def _iter_run_dirs(results_root: Path) -> list[Path]:
    out: set[Path] = set()
    for cfg_path in results_root.rglob("run_config.json"):
        run_dir = cfg_path.parent
        cfg = _safe_load_json(cfg_path)
        task = str(cfg.get("task") or "").strip().lower()
        if task != "kg":
            continue
        if RUN_RE.match(run_dir.name):
            out.add(run_dir.resolve())
        else:
            out.add(run_dir.resolve())
    return sorted(out)


def _iter_sample_dirs(run_dir: Path) -> list[Path]:
    sample_dirs: list[Path] = []
    row_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("row") and "_idx" in p.name])
    for row_dir in row_dirs:
        rollout_dirs = sorted([p for p in row_dir.iterdir() if p.is_dir() and ROLLOUT_RE.match(p.name)])
        if rollout_dirs:
            sample_dirs.extend(rollout_dirs)
        else:
            sample_dirs.append(row_dir)
    return sample_dirs


def _rollout_index(sample_dir: Path) -> int:
    if ROLLOUT_RE.match(sample_dir.name):
        try:
            return int(sample_dir.name.replace("rollout", ""))
        except Exception:
            return 1
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan task=kg rollout outputs and produce audit metrics.")
    parser.add_argument("--results-root", default="", help="Root containing kg runs (default: results/kg_sampled)")
    parser.add_argument("--output-dir", required=True, help="Output directory for audit files")
    parser.add_argument(
        "--feedback-root",
        default="",
        help="Root for kg_execution_feedback.jsonl outputs (default: <repo>/pipeline/kg/data)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]
    default_results_root = repo_root / "results" / "kg_sampled"
    default_feedback_root = repo_root / "pipeline" / "kg" / "data"

    results_root = Path(args.results_root).expanduser().resolve() if args.results_root.strip() else default_results_root
    output_dir = Path(args.output_dir).expanduser().resolve()
    feedback_root = Path(args.feedback_root).expanduser().resolve() if args.feedback_root.strip() else default_feedback_root
    output_dir.mkdir(parents=True, exist_ok=True)
    feedback_root.mkdir(parents=True, exist_ok=True)

    run_dirs = _iter_run_dirs(results_root)

    detailed_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        sample_dirs = _iter_sample_dirs(run_dir)
        for sdir in sample_dirs:
            question = _safe_load_json(sdir / "question.json")
            parsed = _safe_load_json(sdir / "parsed_answer.json")
            run_meta = _safe_load_json(sdir / "run_meta.json")

            kg_spec = question.get("kg_task_spec") if isinstance(question.get("kg_task_spec"), dict) else {}
            if not kg_spec:
                raw_q = question.get("raw_question_json")
                if isinstance(raw_q, str) and raw_q.strip():
                    try:
                        tmp = json.loads(raw_q)
                    except Exception:
                        tmp = {}
                    if isinstance(tmp, dict):
                        kg_spec = tmp

            expected_tools_seq = _extract_expected_tools(kg_spec)
            expected_tools_unique = sorted(set(expected_tools_seq))
            question_text = str(question.get("question_text") or question.get("question") or kg_spec.get("question") or "")
            prompt_text = ""
            prompt_path = sdir / "prompt.txt"
            if prompt_path.is_file():
                prompt_text = prompt_path.read_text(encoding="utf-8", errors="ignore")

            session_path = sdir / "complete_session.jsonl"
            actual_seq, molclaw_calls = _extract_actual_tool_sequence(session_path)
            failed_tools, first_failed_tool = _extract_tool_failure_info(session_path)
            actual_unique = sorted(set(actual_seq))

            expected_set = set(expected_tools_unique)
            actual_set = set(actual_unique)

            expected_cov = None
            unexpected_n = len(actual_set)
            lcs_score = None
            if expected_set:
                hit = len(expected_set & actual_set)
                expected_cov = hit / len(expected_set)
                unexpected_n = len(actual_set - expected_set)
                filtered_actual = [x for x in actual_seq if x in expected_set]
                lcs = _lcs_len(expected_tools_seq, filtered_actual)
                lcs_score = lcs / len(expected_tools_seq) if expected_tools_seq else 0.0

            return_code = run_meta.get("return_code")
            timed_out = bool(run_meta.get("timed_out", False))
            has_session = session_path.is_file()
            completed = bool(return_code == 0 and not timed_out and has_session)

            parse_error = parsed.get("parse_error")
            parse_source = str(parsed.get("parse_source") or "")
            api_error = parse_source == "api_error" or (isinstance(parse_error, str) and "api_error" in parse_error)
            prompt_leak_detected = _prompt_leak_detected(prompt_text)
            question_requires_tool_use = _question_requires_tool_use(question_text, expected_tools_unique)
            actual_tool_count = len(actual_unique)
            failed_tool_count = len(failed_tools)
            failure_type = _classify_failure_type(
                completed=completed,
                used_any_molclaw=molclaw_calls > 0,
                failed_tool_count=failed_tool_count,
            )
            agent_deviated = None
            if expected_set:
                cov_val = expected_cov if isinstance(expected_cov, (int, float)) else 0.0
                lcs_val = lcs_score if isinstance(lcs_score, (int, float)) else 0.0
                agent_deviated = bool(cov_val < 1.0 or lcs_val < 1.0)
            recommendation = _recommendation(
                failure_type=failure_type,
                expected_tool_coverage=float(expected_cov) if isinstance(expected_cov, (int, float)) else None,
            )

            kg_source = kg_spec.get("source") if isinstance(kg_spec.get("source"), dict) else {}
            kg_toolchain = kg_spec.get("toolchain") if isinstance(kg_spec.get("toolchain"), dict) else {}

            row = {
                "run_dir": str(run_dir),
                "sample_dir": str(sdir),
                "row_number": question.get("row_number"),
                "dataset_index": question.get("dataset_index"),
                "rollout_index": _rollout_index(sdir),
                "kg_task_id": kg_spec.get("task_id"),
                "kg_run_id": kg_source.get("kg_run_id"),
                "has_complete_session": has_session,
                "completed": completed,
                "failed": not completed,
                "return_code": return_code,
                "timed_out": timed_out,
                "parse_error": parse_error,
                "parse_source": parse_source,
                "api_error": bool(api_error),
                "question_requires_tool_use": question_requires_tool_use,
                "prompt_leak_detected": prompt_leak_detected,
                "molclaw_tool_call_count": molclaw_calls,
                "used_any_molclaw": molclaw_calls > 0,
                "expected_tools": expected_tools_unique,
                "actual_tools": actual_unique,
                "actual_tool_count": actual_tool_count,
                "expected_tool_coverage": expected_cov,
                "unexpected_tool_count": unexpected_n,
                "tool_order_lcs_score": lcs_score,
                "failed_tools": failed_tools,
                "failed_tool_count": failed_tool_count,
                "first_failed_tool": first_failed_tool,
                "agent_deviated_from_expected_toolchain": agent_deviated,
                "failure_type": failure_type,
                "recommendation": recommendation,
                "expected_hops": kg_toolchain.get("hops"),
                "expected_trajectory_available": bool(kg_spec.get("expected_trajectory")),
            }
            detailed_rows.append(row)

    detailed_path = output_dir / "kg_rollout_detailed.jsonl"
    with detailed_path.open("w", encoding="utf-8") as f:
        for row in detailed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_csv = output_dir / "kg_rollout_summary.csv"
    csv_fields = [
        "run_dir",
        "sample_dir",
        "row_number",
        "dataset_index",
        "rollout_index",
        "kg_task_id",
        "kg_run_id",
        "completed",
        "failed",
        "return_code",
        "timed_out",
        "has_complete_session",
        "parse_error",
        "parse_source",
        "api_error",
        "question_requires_tool_use",
        "prompt_leak_detected",
        "molclaw_tool_call_count",
        "used_any_molclaw",
        "actual_tool_count",
        "expected_tool_coverage",
        "unexpected_tool_count",
        "tool_order_lcs_score",
        "failed_tool_count",
        "first_failed_tool",
        "agent_deviated_from_expected_toolchain",
        "failure_type",
        "recommendation",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in detailed_rows:
            writer.writerow({k: row.get(k) for k in csv_fields})

    total = len(detailed_rows)
    completed_n = sum(1 for x in detailed_rows if x.get("completed"))
    failed_n = total - completed_n
    parse_failed_n = sum(1 for x in detailed_rows if x.get("parse_error"))
    api_error_n = sum(1 for x in detailed_rows if x.get("api_error"))
    any_molclaw_n = sum(1 for x in detailed_rows if x.get("used_any_molclaw"))
    prompt_leak_n = sum(1 for x in detailed_rows if x.get("prompt_leak_detected"))
    no_tool_use_n = sum(1 for x in detailed_rows if x.get("failure_type") == "no_tool_use")
    tool_call_failure_n = sum(1 for x in detailed_rows if x.get("failure_type") == "tool_call_failure")
    execution_failure_n = sum(1 for x in detailed_rows if x.get("failure_type") == "execution_failure")
    success_n = sum(1 for x in detailed_rows if x.get("failure_type") == "success")
    question_requires_tool_use_n = sum(1 for x in detailed_rows if x.get("question_requires_tool_use"))
    avg_actual_tool_count = (
        sum(float(x.get("actual_tool_count", 0)) for x in detailed_rows) / total if total > 0 else 0.0
    )

    coverage_vals = [float(x["expected_tool_coverage"]) for x in detailed_rows if isinstance(x.get("expected_tool_coverage"), (int, float))]
    lcs_vals = [float(x["tool_order_lcs_score"]) for x in detailed_rows if isinstance(x.get("tool_order_lcs_score"), (int, float))]

    avg_cov = (sum(coverage_vals) / len(coverage_vals)) if coverage_vals else 0.0
    avg_lcs = (sum(lcs_vals) / len(lcs_vals)) if lcs_vals else 0.0
    feedback_paths = _write_feedback_files(detailed_rows=detailed_rows, feedback_root=feedback_root)

    report_path = output_dir / "kg_tool_usage_report.md"
    lines = [
        "# KG Rollout Audit Report",
        "",
        f"- generated_at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- results_root: `{results_root}`",
        f"- run_count: {len(run_dirs)}",
        f"- total_tasks: {total}",
        f"- completed_tasks: {completed_n}",
        f"- failed_tasks: {failed_n}",
        f"- parse_failed_count: {parse_failed_n}",
        f"- api_error_count: {api_error_n}",
        f"- prompt_leak_detected_count: {prompt_leak_n}",
        f"- question_requires_tool_use_count: {question_requires_tool_use_n}",
        f"- used_any_molclaw_count: {any_molclaw_n}",
        f"- avg_actual_tool_count: {avg_actual_tool_count:.4f}",
        f"- avg_expected_tool_coverage: {avg_cov:.4f}",
        f"- avg_tool_order_lcs_score: {avg_lcs:.4f}",
        "",
        "## Failure Type Breakdown",
        "",
        f"- success: {success_n}",
        f"- no_tool_use: {no_tool_use_n}",
        f"- execution_failure: {execution_failure_n}",
        f"- tool_call_failure: {tool_call_failure_n}",
        "",
        "## Outputs",
        "",
        f"- `{summary_csv}`",
        f"- `{detailed_path}`",
        f"- `{report_path}`",
        "",
        "## Feedback Files",
        "",
    ]
    if feedback_paths:
        for p in feedback_paths:
            lines.append(f"- `{p}`")
    else:
        lines.append("- None")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "results_root": str(results_root),
                "run_count": len(run_dirs),
                "total_tasks": total,
                "completed_tasks": completed_n,
                "failed_tasks": failed_n,
                "summary_csv": str(summary_csv),
                "detailed_jsonl": str(detailed_path),
                "report_md": str(report_path),
                "feedback_files": feedback_paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
