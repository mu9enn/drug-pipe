#!/usr/bin/env python3
"""Backfill trajectories without reward fields for runs containing molclaw calls."""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from postprocess.trajectory_exporter import export_results_dir

RUN_RE = re.compile(r"^molbench_(vs|ac|pf)_.+_run_(\d{8})_(\d{6})(?:_.+)?$")
MOLCLAW_NEEDLE = '"name":"mcp__molclaw'


def _find_run_dir_from_session(session_path: Path) -> Path | None:
    for parent in session_path.parents:
        if RUN_RE.match(parent.name):
            return parent
    return None


def _infer_task_from_run_dir(run_dir: Path) -> str | None:
    m = RUN_RE.match(run_dir.name)
    if not m:
        return None
    return m.group(1)


def _contains_molclaw_call(session_path: Path, needle: str) -> bool:
    try:
        with session_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if needle in line:
                    return True
    except OSError:
        return False
    return False


def _scan_results(results_root: Path, needle: str) -> tuple[dict[Path, int], int, int]:
    run_hits: dict[Path, int] = {}
    scanned = 0
    matched = 0
    for session_path in results_root.rglob("complete_session.jsonl"):
        scanned += 1
        if not _contains_molclaw_call(session_path, needle):
            continue
        run_dir = _find_run_dir_from_session(session_path)
        if run_dir is None:
            continue
        matched += 1
        run_hits[run_dir] = run_hits.get(run_dir, 0) + 1
    return run_hits, scanned, matched


def _count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _verify_no_reward_fields(traj_dir: Path) -> dict[str, Any]:
    checks = {
        "step_reward_key_found": 0,
        "trajectory_reward_key_found": 0,
        "metrics_reward_key_found": 0,
        "task_metrics_reward_key_found": 0,
        "files_checked": {},
    }

    step_path = traj_dir / "step_level.jsonl"
    traj_path = traj_dir / "trajectory_level.jsonl"

    step_lines = 0
    if step_path.is_file():
        with step_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                step_lines += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "reward" in rec:
                    checks["step_reward_key_found"] += 1
    checks["files_checked"]["step_level_lines"] = step_lines

    traj_lines = 0
    if traj_path.is_file():
        with traj_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                traj_lines += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "reward_outcome" in rec:
                    checks["trajectory_reward_key_found"] += 1
                metrics = rec.get("metrics")
                if isinstance(metrics, dict) and "reward_outcome" in metrics:
                    checks["metrics_reward_key_found"] += 1
                task_metrics = rec.get("task_metrics")
                if isinstance(task_metrics, dict) and "reward_outcome" in task_metrics:
                    checks["task_metrics_reward_key_found"] += 1
    checks["files_checked"]["trajectory_level_lines"] = traj_lines

    summary_path = traj_dir / "dataset_summary.json"
    summary_avg_reward_key = 0
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        tma = summary.get("task_metric_averages") if isinstance(summary, dict) else {}
        if isinstance(tma, dict) and "avg_reward_outcome" in tma:
            summary_avg_reward_key = 1
    checks["summary_avg_reward_key_found"] = summary_avg_reward_key

    checks["reward_keys_total_found"] = (
        checks["step_reward_key_found"]
        + checks["trajectory_reward_key_found"]
        + checks["metrics_reward_key_found"]
        + checks["task_metrics_reward_key_found"]
        + checks["summary_avg_reward_key_found"]
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill trajectory exports without reward fields for runs containing mcp__molclaw calls."
    )
    parser.add_argument(
        "--results-root",
        default=str(Path(__file__).resolve().parents[1] / "results"),
        help="Root directory to scan recursively (default: %(default)s)",
    )
    parser.add_argument(
        "--needle",
        default=MOLCLAW_NEEDLE,
        help="Substring to detect molclaw tool usage in complete_session.jsonl (default: %(default)s)",
    )
    parser.add_argument(
        "--report-out",
        default="",
        help="Optional output JSON report path. Default: <results-root>/backfill_no_reward_report_<ts>.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan and print matched runs, do not re-export trajectories.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    if not results_root.is_dir():
        raise NotADirectoryError(results_root)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_out = (
        Path(args.report_out).expanduser().resolve()
        if args.report_out.strip()
        else (results_root / f"backfill_no_reward_report_{ts}.json")
    )

    run_hits, scanned_samples, matched_samples = _scan_results(results_root, args.needle)
    matched_runs = sorted(run_hits.keys())

    report: dict[str, Any] = {
        "results_root": str(results_root),
        "needle": args.needle,
        "scanned_complete_session_files": scanned_samples,
        "matched_complete_session_files": matched_samples,
        "matched_run_count": len(matched_runs),
        "matched_runs": [str(r) for r in matched_runs],
        "run_hit_counts": {str(k): v for k, v in sorted(run_hits.items(), key=lambda kv: str(kv[0]))},
        "dry_run": bool(args.dry_run),
        "backfill": [],
        "totals": {
            "runs_succeeded": 0,
            "runs_failed": 0,
            "reward_keys_found_after_backfill": 0,
            "accepted_lines_after_backfill": 0,
        },
    }

    if args.dry_run:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"REPORT_FILE={report_out}")
        return

    for run_dir in matched_runs:
        entry: dict[str, Any] = {
            "run_dir": str(run_dir),
            "matched_complete_session_files": run_hits.get(run_dir, 0),
            "status": "pending",
            "error": None,
            "traceback": None,
            "summary": None,
            "verify": None,
            "accepted_lines": 0,
        }
        try:
            task = _infer_task_from_run_dir(run_dir)
            summary = export_results_dir(run_dir, task=task)
            traj_dir = run_dir / "trajectories"
            verify = _verify_no_reward_fields(traj_dir)
            accepted_lines = _count_jsonl_lines(traj_dir / "accepted.jsonl")

            entry["status"] = "ok"
            entry["summary"] = summary
            entry["verify"] = verify
            entry["accepted_lines"] = accepted_lines

            report["totals"]["runs_succeeded"] += 1
            report["totals"]["reward_keys_found_after_backfill"] += int(verify["reward_keys_total_found"])
            report["totals"]["accepted_lines_after_backfill"] += accepted_lines
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = repr(e)
            entry["traceback"] = traceback.format_exc()
            report["totals"]["runs_failed"] += 1

        report["backfill"].append(entry)

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"REPORT_FILE={report_out}")


if __name__ == "__main__":
    main()
