#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare no-gpu and gpu reports")
    parser.add_argument("--coverage-report", required=True, help="coverage_report.json")
    parser.add_argument("--no-gpu-report", required=True, help="run_report_no_gpu.json")
    parser.add_argument("--gpu-report", required=True, help="run_report_gpu.json")
    parser.add_argument("--out-md", required=True, help="summary_diff.md")
    return parser.parse_args()


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def classify_failure(row: dict[str, Any]) -> str:
    result_status = str(row.get("result_status", "")).lower()

    if result_status == "fail_not_registered":
        return "fail_not_registered"
    if result_status == "fail_unreachable":
        return "fail_unreachable"
    if result_status == "fail_route_missing":
        return "fail_route_missing"
    if result_status == "unknown_timeout":
        return "unknown_timeout"
    if result_status == "unknown":
        return "unknown"
    return "other"


def compute_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {
            "exists": False,
            "case_count": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "failure_categories": {},
        }

    rows = report.get("results", [])
    success_status = {"pass_ok", "pass_reached"}
    success_count = sum(1 for r in rows if str(r.get("result_status")) in success_status)

    failure_categories: dict[str, int] = {}
    for row in rows:
        if str(row.get("result_status")) in success_status:
            continue
        cat = classify_failure(row)
        failure_categories[cat] = failure_categories.get(cat, 0) + 1

    return {
        "exists": True,
        "case_count": len(rows),
        "success_count": success_count,
        "success_rate": round(success_count / len(rows), 4) if rows else 0.0,
        "failure_categories": failure_categories,
    }


def build_markdown(
    coverage_report: dict[str, Any] | None,
    no_gpu_metrics: dict[str, Any],
    gpu_metrics: dict[str, Any],
    no_gpu_path: Path,
    gpu_path: Path,
) -> str:
    lines = [
        "# MolClaw Summary Diff",
        "",
        "## Coverage",
        "",
    ]

    if coverage_report is None:
        lines.append("- coverage_report.json not found")
    else:
        summary = coverage_report.get("summary", {})
        lines.extend(
            [
                f"- schema_tool_count: **{summary.get('schema_tool_count', 0)}**",
                f"- notebook_called_tool_count: **{summary.get('notebook_called_tool_count', 0)}**",
                f"- matched_schema_tested_count: **{summary.get('matched_schema_tested_count', 0)}**",
                f"- schema_untested_count: **{summary.get('schema_untested_count', 0)}**",
                f"- notebook_called_not_in_schema_count: **{summary.get('notebook_called_not_in_schema_count', 0)}**",
            ]
        )

    lines.extend([
        "",
        "## Worker Comparison",
        "",
        "| worker | report_exists | case_count | preflight_success_count | preflight_success_rate |",
        "|---|---:|---:|---:|---:|",
        f"| no-gpu | {1 if no_gpu_metrics['exists'] else 0} | {no_gpu_metrics['case_count']} | {no_gpu_metrics['success_count']} | {no_gpu_metrics['success_rate']:.2%} |",
        f"| gpu | {1 if gpu_metrics['exists'] else 0} | {gpu_metrics['case_count']} | {gpu_metrics['success_count']} | {gpu_metrics['success_rate']:.2%} |",
        "",
        "## Failure Categories",
        "",
        f"- no-gpu: `{json.dumps(no_gpu_metrics['failure_categories'], ensure_ascii=False)}`",
        f"- gpu: `{json.dumps(gpu_metrics['failure_categories'], ensure_ascii=False)}`",
        "",
        "## Notes",
        "",
        f"- no-gpu report path: `{no_gpu_path}`",
        f"- gpu report path: `{gpu_path}`",
    ])

    if not gpu_metrics["exists"]:
        lines.append("- gpu report is missing; run `gpu_worker/run_gpu_suite.sh` to complete comparison.")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    coverage_path = Path(args.coverage_report)
    no_gpu_path = Path(args.no_gpu_report)
    gpu_path = Path(args.gpu_report)
    out_md = Path(args.out_md)

    coverage_report = load_json_if_exists(coverage_path)
    no_gpu_report = load_json_if_exists(no_gpu_path)
    gpu_report = load_json_if_exists(gpu_path)

    no_gpu_metrics = compute_metrics(no_gpu_report)
    gpu_metrics = compute_metrics(gpu_report)

    markdown = build_markdown(
        coverage_report=coverage_report,
        no_gpu_metrics=no_gpu_metrics,
        gpu_metrics=gpu_metrics,
        no_gpu_path=no_gpu_path,
        gpu_path=gpu_path,
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown, encoding="utf-8")
    print(f"saved summary diff: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
