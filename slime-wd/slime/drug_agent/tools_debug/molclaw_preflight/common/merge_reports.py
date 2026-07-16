#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple run_molclaw_cases reports")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input report json files")
    parser.add_argument("--out-json", required=True, help="Merged output json")
    parser.add_argument("--out-md", required=True, help="Merged output markdown")
    parser.add_argument("--worker-mode", required=True, help="Worker mode label")
    return parser.parse_args()


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    success_status = {"pass_ok", "pass_reached"}
    success_count = sum(1 for r in results if r.get("result_status") in success_status)
    transport_ok_count = sum(1 for r in results if r.get("transport_ok"))

    tool_status_counts: dict[str, int] = {}
    result_status_counts: dict[str, int] = {}

    for row in results:
        t = str(row.get("tool_status", "unknown"))
        s = str(row.get("result_status", "unknown"))
        tool_status_counts[t] = tool_status_counts.get(t, 0) + 1
        result_status_counts[s] = result_status_counts.get(s, 0) + 1

    summary = {
        "case_count": total,
        "transport_ok_count": transport_ok_count,
        "success_count": success_count,
        "success_rate": round(success_count / total, 4) if total else 0.0,
        "preflight_success_count": success_count,
        "preflight_success_rate": round(success_count / total, 4) if total else 0.0,
        "tool_status_counts": tool_status_counts,
        "result_status_counts": result_status_counts,
    }
    return summary


def write_markdown(report: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Merged MolClaw Run Report",
        "",
        f"- worker_mode: `{report['meta']['worker_mode']}`",
        f"- generated_at: `{report['meta']['generated_at']}`",
        f"- input_reports: `{', '.join(report['meta']['input_reports'])}`",
        "",
        "## Summary",
        "",
        f"- case_count: **{report['summary']['case_count']}**",
        f"- transport_ok_count: **{report['summary']['transport_ok_count']}**",
        f"- preflight_success_count: **{report['summary']['preflight_success_count']}**",
        f"- preflight_success_rate: **{report['summary']['preflight_success_rate']:.2%}**",
        f"- tool_status_counts: `{json.dumps(report['summary']['tool_status_counts'], ensure_ascii=False)}`",
        f"- result_status_counts: `{json.dumps(report['summary']['result_status_counts'], ensure_ascii=False)}`",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    merged_results: list[dict[str, Any]] = []
    input_reports: list[str] = []
    generated_at = None

    for input_path in args.inputs:
        path = Path(input_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        merged_results.extend(data.get("results", []))
        input_reports.append(str(path))
        if generated_at is None:
            generated_at = data.get("meta", {}).get("generated_at")

    report = {
        "meta": {
            "worker_mode": args.worker_mode,
            "generated_at": generated_at,
            "input_reports": input_reports,
        },
        "summary": summarize(merged_results),
        "results": merged_results,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, out_md)

    print(f"saved merged json: {out_json}")
    print(f"saved merged md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
