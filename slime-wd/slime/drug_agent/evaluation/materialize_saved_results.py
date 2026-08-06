from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from drug_agent.evaluation.logger import _prediction_payload
from drug_agent.evaluation.official_eval import run_official_evaluation
from drug_agent.evaluation.task_store import load_records
from drug_agent.utils import to_jsonable, utc_now_iso, write_json, write_jsonl


def _record_entry(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    trace = record.get("trace") if isinstance(record.get("trace"), dict) else {}
    trace_copy = to_jsonable(trace)
    artifact_audit = trace_copy.pop("artifact_audit", {}) if isinstance(trace_copy, dict) else {}
    common = {
        "id": record.get("id"),
        "task_type": record.get("task_type"),
        "suite": record.get("suite"),
        "subtask": record.get("subtask"),
        "prediction": to_jsonable(
            trace.get("projected_final_answer") if trace.get("done_reason") == "final_answer" else None
        ),
        "label": to_jsonable(record.get("label") if isinstance(record.get("label"), dict) else {}),
        "done_reason": trace.get("done_reason"),
        "error": trace.get("error"),
        "status": record.get("status"),
        "saved_at": record.get("saved_at"),
    }
    return common, trace_copy, artifact_audit


def materialize_records(
    records: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path,
    molbench_root: str | Path,
    source_run_dir: str | Path,
    selected_suites: set[str] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected = [
        record
        for record in records
        if not selected_suites or str(record.get("suite") or "") in selected_suites
    ]
    if not selected:
        raise ValueError("no saved evaluation records matched the requested suites")

    predictions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        common, trace, artifact_audit = _record_entry(record)
        predictions.append(common)
        traces.append({"id": common["id"], **trace})
        artifacts.append({"id": common["id"], **artifact_audit})
        group, subtask, prediction = _prediction_payload(common)
        prediction["id"] = common["id"]
        grouped[(group, subtask)].append(prediction)
        if common.get("done_reason") != "final_answer":
            failures.append(common)

    write_jsonl(output_root / "predictions.jsonl", predictions)
    write_jsonl(output_root / "traces.jsonl", traces)
    write_jsonl(output_root / "artifact_audit.jsonl", artifacts)
    write_jsonl(output_root / "failures.jsonl", failures)
    for (group, subtask), rows in grouped.items():
        write_json(output_root / "preds" / group / f"{subtask}.json", rows)

    metrics = run_official_evaluation(output_root, molbench_root)
    by_suite = Counter(str(row.get("suite") or "unknown") for row in predictions)
    by_status = Counter(str(row.get("status") or "unknown") for row in predictions)
    by_done_reason = Counter(str(row.get("done_reason") or "unknown") for row in predictions)
    summary = {
        "schema_version": "drug_agent_saved_eval_materialization_v1",
        "created_at": utc_now_iso(),
        "source_run_dir": str(Path(source_run_dir).expanduser().resolve()),
        "output_dir": str(output_root),
        "selected_suites": sorted(selected_suites or set(by_suite)),
        "sample_count": len(predictions),
        "final_answer_count": len(predictions) - len(failures),
        "failure_count": len(failures),
        "by_suite": dict(sorted(by_suite.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_done_reason": dict(sorted(by_done_reason.items())),
        "metrics": metrics,
        "denominator_policy": "all selected saved tasks; non-final predictions are empty and score zero",
    }
    write_json(output_root / "evaluation_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize official MolBench metrics from atomically saved per-task results."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--molbench-root", type=Path, required=True)
    parser.add_argument("--suite", action="append", default=[])
    args = parser.parse_args()

    manifest_path = args.run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_fingerprint = manifest.get("resume_fingerprint")
    if not isinstance(run_fingerprint, str) or not run_fingerprint:
        raise ValueError("run manifest is missing resume_fingerprint")
    records = load_records(args.run_dir, run_fingerprint=run_fingerprint)
    summary = materialize_records(
        records,
        output_dir=args.output_dir,
        molbench_root=args.molbench_root,
        source_run_dir=args.run_dir,
        selected_suites=set(args.suite),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
