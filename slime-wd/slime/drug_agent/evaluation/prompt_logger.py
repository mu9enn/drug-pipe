from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from drug_agent.utils import to_jsonable, utc_now_iso, write_json, write_jsonl


def _sample_entry(sample: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    env = metadata.get("env_kwargs") if isinstance(metadata.get("env_kwargs"), dict) else {}
    trace = metadata.get("drug_agent_trace") if isinstance(metadata.get("drug_agent_trace"), dict) else {}
    trace_copy = to_jsonable(trace)
    artifact_audit = trace_copy.pop("artifact_audit", {}) if isinstance(trace_copy, dict) else {}
    result = {
        "id": env.get("task_id"),
        "task_type": env.get("task_type"),
        "question": metadata.get("manual_prompt"),
        "done_reason": trace.get("done_reason"),
        "status": str(getattr(sample, "status", "")),
        "projected_final_answer": to_jsonable(trace.get("projected_final_answer")),
        "final_answer": to_jsonable(trace.get("final_answer")),
        "error": trace.get("error"),
        "num_steps": trace.get("num_steps"),
        "num_tool_success": trace.get("num_tool_success"),
        "num_tool_error": trace.get("num_tool_error"),
    }
    return result, trace_copy, artifact_audit


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    run_dir_raw = os.environ.get("DRUG_AGENT_EVAL_RUN_DIR", "").strip()
    if not run_dir_raw:
        raise RuntimeError("DRUG_AGENT_EVAL_RUN_DIR is required for prompt evaluation logging")
    run_dir = Path(run_dir_raw).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for dataset in data.values():
        for sample in dataset.get("samples") or []:
            result, trace, artifact_audit = _sample_entry(sample)
            results.append(result)
            traces.append({"id": result["id"], **trace})
            artifacts.append({"id": result["id"], **artifact_audit})
            if result.get("done_reason") != "final_answer":
                failures.append(result)

    write_jsonl(run_dir / "prompt_results.jsonl", results)
    write_jsonl(run_dir / "traces.jsonl", traces)
    write_jsonl(run_dir / "artifact_audit.jsonl", artifacts)
    write_jsonl(run_dir / "failures.jsonl", failures)
    if len(results) == 1:
        write_json(
            run_dir / "prompt_result.json",
            {
                "result": results[0],
                "trace": traces[0],
                "artifact_audit": artifacts[0],
            },
        )

    summary = {
        "rollout_id": rollout_id,
        "sample_count": len(results),
        "final_answer_count": len(results) - len(failures),
        "failure_count": len(failures),
        "extra_metrics": to_jsonable(extra_metrics or {}),
    }
    write_json(run_dir / "prompt_evaluation_summary.json", summary)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["completed_at"] = utc_now_iso()
        manifest["results"] = summary
        write_json(manifest_path, manifest)
    return True
