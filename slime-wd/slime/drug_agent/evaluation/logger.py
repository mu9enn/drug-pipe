from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from drug_agent.evaluation.official_eval import run_official_evaluation
from drug_agent.evaluation.task_store import RUN_FINGERPRINT_ENV, load_records
from drug_agent.utils import to_jsonable, utc_now_iso, write_json, write_jsonl


def _entry(sample: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    env = metadata.get("env_kwargs") if isinstance(metadata.get("env_kwargs"), dict) else {}
    bench = metadata.get("benchmark") if isinstance(metadata.get("benchmark"), dict) else {}
    label = sample.label if isinstance(getattr(sample, "label", None), dict) else {}
    trace = metadata.get("drug_agent_trace") if isinstance(metadata.get("drug_agent_trace"), dict) else {}
    trace_copy = to_jsonable(trace)
    artifact_audit = trace_copy.pop("artifact_audit", {}) if isinstance(trace_copy, dict) else {}
    # A partial or failed trajectory is not a benchmark answer even if a
    # stale/intermediate projection happens to be present in metadata.
    prediction = trace.get("projected_final_answer") if trace.get("done_reason") == "final_answer" else None
    common = {
        "id": env.get("task_id"),
        "task_type": env.get("task_type"),
        "suite": bench.get("suite"),
        "subtask": bench.get("subtask"),
        "prediction": to_jsonable(prediction),
        "label": to_jsonable(label),
        "done_reason": trace.get("done_reason"),
        "error": trace.get("error"),
        "status": str(getattr(sample, "status", "")),
    }
    return common, trace_copy, artifact_audit


def _prediction_payload(common: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    suite = common["suite"]
    subtask = str(common.get("subtask") or "all")
    label = common.get("label") if isinstance(common.get("label"), dict) else {}
    prediction = common.get("prediction")
    if suite == "molbench_ms1":
        values = prediction if isinstance(prediction, list) else []
        return "rdkit_bench", "all", {"gt": label.get("answer", ""), "json_results": {"output": "\n".join(map(str, values))}}
    if suite == "molbench_ms2":
        return "acnet_curated", "all", {
            "gt": label.get("answer", ""), "s1": label.get("s1", ""), "s2": label.get("s2", ""),
            "json_results": {"output": prediction if isinstance(prediction, str) else ""},
        }
    if suite == "molbench_ms3":
        values = prediction if isinstance(prediction, list) else []
        return "molbench_vs", "all", {
            "index": label.get("index"), "answer": label.get("answer", []), "candidates": label.get("candidates", []),
            "json_results": {"ranking": values, "top3": values[:3]},
        }
    if suite == "molbench_mo_edit":
        return "molbench-mo-edit", subtask, {
            "molecule": label.get("source_smiles", ""), "added_group": label.get("added_group", ""),
            "removed_group": label.get("removed_group", ""),
            "json_results": {"output": prediction if isinstance(prediction, str) else ""},
        }
    if suite == "molbench_mo_opt":
        return "molbench-mo-opt", subtask, {
            "src_smiles": label.get("source_smiles", ""), "prop": subtask,
            "json_results": {"Final Target Molecule": prediction if isinstance(prediction, str) else ""},
        }
    raise ValueError(f"Unsupported benchmark suite in sample metadata: {suite}")


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    run_dir_raw = os.environ.get("DRUG_AGENT_EVAL_RUN_DIR", "").strip()
    molbench_root = os.environ.get("MOLBENCH_ROOT", "").strip()
    if not run_dir_raw or not molbench_root:
        raise RuntimeError("DRUG_AGENT_EVAL_RUN_DIR and MOLBENCH_ROOT are required for MolBench logging")
    run_dir = Path(run_dir_raw).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for dataset in data.values():
        for sample in dataset.get("samples") or []:
            common, trace, artifact_audit = _entry(sample)
            predictions.append(common)
            traces.append({"id": common["id"], **trace})
            artifacts.append({"id": common["id"], **artifact_audit})
            group, subtask, pred = _prediction_payload(common)
            pred["id"] = common["id"]
            grouped[(group, subtask)].append(pred)
            if common.get("done_reason") != "final_answer":
                failures.append(common)

    run_fingerprint = os.environ.get(RUN_FINGERPRINT_ENV, "").strip()
    if run_fingerprint:
        checkpoint_ids = {record.get("id") for record in load_records(run_dir, run_fingerprint=run_fingerprint)}
        result_ids = {row.get("id") for row in predictions}
        if checkpoint_ids != result_ids or len(result_ids) != len(predictions):
            raise RuntimeError(
                "final evaluation data does not match atomically checkpointed tasks: "
                f"results={len(predictions)}, checkpoints={len(checkpoint_ids)}, "
                f"missing_checkpoints={sorted(result_ids - checkpoint_ids)}, "
                f"missing_results={sorted(checkpoint_ids - result_ids)}"
            )

    write_jsonl(run_dir / "predictions.jsonl", predictions)
    write_jsonl(run_dir / "traces.jsonl", traces)
    write_jsonl(run_dir / "artifact_audit.jsonl", artifacts)
    write_jsonl(run_dir / "failures.jsonl", failures)
    for (group, subtask), rows in grouped.items():
        write_json(run_dir / "preds" / group / f"{subtask}.json", rows)

    inference_only = os.environ.get("DRUG_AGENT_EVAL_INFERENCE_ONLY", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    metrics = {} if inference_only else run_official_evaluation(run_dir, molbench_root)
    summary = {
        "rollout_id": rollout_id,
        "sample_count": len(predictions),
        "final_answer_count": len(predictions) - len(failures),
        "failure_count": len(failures),
        "metric_groups": sorted(metrics),
        "metrics_deferred": inference_only,
        "extra_metrics": to_jsonable(extra_metrics or {}),
    }
    write_json(run_dir / "evaluation_summary.json", summary)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["completed_at"] = utc_now_iso()
        manifest["results"] = {
            "sample_count": len(predictions),
            "final_answer_count": len(predictions) - len(failures),
            "failure_count": len(failures),
            "metric_group_count": len(metrics),
        }
        write_json(manifest_path, manifest)
    return True
