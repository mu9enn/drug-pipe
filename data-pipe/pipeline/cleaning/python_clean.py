from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pipeline.cleaning.invariants import collect_repair_hints, validate_final_record
from pipeline.cleaning.io import base_manifest, write_json, write_jsonl
from pipeline.cleaning.models import REACT_SCHEMA_VERSION
from pipeline.cleaning.react_builder import reconstruct_react_messages
from pipeline.cleaning.trace_parser import (
    RolloutSample,
    TASK_CHOICES,
    candidate_values,
    discover_rollout_samples,
    discover_run_dirs,
    infer_task,
    load_session_events,
    question_text,
    safe_load_json,
    source_labels,
)
from pipeline.evaluate.task_evaluator import evaluate_task_answer, load_chemistry_module


def clean_sample(
    sample: RolloutSample,
    *,
    default_task: str,
    chemistry: Any | None,
) -> dict[str, Any]:
    question = safe_load_json(sample.sample_dir / "question.json") or safe_load_json(sample.row_dir / "question.json")
    parsed = safe_load_json(sample.sample_dir / "parsed_answer.json")
    run_meta = safe_load_json(sample.sample_dir / "run_meta.json")
    task = str(question.get("task") or run_meta.get("task") or default_task).strip().lower()
    if task not in TASK_CHOICES:
        task = default_task
    session_path = sample.sample_dir / "complete_session.jsonl"
    events, malformed_line_count, runner_error = load_session_events(session_path)
    final_answer = parsed.get("answer")
    if final_answer in (None, "", []):
        final_answer = parsed.get("answer_block")
    public_question = question_text(question)
    messages, trace_stats = reconstruct_react_messages(
        events,
        question_text=public_question,
        final_answer=final_answer,
        task=task,
    )

    evaluator_error = ""
    try:
        evaluation = evaluate_task_answer(
            task,
            prediction=trace_stats["resolved_final_answer"],
            ground_truth=question.get("answer"),
            candidates=candidate_values(question),
            chemistry=chemistry,
            parse_error=parsed.get("parse_error"),
            task_contract=(
                question.get("evaluation")
                if isinstance(question.get("evaluation"), dict)
                else question.get("task_contract")
                if isinstance(question.get("task_contract"), dict)
                else {}
            ),
        )
    except RuntimeError as exc:
        evaluator_error = str(exc)
        evaluation = {
            "task_answer_valid": False,
            "invalid_reasons": [f"evaluator_error:{exc}"],
            "metrics": {},
            "audit": {"evaluator_error": str(exc)},
            "canonical": {},
        }

    return_code = run_meta.get("return_code")
    timed_out = bool(parsed.get("timed_out") or run_meta.get("timed_out"))
    execution_reasons: list[str] = []
    if not session_path.is_file():
        execution_reasons.append("missing_session")
    if not (sample.sample_dir / "parsed_answer.json").is_file():
        execution_reasons.append("missing_parsed_answer")
    if return_code not in (None, 0):
        execution_reasons.append(f"runner_nonzero_rc:{return_code}")
    if timed_out:
        execution_reasons.append("timeout")
    if runner_error:
        execution_reasons.append("runner_error_last_line")
    if not events:
        execution_reasons.append("no_parseable_session_events")

    training_reasons: list[str] = []
    if trace_stats["molclaw_usage_count"] <= 0:
        training_reasons.append("missing_molclaw_usage")
    if trace_stats["missing_observation_count"]:
        training_reasons.append(f"missing_tool_observations:{trace_stats['missing_observation_count']}")
    if trace_stats["resolved_final_answer"] in (None, "", []):
        training_reasons.append("missing_final_answer")
    if not public_question:
        training_reasons.append("missing_task_prompt")

    sample_key = f"{task}:{sample.row_number}:{sample.dataset_index}:{sample.rollout_index}:{session_path.resolve()}"
    record_id = f"react_{task}_{hashlib.sha256(sample_key.encode()).hexdigest()[:16]}"
    draft = {"schema_version": REACT_SCHEMA_VERSION, "id": record_id, "messages": messages}
    invariant_report = validate_final_record(draft)
    if invariant_report["errors"]:
        training_reasons.extend(f"invariant:{finding}" for finding in invariant_report["errors"])
    python_reasons: list[str] = []
    if execution_reasons:
        python_reasons.append("execution_invalid")
    if not evaluation["task_answer_valid"]:
        python_reasons.append("task_answer_invalid")
    if training_reasons:
        python_reasons.append("training_trace_invalid")
    python_status = "rejected" if python_reasons else "python_valid"

    audit = {
        "id": record_id,
        "python_status": python_status,
        "python_status_authority": "python_clean",
        "python_status_reasons": list(dict.fromkeys(python_reasons)),
        "execution_valid": not execution_reasons,
        "task_answer_valid": bool(evaluation["task_answer_valid"]),
        "training_trace_valid": not training_reasons,
        "task": task,
        "task_id": f"{task}_row{sample.row_number:04d}_idx{sample.dataset_index}_r{sample.rollout_index:04d}",
        "source_session": str(session_path.resolve()),
        "execution_invalid_reasons": execution_reasons,
        "task_answer_invalid_reasons": evaluation["invalid_reasons"],
        "training_trace_invalid_reasons": training_reasons,
        "trace_stats": trace_stats,
        "session_event_count": len(events),
        "malformed_session_line_count": malformed_line_count,
        "return_code": return_code,
        "timed_out": timed_out,
        "task_metrics": evaluation["metrics"],
        "task_evaluator_audit": evaluation.get("audit", {}),
        "task_evaluator_canonical": evaluation.get("canonical", {}),
        "source_labels": source_labels(question),
        "task_evaluator_error": evaluator_error,
        "python_invariants": invariant_report,
        "repair_hints": collect_repair_hints(draft),
    }
    return {"draft": draft, "audit": audit}


def python_clean(results_root: Path, output_root: Path) -> dict[str, Any]:
    results_root = results_root.resolve()
    output_root = output_root.resolve()
    run_dirs = discover_run_dirs(results_root)
    if not run_dirs:
        raise FileNotFoundError(f"no run_config.json found under {results_root}")
    chemistry, chemistry_error = load_chemistry_module()
    processed: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        task = infer_task(run_dir)
        for sample in discover_rollout_samples(run_dir):
            processed.append(clean_sample(sample, default_task=task, chemistry=chemistry))

    valid = [item for item in processed if item["audit"]["python_status"] == "python_valid"]
    rejected = [item["audit"] for item in processed if item["audit"]["python_status"] == "rejected"]
    audits = [item["audit"] for item in processed]
    outputs = {
        "python_drafts": output_root / "python_drafts.jsonl",
        "python_audit": output_root / "python_audit.jsonl",
        "rejected": output_root / "rejected.jsonl",
        "run_manifest": output_root / "run_manifest.json",
    }
    write_jsonl(outputs["python_drafts"], [item["draft"] for item in valid])
    write_jsonl(outputs["python_audit"], audits)
    write_jsonl(outputs["rejected"], rejected)
    repo_root = Path(__file__).resolve().parents[3]
    manifest = {
        **base_manifest(step="python_clean", source=results_root, repo_root=repo_root),
        "run_dirs": [str(path) for path in run_dirs],
        "input_count": len(processed),
        "python_valid_count": len(valid),
        "rejected_count": len(rejected),
        "chemistry_available": chemistry is not None,
        "chemistry_error": chemistry_error,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    write_json(outputs["run_manifest"], manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: deterministically clean raw Drug-Pipe sessions once.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = python_clean(Path(args.results_root), Path(args.output_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
