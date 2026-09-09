from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from pipeline.cleaning.invariants import validate_semantic_record
from pipeline.cleaning.io import base_manifest, write_json, write_jsonl, write_pretty_json
from pipeline.cleaning.semantic_builder import SemanticConstructionError, build_semantic_trajectory
from pipeline.cleaning.trace_parser import (
    RolloutSample,
    TASK_CHOICES,
    discover_rollout_samples,
    discover_run_dirs,
    infer_task,
    load_session_events,
    question_text,
    safe_load_json,
    source_labels,
    terminal_execution_findings,
)
from pipeline.output_contracts import CONTRACT_VERSION, normalize_final_answer, normalize_task_prompt, task_constraints


CAPTURE_HASH_KEYS = (
    "question_sha256",
    "user_prompt_sha256",
    "system_prompt_sha256",
    "selected_session_sha256",
    "source_dataset_sha256",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capture_binding_audit(sample: RolloutSample, run_meta: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    expected_files = {
        "question_sha256": sample.sample_dir / "question.json",
        "user_prompt_sha256": sample.sample_dir / "prompt.txt",
        "selected_session_sha256": sample.sample_dir / "complete_session.jsonl",
    }
    for key, path in expected_files.items():
        expected = str(run_meta.get(key) or "")
        checks[key] = bool(expected) and path.is_file() and _sha256(path) == expected

    run_config = safe_load_json(sample.run_dir / "run_config.json")
    for key in ("system_prompt_sha256", "source_dataset_sha256"):
        expected = str(run_meta.get(key) or "")
        configured = str(run_config.get(key) or "")
        checks[key] = bool(expected) and bool(configured) and configured == expected

    complete = all(str(run_meta.get(key) or "") for key in CAPTURE_HASH_KEYS)
    selected_manifest = safe_load_json(sample.sample_dir / "selected_attempt_artifacts.json")
    for key in CAPTURE_HASH_KEYS:
        expected = str(run_meta.get(key) or "")
        selected = str(selected_manifest.get(key) or "")
        checks[f"selected_manifest:{key}"] = bool(expected) and selected == expected
    return {
        "status": "complete" if complete else "missing_required_hashes",
        "checks": checks,
        "ok": complete and all(checks.values()),
    }


def clean_sample(
    sample: RolloutSample,
    *,
    default_task: str,
    max_observation_chars: int = 6000,
) -> dict[str, Any]:
    question = safe_load_json(sample.sample_dir / "question.json") or safe_load_json(sample.row_dir / "question.json")
    run_meta = safe_load_json(sample.sample_dir / "run_meta.json")
    parsed = safe_load_json(sample.sample_dir / "parsed_answer.json")
    task = str(question.get("task") or run_meta.get("task") or default_task).strip().lower()
    if task not in TASK_CHOICES:
        task = default_task
    session_path = sample.sample_dir / "complete_session.jsonl"
    events, malformed_line_count, runner_error = load_session_events(session_path)
    execution_reasons: list[str] = []
    if not session_path.is_file():
        execution_reasons.append("missing_session")
    if run_meta.get("return_code") not in (None, 0):
        execution_reasons.append(f"runner_nonzero_rc:{run_meta.get('return_code')}")
    if run_meta.get("timed_out") is True:
        execution_reasons.append("timeout")
    if runner_error:
        execution_reasons.append("runner_error_last_line")
    execution_reasons.extend(terminal_execution_findings(events))
    capture_binding = _capture_binding_audit(sample, run_meta)
    if not capture_binding["ok"]:
        execution_reasons.append("capture_hash_binding_mismatch")

    sample_key = f"{task}:{sample.row_number}:{sample.dataset_index}:{sample.rollout_index}:{session_path.resolve()}"
    record_id = f"semantic_{task}_{hashlib.sha256(sample_key.encode()).hexdigest()[:16]}"
    audit: dict[str, Any] = {
        "id": record_id,
        "task": task,
        "task_id": f"{task}_row{sample.row_number:04d}_idx{sample.dataset_index}_r{sample.rollout_index:04d}",
        "source_session": str(session_path.resolve()),
        "session_event_count": len(events),
        "malformed_session_line_count": malformed_line_count,
        "execution_invalid_reasons": list(dict.fromkeys(execution_reasons)),
        "parsed_answer_present": (sample.sample_dir / "parsed_answer.json").is_file(),
        "parsed_answer_audit": {"parse_error": parsed.get("parse_error"), "parse_source": parsed.get("parse_source")},
        "source_labels": source_labels(question),
        "capture_binding": capture_binding,
    }
    if execution_reasons or malformed_line_count:
        audit.update(status="rejected", reasons=["execution_invalid"] if execution_reasons else ["malformed_session"])
        return {"semantic": None, "audit": audit}

    workspace = Path(str(run_meta.get("selected_attempt_workdir") or sample.sample_dir)).resolve()
    session_sha256 = _sha256(session_path)
    try:
        semantic, trace = build_semantic_trajectory(
            events,
            record_id=record_id,
            user_task=question_text(question),
            workspace=workspace,
            task_type=task,
            source_session_sha256=session_sha256,
            max_observation_chars=max_observation_chars,
        )
    except (SemanticConstructionError, ValueError) as exc:
        audit.update(status="rejected", reasons=["semantic_construction_failed"], error=str(exc))
        return {"semantic": None, "audit": audit}

    try:
        from pipeline.benchmark_release import require_training_task
        require_training_task(semantic["user_task"], task, source_task_ids=tuple(semantic.get("metadata", {}).get("source_task_ids", [])))
        # An invalid final is a recovery candidate, not an execution failure.
        from pipeline.cleaning.answer_recovery import recover_answer
        semantic, recovery = recover_answer(semantic)
        audit['answer_recovery'] = recovery
        if recovery['status'] == 'quarantined':
            raise ValueError(recovery['reason'])
        expected_task = normalize_task_prompt(semantic["user_task"], task)
        if semantic["user_task"] != expected_task:
            raise ValueError(
                "source prompt does not use the canonical answer contract; "
                "collect the trajectory with the current Claude runner"
            )
        final_event = next(
            event for event in semantic["events"] if event.get("final_answer") is not None
        )
        if recovery['status'] != 'pending':
            final_event["final_answer"] = normalize_final_answer(final_event["final_answer"], task, constraints=task_constraints(semantic["user_task"], task))
    except (StopIteration, ValueError) as exc:
        audit.update(status="rejected", reasons=["answer_contract_invalid"], error=str(exc))
        return {"semantic": None, "audit": audit}
    semantic.setdefault("metadata", {})["answer_contract_version"] = CONTRACT_VERSION
    invariants = validate_semantic_record(semantic)
    if not invariants["ok"]:
        raise RuntimeError(f"semantic builder violated invariants: {invariants['errors']}")
    audit.update(status="semantic_valid", reasons=[], trace=trace, semantic_invariants=invariants)
    return {"semantic": semantic, "audit": audit}


def python_clean(
    results_root: Path,
    output_root: Path,
    *,
    max_observation_chars: int = 6000,
) -> dict[str, Any]:
    results_root = results_root.resolve()
    output_root = output_root.resolve()
    run_dirs = discover_run_dirs(results_root)
    if not run_dirs:
        raise FileNotFoundError(f"no run_config.json found under {results_root}")
    processed: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        task = infer_task(run_dir)
        for sample in discover_rollout_samples(run_dir):
            processed.append(
                clean_sample(
                    sample,
                    default_task=task,
                    max_observation_chars=max_observation_chars,
                )
            )
    valid = [item for item in processed if item["semantic"] is not None]
    rejected = [item["audit"] for item in processed if item["semantic"] is None]
    outputs = {
        "semantic": output_root / "semantic_trajectories.jsonl",
        "semantic_pretty": output_root / "semantic_trajectories.pretty.json",
        "python_audit": output_root / "python_audit.jsonl",
        "rejected": output_root / "rejected.jsonl",
        "run_manifest": output_root / "run_manifest.json",
    }
    write_jsonl(outputs["semantic"], [item["semantic"] for item in valid])
    write_pretty_json(outputs["semantic_pretty"], [item["semantic"] for item in valid])
    write_jsonl(outputs["python_audit"], [item["audit"] for item in processed])
    write_jsonl(outputs["rejected"], rejected)
    manifest = {
        **base_manifest(step="semantic_python_clean", source=results_root, repo_root=Path(__file__).resolve().parents[3]),
        "run_dirs": [str(path) for path in run_dirs],
        "input_count": len(processed),
        "semantic_valid_count": len(valid),
        "rejected_count": len(rejected),
        "path_contract": "trajectory_source_aware_v1",
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    write_json(outputs["run_manifest"], manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-observation-chars", default=6000, type=int)
    args = parser.parse_args()
    manifest = python_clean(
        args.results_root,
        args.output_root,
        max_observation_chars=args.max_observation_chars,
    )
    print(manifest)


if __name__ == "__main__":
    main()
