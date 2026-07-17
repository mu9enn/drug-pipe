from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from evaluate.task_evaluator import evaluate_task_answer, load_chemistry_module  # noqa: E402
from cleaning.acceptance_gate import decide_final_status  # noqa: E402
from cleaning.hard_cleaner import hard_clean  # noqa: E402
from cleaning.llm_cleaner import RewriteFunction, clean_with_llm  # noqa: E402
from postprocess.react_constructor import (  # noqa: E402
    CANONICAL_SYSTEM_PROMPT,
    reconstruct_react_messages,
)


SCHEMA_VERSION = "drug_agent_sft_react_json_v1"
TASK_CHOICES = {"vs", "ac", "pf", "e2e", "kg"}
SYSTEM_PROMPT = CANONICAL_SYSTEM_PROMPT


@dataclass(frozen=True)
class RolloutSample:
    row_dir: Path
    sample_dir: Path
    row_number: int
    dataset_index: str
    rollout_index: int


def safe_load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def load_session_events(path: Path) -> tuple[list[dict[str, Any]], int, bool]:
    if not path.is_file():
        return [], 0, False
    events: list[dict[str, Any]] = []
    malformed = 0
    last_nonempty = ""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            last_nonempty = text
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(value, dict):
                value["_line_no"] = line_number
                events.append(value)
            else:
                malformed += 1
    return events, malformed, last_nonempty.startswith("[runner-error]")


def discover_rollout_samples(results_dir: Path) -> list[RolloutSample]:
    samples: list[RolloutSample] = []
    row_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir() and path.name.startswith("row") and "_idx" in path.name)
    for row_dir in row_dirs:
        try:
            row_number = int(row_dir.name.split("_idx", 1)[0].removeprefix("row"))
        except ValueError:
            row_number = -1
        row_question = safe_load_json(row_dir / "question.json")
        dataset_index = str(row_question.get("dataset_index") or row_number)
        if (row_dir / "complete_session.jsonl").is_file() or (row_dir / "parsed_answer.json").is_file():
            samples.append(RolloutSample(row_dir, row_dir, row_number, dataset_index, 1))
        for rollout_dir in sorted(path for path in row_dir.iterdir() if path.is_dir() and path.name.startswith("rollout")):
            if not any(
                (rollout_dir / marker).is_file()
                for marker in ("complete_session.jsonl", "parsed_answer.json", "run_meta.json", "question.json")
            ):
                continue
            suffix = rollout_dir.name.removeprefix("rollout")
            rollout_index = int(suffix) if suffix.isdigit() else len(samples) + 1
            question = safe_load_json(rollout_dir / "question.json")
            samples.append(
                RolloutSample(
                    row_dir,
                    rollout_dir,
                    row_number,
                    str(question.get("dataset_index") or dataset_index),
                    rollout_index,
                )
            )
    return sorted(samples, key=lambda item: (item.row_number, item.rollout_index, item.sample_dir.name))


def infer_task(results_dir: Path, explicit_task: str | None = None) -> str:
    if explicit_task:
        value = explicit_task.strip().lower()
        if value in TASK_CHOICES:
            return value
        raise ValueError(f"unsupported task: {explicit_task}")
    configured = str(safe_load_json(results_dir / "run_config.json").get("task") or "").strip().lower()
    if configured in TASK_CHOICES:
        return configured
    for task in ("vs", "ac", "pf", "e2e", "kg"):
        if (results_dir / "preds" / f"molbench_{task}" / f"molbench_{task}.json").is_file():
            return task
    raise FileNotFoundError(f"unable to infer task from {results_dir}")


def _question_text(question: dict[str, Any]) -> str:
    for key in ("question", "question_text", "prompt", "public_question_text"):
        value = question.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raw = question.get("raw_question_json")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return json.dumps(question, ensure_ascii=False, sort_keys=True)


def _candidate_values(question: dict[str, Any]) -> Any:
    if isinstance(question.get("candidates"), list):
        return question["candidates"]
    raw = question.get("raw_question_json")
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except Exception:
            value = {}
        if isinstance(value, dict):
            return value.get("candidates")
    return None


def curate_sample(
    sample: RolloutSample,
    *,
    default_task: str,
    chemistry: Any | None,
    llm_rewrite: RewriteFunction | None = None,
    llm_clean_required: bool = False,
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

    messages, trace_stats = reconstruct_react_messages(
        events,
        question_text=_question_text(question),
        final_answer=final_answer,
        task=task,
    )
    resolved_final_answer = trace_stats["resolved_final_answer"]
    evaluator_error = ""
    try:
        evaluation = evaluate_task_answer(
            task,
            prediction=resolved_final_answer,
            ground_truth=question.get("answer"),
            candidates=_candidate_values(question),
            chemistry=chemistry,
            parse_error=parsed.get("parse_error"),
            task_contract=(
                question.get("evaluation")
                if isinstance(question.get("evaluation"), dict)
                else question.get("task_contract")
                if isinstance(question.get("task_contract"), dict)
                else {}
            ),
            execution_evidence={
                "molclaw_usage_count": trace_stats["molclaw_usage_count"],
                "observation_count": trace_stats["observed_tool_call_count"],
            },
        )
    except RuntimeError as exc:
        evaluator_error = str(exc)
        evaluation = {
            "task_answer_valid": False,
            "invalid_reasons": [f"evaluator_error:{exc}"],
            "metrics": {},
            "audit": {"evaluator_error": str(exc)},
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
    execution_valid = not execution_reasons

    training_reasons: list[str] = []
    if trace_stats["molclaw_usage_count"] <= 0:
        training_reasons.append("missing_molclaw_usage")
    if trace_stats["missing_observation_count"]:
        training_reasons.append(f"missing_tool_observations:{trace_stats['missing_observation_count']}")
    if resolved_final_answer in (None, "", []):
        training_reasons.append("missing_final_answer")
    training_trace_valid = not training_reasons
    sample_key = f"{task}:{sample.row_number}:{sample.dataset_index}:{sample.rollout_index}:{session_path.resolve()}"
    record_id = f"react_{task}_{hashlib.sha256(sample_key.encode()).hexdigest()[:16]}"
    training_record = {
        "schema_version": SCHEMA_VERSION,
        "id": record_id,
        "messages": messages,
    }
    llm_cleaned, llm_report = clean_with_llm(training_record, llm_rewrite)
    hard_cleaned, hard_report = hard_clean(llm_cleaned)
    if trace_stats["error_status_conflicts"]:
        hard_report["errors"].append("source_error_status_conflict")
        hard_report["errors"] = list(dict.fromkeys(hard_report["errors"]))
    decision = decide_final_status(
        execution_valid=execution_valid,
        task_answer_valid=bool(evaluation["task_answer_valid"]),
        training_trace_valid=training_trace_valid,
        llm_clean_status=str(llm_report["status"]),
        llm_clean_findings=list(llm_report["findings"]),
        hard_clean_findings=list(hard_report["errors"]),
        llm_clean_required=llm_clean_required,
    )
    return {
        "training_record": hard_cleaned,
        "audit": {
            "id": record_id,
            "final_status": decision["final_status"],
            "final_status_authority": decision["authority"],
            "final_status_reasons": decision["reasons"],
            "execution_valid": execution_valid,
            "task_answer_valid": evaluation["task_answer_valid"],
            "training_trace_valid": training_trace_valid,
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
            "task_evaluator_error": evaluator_error,
            "llm_clean": llm_report,
            "hard_clean": hard_report,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def curate_results_dir(
    results_dir: Path,
    task: str | None = None,
    *,
    llm_rewrite: RewriteFunction | None = None,
    llm_clean_required: bool = False,
) -> dict[str, Any]:
    results_dir = results_dir.resolve()
    task_name = infer_task(results_dir, task)
    chemistry, chemistry_error = load_chemistry_module()
    records = [
        curate_sample(
            sample,
            default_task=task_name,
            chemistry=chemistry,
            llm_rewrite=llm_rewrite,
            llm_clean_required=llm_clean_required,
        )
        for sample in discover_rollout_samples(results_dir)
    ]
    accepted = [record for record in records if record["audit"]["final_status"] == "accepted"]
    rejected = [record["audit"] for record in records if record["audit"]["final_status"] == "rejected"]
    quarantine = [record["audit"] for record in records if record["audit"]["final_status"] == "quarantine"]
    audits = [record["audit"] for record in records]
    output_dir = results_dir / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "react_trajectories": output_dir / "react_trajectories.jsonl",
        "curation_audit": output_dir / "curation_audit.jsonl",
        "rejected": output_dir / "rejected.jsonl",
    }
    _write_jsonl(outputs["react_trajectories"], [record["training_record"] for record in accepted])
    _write_jsonl(outputs["curation_audit"], audits)
    _write_jsonl(outputs["rejected"], rejected)
    quarantine_path = output_dir / "quarantine.jsonl"
    if quarantine:
        _write_jsonl(quarantine_path, quarantine)
        outputs["quarantine"] = quarantine_path
    elif quarantine_path.exists():
        quarantine_path.unlink()

    summary = {
        "schema_version": SCHEMA_VERSION,
        "task": task_name,
        "results_dir": str(results_dir),
        "input_count": len(records),
        "output_count": len(accepted),
        "rejected_count": len(rejected),
        "quarantine_count": len(quarantine),
        "chemistry_available": chemistry is not None,
        "chemistry_error": chemistry_error,
        "authority": {
            "execution_valid": "trace_curator",
            "task_answer_valid": "task_evaluator",
            "training_trace_valid": "trace_curator",
            "final_status": "final_acceptance_gate",
            "molclaw_usage": "trace_curator",
            "react_messages": "react_constructor",
            "llm_clean": "llm_cleaner",
            "hard_clean_findings": "hard_cleaner",
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    (output_dir / "curation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
