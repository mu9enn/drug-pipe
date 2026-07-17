from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from evaluate.task_evaluator import evaluate_task_answer, load_chemistry_module  # noqa: E402


SCHEMA_VERSION = "react_trajectory_v2"
TASK_CHOICES = {"vs", "ac", "pf", "e2e", "kg"}
MOLCLAW_PREFIXES = ("mcp__molclaw-scp__", "mcp__molclaw-vs__")
ROLLOUT_RE = re.compile(r"rollout(\d+)$")
SYSTEM_PROMPT = (
    "You are a scientific agent. Use real MolClaw tools, reason before each decision, "
    "ground conclusions in tool observations, and return a concise final answer."
)


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
        if (row_dir / "parsed_answer.json").is_file():
            samples.append(RolloutSample(row_dir, row_dir, row_number, dataset_index, 1))
            continue
        for rollout_dir in sorted(path for path in row_dir.iterdir() if path.is_dir() and path.name.startswith("rollout")):
            if not (rollout_dir / "parsed_answer.json").is_file():
                continue
            match = ROLLOUT_RE.search(rollout_dir.name)
            rollout_index = int(match.group(1)) if match else len(samples) + 1
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


def _serialize_payload(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tool_call_tag(name: str, arguments: Any, call_id: str) -> str:
    payload = {"name": name, "arguments": arguments if isinstance(arguments, dict) else {}, "id": call_id}
    return f"<tool_call>{_serialize_payload(payload)}</tool_call>"


def _observation_tag(name: str, content: Any, is_error: bool) -> str:
    payload = {"tool_name": name, "content": content, "is_error": bool(is_error)}
    return f'<observation tool_name="{name}">{_serialize_payload(payload)}</observation>'


def reconstruct_react_messages(
    events: list[dict[str, Any]],
    *,
    question_text: str,
    final_answer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT, "step_loss_mask": 0},
        {"role": "user", "content": question_text, "step_loss_mask": 0},
    ]
    retained_calls: dict[str, str] = {}
    observed_calls: set[str] = set()
    tool_hist: Counter[str] = Counter()
    dropped_non_molclaw_calls = 0

    def append_turn(turn: dict[str, Any]) -> None:
        if len(messages) > 2 and messages[-1]["role"] == turn["role"]:
            messages[-1]["content"] = f"{messages[-1]['content']}\n{turn['content']}"
            return
        messages.append(turn)

    for event in events:
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if event.get("type") == "assistant":
            thought_parts: list[str] = []
            call_parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "")
                if item_type in {"thinking", "text"}:
                    value = item.get("thinking") if item_type == "thinking" else item.get("text")
                    if isinstance(value, str) and value.strip():
                        thought_parts.append(value.strip())
                elif item_type == "tool_use":
                    name = str(item.get("name") or "")
                    call_id = str(item.get("id") or "")
                    if not name.startswith(MOLCLAW_PREFIXES):
                        dropped_non_molclaw_calls += 1
                        continue
                    retained_calls[call_id] = name
                    tool_hist[name] += 1
                    call_parts.append(_tool_call_tag(name, item.get("input"), call_id))
            if call_parts:
                thought = "\n".join(thought_parts).strip()
                rendered = (f"<thought>{thought}</thought>\n" if thought else "") + "\n".join(call_parts)
                append_turn({"role": "assistant", "content": rendered, "step_loss_mask": 1})
        elif event.get("type") == "user":
            observations: list[str] = []
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "tool_result":
                    continue
                call_id = str(item.get("tool_use_id") or "")
                name = retained_calls.get(call_id)
                if not name:
                    continue
                observed_calls.add(call_id)
                observations.append(_observation_tag(name, item.get("content"), bool(item.get("is_error"))))
            if observations:
                append_turn({"role": "user", "content": "\n".join(observations), "step_loss_mask": 0})

    final_payload = final_answer if isinstance(final_answer, (dict, list)) else {"answer": final_answer}
    messages.append(
        {
            "role": "assistant",
            "content": f"<final_answer>{_serialize_payload(final_payload)}</final_answer>",
            "step_loss_mask": 1,
        }
    )
    return messages, {
        "molclaw_usage_count": len(retained_calls),
        "molclaw_usage_computation_count": 1,
        "tool_name_hist": dict(tool_hist),
        "observed_tool_call_count": len(observed_calls),
        "missing_observation_count": len(set(retained_calls) - observed_calls),
        "dropped_non_molclaw_call_count": dropped_non_molclaw_calls,
    }


def curate_sample(
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

    evaluation = evaluate_task_answer(
        task,
        prediction=final_answer,
        ground_truth=question.get("answer"),
        candidates=_candidate_values(question),
        chemistry=chemistry,
        parse_error=parsed.get("parse_error"),
    )
    messages, trace_stats = reconstruct_react_messages(
        events,
        question_text=_question_text(question),
        final_answer=final_answer,
    )

    return_code = run_meta.get("return_code")
    timed_out = bool(parsed.get("timed_out") or run_meta.get("timed_out"))
    execution_reasons: list[str] = []
    if not session_path.is_file():
        execution_reasons.append("missing_session")
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
    if final_answer in (None, "", []):
        training_reasons.append("missing_final_answer")
    training_trace_valid = not training_reasons
    accepted = execution_valid and evaluation["task_answer_valid"] and training_trace_valid
    sample_key = f"{task}:{sample.row_number}:{sample.dataset_index}:{sample.rollout_index}:{session_path.resolve()}"
    record_id = f"react_{task}_{hashlib.sha256(sample_key.encode()).hexdigest()[:16]}"
    return {
        "schema_version": SCHEMA_VERSION,
        "id": record_id,
        "messages": messages,
        "status": {
            "accepted": accepted,
            "execution_valid": execution_valid,
            "task_answer_valid": evaluation["task_answer_valid"],
            "training_trace_valid": training_trace_valid,
        },
        "metrics": evaluation["metrics"],
        "metadata": {
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
        },
    }


def curate_results_dir(results_dir: Path, task: str | None = None) -> dict[str, Any]:
    results_dir = results_dir.resolve()
    task_name = infer_task(results_dir, task)
    chemistry, chemistry_error = load_chemistry_module()
    records = [
        curate_sample(sample, default_task=task_name, chemistry=chemistry)
        for sample in discover_rollout_samples(results_dir)
    ]
    accepted = [record for record in records if record["status"]["accepted"]]
    rejected = [record for record in records if not record["status"]["accepted"]]
    output_dir = results_dir / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "react_trajectories": output_dir / "react_trajectories.jsonl",
        "rejected": output_dir / "react_rejected.jsonl",
        "trajectory_level": output_dir / "trajectory_level.jsonl",
        "accepted": output_dir / "accepted.jsonl",
    }
    for name, path in outputs.items():
        rows = accepted if name in {"react_trajectories", "accepted"} else rejected if name == "rejected" else records
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "task": task_name,
        "results_dir": str(results_dir),
        "input_count": len(records),
        "output_count": len(accepted),
        "rejected_count": len(rejected),
        "chemistry_available": chemistry is not None,
        "chemistry_error": chemistry_error,
        "authority": {
            "execution_valid": "trace_curator",
            "task_answer_valid": "task_evaluator",
            "training_trace_valid": "trace_curator",
            "accepted": "trace_curator",
            "molclaw_usage": "trace_curator",
            "react_messages": "trace_curator",
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
