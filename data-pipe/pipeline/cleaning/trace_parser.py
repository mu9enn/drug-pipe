from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TASK_CHOICES = {"vs", "ac", "pf", "e2e", "kg"}
PUBLIC_QUESTION_KEYS = ("question", "question_text", "prompt", "public_question_text")
TRAINING_INPUT_DENY_KEYS = {
    "accepted", "answer", "answer_hit", "exact_match", "f1", "final_status",
    "ground_truth", "metrics", "precision", "recall", "reference_answer",
    "task_answer_valid", "top3", "top3_hit", "top10", "top10_hit",
}
SOURCE_LABEL_KEYS = TRAINING_INPUT_DENY_KEYS - {"accepted", "final_status"}


@dataclass(frozen=True)
class RolloutSample:
    run_dir: Path
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


def terminal_execution_findings(events: list[dict[str, Any]]) -> list[str]:
    """Report agent-level terminal failures without inspecting tool results."""
    terminal = next(
        (
            event
            for event in reversed(events)
            if str(event.get("type") or "").strip().lower() == "result"
        ),
        None,
    )
    if terminal is None:
        return []

    findings: list[str] = []
    if terminal.get("is_error") is True:
        findings.append("terminal_result_error")

    api_status = terminal.get("api_error_status")
    if api_status not in (None, ""):
        findings.append(f"terminal_api_error:{api_status}")

    subtype = str(terminal.get("subtype") or "").strip().lower()
    if subtype.startswith("error") or subtype.startswith("fail"):
        findings.append(f"terminal_subtype:{subtype}")

    reason = str(terminal.get("terminal_reason") or "").strip().lower()
    if reason.startswith(("abort", "error", "fail")):
        findings.append(f"terminal_reason:{reason}")
    return list(dict.fromkeys(findings))


def discover_run_dirs(results_root: Path) -> list[Path]:
    root = results_root.resolve()
    if (root / "run_config.json").is_file():
        return [root]
    return sorted({path.parent for path in root.rglob("run_config.json")})


def discover_rollout_samples(run_dir: Path) -> list[RolloutSample]:
    samples: list[RolloutSample] = []
    row_dirs = sorted(
        path for path in run_dir.iterdir()
        if path.is_dir() and path.name.startswith("row") and "_idx" in path.name
    )
    for row_dir in row_dirs:
        try:
            row_number = int(row_dir.name.split("_idx", 1)[0].removeprefix("row"))
        except ValueError:
            row_number = -1
        row_question = safe_load_json(row_dir / "question.json")
        dataset_index = str(row_question.get("dataset_index") or row_number)
        if (row_dir / "complete_session.jsonl").is_file() or (row_dir / "parsed_answer.json").is_file():
            samples.append(RolloutSample(run_dir, row_dir, row_dir, row_number, dataset_index, 1))
        for rollout_dir in sorted(
            path for path in row_dir.iterdir()
            if path.is_dir() and path.name.startswith("rollout")
        ):
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
                    run_dir, row_dir, rollout_dir, row_number,
                    str(question.get("dataset_index") or dataset_index), rollout_index,
                )
            )
    return sorted(samples, key=lambda item: (str(item.run_dir), item.row_number, item.rollout_index))


def infer_task(run_dir: Path) -> str:
    configured = str(safe_load_json(run_dir / "run_config.json").get("task") or "").strip().lower()
    if configured in TASK_CHOICES:
        return configured
    for task in ("vs", "ac", "pf", "e2e", "kg"):
        if (run_dir / "preds" / f"molbench_{task}" / f"molbench_{task}.json").is_file():
            return task
    raise FileNotFoundError(f"unable to infer task from {run_dir}")


def public_task_input(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): public_task_input(item)
            for key, item in value.items()
            if str(key).strip().lower() not in TRAINING_INPUT_DENY_KEYS and str(key) != "raw_question_json"
        }
    if isinstance(value, list):
        return [public_task_input(item) for item in value]
    return value


def raw_question_value(question: dict[str, Any]) -> Any:
    raw = question.get("raw_question_json")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def question_text(question: dict[str, Any]) -> str:
    for key in PUBLIC_QUESTION_KEYS:
        value = question.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raw = raw_question_value(question)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    projected = public_task_input(raw if isinstance(raw, dict) else question)
    if not isinstance(projected, dict) or not projected:
        return ""
    return json.dumps(projected, ensure_ascii=False, sort_keys=True)


def source_labels(question: dict[str, Any]) -> dict[str, Any]:
    labels = {
        str(key): value for key, value in question.items()
        if str(key).strip().lower() in SOURCE_LABEL_KEYS
    }
    raw = raw_question_value(question)
    if isinstance(raw, dict):
        for key, value in raw.items():
            if str(key).strip().lower() in SOURCE_LABEL_KEYS:
                labels.setdefault(str(key), value)
    return labels


def candidate_values(question: dict[str, Any]) -> Any:
    if isinstance(question.get("candidates"), list):
        return question["candidates"]
    raw = raw_question_value(question)
    return raw.get("candidates") if isinstance(raw, dict) else None
