from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import fcntl

from drug_agent.utils import ensure_dir, to_jsonable, utc_now_iso


SCHEMA_VERSION = "drug_agent_eval_task_result_v1"
RUN_FINGERPRINT_ENV = "DRUG_AGENT_EVAL_RUN_FINGERPRINT"
RESUME_ENV = "DRUG_AGENT_EVAL_RESUME"
RETRY_NON_FINAL_ENV = "DRUG_AGENT_EVAL_RETRY_NON_FINAL"
EXPECTED_COUNT_ENV = "DRUG_AGENT_EVAL_EXPECTED_TASK_COUNT"

_STORE_LOCK = threading.Lock()
_BOUND_TASK_FINGERPRINT_KEY = "_drug_agent_eval_task_fingerprint"


def _canonical_json(payload: Any) -> str:
    return json.dumps(to_jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _run_dir() -> Path | None:
    raw = os.environ.get("DRUG_AGENT_EVAL_RUN_DIR", "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _run_fingerprint() -> str:
    return os.environ.get(RUN_FINGERPRINT_ENV, "").strip()


def _sample_identity(sample: Any) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    env = metadata.get("env_kwargs") if isinstance(metadata.get("env_kwargs"), dict) else {}
    benchmark = metadata.get("benchmark") if isinstance(metadata.get("benchmark"), dict) else {}
    return {
        "id": env.get("task_id"),
        "task_type": env.get("task_type"),
        "data_source": env.get("data_source"),
        "prompt": to_jsonable(getattr(sample, "prompt", None)),
        "label": to_jsonable(getattr(sample, "label", None)),
        "benchmark": to_jsonable(benchmark),
        "manual_prompt": metadata.get("manual_prompt"),
    }


def task_fingerprint(sample: Any) -> str:
    metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    bound = metadata.get(_BOUND_TASK_FINGERPRINT_KEY)
    if isinstance(bound, str) and bound:
        return bound
    return _sha256(_sample_identity(sample))


def bind_task_identity(sample: Any) -> str:
    """Freeze identity before generation mutates ``sample.prompt``.

    Slime replaces the source message list with the rendered prompt after a
    rollout.  Recovery starts from the original dataset row, so computing the
    identity only at checkpoint time would make every completed task appear to
    belong to a different input on resume.
    """

    if not isinstance(getattr(sample, "metadata", None), dict):
        sample.metadata = {}
    bound = sample.metadata.get(_BOUND_TASK_FINGERPRINT_KEY)
    if isinstance(bound, str) and bound:
        return bound
    fingerprint = _sha256(_sample_identity(sample))
    sample.metadata[_BOUND_TASK_FINGERPRINT_KEY] = fingerprint
    return fingerprint


def _task_id(sample: Any) -> str:
    value = _sample_identity(sample).get("id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("evaluation sample is missing metadata.env_kwargs.task_id")
    return value.strip()


def _record_path(run_dir: Path, task_id: str) -> Path:
    name = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return run_dir / "task_results" / f"{name}.json"


def _status_value(sample: Any) -> str:
    status = getattr(sample, "status", None)
    value = getattr(status, "value", status)
    return str(value or "pending")


def _record_from_sample(sample: Any, run_fingerprint: str) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    env = metadata.get("env_kwargs") if isinstance(metadata.get("env_kwargs"), dict) else {}
    benchmark = metadata.get("benchmark") if isinstance(metadata.get("benchmark"), dict) else {}
    trace = metadata.get("drug_agent_trace") if isinstance(metadata.get("drug_agent_trace"), dict) else {}
    if not trace:
        raise ValueError(f"evaluation sample {_task_id(sample)!r} has no drug_agent_trace to checkpoint")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "task_fingerprint": task_fingerprint(sample),
        "id": _task_id(sample),
        "sample_index": getattr(sample, "index", None),
        "task_type": env.get("task_type"),
        "data_source": env.get("data_source"),
        "suite": benchmark.get("suite"),
        "subtask": benchmark.get("subtask"),
        "status": _status_value(sample),
        "label": to_jsonable(getattr(sample, "label", None)),
        "response": to_jsonable(getattr(sample, "response", "")),
        "response_length": int(getattr(sample, "response_length", 0) or 0),
        "trace": to_jsonable(trace),
        "saved_at": utc_now_iso(),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        # Evaluation runs execute as root on GPU workers while official scoring
        # is intentionally performed by the owning user on the shared CPU
        # development host. NamedTemporaryFile defaults to 0600, so make the
        # published JSON artifact readable after the atomic rename.
        os.chmod(path, 0o644)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n")


def _load_record(path: Path, *, expected_run_fingerprint: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"invalid evaluation task checkpoint schema: {path}")
    if expected_run_fingerprint and payload.get("run_fingerprint") != expected_run_fingerprint:
        raise ValueError(f"evaluation task checkpoint belongs to a different run: {path}")
    return payload


@contextmanager
def _store_lock(run_dir: Path):
    """Serialize materialization across threads and Ray worker processes."""

    ensure_dir(run_dir)
    lock_path = run_dir / ".task_results.lock"
    with _STORE_LOCK:
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_records(run_dir: str | Path, *, run_fingerprint: str | None = None) -> list[dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve() / "task_results"
    if not root.is_dir():
        return []
    records = [_load_record(path, expected_run_fingerprint=run_fingerprint) for path in sorted(root.glob("*.json"))]
    ids = [record.get("id") for record in records]
    duplicates = [task_id for task_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate task ids in evaluation checkpoints: {duplicates}")
    return sorted(
        records,
        key=lambda record: (
            record.get("sample_index") is None,
            record.get("sample_index") if record.get("sample_index") is not None else 0,
            str(record.get("id") or ""),
        ),
    )


def _materialize_progress(run_dir: Path, run_fingerprint: str) -> None:
    records = load_records(run_dir, run_fingerprint=run_fingerprint)
    # Keep the aggregate view intentionally small.  Full traces remain in the
    # authoritative per-task files; rewriting every prior trace after every
    # completion would create quadratic I/O for long agent trajectories.
    partial_rows = []
    for record in records:
        trace = record.get("trace") if isinstance(record.get("trace"), dict) else {}
        partial_rows.append(
            {
                "id": record.get("id"),
                "sample_index": record.get("sample_index"),
                "task_type": record.get("task_type"),
                "data_source": record.get("data_source"),
                "suite": record.get("suite"),
                "subtask": record.get("subtask"),
                "status": record.get("status"),
                "done_reason": trace.get("done_reason"),
                "prediction": trace.get("projected_final_answer"),
                "error": trace.get("error"),
                "saved_at": record.get("saved_at"),
            }
        )
    lines = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in partial_rows)
    _atomic_write_text(run_dir / "partial_results.jsonl", lines)
    expected_raw = os.environ.get(EXPECTED_COUNT_ENV, "").strip()
    expected = int(expected_raw) if expected_raw else None
    by_suite = Counter(str(record.get("suite") or record.get("data_source") or "unknown") for record in records)
    by_done_reason = Counter(str((record.get("trace") or {}).get("done_reason") or "unknown") for record in records)
    by_status = Counter(str(record.get("status") or "unknown") for record in records)
    successful_final_count = sum(
        str(record.get("status") or "") == "completed"
        and str((record.get("trace") or {}).get("done_reason") or "") == "final_answer"
        for record in records
    )
    _atomic_write_json(
        run_dir / "progress.json",
        {
            "schema_version": "drug_agent_eval_progress_v1",
            "updated_at": utc_now_iso(),
            "run_fingerprint": run_fingerprint,
            "expected_count": expected,
            "checkpointed_count": len(records),
            "remaining_count": max(expected - len(records), 0) if expected is not None else None,
            "final_answer_count": by_done_reason.get("final_answer", 0),
            "non_final_count": len(records) - by_done_reason.get("final_answer", 0),
            "successful_final_count": successful_final_count,
            "retryable_non_final_count": len(records) - successful_final_count,
            "remaining_to_success_count": (
                max(expected - successful_final_count, 0) if expected is not None else None
            ),
            "by_suite": dict(sorted(by_suite.items())),
            "by_status": dict(sorted(by_status.items())),
            "by_done_reason": dict(sorted(by_done_reason.items())),
        },
    )


def checkpoint_sample(sample: Any, *, evaluation: bool) -> None:
    """Atomically persist one completed online-evaluation sample.

    The per-task file is the recovery authority.  The aggregate partial files
    are rebuilt from those immutable records and may lag by at most one task if
    a process dies between the two atomic renames.
    """

    if not evaluation:
        return
    run_dir = _run_dir()
    run_fingerprint = _run_fingerprint()
    if run_dir is None or not run_fingerprint:
        return
    record = _record_from_sample(sample, run_fingerprint)
    path = _record_path(run_dir, record["id"])
    with _store_lock(run_dir):
        if path.is_file():
            existing = _load_record(path, expected_run_fingerprint=run_fingerprint)
            if existing.get("task_fingerprint") != record["task_fingerprint"]:
                raise ValueError(f"refusing to overwrite mismatched task checkpoint for {record['id']!r}")
        _atomic_write_json(path, record)
        _materialize_progress(run_dir, run_fingerprint)


def restore_sample(sample: Any, *, evaluation: bool) -> Any | None:
    if not evaluation or not _truthy_env(RESUME_ENV):
        return None
    run_dir = _run_dir()
    run_fingerprint = _run_fingerprint()
    if run_dir is None or not run_fingerprint:
        raise RuntimeError("evaluation resume requires run directory and run fingerprint")
    task_id = _task_id(sample)
    path = _record_path(run_dir, task_id)
    if not path.is_file():
        return None
    record = _load_record(path, expected_run_fingerprint=run_fingerprint)
    if record.get("task_fingerprint") != task_fingerprint(sample):
        raise ValueError(f"evaluation task checkpoint fingerprint mismatch for {task_id!r}")
    trace = record.get("trace") if isinstance(record.get("trace"), dict) else {}
    retryable = trace.get("retryable") is not False
    if _truthy_env(RETRY_NON_FINAL_ENV) and retryable and not (
        str(record.get("status") or "") == "completed"
        and str(trace.get("done_reason") or "") == "final_answer"
    ):
        return None

    status_type = type(getattr(sample, "status", None))
    try:
        sample.status = status_type(record["status"])
    except Exception as exc:
        raise ValueError(f"invalid saved sample status for {task_id!r}: {record.get('status')!r}") from exc
    if not isinstance(getattr(sample, "metadata", None), dict):
        sample.metadata = {}
    sample.metadata["drug_agent_trace"] = to_jsonable(record["trace"])
    sample.metadata["drug_agent_eval_resume"] = {
        "restored": True,
        "checkpoint": str(path),
        "saved_at": record.get("saved_at"),
    }
    sample.response = record.get("response") or ""
    sample.response_length = int(record.get("response_length") or 0)
    return sample
