"""Immutable capture helpers for Claude CLI stream-json invocations."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


CLAUDE_CODE_EXECUTION_ENV = {
    "CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY": "2",
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
}

HTTP_500_RE = re.compile(r"\b(?:code|status(?:_code)?)\b.{0,24}\b500\b", re.I | re.S)


def claude_code_environment() -> dict[str, str]:
    """Return the controlled environment shared by all Data-Pipe Claude runs."""
    env = os.environ.copy()
    env.update(CLAUDE_CODE_EXECUTION_ENV)
    return env


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_session(path: Path) -> dict[str, Any]:
    byte_count = path.stat().st_size if path.is_file() else 0
    parseable_event_count = 0
    if path.is_file():
        with path.open("rb") as stream:
            for raw_line in stream:
                try:
                    value = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    parseable_event_count += 1
    return {
        "byte_count": byte_count,
        "sha256": _sha256(path) if path.is_file() else None,
        "parseable_event_count": parseable_event_count,
        "raw_session_valid": byte_count > 0 and parseable_event_count > 0,
    }


def session_has_retryable_http_500(path: Path) -> bool:
    """Return true only for a terminal upstream HTTP-500 API failure."""
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "result" or not event.get("is_error"):
                continue
            text = json.dumps(
                {"result": event.get("result"), "error": event.get("error")},
                ensure_ascii=False,
                default=str,
            )
            if ("API Error" in text or "ChatCompletionStreamResponse" in text) and HTTP_500_RE.search(text):
                return True
    return False


def http_500_retry_delay(retry_count: int) -> int:
    """Exponential retry delay capped at five minutes."""
    if retry_count < 1:
        raise ValueError("retry_count must be >= 1")
    return min(300, 30 * (2 ** min(retry_count - 1, 4)))


def next_attempt_index(workdir: Path) -> int:
    attempts_root = workdir / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    indexes: list[int] = []
    for child in attempts_root.glob("attempt_*"):
        try:
            indexes.append(int(child.name.removeprefix("attempt_")))
        except ValueError:
            continue
    return max(indexes, default=0) + 1


def next_attempt(workdir: Path) -> tuple[int, Path]:
    attempts_root = workdir / "attempts"
    index = next_attempt_index(workdir)
    attempt_dir = attempts_root / f"attempt_{index:04d}"
    attempt_dir.mkdir(parents=False, exist_ok=False)
    session_path = attempt_dir / "complete_session.jsonl"
    session_path.touch()
    return index, session_path


def run_stream_json(
    command: list[str],
    *,
    cwd: Path,
    archive_root: Path,
    attempt_index: int | None = None,
    input_text: str | None = None,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    if "--verbose" not in command or "--output-format" not in command:
        raise ValueError("Claude command must request verbose stream-json output")
    output_index = command.index("--output-format") + 1
    if output_index >= len(command) or command[output_index] != "stream-json":
        raise ValueError("Claude command must request --output-format stream-json")

    if attempt_index is None:
        attempt_index, session_path = next_attempt(archive_root)
    else:
        if attempt_index < 1:
            raise ValueError("attempt_index must be >= 1")
        attempt_dir = archive_root / "attempts" / f"attempt_{attempt_index:04d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        session_path = attempt_dir / "complete_session.jsonl"
        if session_path.exists():
            raise FileExistsError(f"Claude attempt session already exists: {session_path}")
        session_path.touch()
    started = time.time()
    return_code = 1
    timed_out = False
    failure: str | None = None
    with session_path.open("wb") as session_stream:
        try:
            process = subprocess.run(
                command,
                cwd=str(cwd),
                env=claude_code_environment(),
                input=input_text.encode("utf-8") if input_text is not None else None,
                stdout=session_stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_sec,
            )
            return_code = int(process.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = 124
            failure = "timeout"
        except FileNotFoundError:
            return_code = 127
            failure = "executable_not_found"

    metadata = {
        "attempt_index": attempt_index,
        "workdir": str(cwd),
        "session_file": str(session_path),
        "return_code": return_code,
        "timed_out": timed_out,
        "timeout_sec": timeout_sec,
        "duration_sec": round(time.time() - started, 3),
        "failure": failure,
    }
    metadata.update(inspect_session(session_path))
    return metadata


def select_attempt(attempt: dict[str, Any], canonical_path: Path) -> dict[str, Any]:
    source = Path(str(attempt["session_file"]))
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, canonical_path)
    selected = inspect_session(canonical_path)
    if selected["sha256"] != attempt.get("sha256"):
        raise RuntimeError(
            f"selected Claude session checksum mismatch: {source} -> {canonical_path}"
        )
    return selected
