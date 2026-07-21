from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected.append({"line_number": line_number, "reason": f"JSONDecodeError:{exc}"})
                continue
            if not isinstance(value, dict):
                rejected.append({"line_number": line_number, "reason": "row_not_object"})
                continue
            rows.append(value)
    return rows, rejected


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_commit(repo_root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    return process.stdout.strip() if process.returncode == 0 else "unknown"


def base_manifest(*, step: str, source: Path, repo_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "drug_pipe_cleaning_manifest_v1",
        "step": step,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(repo_root),
        "source": str(source.resolve()),
    }
