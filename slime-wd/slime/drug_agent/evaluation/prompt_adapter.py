from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from drug_agent.constants import DEFAULT_SYSTEM_PROMPT
from drug_agent.utils import ensure_dir, write_json, write_jsonl

SUPPORTED_TASK_TYPES = {"vs", "ac", "pf", "kg", "e2e", "mol_edit", "mol_opt", "mol_opt_physchem"}


def _sample(*, prompt: str, task_type: str, task_id: str, max_steps: int, source: Path) -> dict[str, Any]:
    normalized_task_type = str(task_type or "").strip().lower()
    if normalized_task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"Unsupported task type {task_type!r}; choose one of {sorted(SUPPORTED_TASK_TYPES)}"
        )
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Prompt is empty for task {task_id!r}")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative (0 means unlimited)")
    return {
        "id": task_id,
        "prompt": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
        ],
        "label": {},
        "metadata": {
            "manual_prompt": prompt.strip(),
            "env_kwargs": {
                "task_id": task_id,
                "task_type": normalized_task_type,
                "data_source": "manual_prompt",
                "max_steps": max_steps,
            },
            "benchmark": {
                "suite": "manual_prompt",
                "subtask": normalized_task_type,
                "source_path": str(source),
            },
        },
    }


def build_single_prompt_dataset(
    prompt_file: str | Path,
    output_dir: str | Path,
    *,
    task_type: str = "e2e",
    task_id: str = "manual_prompt_001",
    max_steps: int = 0,
) -> dict[str, Any]:
    source = Path(prompt_file).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Prompt file not found: {source}")
    prompt = source.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {source}")
    output = ensure_dir(output_dir)
    dataset_path = output / "prompt_eval.jsonl"
    row = _sample(
        prompt=prompt, task_type=task_type, task_id=task_id, max_steps=max_steps, source=source
    )
    write_jsonl(dataset_path, [row])
    manifest = {
        "schema_version": "drug_agent_single_prompt_input_v1",
        "dataset_path": str(dataset_path),
        "sample_count": 1,
        "task_id": task_id,
        "task_type": row["metadata"]["env_kwargs"]["task_type"],
        "prompt_source": str(source),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "max_steps": max_steps,
    }
    write_json(output / "prompt_manifest.json", manifest)
    return manifest


def build_prompt_suite_dataset(
    suite_file: str | Path,
    output_dir: str | Path,
    *,
    max_steps: int = 0,
) -> dict[str, Any]:
    source = Path(suite_file).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Prompt suite file not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Prompt suite must be a non-empty JSON list")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt suite row {index} must be an object")
        task_id = str(item.get("id") or f"manual_prompt_{index:03d}")
        if task_id in seen:
            raise ValueError(f"Duplicate prompt task id: {task_id}")
        seen.add(task_id)
        rows.append(
            _sample(
                prompt=item.get("prompt"),
                task_type=str(item.get("task_type") or "e2e"),
                task_id=task_id,
                max_steps=max_steps,
                source=source,
            )
        )
    output = ensure_dir(output_dir)
    dataset_path = output / "prompt_eval.jsonl"
    write_jsonl(dataset_path, rows)
    manifest = {
        "schema_version": "drug_agent_prompt_suite_input_v1",
        "dataset_path": str(dataset_path),
        "sample_count": len(rows),
        "task_ids": [row["id"] for row in rows],
        "suite_source": str(source),
        "suite_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "max_steps": max_steps,
    }
    write_json(output / "prompt_manifest.json", manifest)
    return manifest
