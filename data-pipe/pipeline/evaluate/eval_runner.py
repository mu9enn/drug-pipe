from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from .task_evaluator import evaluate_task_answer, load_chemistry_module


def _evaluate_file(path: Path, task: str) -> Dict[str, Any]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"Prediction file must be a JSON list: {path}")

    chemistry, chemistry_error = load_chemistry_module()
    metric_values: dict[str, list[float]] = {}
    invalid_hist: Counter[str] = Counter()
    valid_count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        result = evaluate_task_answer(
            task,
            prediction=(entry.get("json_results") or {}).get("ranking" if task == "vs" else "prediction"),
            ground_truth=entry.get("answer"),
            candidates=entry.get("candidates"),
            chemistry=chemistry,
        )
        entry["metrics"] = result["metrics"]
        entry["task_answer_valid"] = result["task_answer_valid"]
        entry["eval_audit"] = result["audit"]
        entry["eval_audit"]["invalid_reasons"] = result["invalid_reasons"]
        eligible = bool(result["aggregate_eligible"])
        valid_count += int(eligible)
        invalid_hist.update(result["invalid_reasons"])
        for name, value in result["metrics"].items():
            if eligible and isinstance(value, (int, float, bool)):
                metric_values.setdefault(name, []).append(float(value))

    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    averages = {
        name: (sum(values) / len(values) if values else 0.0)
        for name, values in metric_values.items()
    }
    if task == "vs":
        task_summary = {
            "top3_avg_hit_num": averages.get("top3_hit_num", 0.0),
            "top10_avg_hit_num": averages.get("top10_hit_num", 0.0),
            "n_samples": len(entries),
            "n_valid_scored": valid_count,
        }
    elif task == "ac":
        task_summary = {
            "accuracy": averages.get("is_correct", 0.0),
            "n_samples": len(entries),
            "n_valid_scored": valid_count,
        }
    else:
        task_summary = {
            "exact_set_match_rate": averages.get("exact_set_match", 0.0),
            "avg_f1": averages.get("f1", 0.0),
            "single_answer_accuracy": averages.get("acc", 0.0),
            "n_samples": len(entries),
            "n_valid_scored": valid_count,
        }
    return {
        f"molbench_{task}_molbench_{task}": task_summary,
        "audit": {
            "rdkit_available": chemistry is not None,
            "rdkit_error": chemistry_error,
            "task_answer_valid_count": valid_count,
            "invalid_reason_hist": dict(invalid_hist),
        },
    }


def eval_molbench_vs_file(pred_json_path: str) -> Dict[str, Any]:
    return _evaluate_file(Path(pred_json_path), "vs")


def eval_molbench_ac_file(pred_json_path: str) -> Dict[str, Any]:
    return _evaluate_file(Path(pred_json_path), "ac")


def eval_molbench_pf_file(pred_json_path: str) -> Dict[str, Any]:
    return _evaluate_file(Path(pred_json_path), "pf")


def _infer_task_from_results_dir(results_dir: str) -> str:
    root = Path(results_dir)
    config_path = root / "run_config.json"
    if config_path.is_file():
        try:
            task = str(json.loads(config_path.read_text(encoding="utf-8")).get("task") or "").lower()
        except Exception:
            task = ""
        if task in {"vs", "ac", "pf"}:
            return task
    for task in ("vs", "ac", "pf"):
        if (root / "preds" / f"molbench_{task}" / f"molbench_{task}.json").is_file():
            return task
    raise FileNotFoundError(f"Cannot infer task from results_dir: {results_dir}")


def eval_results_dir(results_dir: str, task: str | None = None) -> Dict[str, Any]:
    root = Path(os.path.abspath(results_dir))
    resolved_task = (task or "").strip().lower() or _infer_task_from_results_dir(str(root))
    if resolved_task not in {"vs", "ac", "pf"}:
        raise ValueError(f"Unsupported task: {resolved_task}")
    path = root / "preds" / f"molbench_{resolved_task}" / f"molbench_{resolved_task}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    return _evaluate_file(path, resolved_task)
