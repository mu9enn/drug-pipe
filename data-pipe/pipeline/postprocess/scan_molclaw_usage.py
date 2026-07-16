#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

RUN_RE = re.compile(r"^molbench_(vs|ac|pf|kg|e2e)_.+_run_(\d{8})_(\d{6})(?:_.+)?$")
ROW_RE = re.compile(r"^row\d+_idx.+$")
ROLLOUT_RE = re.compile(r"^rollout\d+$")
SUPPORTED_TASKS = {"vs", "ac", "pf", "kg", "e2e"}


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _safe_load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _session_ends_with_runner_error(path: Path) -> bool:
    if not path.is_file():
        return False
    last_nonempty = ""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                last_nonempty = stripped
    return last_nonempty.startswith("[runner-error]")


def _find_run_dir(path: Path) -> Path | None:
    for anc in [path.parent] + list(path.parents):
        if RUN_RE.match(anc.name):
            return anc
    return None


def _find_row_dir(path: Path) -> Path | None:
    for anc in [path.parent] + list(path.parents):
        if ROW_RE.match(anc.name):
            return anc
    return None


def _infer_task(run_dir: Path) -> str:
    cfg = _safe_load_json(run_dir / "run_config.json")
    task = str(cfg.get("task") or "").strip().lower()
    if task in SUPPORTED_TASKS:
        return task
    m = RUN_RE.match(run_dir.name)
    if m:
        return m.group(1)
    return "unknown"


def _session_key(path: Path) -> str:
    return str(path.resolve())


def _resolve_session_from_record(rec: dict[str, Any], run_dir: Path) -> Path | None:
    sample_dir = rec.get("sample_dir")
    if isinstance(sample_dir, str) and sample_dir.strip():
        p = Path(sample_dir)
        if not p.is_absolute():
            p = (run_dir / p).resolve()
        else:
            p = p.resolve()
        direct = p / "complete_session.jsonl"
        if direct.is_file():
            return direct

    task_id = str(rec.get("task_id") or "")
    row_match = re.search(r"(row\d+_idx[^_]+)", task_id)
    row_dir_name = row_match.group(1) if row_match else ""
    rollout_idx = rec.get("rollout_index")

    if row_dir_name:
        row_dir = run_dir / row_dir_name
        if isinstance(rollout_idx, int):
            rp = row_dir / f"rollout{rollout_idx:04d}" / "complete_session.jsonl"
            if rp.is_file():
                return rp
        direct = row_dir / "complete_session.jsonl"
        if direct.is_file():
            return direct
        any_roll = sorted(row_dir.glob("rollout*/complete_session.jsonl"))
        if any_roll:
            return any_roll[0]

    return None


class RunTrajectoryIndex:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.by_session: dict[str, dict[str, Any]] = {}
        traj_path = run_dir / "trajectories" / "trajectory_level.jsonl"
        for rec in _safe_load_jsonl(traj_path):
            sample_dir_val = rec.get("sample_dir")
            if not isinstance(sample_dir_val, str) or not sample_dir_val.strip():
                continue
            sample_dir = Path(sample_dir_val)
            if not sample_dir.is_absolute():
                sample_dir = (run_dir / sample_dir).resolve()
            else:
                sample_dir = sample_dir.resolve()
            key = _session_key(sample_dir / "complete_session.jsonl")
            self.by_session[key] = rec

    def get_by_session(self, session_path: Path) -> dict[str, Any] | None:
        return self.by_session.get(_session_key(session_path))


def _build_indices(results_root: Path) -> dict[Path, RunTrajectoryIndex]:
    indices: dict[Path, RunTrajectoryIndex] = {}
    for traj_path in sorted(results_root.rglob("trajectories/trajectory_level.jsonl")):
        run_dir = traj_path.parent.parent.resolve()
        indices[run_dir] = RunTrajectoryIndex(run_dir)
    return indices


def _as_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return None


def _metrics_from_record(task: str, rec: dict[str, Any]) -> dict[str, Any]:
    tm = rec.get("task_metrics") if isinstance(rec.get("task_metrics"), dict) else {}
    out: dict[str, Any] = {
        "vs_top3_hit_num": None,
        "vs_top10_hit_num": None,
        "ac_is_correct": None,
        "pf_precision": None,
        "pf_recall": None,
        "pf_f1": None,
        "pf_is_correct": None,
    }
    if task == "vs":
        out["vs_top3_hit_num"] = _as_float(tm.get("top3_hit_num"))
        out["vs_top10_hit_num"] = _as_float(tm.get("top10_hit_num"))
    elif task == "ac":
        out["ac_is_correct"] = _as_bool(tm.get("is_correct"))
    elif task == "pf":
        out["pf_precision"] = _as_float(tm.get("precision"))
        out["pf_recall"] = _as_float(tm.get("recall"))
        out["pf_f1"] = _as_float(tm.get("f1"))
        out["pf_is_correct"] = bool(tm.get("acc")) if tm.get("acc") is not None else None
    return out


def _answer_hit_pass(task: str, m: dict[str, Any]) -> bool | None:
    if task == "vs":
        v = m.get("vs_top3_hit_num")
        return bool(isinstance(v, (int, float)) and float(v) >= 1.0)
    if task == "ac":
        v = m.get("ac_is_correct")
        return bool(v is True)
    if task == "pf":
        v = m.get("pf_is_correct")
        return bool(v is True)
    return None


def _missing_metric_reason(task: str, metrics: dict[str, Any]) -> str | None:
    required = {
        "vs": ("vs_top3_hit_num", "vs_top10_hit_num"),
        "ac": ("ac_is_correct",),
        "pf": ("pf_precision", "pf_recall", "pf_f1", "pf_is_correct"),
    }.get(task, ())
    for field in required:
        if metrics.get(field) is None:
            if field == "pf_is_correct":
                return "missing_pf_exact_match"
            return f"missing_{field}"
    return None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _unique_dst(path: Path) -> Path:
    if not path.exists():
        return path
    i = 2
    while True:
        cand = path.with_name(f"{path.stem}__dup{i}{path.suffix}")
        if not cand.exists():
            return cand
        i += 1


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    default_results_root = repo_root / "results"
    default_output_root = default_results_root / "postprocess_candidates"

    parser = argparse.ArgumentParser(
        description="Collect accepted complete_session.jsonl files and usage/metric summary for post-processing."
    )
    parser.add_argument("--results-root", default=str(default_results_root))
    parser.add_argument("--output-root", default=str(default_output_root))
    parser.add_argument("--csv-name", default="molclaw_usage_summary.csv")
    parser.add_argument(
        "--use-accepted-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use trajectories/accepted.jsonl as source set (default: true).",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    csv_path = output_root / args.csv_name

    if not results_root.is_dir():
        raise NotADirectoryError(results_root)

    output_root.mkdir(parents=True, exist_ok=True)
    task_dirs = {t: output_root / t for t in SUPPORTED_TASKS}
    for d in task_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    run_indices = _build_indices(results_root)

    rows: list[dict[str, Any]] = []
    copied = 0
    skipped_unknown = 0
    skipped_no_session = 0
    skipped_runner_error_last_line = 0
    rejected_missing_metrics: list[dict[str, Any]] = []

    if args.use_accepted_only:
        accepted_files = sorted(results_root.rglob("trajectories/accepted.jsonl"))
        for accepted_path in accepted_files:
            run_dir = accepted_path.parent.parent.resolve()
            task = _infer_task(run_dir)
            if task not in SUPPORTED_TASKS:
                skipped_unknown += 1
                continue

            for rec in _safe_load_jsonl(accepted_path):
                session_path = _resolve_session_from_record(rec, run_dir)
                if session_path is None or not session_path.is_file():
                    skipped_no_session += 1
                    continue
                if _session_ends_with_runner_error(session_path):
                    skipped_runner_error_last_line += 1
                    continue

                traj_rec = rec
                idx = run_indices.get(run_dir)
                if idx is not None:
                    matched = idx.get_by_session(session_path)
                    if matched:
                        traj_rec = matched

                metric_cols = _metrics_from_record(task, traj_rec)
                missing_reason = _missing_metric_reason(task, metric_cols)
                if missing_reason:
                    rejected_missing_metrics.append(
                        {"task": task, "source": str(session_path), "run_dir": str(run_dir), "reason": missing_reason}
                    )
                    continue

                row_dir = _find_row_dir(session_path)
                row_name = row_dir.name if row_dir else "row_unknown"
                rollout_name = session_path.parent.name if ROLLOUT_RE.match(session_path.parent.name) else "rollout0001"
                dst_name = f"{run_dir.name}__{row_name}__{rollout_name}.jsonl"
                dst_path = _unique_dst(task_dirs[task] / dst_name)
                shutil.copy2(session_path, dst_path)

                rows.append(
                    {
                        "task": task,
                        "run_dir": run_dir.name,
                        "status": str(traj_rec.get("status") or "accepted"),
                        "is_accepted": str(traj_rec.get("status") or "accepted") == "accepted",
                        "original_path": str(session_path),
                        "copied_path": str(dst_path),
                        "molclaw_usage_count": int(traj_rec.get("molclaw_usage_count") or 0),
                        "answer_hit_pass": _answer_hit_pass(task, metric_cols),
                        **metric_cols,
                    }
                )
                copied += 1
    else:
        for session_path in sorted(results_root.rglob("complete_session.jsonl")):
            run_dir = _find_run_dir(session_path)
            if run_dir is None:
                skipped_unknown += 1
                continue
            task = _infer_task(run_dir)
            if task not in SUPPORTED_TASKS:
                skipped_unknown += 1
                continue
            if _session_ends_with_runner_error(session_path):
                skipped_runner_error_last_line += 1
                continue

            traj_rec: dict[str, Any] = {}
            idx = run_indices.get(run_dir.resolve())
            if idx is not None:
                matched = idx.get_by_session(session_path)
                if matched:
                    traj_rec = matched

            metric_cols = _metrics_from_record(task, traj_rec)
            missing_reason = _missing_metric_reason(task, metric_cols)
            if missing_reason:
                rejected_missing_metrics.append(
                    {"task": task, "source": str(session_path), "run_dir": str(run_dir), "reason": missing_reason}
                )
                continue

            row_dir = _find_row_dir(session_path)
            row_name = row_dir.name if row_dir else "row_unknown"
            rollout_name = session_path.parent.name if ROLLOUT_RE.match(session_path.parent.name) else "rollout0001"
            dst_name = f"{run_dir.name}__{row_name}__{rollout_name}.jsonl"
            dst_path = _unique_dst(task_dirs[task] / dst_name)
            shutil.copy2(session_path, dst_path)

            rows.append(
                {
                    "task": task,
                    "run_dir": run_dir.name,
                    "status": str(traj_rec.get("status") or "unknown"),
                    "is_accepted": str(traj_rec.get("status") or "") == "accepted",
                    "original_path": str(session_path),
                    "copied_path": str(dst_path),
                    "molclaw_usage_count": int(traj_rec.get("molclaw_usage_count") or 0),
                    "answer_hit_pass": _answer_hit_pass(task, metric_cols),
                    **metric_cols,
                }
            )
            copied += 1

    fieldnames = [
        "task",
        "run_dir",
        "status",
        "is_accepted",
        "original_path",
        "copied_path",
        "molclaw_usage_count",
        "answer_hit_pass",
        "vs_top3_hit_num",
        "vs_top10_hit_num",
        "ac_is_correct",
        "pf_precision",
        "pf_recall",
        "pf_f1",
        "pf_is_correct",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    rejected_path = output_root / "stage2_rejected_candidates.jsonl"
    _write_jsonl(rejected_path, rejected_missing_metrics)

    print(f"results_root={results_root}")
    print(f"output_root={output_root}")
    print(f"csv={csv_path}")
    print(f"use_accepted_only={int(bool(args.use_accepted_only))}")
    print(f"copied_files={copied}")
    print(f"skipped_unknown={skipped_unknown}")
    print(f"skipped_no_session={skipped_no_session}")
    print(f"skipped_runner_error_last_line={skipped_runner_error_last_line}")
    print(f"stage2_rejected_missing_metrics={len(rejected_missing_metrics)}")
    print(f"stage2_rejected_file={rejected_path}")


if __name__ == "__main__":
    main()
