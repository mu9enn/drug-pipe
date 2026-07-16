#!/usr/bin/env python3
"""Export MolBench Claude session artifacts into trajectory datasets (VS/AC/PF/E2E/KG)."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROLLOUT_RE = re.compile(r"rollout(\d+)$")
AFFINITY_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:kcal/?mol)", re.IGNORECASE)
TASK_CHOICES = {"vs", "ac", "pf", "e2e", "kg"}


@dataclass
class RolloutSample:
    row_dir: Path
    sample_dir: Path
    row_number: int
    dataset_index: str
    rollout_index: int


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _iter_row_dirs(results_dir: Path) -> list[Path]:
    rows = [p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("row") and "_idx" in p.name]
    return sorted(rows)


def _infer_row_number(row_dir: Path) -> int:
    prefix = row_dir.name.split("_idx", 1)[0]
    try:
        return int(prefix.replace("row", ""))
    except Exception:
        return -1


def _discover_rollout_samples(results_dir: Path) -> list[RolloutSample]:
    samples: list[RolloutSample] = []
    for row_dir in _iter_row_dirs(results_dir):
        row_number = _infer_row_number(row_dir)
        row_question = _safe_load_json(row_dir / "question.json")
        dataset_index = str(row_question.get("dataset_index") or "")

        direct_parsed = row_dir / "parsed_answer.json"
        if direct_parsed.is_file():
            if not dataset_index:
                dataset_index = str(_safe_load_json(direct_parsed).get("dataset_index") or "")
            samples.append(
                RolloutSample(
                    row_dir=row_dir,
                    sample_dir=row_dir,
                    row_number=row_number,
                    dataset_index=dataset_index or str(row_number),
                    rollout_index=1,
                )
            )
            continue

        rollout_dirs = sorted([p for p in row_dir.iterdir() if p.is_dir() and p.name.startswith("rollout")])
        for rd in rollout_dirs:
            parsed = rd / "parsed_answer.json"
            if not parsed.is_file():
                continue
            m = ROLLOUT_RE.search(rd.name)
            if m:
                r_idx = int(m.group(1))
            else:
                r_idx = len(samples) + 1

            q = _safe_load_json(rd / "question.json")
            idx = str(q.get("dataset_index") or dataset_index or row_number)
            samples.append(
                RolloutSample(
                    row_dir=row_dir,
                    sample_dir=rd,
                    row_number=row_number,
                    dataset_index=idx,
                    rollout_index=r_idx,
                )
            )

    samples.sort(key=lambda x: (x.row_number, x.rollout_index, x.sample_dir.name))
    return samples


def _split_lines(s: str) -> list[str]:
    return [ln.strip() for ln in (s or "").strip().splitlines() if ln.strip()]


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            v = json.loads(s)
        except Exception:
            return _split_lines(s)
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return _split_lines(v)
    return []


def _extract_candidates(question_obj: dict[str, Any]) -> list[str]:
    cands = question_obj.get("candidates")
    if isinstance(cands, list):
        return [str(x).strip() for x in cands if str(x).strip()]

    raw_q = question_obj.get("raw_question_json")
    if isinstance(raw_q, str) and raw_q.strip():
        try:
            q_obj = json.loads(raw_q)
        except Exception:
            return []
        cands2 = q_obj.get("candidates") if isinstance(q_obj, dict) else None
        if isinstance(cands2, list):
            return [str(x).strip() for x in cands2 if str(x).strip()]
    return []


def _canonicalize_list(smiles_list: list[str], chem_module: Any) -> tuple[list[str], list[dict[str, Any]]]:
    out: list[str] = []
    errors: list[dict[str, Any]] = []
    for i, s in enumerate(smiles_list):
        mol = chem_module.MolFromSmiles(s)
        if mol is None:
            errors.append({"index": i, "smiles": s, "reason": "invalid_smiles"})
            continue
        out.append(chem_module.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return out, errors


def _tool_category(tool_name: str) -> str:
    if tool_name.startswith("mcp__"):
        return "scientific"
    if tool_name in {"TodoWrite", "Write", "Read", "Edit", "Bash", "Task", "Glob", "Grep"}:
        return "workspace"
    return "other"


def _load_session_events(session_path: Path) -> list[dict[str, Any]]:
    if not session_path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with session_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                obj["_line_no"] = line_no
                events.append(obj)
    return events


def _session_ends_with_runner_error(session_path: Path) -> bool:
    if not session_path.is_file():
        return False
    last_nonempty = ""
    with session_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                last_nonempty = stripped
    return last_nonempty.startswith("[runner-error]")


def _sum_counter_by_tool_suffix(counter: Counter[str], tool_suffix: str) -> int:
    total = 0
    for name, cnt in counter.items():
        parts = str(name).split("__")
        last = parts[-1] if parts else str(name)
        if last == tool_suffix:
            total += int(cnt)
    return total


def _build_artifact_audit(events: list[dict[str, Any]]) -> dict[str, Any]:
    tool_use_name_by_id: dict[str, str] = {}
    tool_use_counter: Counter[str] = Counter()
    tool_result_counter: Counter[str] = Counter()
    assistant_text_blocks = 0
    assistant_affinity_mentions = 0
    answer_tag_mentions = 0

    for ev in events:
        ev_type = str(ev.get("type") or "")
        if ev_type == "assistant":
            msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "")
                if item_type == "tool_use":
                    tool_name = str(item.get("name") or "")
                    tool_use_id = str(item.get("id") or "")
                    tool_use_name_by_id[tool_use_id] = tool_name
                    tool_use_counter[tool_name] += 1
                elif item_type in {"text", "thinking"}:
                    txt = item.get("text") if isinstance(item.get("text"), str) else item.get("thinking")
                    if isinstance(txt, str) and txt.strip():
                        assistant_text_blocks += 1
                        assistant_affinity_mentions += len(AFFINITY_RE.findall(txt))
                        if "<answer>" in txt.lower():
                            answer_tag_mentions += 1
        elif ev_type == "user":
            msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if str(item.get("type") or "") != "tool_result":
                    continue
                tool_use_id = str(item.get("tool_use_id") or "")
                name = tool_use_name_by_id.get(tool_use_id, "")
                tool_result_counter[name] += 1

    scientific_tool_results = sum(v for k, v in tool_result_counter.items() if k.startswith("mcp__"))
    scientific_tool_uses = sum(v for k, v in tool_use_counter.items() if k.startswith("mcp__"))
    docking_suffix = "molecule_docking_quickvina_fullprocess"

    return {
        "tool_generated": {
            "tool_use_hist": dict(tool_use_counter),
            "tool_result_hist": dict(tool_result_counter),
            "scientific_tool_use_count": scientific_tool_uses,
            "scientific_tool_result_count": scientific_tool_results,
            "docking_use_count": _sum_counter_by_tool_suffix(tool_use_counter, docking_suffix),
            "docking_result_count": _sum_counter_by_tool_suffix(tool_result_counter, docking_suffix),
        },
        "llm_generated": {
            "assistant_text_block_count": assistant_text_blocks,
            "assistant_affinity_mentions_count": assistant_affinity_mentions,
            "answer_tag_mentions_count": answer_tag_mentions,
        },
    }


def _build_step_records(events: list[dict[str, Any]], task_id: str, accepted: bool) -> tuple[list[dict[str, Any]], Counter[str]]:
    steps: list[dict[str, Any]] = []
    tool_use_name_by_id: dict[str, str] = {}
    tool_counter: Counter[str] = Counter()
    step_id = 0

    for ev in events:
        ev_type = str(ev.get("type") or "")

        if ev_type == "assistant":
            msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "")
                if item_type == "tool_use":
                    step_id += 1
                    tool_name = str(item.get("name") or "")
                    tool_use_id = str(item.get("id") or "")
                    tool_use_name_by_id[tool_use_id] = tool_name
                    tool_counter[tool_name] += 1
                    steps.append(
                        {
                            "task_id": task_id,
                            "step_id": step_id,
                            "line_no": ev.get("_line_no"),
                            "action_type": "tool_use",
                            "tool_name": tool_name,
                            "tool_category": _tool_category(tool_name),
                            "tool_use_id": tool_use_id,
                            "tool_args": item.get("input"),
                            "assistant_text": None,
                            "observation": None,
                            "accepted": accepted,
                            "done": False,
                        }
                    )
                elif item_type in {"text", "thinking"}:
                    text = item.get("text") if isinstance(item.get("text"), str) else item.get("thinking")
                    if isinstance(text, str) and text.strip():
                        step_id += 1
                        steps.append(
                            {
                                "task_id": task_id,
                                "step_id": step_id,
                                "line_no": ev.get("_line_no"),
                                "action_type": "assistant_content",
                                "tool_name": None,
                                "tool_category": None,
                                "tool_use_id": None,
                                "tool_args": None,
                                "assistant_text": text,
                                "observation": None,
                                "accepted": accepted,
                                "done": False,
                            }
                        )

        elif ev_type == "user":
            msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if str(item.get("type") or "") != "tool_result":
                    continue
                step_id += 1
                tool_use_id = str(item.get("tool_use_id") or "")
                tool_name = tool_use_name_by_id.get(tool_use_id)
                steps.append(
                    {
                        "task_id": task_id,
                        "step_id": step_id,
                        "line_no": ev.get("_line_no"),
                        "action_type": "tool_result",
                        "tool_name": tool_name,
                        "tool_category": _tool_category(tool_name) if tool_name else None,
                        "tool_use_id": tool_use_id,
                        "tool_args": None,
                        "assistant_text": None,
                        "observation": item.get("content"),
                        "tool_result_error": bool(item.get("is_error")),
                        "accepted": accepted,
                        "done": False,
                    }
                )

    if steps:
        steps[-1]["done"] = True

    return steps, tool_counter


def _compute_set_metrics(pred_set: set[str], gt_set: set[str]) -> dict[str, float]:
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "acc": float(1.0 if pred_set == gt_set else 0.0),
    }


def _compute_hit_num(pred: list[str], gt: list[str], k: int) -> float:
    pred_topk = pred[:k]
    gt_set = set(gt)
    return float(sum(1 for x in pred_topk if x in gt_set))


def _infer_task(results_dir: Path, explicit_task: str | None = None) -> str:
    if explicit_task:
        task = explicit_task.strip().lower()
        if task in TASK_CHOICES:
            return task
        raise ValueError(f"Unsupported task override: {explicit_task}")

    cfg = _safe_load_json(results_dir / "run_config.json")
    cfg_task = str(cfg.get("task") or "").strip().lower()
    if cfg_task in TASK_CHOICES:
        return cfg_task

    for t in ("vs", "ac", "pf", "e2e", "kg"):
        pred_file = results_dir / "preds" / f"molbench_{t}" / f"molbench_{t}.json"
        if pred_file.is_file():
            return t

    raise FileNotFoundError(f"Unable to infer task from {results_dir}")


def _aggregate_task_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        tm = rec.get("task_metrics")
        if not isinstance(tm, dict):
            continue
        for k, v in tm.items():
            if isinstance(v, (int, float)):
                sums[k] += float(v)
                counts[k] += 1
    out: dict[str, float] = {}
    for k, total in sums.items():
        c = counts.get(k, 0)
        out[f"avg_{k}"] = (total / c) if c else 0.0
    return out


def export_results_dir(results_dir: Path, task: str | None = None) -> dict[str, Any]:
    if not results_dir.is_dir():
        raise NotADirectoryError(results_dir)

    task_name = _infer_task(results_dir, explicit_task=task)

    try:
        from rdkit import Chem  # type: ignore

        rdkit_available = True
        rdkit_error = None
    except Exception as e:  # pragma: no cover - environment-dependent
        Chem = None
        rdkit_available = False
        rdkit_error = str(e)

    out_dir = results_dir / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = _discover_rollout_samples(results_dir)

    trajectory_records: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    reject_reason_counter: Counter[str] = Counter()

    for s in samples:
        question = _safe_load_json(s.sample_dir / "question.json")
        if not question:
            question = _safe_load_json(s.row_dir / "question.json")

        parsed = _safe_load_json(s.sample_dir / "parsed_answer.json")
        run_meta = _safe_load_json(s.sample_dir / "run_meta.json")
        session_path = s.sample_dir / "complete_session.jsonl"
        session_ends_with_runner_error = _session_ends_with_runner_error(session_path)
        timed_out_value = bool(parsed.get("timed_out", False) or run_meta.get("timed_out", False))
        return_code_value = run_meta.get("return_code")

        task_for_sample = str(question.get("task") or run_meta.get("task") or task_name).strip().lower()
        if task_for_sample not in TASK_CHOICES:
            task_for_sample = task_name

        candidates = _extract_candidates(question)
        gt_answers = _as_str_list(question.get("answer"))
        pred_answers = _as_str_list(parsed.get("answer"))

        kg_metadata: dict[str, Any] | None = None

        if task_for_sample == "e2e":
            parse_error = None
            if not pred_answers:
                raw = parsed.get("answer_block")
                if isinstance(raw, str) and raw.strip():
                    pred_answers = [raw.strip()]
            has_session = session_path.is_file()
            rc_ok = return_code_value == 0
            reject_reasons = []
            if not rc_ok:
                reject_reasons.append(f"runner_nonzero_rc:{return_code_value}")
            if timed_out_value:
                reject_reasons.append("timeout")
            if not has_session:
                reject_reasons.append("missing_session")
            if session_ends_with_runner_error:
                reject_reasons.append("runner_error_last_line")
            task_reject_checks = {
                "return_code": return_code_value,
                "timed_out": timed_out_value,
                "has_complete_session": has_session,
                "session_ends_with_runner_error": session_ends_with_runner_error,
                "ground_truth_size": len(gt_answers),
                "prediction_size": len(pred_answers),
            }
            task_metrics = {}
            cand_canon = [x.strip() for x in candidates if x.strip()]
            gt_canon = [x.strip() for x in gt_answers if x.strip()]
            pred_canon = [x.strip() for x in pred_answers if x.strip()]
            accepted = len(reject_reasons) == 0
        elif task_for_sample == "kg":
            parse_error = parsed.get("parse_error")
            if not pred_answers:
                raw = parsed.get("answer_block")
                if isinstance(raw, str) and raw.strip():
                    pred_answers = [raw.strip()]

            has_session = session_path.is_file()
            rc_ok = return_code_value == 0
            accepted = bool(rc_ok and not timed_out_value and has_session)

            reject_reasons = []
            if not rc_ok:
                reject_reasons.append(f"runner_nonzero_rc:{return_code_value}")
            if timed_out_value:
                reject_reasons.append("timeout")
            if not has_session:
                reject_reasons.append("missing_session")
            if session_ends_with_runner_error:
                reject_reasons.append("runner_error_last_line")
            accepted = len(reject_reasons) == 0

            task_reject_checks = {
                "return_code": return_code_value,
                "timed_out": timed_out_value,
                "has_complete_session": has_session,
                "session_ends_with_runner_error": session_ends_with_runner_error,
                "prediction_size": len(pred_answers),
                "ground_truth_size": len(gt_answers),
            }
            task_metrics = {}
            cand_canon = [x.strip() for x in candidates if x.strip()]
            gt_canon = [x.strip() for x in gt_answers if x.strip()]
            pred_canon = [x.strip() for x in pred_answers if x.strip()]

            kg_task_spec = question.get("kg_task_spec") if isinstance(question.get("kg_task_spec"), dict) else {}
            if not kg_task_spec:
                raw_q = question.get("raw_question_json")
                if isinstance(raw_q, str) and raw_q.strip():
                    try:
                        tmp = json.loads(raw_q)
                    except Exception:
                        tmp = {}
                    if isinstance(tmp, dict):
                        kg_task_spec = tmp

            kg_source = kg_task_spec.get("source") if isinstance(kg_task_spec.get("source"), dict) else {}
            kg_toolchain = kg_task_spec.get("toolchain") if isinstance(kg_task_spec.get("toolchain"), dict) else {}
            expected_tools = kg_toolchain.get("tools") if isinstance(kg_toolchain.get("tools"), list) else question.get("toolchain_nodes")
            if not isinstance(expected_tools, list):
                expected_tools = []
            kg_metadata = {
                "source": "molclaw_kg",
                "kg_run_id": kg_source.get("kg_run_id"),
                "kg_task_id": kg_task_spec.get("task_id"),
                "expected_toolchain": expected_tools,
                "expected_trajectory_available": bool(kg_task_spec.get("expected_trajectory")),
            }
        else:
            parse_error = parsed.get("parse_error")
            reject_reasons: list[str] = []
            task_reject_checks: dict[str, Any] = {
                "parse_error": bool(parse_error),
                "session_ends_with_runner_error": session_ends_with_runner_error,
                "ground_truth_size": len(gt_answers),
                "prediction_size": len(pred_answers),
            }

            if parse_error:
                reject_reasons.append("parse_error")
            if session_ends_with_runner_error:
                reject_reasons.append("runner_error_last_line")

            if not rdkit_available:
                cand_canon = [x.strip() for x in candidates if x.strip()]
                gt_canon = [x.strip() for x in gt_answers if x.strip()]
                pred_canon = [x.strip() for x in pred_answers if x.strip()]
                cand_err: list[dict[str, Any]] = []
                gt_err: list[dict[str, Any]] = []
                pred_err: list[dict[str, Any]] = []
            else:
                cand_canon, cand_err = _canonicalize_list(candidates, Chem)
                gt_canon, gt_err = _canonicalize_list(gt_answers, Chem)
                pred_canon, pred_err = _canonicalize_list(pred_answers, Chem)

            if gt_err:
                reject_reasons.append(f"invalid_ground_truth_smiles:{len(gt_err)}")
            if pred_err:
                reject_reasons.append(f"invalid_prediction_smiles:{len(pred_err)}")
            if task_for_sample == "vs" and cand_err:
                reject_reasons.append(f"invalid_candidate_smiles:{len(cand_err)}")

            task_metrics = {}

            if task_for_sample == "vs":
                expected_n = len(cand_canon)
                if expected_n == 0:
                    reject_reasons.append("empty_candidate_set")
                if len(pred_canon) != expected_n:
                    reject_reasons.append(f"length_mismatch:{len(pred_canon)}!={expected_n}")

                unique_pred_n = len(set(pred_canon))
                if unique_pred_n != len(pred_canon):
                    reject_reasons.append("duplicate_predictions")

                cand_set = set(cand_canon)
                outside_n = sum(1 for x in pred_canon if x not in cand_set)
                if outside_n > 0:
                    reject_reasons.append(f"outside_candidate_set:{outside_n}")

                top3 = _compute_hit_num(pred_canon, gt_canon, 3) if (gt_canon and pred_canon) else 0.0
                top10 = _compute_hit_num(pred_canon, gt_canon, 10) if (gt_canon and pred_canon) else 0.0
                task_metrics = {
                    "top3_hit_num": float(top3),
                    "top10_hit_num": float(top10),
                }
                task_reject_checks.update(
                    {
                        "expected_candidate_size": expected_n,
                        "prediction_unique_size": unique_pred_n,
                        "outside_candidate_count": outside_n,
                    }
                )

            elif task_for_sample == "ac":
                if len(pred_answers) == 0:
                    reject_reasons.append("empty_prediction")
                if len(pred_answers) != 1:
                    reject_reasons.append(f"invalid_prediction_count:{len(pred_answers)}")

                pred_one = pred_canon[0] if pred_canon else ""
                gt_one = gt_canon[0] if gt_canon else ""
                acc = float(1.0 if (pred_one and gt_one and pred_one == gt_one) else 0.0)
                task_metrics = {
                    "acc": acc,
                    "is_correct": bool(acc),
                }

            elif task_for_sample == "pf":
                if len(pred_answers) == 0:
                    reject_reasons.append("empty_prediction")

                pred_set = set(pred_canon)
                gt_set = set(gt_canon)
                set_metrics = _compute_set_metrics(pred_set, gt_set)
                task_metrics = dict(set_metrics)
                task_reject_checks.update(
                    {
                        "prediction_unique_size": len(pred_set),
                        "ground_truth_unique_size": len(gt_set),
                    }
                )

            accepted = len(reject_reasons) == 0
        task_id = f"{task_for_sample}_row{s.row_number:04d}_idx{s.dataset_index}_r{s.rollout_index:04d}"

        events = _load_session_events(session_path)
        steps, tool_counter = _build_step_records(events, task_id, accepted)
        step_records.extend(steps)
        artifact_audit = _build_artifact_audit(events)
        molclaw_usage_count = int(sum(cnt for name, cnt in tool_counter.items() if str(name).startswith("mcp__molclaw")))
        if molclaw_usage_count <= 0:
            if "missing_molclaw_usage" not in reject_reasons:
                reject_reasons.append("missing_molclaw_usage")
            accepted = False
        for st in steps:
            st["accepted"] = accepted
        task_reject_checks["molclaw_usage_count"] = molclaw_usage_count
        for rr in reject_reasons:
            reject_reason_counter[rr] += 1

        traj = {
            "task": task_for_sample,
            "task_id": task_id,
            "row_number": s.row_number,
            "dataset_index": s.dataset_index,
            "rollout_index": s.rollout_index,
            "sample_dir": str(s.sample_dir),
            "status": "accepted" if accepted else "rejected",
            "reject_reasons": reject_reasons,
            "task_reject_checks": task_reject_checks,
            "candidates": candidates,
            "ground_truth": gt_answers,
            "final_answer": pred_answers,
            "canonical": {
                "candidates": cand_canon,
                "ground_truth": gt_canon,
                "final_answer": pred_canon,
            },
            "task_metrics": task_metrics,
            "metrics": dict(task_metrics),
            "parse_error": parse_error,
            "timed_out": timed_out_value,
            "return_code": return_code_value,
            "tool_stats": dict(tool_counter),
            "artifact_audit": artifact_audit,
            "molclaw_usage_count": molclaw_usage_count,
            "session_event_count": len(events),
            "step_count": len(steps),
        }
        if kg_metadata is not None:
            traj["kg_metadata"] = kg_metadata
        trajectory_records.append(traj)

    traj_path = out_dir / "trajectory_level.jsonl"
    with traj_path.open("w", encoding="utf-8") as f:
        for rec in trajectory_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    step_path = out_dir / "step_level.jsonl"
    with step_path.open("w", encoding="utf-8") as f:
        for rec in step_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    accepted_path = out_dir / "accepted.jsonl"
    rejected_path = out_dir / "rejected.jsonl"
    with accepted_path.open("w", encoding="utf-8") as fa, rejected_path.open("w", encoding="utf-8") as fr:
        for rec in trajectory_records:
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            if rec.get("status") == "accepted":
                fa.write(line)
            else:
                fr.write(line)

    summary = {
        "task": task_name,
        "results_dir": str(results_dir),
        "rdkit_available": rdkit_available,
        "rdkit_error": rdkit_error,
        "n_samples": len(trajectory_records),
        "n_steps": len(step_records),
        "n_accepted": sum(1 for x in trajectory_records if x.get("status") == "accepted"),
        "n_rejected": sum(1 for x in trajectory_records if x.get("status") == "rejected"),
        "reject_reason_hist": dict(reject_reason_counter),
        "task_metric_averages": _aggregate_task_metrics(trajectory_records),
        "files": {
            "trajectory_level": str(traj_path),
            "step_level": str(step_path),
            "accepted": str(accepted_path),
            "rejected": str(rejected_path),
        },
    }
    summary_path = out_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export run artifacts to trajectory datasets.")
    parser.add_argument("results_dir", help="Run directory produced by claude_agent/run_claude.py")
    parser.add_argument("--task", choices=sorted(TASK_CHOICES), default="", help="Optional task override")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    summary = export_results_dir(results_dir, task=args.task or None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
