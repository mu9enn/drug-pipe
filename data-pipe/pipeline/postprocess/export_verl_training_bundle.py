#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BUNDLE_SCHEMA_VERSION = "mol_pipeline_to_verl_bundle_v0.1"
SFT_SCHEMA_VERSION = "sft_messages_schema_v0.1"
RL_SCHEMA_VERSION = "rl_prompt_schema_v0.1"
RUN_RE = re.compile(r"^molbench_(vs|ac|pf|kg|e2e)_.+_run_(\d{8})_(\d{6})(?:_.+)?$")


@dataclass
class NormalizeStats:
    total: int = 0
    valid: int = 0
    invalid: int = 0
    assistant_json_parse_fail: int = 0
    assistant_action_count: int = 0
    assistant_tool_call_count: int = 0
    assistant_final_answer_count: int = 0
    observation_count: int = 0
    assistant_text_dropped_count: int = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _split_from_id(stable_id: str, valid_ratio: float = 0.1) -> str:
    h = int(hashlib.md5(stable_id.encode("utf-8")).hexdigest()[:8], 16)
    return "valid" if (h % 10_000) < int(valid_ratio * 10_000) else "train"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        return _load_jsonl(path)
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for p in sorted(path.glob("*.json")):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
        return rows
    return []


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _git_head(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
        return out
    except Exception:
        return "unknown"


def _git_dirty(repo_root: Path) -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root), text=True)
        return bool(out.strip())
    except Exception:
        return True


def _task_from_id_or_meta(rec: dict[str, Any]) -> str:
    meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
    task_meta = meta.get("task") if isinstance(meta.get("task"), dict) else {}
    t = str(task_meta.get("task_type") or "").strip().lower()
    if t:
        return t
    rid = str(rec.get("id") or "")
    m = re.search(r"_([a-z]+)_\d{8}_\d{6}_row", rid)
    if m:
        return m.group(1)
    return "unknown"


def _source_run_from_record(rec: dict[str, Any]) -> str:
    meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
    src = meta.get("source") if isinstance(meta.get("source"), dict) else {}
    p = str(src.get("raw_trajectory_path") or "")
    if not p:
        return ""
    path = Path(p)
    for anc in [path.parent] + list(path.parents):
        if RUN_RE.match(anc.name):
            return anc.name
    return ""


def _extract_tool_call_json(content: str) -> dict[str, Any] | None:
    content = content.strip()
    m = re.search(r"<tool_call>([\s\S]*?)</tool_call>", content)
    if m:
        payload = m.group(1).strip()
    else:
        payload = content

    try:
        obj = json.loads(payload)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    if obj.get("type") == "tool_call":
        tn = obj.get("tool_name")
        args = obj.get("arguments")
        if isinstance(tn, str) and isinstance(args, dict):
            return obj
        return None

    name = obj.get("name")
    args = obj.get("arguments")
    if isinstance(name, str):
        return {
            "type": "tool_call",
            "tool_name": name,
            "arguments": args if isinstance(args, dict) else {},
        }
    return None


def _extract_answer_payload(content: str) -> Any | None:
    content = content.strip()
    m = re.search(r"<final_answer>([\s\S]*?)</final_answer>", content, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"<answer>([\s\S]*?)</answer>", content, flags=re.IGNORECASE)
    payload = m.group(1).strip() if m else ""
    if not payload:
        return None
    try:
        return json.loads(payload)
    except Exception:
        return payload


def _task_specific_result(task_type: str, values: list[str], fallback_text: str = "") -> dict[str, Any]:
    vals = [v for v in values if isinstance(v, str) and v.strip()]
    if task_type == "ac":
        answer_smiles = vals[0] if vals else ""
        short_reason = f"Selected {answer_smiles} as the predicted molecule." if answer_smiles else "Selected the predicted molecule."
        return {
            "task_type": "ac",
            "answer_smiles": answer_smiles,
            "short_reason": short_reason,
            "evidence": [],
        }
    if task_type == "vs":
        ranked_smiles = vals
        selected_smiles = ranked_smiles[0] if ranked_smiles else ""
        short_reason = (
            f"Ranked the candidate SMILES and selected {selected_smiles} as the top candidate."
            if selected_smiles
            else "Ranked the candidate SMILES and selected the top candidate."
        )
        return {
            "task_type": "vs",
            "ranked_smiles": ranked_smiles,
            "selected_smiles": selected_smiles,
            "short_reason": short_reason,
            "evidence": [],
        }
    if task_type == "pf":
        selected_smiles = vals
        short_reason = f"Extracted {len(selected_smiles)} predicted SMILES from the final response."
        return {
            "task_type": "pf",
            "selected_smiles": selected_smiles,
            "short_reason": short_reason,
            "evidence": [],
        }
    if task_type in {"kg", "e2e"}:
        answer_text = fallback_text.strip() or (vals[0] if vals else "")
        short_reason = answer_text[:180] if answer_text else "Final response extracted from the completed session."
        return {
            "task_type": task_type,
            "answer": answer_text,
            "steps_summary": short_reason,
            "evidence": [],
        }
    answer_text = fallback_text.strip() or (vals[0] if vals else "")
    short_reason = answer_text[:180] if answer_text else "Final response extracted from the completed session."
    return {
        "task_type": task_type,
        "answer": answer_text,
        "short_reason": short_reason,
        "evidence": [],
    }


def _build_final_answer_action(rec: dict[str, Any], answer_obj: Any | None = None, fallback_text: str = "") -> dict[str, Any]:
    task_type = _task_from_id_or_meta(rec)
    if isinstance(answer_obj, dict) and str(answer_obj.get("type") or "") == "final_answer" and isinstance(answer_obj.get("answer"), dict):
        return answer_obj

    values: list[str] = []
    if isinstance(answer_obj, list):
        values = _ensure_smiles_list(answer_obj)
    elif isinstance(answer_obj, dict):
        if task_type == "ac":
            answer_smiles = _ensure_smiles_list(answer_obj.get("answer_smiles"))
            selected_molecule = _ensure_smiles_list(answer_obj.get("selected_molecule"))
            if answer_smiles and selected_molecule and answer_smiles[0] != selected_molecule[0]:
                result = dict(answer_obj)
                result.setdefault("task_type", task_type)
                summary = str(result.get("short_reason") or result.get("summary") or "Final answer generated from accepted trajectory.")
                return {
                    "type": "final_answer",
                    "task_type": task_type,
                    "answer": {
                        "summary": summary,
                        "evidence": list(result.get("evidence") or []) if isinstance(result.get("evidence"), list) else [],
                        "result": result,
                    },
                }
        for key in ("ranking", "ranked", "ranked_smiles", "ordered", "predicted_ranking", "top3", "prediction", "output", "answer"):
            values = _ensure_smiles_list(answer_obj.get(key))
            if values:
                break
        if not values:
            for key in ("answer_smiles", "selected_molecule", "selected_smiles"):
                values = _ensure_smiles_list(answer_obj.get(key))
                if values:
                    break
        if not values:
            nested = answer_obj.get("answer")
            if isinstance(nested, str) and nested.strip():
                values = [nested.strip()]
    elif isinstance(answer_obj, str):
        values = _ensure_smiles_list(answer_obj)
    if not values and fallback_text.strip():
        values = _ensure_smiles_list(fallback_text)
        if not values:
            values = [ln.strip() for ln in fallback_text.splitlines() if ln.strip()]

    result = _task_specific_result(task_type, values, fallback_text=fallback_text)
    if isinstance(answer_obj, dict) and task_type == "pf":
        labels = answer_obj.get("labels")
        if isinstance(labels, list) and labels:
            result["labels"] = [str(v).strip() for v in labels if isinstance(v, str) and v.strip()]
    return {
        "type": "final_answer",
        "task_type": task_type,
        "answer": {
            "summary": str(result.get("short_reason") or "Final answer generated from accepted trajectory."),
            "evidence": [],
            "result": result,
        },
    }


def _ensure_observation(role: str, content: str, name: str = "") -> str:
    if role == "tool":
        return f"<observation tool_name=\"{name or 'unknown'}\">{content}</observation>"
    if "<observation" in content:
        return content
    if "<tool_response>" in content:
        return f"<observation>{content}</observation>"
    return f"<observation>{content}</observation>"


def _normalize_sft_record(rec: dict[str, Any], stats: NormalizeStats) -> tuple[dict[str, Any] | None, str | None]:
    stats.total += 1

    rid = str(rec.get("id") or "")
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        stats.invalid += 1
        return None, "missing_messages"

    normalized: list[dict[str, Any]] = []
    first_system_added = False
    first_user_added = False
    final_action_added = False
    last_assistant_text = ""

    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        content = str(m.get("content") or "")
        name = str(m.get("name") or "")
        if not content.strip():
            continue

        if role == "system":
            if not first_system_added:
                normalized.append({"role": "system", "content": content, "step_loss_mask": 0})
                first_system_added = True
            continue

        if role == "user":
            if not first_user_added:
                normalized.append({"role": "user", "content": content, "step_loss_mask": 0})
                first_user_added = True
            else:
                obs = _ensure_observation("user", content)
                normalized.append({"role": "user", "content": obs, "step_loss_mask": 0})
                stats.observation_count += 1
            continue

        if role == "tool":
            obs = _ensure_observation("tool", content, name=name)
            normalized.append({"role": "user", "content": obs, "step_loss_mask": 0})
            stats.observation_count += 1
            continue

        if role == "assistant":
            last_assistant_text = content
            tool_call = _extract_tool_call_json(content)
            if tool_call is not None:
                normalized.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(tool_call, ensure_ascii=False, separators=(",", ":")),
                        "step_loss_mask": 1,
                    }
                )
                stats.assistant_action_count += 1
                stats.assistant_tool_call_count += 1
                continue

            answer_obj = _extract_answer_payload(content)
            if answer_obj is not None:
                if isinstance(answer_obj, dict) and str(answer_obj.get("type") or "") == "final_answer" and isinstance(answer_obj.get("answer"), dict):
                    action = answer_obj
                else:
                    action = _build_final_answer_action(rec, answer_obj=answer_obj, fallback_text=content)
                normalized.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(action, ensure_ascii=False, separators=(",", ":")),
                        "step_loss_mask": 1,
                    }
                )
                stats.assistant_action_count += 1
                stats.assistant_final_answer_count += 1
                final_action_added = True
                continue

            stats.assistant_text_dropped_count += 1
            continue

    if not first_system_added or not first_user_added:
        stats.invalid += 1
        return None, "missing_system_or_user"

    if not final_action_added:
        action = _build_final_answer_action(rec, answer_obj=None, fallback_text=last_assistant_text)
        normalized.append(
            {
                "role": "assistant",
                "content": json.dumps(action, ensure_ascii=False, separators=(",", ":")),
                "step_loss_mask": 1,
            }
        )
        stats.assistant_action_count += 1
        stats.assistant_final_answer_count += 1

    for m in normalized:
        if m.get("role") != "assistant":
            continue
        c = str(m.get("content") or "")
        try:
            obj = json.loads(c)
        except Exception:
            stats.assistant_json_parse_fail += 1
            stats.invalid += 1
            return None, "assistant_json_parse_fail"
        if not isinstance(obj, dict):
            stats.invalid += 1
            return None, "assistant_json_not_object"
        t = str(obj.get("type") or "")
        if t not in {"tool_call", "final_answer"}:
            stats.invalid += 1
            return None, "assistant_json_unknown_type"
        if t == "tool_call":
            if not isinstance(obj.get("tool_name"), str) or not isinstance(obj.get("arguments"), dict):
                stats.invalid += 1
                return None, "tool_call_schema_invalid"
        if t == "final_answer":
            ans = obj.get("answer")
            if not isinstance(ans, dict):
                stats.invalid += 1
                return None, "final_answer_schema_invalid"
            if not isinstance(obj.get("task_type"), str) or not str(obj.get("task_type") or "").strip():
                stats.invalid += 1
                return None, "final_answer_task_type_invalid"
            if not isinstance(ans.get("summary"), str):
                stats.invalid += 1
                return None, "final_answer_summary_invalid"
            if not isinstance(ans.get("evidence"), list):
                stats.invalid += 1
                return None, "final_answer_evidence_invalid"
            if not isinstance(ans.get("result"), dict):
                stats.invalid += 1
                return None, "final_answer_result_invalid"
            result = ans.get("result")
            task_type = str(obj.get("task_type") or _task_from_id_or_meta(rec) or "").strip().lower()
            result_task = str(result.get("task_type") or task_type).strip().lower()
            if task_type and result_task and task_type != result_task:
                stats.invalid += 1
                return None, "final_answer_task_mismatch"
            if task_type == "ac":
                if not isinstance(result.get("answer_smiles"), str) or not str(result.get("answer_smiles") or "").strip():
                    stats.invalid += 1
                    return None, "final_answer_ac_answer_smiles_invalid"
                if not isinstance(result.get("short_reason"), str) or not str(result.get("short_reason") or "").strip():
                    stats.invalid += 1
                    return None, "final_answer_ac_short_reason_invalid"
            elif task_type == "vs":
                ranked = result.get("ranked_smiles")
                selected = result.get("selected_smiles")
                ranked_ok = isinstance(ranked, list) and any(isinstance(v, str) and v.strip() for v in ranked)
                selected_ok = isinstance(selected, str) and bool(selected.strip())
                if not (ranked_ok or selected_ok):
                    stats.invalid += 1
                    return None, "final_answer_vs_ranking_invalid"
                if not isinstance(result.get("short_reason"), str) or not str(result.get("short_reason") or "").strip():
                    stats.invalid += 1
                    return None, "final_answer_vs_short_reason_invalid"
            elif task_type == "pf":
                selected_smiles = result.get("selected_smiles")
                prediction = result.get("prediction")
                selected_ok = isinstance(selected_smiles, list) and any(isinstance(v, str) and v.strip() for v in selected_smiles)
                prediction_ok = isinstance(prediction, list) and any(isinstance(v, str) and v.strip() for v in prediction)
                if not (selected_ok or prediction_ok):
                    stats.invalid += 1
                    return None, "final_answer_pf_selected_smiles_invalid"
                labels = result.get("labels")
                if labels is not None and not isinstance(labels, list):
                    stats.invalid += 1
                    return None, "final_answer_pf_labels_invalid"
                if not isinstance(result.get("short_reason"), str) or not str(result.get("short_reason") or "").strip():
                    stats.invalid += 1
                    return None, "final_answer_pf_short_reason_invalid"
            elif task_type in {"kg", "e2e"}:
                if task_type == "e2e" and "steps_summary" in result and not isinstance(result.get("steps_summary"), str):
                    stats.invalid += 1
                    return None, "final_answer_steps_summary_invalid"

    task_type = _task_from_id_or_meta(rec)
    out = {
        "schema_version": SFT_SCHEMA_VERSION,
        "id": rid,
        "messages": normalized,
    }
    stats.valid += 1
    return out, None


def _build_rl_prompt_record_from_sft(
    rec: dict[str, Any],
    index: int,
    *,
    task: str | None = None,
    summary_row: dict[str, Any] | None = None,
    cleaning_report: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None, "missing_messages"

    prompt: list[dict[str, Any]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        if role in {"system", "user"}:
            prompt.append({"role": role, "content": str(m.get("content") or "")})
        if len(prompt) >= 2:
            break
    if len(prompt) < 2:
        return None, "prompt_too_short"

    task_type = str(task or _task_from_id_or_meta(rec) or "unknown")
    data_source = f"mol_pipeline_{task_type}"
    allowed_tools = []
    if cleaning_report and isinstance(cleaning_report.get("raw_tool_name_map"), list):
        for item in cleaning_report["raw_tool_name_map"]:
            if not isinstance(item, dict) or not item.get("kept"):
                continue
            tool_name = str(item.get("tool_name") or "").strip()
            if tool_name:
                allowed_tools.append(tool_name)

    out = {
        "id": str(rec.get("id") or ""),
        "data_source": data_source,
        "prompt": prompt,
        "ability": "mol_pipeline_tool_use",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "index": index,
            "task_type": task_type,
            "source_run": str((summary_row or {}).get("run_dir") or ""),
            "trajectory_id": str(rec.get("id") or ""),
            "has_reference_trajectory": True,
            "used_molclaw": True,
            "answer_hit": (summary_row or {}).get("answer_hit_pass"),
            "tool_call_count": int((cleaning_report or {}).get("counts", {}).get("retained_mcp_tool_calls", 0)),
            "final_answer_source": (cleaning_report or {}).get("final_answer_source"),
        },
        "env_kwargs": {
            "task": {
                "task_id": str(rec.get("id") or ""),
                "task_type": task_type,
                "instruction": str(prompt[1].get("content") or ""),
                "inputs": {},
                "allowed_tools": allowed_tools,
                "max_steps": 8,
                "data_source": data_source,
            }
        },
        "metadata": {
            "source_project": "data-pipe",
            "schema_version": "rl_prompt_verl_ready_v0.1",
        },
    }
    return out, None


def _iter_kg_sample_dirs(kg_results_root: Path) -> list[Path]:
    out: list[Path] = []
    if not kg_results_root.exists():
        return out
    for run_dir in sorted([p for p in kg_results_root.iterdir() if p.is_dir() and RUN_RE.match(p.name)]):
        for row_dir in sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("row")]):
            rollout_dirs = sorted([p for p in row_dir.iterdir() if p.is_dir() and p.name.startswith("rollout")])
            if rollout_dirs:
                out.extend(rollout_dirs)
            else:
                out.append(row_dir)
    return out


def _build_kg_rl_prompt_records(kg_results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_dirs = _iter_kg_sample_dirs(kg_results_root)
    for idx, sdir in enumerate(sample_dirs):
        q = _load_json(sdir / "question.json")
        run_meta = _load_json(sdir / "run_meta.json")
        kg_spec = q.get("kg_task_spec") if isinstance(q.get("kg_task_spec"), dict) else {}
        task_id = str(kg_spec.get("task_id") or q.get("dataset_index") or sdir.name)
        question_text = str(q.get("question_text") or kg_spec.get("question") or "").strip()
        if not question_text:
            continue
        expected_tools: list[str] = []
        tc = kg_spec.get("toolchain") if isinstance(kg_spec.get("toolchain"), dict) else {}
        for t in tc.get("tools") or []:
            if isinstance(t, str) and t.strip():
                expected_tools.append(t.strip())

        split = _split_from_id(task_id, valid_ratio=0.1)
        data_source = "mol_pipeline_kg"
        rec = {
            "id": f"kg_{task_id}",
            "split": split,
            "data_source": data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": "You are a computational drug-discovery assistant. Use available tools when needed and avoid fabricating outputs.",
                },
                {"role": "user", "content": question_text},
            ],
            "ability": "mol_pipeline_tool_use",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {
                "index": idx,
                "task_type": "kg",
                "source_run": str(kg_spec.get("source", {}).get("kg_run_id", "")) if isinstance(kg_spec.get("source"), dict) else "",
                "trajectory_id": task_id,
                "has_reference_trajectory": True,
                "used_molclaw": True,
                "answer_hit": None,
                "tool_call_count": None,
                "run_return_code": run_meta.get("return_code"),
                "run_timed_out": run_meta.get("timed_out"),
            },
            "env_kwargs": {
                "task": {
                    "task_id": task_id,
                    "task_type": "kg",
                    "instruction": question_text,
                    "inputs": {},
                    "allowed_tools": expected_tools,
                    "max_steps": 8,
                    "data_source": data_source,
                }
            },
            "metadata": {
                "source_project": "data-pipe",
                "schema_version": "rl_prompt_verl_ready_v0.1",
                "kg_task_spec": kg_spec,
            },
        }
        rows.append(rec)
    return rows


def _validate_normalized_sft(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    invalid_rows: list[dict[str, Any]] = []
    invalid_reason_hist: dict[str, int] = {}
    tool_call_count = 0
    final_answer_count = 0
    observation_count = 0
    task_hist: dict[str, int] = {}

    for r in rows:
        rid = str(r.get("id") or "")
        task = _task_from_id_or_meta(r)
        msgs = r.get("messages")
        if not isinstance(msgs, list) or not msgs:
            reason = "missing_messages"
            invalid_reason_hist[reason] = invalid_reason_hist.get(reason, 0) + 1
            invalid_rows.append({"id": rid, "reason": reason})
            continue

        has_assistant = False
        ok = True
        reason = ""
        for m in msgs:
            if not isinstance(m, dict):
                ok = False
                reason = "message_not_object"
                break
            role = str(m.get("role") or "")
            if role not in {"system", "user", "assistant"}:
                ok = False
                reason = "invalid_role"
                break
            if role == "user" and "<observation" in str(m.get("content") or ""):
                observation_count += 1
            if role != "assistant":
                continue
            has_assistant = True
            try:
                obj = json.loads(str(m.get("content") or ""))
            except Exception:
                ok = False
                reason = "assistant_json_parse_fail"
                break
            if not isinstance(obj, dict):
                ok = False
                reason = "assistant_json_not_object"
                break
            t = str(obj.get("type") or "")
            if t == "tool_call":
                tool_call_count += 1
            elif t == "final_answer":
                final_answer_count += 1
                ans = obj.get("answer")
                if not isinstance(ans, dict):
                    ok = False
                    reason = "final_answer_schema_invalid"
                    break
                if not isinstance(ans.get("summary"), str) or not str(ans.get("summary") or "").strip():
                    ok = False
                    reason = "final_answer_summary_invalid"
                    break
                if not isinstance(ans.get("evidence"), list):
                    ok = False
                    reason = "final_answer_evidence_invalid"
                    break
                result = ans.get("result")
                if not isinstance(result, dict):
                    ok = False
                    reason = "final_answer_result_invalid"
                    break
                result_task = str(obj.get("task_type") or result.get("task_type") or task).strip().lower()
                if result_task and task not in {"unknown", ""} and result_task != task:
                    ok = False
                    reason = "final_answer_task_mismatch"
                    break
                if task == "ac":
                    if not isinstance(result.get("answer_smiles"), str) or not str(result.get("answer_smiles") or "").strip():
                        ok = False
                        reason = "final_answer_ac_answer_smiles_invalid"
                        break
                    if not isinstance(result.get("short_reason"), str) or not str(result.get("short_reason") or "").strip():
                        ok = False
                        reason = "final_answer_ac_short_reason_invalid"
                        break
                elif task == "vs":
                    ranked = result.get("ranked_smiles")
                    selected = result.get("selected_smiles")
                    ranked_ok = isinstance(ranked, list) and any(isinstance(v, str) and v.strip() for v in ranked)
                    selected_ok = isinstance(selected, str) and bool(selected.strip())
                    if not (ranked_ok or selected_ok):
                        ok = False
                        reason = "final_answer_vs_ranking_invalid"
                        break
                    if not isinstance(result.get("short_reason"), str) or not str(result.get("short_reason") or "").strip():
                        ok = False
                        reason = "final_answer_vs_short_reason_invalid"
                        break
                elif task == "pf":
                    selected_smiles = result.get("selected_smiles")
                    prediction = result.get("prediction")
                    selected_ok = isinstance(selected_smiles, list) and any(isinstance(v, str) and v.strip() for v in selected_smiles)
                    prediction_ok = isinstance(prediction, list) and any(isinstance(v, str) and v.strip() for v in prediction)
                    if not (selected_ok or prediction_ok):
                        ok = False
                        reason = "final_answer_pf_selected_smiles_invalid"
                        break
                    labels = result.get("labels")
                    if labels is not None and not isinstance(labels, list):
                        ok = False
                        reason = "final_answer_pf_labels_invalid"
                        break
                    if not isinstance(result.get("short_reason"), str) or not str(result.get("short_reason") or "").strip():
                        ok = False
                        reason = "final_answer_pf_short_reason_invalid"
                        break
            else:
                ok = False
                reason = "assistant_action_type_invalid"
                break
        if not has_assistant and ok:
            ok = False
            reason = "no_assistant_turn"
        if not ok:
            invalid_reason_hist[reason] = invalid_reason_hist.get(reason, 0) + 1
            invalid_rows.append({"id": rid, "reason": reason})
            continue
        task_hist[task] = task_hist.get(task, 0) + 1

    report = {
        "total_samples": len(rows),
        "valid_samples": len(rows) - len(invalid_rows),
        "invalid_samples": len(invalid_rows),
        "invalid_reasons": invalid_reason_hist,
        "assistant_json_parse_fail_count": invalid_reason_hist.get("assistant_json_parse_fail", 0),
        "tool_call_count": tool_call_count,
        "final_answer_count": final_answer_count,
        "observation_count": observation_count,
        "task_type_distribution": task_hist,
    }
    return report, invalid_rows


def _validate_rl_prompts(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = ["data_source", "prompt", "ability", "reward_model", "extra_info", "env_kwargs"]
    invalid_rows: list[dict[str, Any]] = []
    invalid_hist: dict[str, int] = {}
    task_hist: dict[str, int] = {}

    for r in rows:
        rid = str(r.get("id") or "")
        reason = ""
        for k in required:
            if k not in r:
                reason = f"missing_{k}"
                break
        if not reason:
            if not isinstance(r.get("prompt"), list):
                reason = "prompt_not_list"
            else:
                for m in r["prompt"]:
                    if not isinstance(m, dict):
                        reason = "prompt_item_not_object"
                        break
                    if not isinstance(m.get("role"), str) or not isinstance(m.get("content"), str):
                        reason = "prompt_item_schema_invalid"
                        break
        if not reason:
            ds = str(r.get("data_source") or "")
            if not ds:
                reason = "empty_data_source"
        if not reason:
            env_kwargs = r.get("env_kwargs")
            if not isinstance(env_kwargs, dict) or not isinstance(env_kwargs.get("task"), dict):
                reason = "missing_env_task"
        if reason:
            invalid_hist[reason] = invalid_hist.get(reason, 0) + 1
            invalid_rows.append({"id": rid, "reason": reason})
            continue
        task = str(r.get("extra_info", {}).get("task_type", "unknown")) if isinstance(r.get("extra_info"), dict) else "unknown"
        task_hist[task] = task_hist.get(task, 0) + 1

    report = {
        "total_samples": len(rows),
        "valid_samples": len(rows) - len(invalid_rows),
        "invalid_samples": len(invalid_rows),
        "invalid_reasons": invalid_hist,
        "task_type_distribution": task_hist,
    }
    return report, invalid_rows


def _write_markdown_report(path: Path, title: str, payload: dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for k, v in payload.items():
        lines.append(f"- {k}: {json.dumps(v, ensure_ascii=False)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_raw_refs(
    out_dir: Path,
    include_raw_samples: int,
    results_root: Path,
    scan_output_root: Path | None,
    kg_results_root: Path | None,
) -> dict[str, int]:
    ref_dir = out_dir / "raw_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "trajectory_level_samples": 0,
        "step_level_samples": 0,
        "complete_session_samples": 0,
        "question_samples": 0,
        "run_meta_samples": 0,
    }

    sample_count = 0
    if scan_output_root and scan_output_root.exists():
        for task in ["vs", "ac", "pf"]:
            tdir = scan_output_root / task
            if not tdir.is_dir():
                continue
            for p in sorted(tdir.glob("*.jsonl")):
                if sample_count >= include_raw_samples:
                    break
                rows = _load_jsonl(p)
                if not rows:
                    continue
                if rows:
                    tgt = ref_dir / "complete_session.sample.jsonl"
                    with tgt.open("a", encoding="utf-8") as f:
                        for row in rows[:40]:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats["complete_session_samples"] += 1
                sample_count += 1
            if sample_count >= include_raw_samples:
                break

    traj_sample_budget = max(3, include_raw_samples // 2)
    tl_count = 0
    for p in sorted(results_root.rglob("trajectories/trajectory_level.jsonl")):
        if tl_count >= traj_sample_budget:
            break
        rows = _load_jsonl(p)
        if not rows:
            continue
        with (ref_dir / "trajectory_level.sample.jsonl").open("a", encoding="utf-8") as f:
            for row in rows[:10]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats["trajectory_level_samples"] += 1
        tl_count += 1

    sl_count = 0
    for p in sorted(results_root.rglob("trajectories/step_level.jsonl")):
        if sl_count >= traj_sample_budget:
            break
        rows = _load_jsonl(p)
        if not rows:
            continue
        with (ref_dir / "step_level.sample.jsonl").open("a", encoding="utf-8") as f:
            for row in rows[:40]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats["step_level_samples"] += 1
        sl_count += 1

    if kg_results_root and kg_results_root.exists():
        for sdir in _iter_kg_sample_dirs(kg_results_root)[: max(3, include_raw_samples // 4)]:
            q = _load_json(sdir / "question.json")
            rm = _load_json(sdir / "run_meta.json")
            if q:
                with (ref_dir / "question.sample.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(q, ensure_ascii=False) + "\n")
                stats["question_samples"] += 1
            if rm:
                with (ref_dir / "run_meta.sample.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rm, ensure_ascii=False) + "\n")
                stats["run_meta_samples"] += 1

    return stats


def _write_schema_docs(out_dir: Path) -> None:
    schema_dir = out_dir / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    (schema_dir / "mol_pipeline_to_verl_bundle_v0.1.md").write_text(
        "# data-pipe -> verl bundle schema v0.1\n\n"
        "- Top-level bundle contains `sft/`, `rl_prompts/`, `raw_refs/`, `reports/`, and `bundle_manifest.json`.\n"
        "- No reward labels and no runtime secrets are included.\n",
        encoding="utf-8",
    )
    (schema_dir / "sft_messages_schema_v0.1.md").write_text(
        "# SFT messages schema v0.1\n\n"
        "- `messages` roles: `system|user|assistant`.\n"
        "- assistant content is strict JSON string with action type `tool_call|final_answer`.\n"
        "- observation is represented as `role=user` content wrapped by `<observation>...</observation>`.\n",
        encoding="utf-8",
    )
    (schema_dir / "rl_prompt_schema_v0.1.md").write_text(
        "# RL prompt schema v0.1\n\n"
        "- Required top-level keys: `data_source,prompt,ability,reward_model,extra_info,env_kwargs`.\n"
        "- `prompt` is `list[dict(role,content)]`.\n"
        "- `env_kwargs.task` includes task payload only and carries no endpoint/token.\n",
        encoding="utf-8",
    )
    (schema_dir / "verl_rlhf_parquet_contract_v0.1.md").write_text(
        "# VERL RLHF parquet contract v0.1\n\n"
        "- JSONL in this bundle is the authoritative contract.\n"
        "- Optional preview parquet is best-effort and should be re-validated on verl side.\n",
        encoding="utf-8",
    )


def _try_write_preview_parquet(out_jsonl: Path, out_parquet: Path) -> tuple[bool, str]:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        return False, f"pyarrow unavailable: {e}"
    rows = _load_jsonl(out_jsonl)
    try:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, out_parquet)
    except Exception as e:
        return False, f"parquet write failed: {e}"
    return True, "ok"


def _scan_security(bundle_dir: Path) -> dict[str, Any]:
    forbidden_files: list[str] = []
    leaked_pattern_files: list[str] = []
    secret_res = [
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        re.compile(r"ghp_[a-zA-Z0-9]{20,}"),
        re.compile(r"AIza[0-9A-Za-z\\-_]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"authorization\\s*:\\s*bearer\\s+[A-Za-z0-9\\-\\._]{20,}", re.IGNORECASE),
        re.compile(r"x-api-key\\s*[:=]\\s*[A-Za-z0-9\\-\\._]{20,}", re.IGNORECASE),
        re.compile(r"(api[_-]?key|token)\"?\\s*[:=]\\s*\"?[A-Za-z0-9\\-\\._]{20,}", re.IGNORECASE),
    ]
    for p in bundle_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(bundle_dir))
        if rel.endswith(".env") or rel.endswith(".env.template") or rel.endswith(".mcp.json"):
            forbidden_files.append(rel)
        if p.suffix.lower() in {".jsonl", ".json", ".md", ".txt"}:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if any(rx.search(txt) for rx in secret_res):
                leaked_pattern_files.append(rel)
    return {
        "contains_forbidden_files": bool(forbidden_files),
        "forbidden_files": forbidden_files,
        "contains_secret_pattern": bool(leaked_pattern_files),
        "secret_pattern_files": leaked_pattern_files,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    postprocess_dir = repo_root / "pipeline" / "postprocess"
    default_results_root = repo_root / "results"
    default_output = repo_root / "exports" / "mol_pipeline_to_verl_bundle_v0.1"

    ap = argparse.ArgumentParser(description="Export portable training bundle from data-pipe outputs for verl consumption.")
    ap.add_argument("--sft-output-dir", default="", help="Directory containing mcp_sft_{train,valid}.jsonl and mcp_rl_prompts_{train,valid}.jsonl")
    ap.add_argument("--scan-output-root", default="", help="Root containing extracted vs/ac/pf JSONL files.")
    ap.add_argument("--results-root", default=str(default_results_root), help="results root.")
    ap.add_argument("--run-scan", action="store_true", help="Run scan_molclaw_usage.py internally before export.")
    ap.add_argument("--run-postprocess", action="store_true", help="Run post_process_sft.py internally before export.")
    ap.add_argument("--use-accepted-only", action="store_true")
    ap.add_argument("--answer-hit-only", action="store_true")
    ap.add_argument("--output-dir", default=str(default_output), help="Bundle output directory.")
    ap.add_argument("--include-raw-samples", type=int, default=20)
    ap.add_argument("--include-kg-if-present", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-e2e", action="store_true")
    ap.add_argument("--write-preview-parquet", action="store_true")
    args = ap.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    bundle_dir = Path(args.output_dir).expanduser().resolve()
    scan_output_root = Path(args.scan_output_root).expanduser().resolve() if args.scan_output_root.strip() else None
    sft_output_dir = Path(args.sft_output_dir).expanduser().resolve() if args.sft_output_dir.strip() else None

    if args.run_scan:
        if scan_output_root is None:
            scan_output_root = results_root / "used_molclaw_accepted_hit_export"
        cmd = [
            "python",
            str(postprocess_dir / "scan_molclaw_usage.py"),
            "--results-root",
            str(results_root),
            "--output-root",
            str(scan_output_root),
        ]
        if args.use_accepted_only:
            cmd.append("--use-accepted-only")
        _run_cmd(cmd, cwd=repo_root)

    if args.run_postprocess:
        if scan_output_root is None:
            raise ValueError("--run-postprocess requires --scan-output-root or --run-scan")
        if sft_output_dir is None:
            sft_output_dir = scan_output_root / "sft_outputs"
        cmd = [
            "python",
            str(postprocess_dir / "post_process_sft.py"),
            "--input-root",
            str(scan_output_root),
            "--output-dir",
            str(sft_output_dir),
        ]
        if args.answer_hit_only:
            cmd.append("--answer-hit-only")
        _run_cmd(cmd, cwd=repo_root)

    if sft_output_dir is None:
        raise ValueError("missing --sft-output-dir (or use --run-postprocess to generate it)")
    if not sft_output_dir.exists():
        raise FileNotFoundError(f"sft output dir not found: {sft_output_dir}")

    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "sft").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "rl_prompts").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "reports").mkdir(parents=True, exist_ok=True)

    raw_train_path = sft_output_dir / "mcp_sft_train.jsonl"
    raw_valid_path = sft_output_dir / "mcp_sft_valid.jsonl"
    raw_rl_train_path = sft_output_dir / "mcp_rl_prompts_train.jsonl"
    raw_rl_valid_path = sft_output_dir / "mcp_rl_prompts_valid.jsonl"
    raw_all_path = sft_output_dir / "mcp_sft_all.jsonl"
    raw_rl_all_path = sft_output_dir / "mcp_rl_prompts_all.jsonl"

    if (not raw_train_path.is_file() or not raw_valid_path.is_file()) and not raw_all_path.is_file():
        raise FileNotFoundError(
            "missing SFT files: expected mcp_sft_train.jsonl+mcp_sft_valid.jsonl or mcp_sft_all.jsonl"
        )

    if (raw_all_path.is_file() or raw_all_path.is_dir()) and (not raw_train_path.is_file() or not raw_valid_path.is_file()):
        all_rows = _load_json_records(raw_all_path)
        raw_train: list[dict[str, Any]] = []
        raw_valid: list[dict[str, Any]] = []
        for rec in all_rows:
            rid = str(rec.get("id") or _sha256_text(json.dumps(rec, ensure_ascii=False, sort_keys=True)))
            if _split_from_id(rid) == "valid":
                raw_valid.append(rec)
            else:
                raw_train.append(rec)
        _write_jsonl(raw_train_path, raw_train)
        _write_jsonl(raw_valid_path, raw_valid)

    if (raw_rl_all_path.is_file() or raw_rl_all_path.is_dir()) and (not raw_rl_train_path.is_file() or not raw_rl_valid_path.is_file()):
        all_rl = _load_json_records(raw_rl_all_path)
        rl_train_rows: list[dict[str, Any]] = []
        rl_valid_rows: list[dict[str, Any]] = []
        for rec in all_rl:
            rid = str(rec.get("id") or _sha256_text(json.dumps(rec, ensure_ascii=False, sort_keys=True)))
            if _split_from_id(rid) == "valid":
                rl_valid_rows.append(rec)
            else:
                rl_train_rows.append(rec)
        _write_jsonl(raw_rl_train_path, rl_train_rows)
        _write_jsonl(raw_rl_valid_path, rl_valid_rows)

    shutil.copy2(raw_train_path, bundle_dir / "sft" / "mcp_sft_train.raw.jsonl")
    shutil.copy2(raw_valid_path, bundle_dir / "sft" / "mcp_sft_valid.raw.jsonl")
    if raw_rl_train_path.is_file():
        shutil.copy2(raw_rl_train_path, bundle_dir / "rl_prompts" / "mcp_rl_prompts_train.raw.jsonl")
    if raw_rl_valid_path.is_file():
        shutil.copy2(raw_rl_valid_path, bundle_dir / "rl_prompts" / "mcp_rl_prompts_valid.raw.jsonl")

    raw_train = _load_jsonl(raw_train_path)
    raw_valid = _load_jsonl(raw_valid_path)

    nstats = NormalizeStats()
    norm_train: list[dict[str, Any]] = []
    norm_valid: list[dict[str, Any]] = []
    rejected_samples: list[dict[str, Any]] = []

    for rec in raw_train:
        out, err = _normalize_sft_record(rec, nstats)
        if out is None:
            rejected_samples.append({"split": "train", "id": str(rec.get("id") or ""), "reason": err or "normalize_failed"})
            continue
        norm_train.append(out)
    for rec in raw_valid:
        out, err = _normalize_sft_record(rec, nstats)
        if out is None:
            rejected_samples.append({"split": "valid", "id": str(rec.get("id") or ""), "reason": err or "normalize_failed"})
            continue
        norm_valid.append(out)

    _write_jsonl(bundle_dir / "sft" / "mcp_sft_train.normalized_json_action.jsonl", norm_train)
    _write_jsonl(bundle_dir / "sft" / "mcp_sft_valid.normalized_json_action.jsonl", norm_valid)

    sft_report_train, sft_invalid_train = _validate_normalized_sft(norm_train)
    sft_report_valid, sft_invalid_valid = _validate_normalized_sft(norm_valid)
    sft_report = {
        "train": sft_report_train,
        "valid": sft_report_valid,
        "normalize_stats": nstats.__dict__,
    }
    _write_json(bundle_dir / "sft" / "sft_validation_report.json", sft_report)
    _write_markdown_report(bundle_dir / "sft" / "sft_validation_report.md", "SFT Validation Report", sft_report)

    rl_train: list[dict[str, Any]] = []
    rl_valid: list[dict[str, Any]] = []
    rl_rejected: list[dict[str, Any]] = []

    for i, rec in enumerate(norm_train):
        out, err = _build_rl_prompt_record_from_sft(rec, i)
        if out is None:
            rl_rejected.append({"split": "train", "id": str(rec.get("id") or ""), "reason": err or "rl_build_failed"})
            continue
        rl_train.append(out)
    for i, rec in enumerate(norm_valid):
        out, err = _build_rl_prompt_record_from_sft(rec, i)
        if out is None:
            rl_rejected.append({"split": "valid", "id": str(rec.get("id") or ""), "reason": err or "rl_build_failed"})
            continue
        rl_valid.append(out)

    kg_added = {"train": 0, "valid": 0}
    kg_results_root = results_root / "kg_sampled"
    if args.include_kg_if_present and kg_results_root.exists():
        kg_records = _build_kg_rl_prompt_records(kg_results_root)
        for rec in kg_records:
            if rec.get("split") == "valid":
                rl_valid.append(rec)
                kg_added["valid"] += 1
            else:
                rl_train.append(rec)
                kg_added["train"] += 1

    _write_jsonl(bundle_dir / "rl_prompts" / "mcp_rl_prompts_train.verl_ready.jsonl", rl_train)
    _write_jsonl(bundle_dir / "rl_prompts" / "mcp_rl_prompts_valid.verl_ready.jsonl", rl_valid)

    rl_report_train, rl_invalid_train = _validate_rl_prompts(rl_train)
    rl_report_valid, rl_invalid_valid = _validate_rl_prompts(rl_valid)
    rl_report = {
        "train": rl_report_train,
        "valid": rl_report_valid,
        "kg_records_added": kg_added,
    }
    _write_json(bundle_dir / "rl_prompts" / "rl_prompt_validation_report.json", rl_report)
    _write_markdown_report(bundle_dir / "rl_prompts" / "rl_prompt_validation_report.md", "RL Prompt Validation Report", rl_report)

    preview_parquet = {"enabled": bool(args.write_preview_parquet), "status": "disabled", "note": ""}
    if args.write_preview_parquet:
        ok_t, msg_t = _try_write_preview_parquet(
            bundle_dir / "rl_prompts" / "mcp_rl_prompts_train.verl_ready.jsonl",
            bundle_dir / "rl_prompts" / "mcp_rl_prompts_train.verl_ready.preview.parquet",
        )
        ok_v, msg_v = _try_write_preview_parquet(
            bundle_dir / "rl_prompts" / "mcp_rl_prompts_valid.verl_ready.jsonl",
            bundle_dir / "rl_prompts" / "mcp_rl_prompts_valid.verl_ready.preview.parquet",
        )
        if ok_t and ok_v:
            preview_parquet["status"] = "ok"
            preview_parquet["note"] = "best_effort parquet files written."
        else:
            preview_parquet["status"] = "failed"
            preview_parquet["note"] = f"train={msg_t}; valid={msg_v}"

    raw_ref_stats = _collect_raw_refs(
        bundle_dir,
        include_raw_samples=max(1, args.include_raw_samples),
        results_root=results_root,
        scan_output_root=scan_output_root,
        kg_results_root=kg_results_root if args.include_kg_if_present else None,
    )
    _write_schema_docs(bundle_dir)

    if (sft_output_dir / "dataset_manifest.json").is_file():
        shutil.copy2(sft_output_dir / "dataset_manifest.json", bundle_dir / "reports" / "dataset_manifest.json")
    else:
        _write_json(bundle_dir / "reports" / "dataset_manifest.json", {"note": "missing source dataset_manifest.json"})

    filtering_md = (
        "# Filtering Report\n\n"
        "- source: `scan_molclaw_usage.py` + `post_process_sft.py`\n"
        "- default sample constraints: accepted + answer-hit + molclaw-used (from upstream run config)\n"
        f"- rejected_on_normalize: {len(rejected_samples)}\n"
        f"- rejected_on_rl_prompt_build: {len(rl_rejected)}\n"
    )
    (bundle_dir / "reports" / "filtering_report.md").write_text(filtering_md, encoding="utf-8")
    _write_jsonl(bundle_dir / "reports" / "rejected_samples.jsonl", rejected_samples + rl_rejected + sft_invalid_train + sft_invalid_valid + rl_invalid_train + rl_invalid_valid)

    sec = _scan_security(bundle_dir)
    security_ok = not sec["contains_forbidden_files"] and not sec["contains_secret_pattern"]

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "source_project": "data-pipe",
        "source_root": str(repo_root),
        "source_commit": _git_head(repo_root),
        "dirty_worktree": _git_dirty(repo_root),
        "reward_included": False,
        "tool_role_mode": "user_observation",
        "normalization": {
            "sft": "json_action_v0.1",
            "rl_prompt": "verl_ready_v0.1",
        },
        "inputs": {
            "results_root": str(results_root),
            "scan_output_root": str(scan_output_root) if scan_output_root else "",
            "sft_output_dir": str(sft_output_dir),
            "include_kg_if_present": bool(args.include_kg_if_present),
            "include_e2e": bool(args.include_e2e),
        },
        "outputs": {
            "sft_train_raw": "sft/mcp_sft_train.raw.jsonl",
            "sft_train_normalized": "sft/mcp_sft_train.normalized_json_action.jsonl",
            "rl_prompts_train_verl_ready": "rl_prompts/mcp_rl_prompts_train.verl_ready.jsonl",
        },
        "counts": {
            "sft_train_raw": len(raw_train),
            "sft_valid_raw": len(raw_valid),
            "sft_train_normalized_valid": sft_report_train["valid_samples"],
            "sft_valid_normalized_valid": sft_report_valid["valid_samples"],
            "rl_train_total": len(rl_train),
            "rl_valid_total": len(rl_valid),
            "kg_records_added_train": kg_added["train"],
            "kg_records_added_valid": kg_added["valid"],
            "by_task_type": {
                "train": rl_report_train.get("task_type_distribution", {}),
                "valid": rl_report_valid.get("task_type_distribution", {}),
            },
        },
        "validation": {
            "sft_valid": sft_report_train["invalid_samples"] == 0 and sft_report_valid["invalid_samples"] == 0,
            "rl_prompt_valid": rl_report_train["invalid_samples"] == 0 and rl_report_valid["invalid_samples"] == 0,
            "num_errors": sft_report_train["invalid_samples"] + sft_report_valid["invalid_samples"] + rl_report_train["invalid_samples"] + rl_report_valid["invalid_samples"],
            "reports": [
                "sft/sft_validation_report.md",
                "rl_prompts/rl_prompt_validation_report.md",
            ],
        },
        "preview_parquet": preview_parquet,
        "raw_ref_stats": raw_ref_stats,
        "security": {
            "contains_api_key": sec["contains_secret_pattern"],
            "contains_mcp_config": sec["contains_forbidden_files"],
            "contains_env_file": sec["contains_forbidden_files"],
            "security_ok": security_ok,
            "details": sec,
        },
    }
    _write_json(bundle_dir / "bundle_manifest.json", manifest)

    summary = {
        "bundle_dir": str(bundle_dir),
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "sft_train_raw": len(raw_train),
        "sft_valid_raw": len(raw_valid),
        "sft_train_normalized": len(norm_train),
        "sft_valid_normalized": len(norm_valid),
        "rl_train": len(rl_train),
        "rl_valid": len(rl_valid),
        "kg_records_added": kg_added,
        "security_ok": security_ok,
    }
    _write_json(bundle_dir / "reports" / "export_summary.json", summary)
    (bundle_dir / "reports" / "export_summary.md").write_text(
        "# Export Summary\n\n"
        f"- bundle_dir: `{bundle_dir}`\n"
        f"- sft_train_raw: {len(raw_train)}\n"
        f"- sft_valid_raw: {len(raw_valid)}\n"
        f"- sft_train_normalized: {len(norm_train)}\n"
        f"- sft_valid_normalized: {len(norm_valid)}\n"
        f"- rl_train: {len(rl_train)}\n"
        f"- rl_valid: {len(rl_valid)}\n"
        f"- kg_records_added: train={kg_added['train']}, valid={kg_added['valid']}\n"
        f"- security_ok: {security_ok}\n",
        encoding="utf-8",
    )

    (bundle_dir / "README.md").write_text(
        "# data-pipe to verl bundle v0.1\n\n"
        "This bundle is a portable handoff package from `data-pipe` for downstream training ingestion.\n\n"
        "## Contents\n\n"
        "- `sft/`: raw and normalized SFT messages.\n"
        "- `rl_prompts/`: raw and verl-ready RL prompt records.\n"
        "- `raw_refs/`: sampled raw references for audit and traceability.\n"
        "- `reports/`: export summary, filtering report, and rejected sample traces.\n"
        "- `schemas/`: compact schema contracts.\n\n"
        "## Notes\n\n"
        "- Reward fields are intentionally excluded.\n"
        "- Endpoint/token or runtime secrets are not included.\n"
        "- RL prompt records are designed to be converted downstream to parquet if needed.\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
