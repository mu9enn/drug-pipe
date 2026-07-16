#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


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


def _err(errors: list[str], msg: str) -> None:
    errors.append(msg)


def _task_from_id(rid: str) -> str:
    m = re.match(r"^mcp_sft_(vs|ac|pf|kg|e2e)_", str(rid or ""))
    return m.group(1) if m else "unknown"


def _validate_sft_file(path: Path, errors: list[str]) -> dict[str, Any]:
    rows = _load_jsonl(path)
    assistant_json_fail = 0
    for i, r in enumerate(rows, 1):
        rid = str(r.get("id") or "")
        task = _task_from_id(rid)
        msgs = r.get("messages")
        if not isinstance(msgs, list) or not msgs:
            _err(errors, f"{path.name}: row {i} missing messages")
            continue
        has_assistant = False
        for m in msgs:
            if not isinstance(m, dict):
                _err(errors, f"{path.name}: row {i} message not object")
                continue
            role = str(m.get("role") or "")
            if role not in {"system", "user", "assistant"}:
                _err(errors, f"{path.name}: row {i} invalid role {role}")
                continue
            if role != "assistant":
                continue
            has_assistant = True
            c = str(m.get("content") or "")
            try:
                obj = json.loads(c)
            except Exception:
                assistant_json_fail += 1
                _err(errors, f"{path.name}: row {i} assistant content not JSON")
                continue
            if not isinstance(obj, dict):
                _err(errors, f"{path.name}: row {i} assistant JSON is not object")
                continue
            t = str(obj.get("type") or "")
            if t not in {"tool_call", "final_answer"}:
                _err(errors, f"{path.name}: row {i} assistant action type invalid ({t})")
                continue
            if t == "tool_call":
                if not isinstance(obj.get("tool_name"), str) or not isinstance(obj.get("arguments"), dict):
                    _err(errors, f"{path.name}: row {i} tool_call schema invalid")
            if t == "final_answer":
                if not isinstance(obj.get("task_type"), str) or not str(obj.get("task_type") or "").strip():
                    _err(errors, f"{path.name}: row {i} final_answer.task_type invalid")
                ans = obj.get("answer")
                if not isinstance(ans, dict):
                    _err(errors, f"{path.name}: row {i} final_answer.answer invalid")
                else:
                    if not isinstance(ans.get("summary"), str):
                        _err(errors, f"{path.name}: row {i} final_answer.summary invalid")
                    if not isinstance(ans.get("evidence"), list):
                        _err(errors, f"{path.name}: row {i} final_answer.evidence invalid")
                    if not isinstance(ans.get("result"), dict):
                        _err(errors, f"{path.name}: row {i} final_answer.result invalid")
                    else:
                        result = ans.get("result")
                        result_task = str(result.get("task_type") or obj.get("task_type") or task).strip().lower()
                        if task != "unknown" and result_task and result_task != task:
                            _err(errors, f"{path.name}: row {i} final_answer task mismatch")
                        if task == "ac":
                            if not isinstance(result.get("answer_smiles"), str) or not str(result.get("answer_smiles") or "").strip():
                                _err(errors, f"{path.name}: row {i} ac answer_smiles invalid")
                            if not isinstance(result.get("short_reason"), str) or not str(result.get("short_reason") or "").strip():
                                _err(errors, f"{path.name}: row {i} ac short_reason invalid")
                        elif task == "vs":
                            ranked = result.get("ranked_smiles")
                            selected = result.get("selected_smiles")
                            ranked_ok = isinstance(ranked, list) and any(isinstance(v, str) and v.strip() for v in ranked)
                            selected_ok = isinstance(selected, str) and bool(selected.strip())
                            if not (ranked_ok or selected_ok):
                                _err(errors, f"{path.name}: row {i} vs ranking invalid")
                            if not isinstance(result.get("short_reason"), str) or not str(result.get("short_reason") or "").strip():
                                _err(errors, f"{path.name}: row {i} vs short_reason invalid")
                        elif task == "pf":
                            selected_smiles = result.get("selected_smiles")
                            prediction = result.get("prediction")
                            selected_ok = isinstance(selected_smiles, list) and any(isinstance(v, str) and v.strip() for v in selected_smiles)
                            prediction_ok = isinstance(prediction, list) and any(isinstance(v, str) and v.strip() for v in prediction)
                            if not (selected_ok or prediction_ok):
                                _err(errors, f"{path.name}: row {i} pf selected_smiles invalid")
                            labels = result.get("labels")
                            if labels is not None and not isinstance(labels, list):
                                _err(errors, f"{path.name}: row {i} pf labels invalid")
                            if not isinstance(result.get("short_reason"), str) or not str(result.get("short_reason") or "").strip():
                                _err(errors, f"{path.name}: row {i} pf short_reason invalid")
        if not has_assistant:
            _err(errors, f"{path.name}: row {i} has no assistant turn")
    return {"rows": len(rows), "assistant_json_parse_fail_count": assistant_json_fail}


def _validate_rl_file(path: Path, errors: list[str]) -> dict[str, Any]:
    rows = _load_jsonl(path)
    required = ["data_source", "prompt", "ability", "reward_model", "extra_info", "env_kwargs"]
    for i, r in enumerate(rows, 1):
        for k in required:
            if k not in r:
                _err(errors, f"{path.name}: row {i} missing key {k}")
        prompt = r.get("prompt")
        if not isinstance(prompt, list):
            _err(errors, f"{path.name}: row {i} prompt is not list")
        else:
            for j, m in enumerate(prompt, 1):
                if not isinstance(m, dict):
                    _err(errors, f"{path.name}: row {i} prompt item {j} not object")
                    continue
                if not isinstance(m.get("role"), str) or not isinstance(m.get("content"), str):
                    _err(errors, f"{path.name}: row {i} prompt item {j} schema invalid")
        ds = str(r.get("data_source") or "")
        if not ds.strip():
            _err(errors, f"{path.name}: row {i} empty data_source")
        env_kwargs = r.get("env_kwargs")
        if not isinstance(env_kwargs, dict) or not isinstance(env_kwargs.get("task"), dict):
            _err(errors, f"{path.name}: row {i} missing env_kwargs.task")
        else:
            task_ds = str(env_kwargs.get("task", {}).get("data_source") or "")
            if task_ds and ds and task_ds != ds:
                _err(errors, f"{path.name}: row {i} env_kwargs.task.data_source mismatch")
        if "reward" in r:
            _err(errors, f"{path.name}: row {i} should not include reward field")
    return {"rows": len(rows)}


def _scan_security(bundle_dir: Path) -> dict[str, Any]:
    forbidden_files: list[str] = []
    secret_pattern_files: list[str] = []
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
        name = p.name.lower()
        if name == ".env" or name == ".mcp.json":
            forbidden_files.append(rel)
        if p.suffix.lower() in {".jsonl", ".json", ".md", ".txt"}:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if any(rx.search(txt) for rx in secret_res):
                secret_pattern_files.append(rel)
    return {
        "forbidden_files": forbidden_files,
        "secret_pattern_files": secret_pattern_files,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate exported data-pipe -> verl training bundle.")
    ap.add_argument("--bundle-dir", required=True)
    args = ap.parse_args()

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    if not bundle_dir.exists():
        raise FileNotFoundError(f"bundle not found: {bundle_dir}")

    errors: list[str] = []
    required_files = [
        "bundle_manifest.json",
        "README.md",
        "schemas/mol_pipeline_to_verl_bundle_v0.1.md",
        "schemas/sft_messages_schema_v0.1.md",
        "schemas/rl_prompt_schema_v0.1.md",
        "sft/mcp_sft_train.raw.jsonl",
        "sft/mcp_sft_valid.raw.jsonl",
        "sft/mcp_sft_train.normalized_json_action.jsonl",
        "sft/mcp_sft_valid.normalized_json_action.jsonl",
        "sft/sft_validation_report.json",
        "rl_prompts/mcp_rl_prompts_train.verl_ready.jsonl",
        "rl_prompts/mcp_rl_prompts_valid.verl_ready.jsonl",
        "rl_prompts/rl_prompt_validation_report.json",
        "reports/export_summary.json",
        "reports/filtering_report.md",
    ]

    for rel in required_files:
        p = bundle_dir / rel
        if not p.is_file():
            _err(errors, f"missing required file: {rel}")

    manifest = _load_json(bundle_dir / "bundle_manifest.json")
    if not manifest:
        _err(errors, "bundle_manifest.json unreadable")
    else:
        if not isinstance(manifest.get("schema_version"), str):
            _err(errors, "manifest.schema_version invalid")

    sft_train_stats = _validate_sft_file(bundle_dir / "sft/mcp_sft_train.normalized_json_action.jsonl", errors)
    sft_valid_stats = _validate_sft_file(bundle_dir / "sft/mcp_sft_valid.normalized_json_action.jsonl", errors)
    rl_train_stats = _validate_rl_file(bundle_dir / "rl_prompts/mcp_rl_prompts_train.verl_ready.jsonl", errors)
    rl_valid_stats = _validate_rl_file(bundle_dir / "rl_prompts/mcp_rl_prompts_valid.verl_ready.jsonl", errors)

    sec = _scan_security(bundle_dir)
    if sec["forbidden_files"]:
        _err(errors, f"forbidden files present: {sec['forbidden_files']}")
    if sec["secret_pattern_files"]:
        _err(errors, f"secret-like patterns found: {sec['secret_pattern_files']}")

    summary = {
        "bundle_dir": str(bundle_dir),
        "ok": len(errors) == 0,
        "num_errors": len(errors),
        "errors": errors,
        "sft_train": sft_train_stats,
        "sft_valid": sft_valid_stats,
        "rl_train": rl_train_stats,
        "rl_valid": rl_valid_stats,
        "security": sec,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_report = bundle_dir / "reports" / "bundle_validation_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
