#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, deque
from pathlib import Path
from typing import Any

TOOL_RE = re.compile(r"<tool_call>([\s\S]*?)</tool_call>")
OBS_RE = re.compile(r'<observation\s+tool_name="([^"]+)">([\s\S]*?)</observation>')
FINAL_RE = re.compile(r"<final_answer>([\s\S]*?)</final_answer>")
ARTIFACT_RE = re.compile(r"<artifact:[^>]+>")
COMMON_EXTENSIONS = (
    "json", "pdb", "cif", "mmcif", "sdf", "mol2", "pdbqt", "csv", "tsv",
    "txt", "md", "png", "jpg", "jpeg", "svg", "html", "npy", "npz", "pt", "pkl",
)
KNOWN_DIRS = (
    "protein_seq", "protein_structures", "pdbfixer_result", "fpocket_result",
    "pocket_result", "docking", "docking_results", "boltz_data", "boltz_results",
    "exp_data", "outputs", "result", "results",
)
RELATIVE_PATH_RE = re.compile(
    rf"(?<![:/A-Za-z0-9])(?P<path>(?:\.\.?/)+(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+\.({'|'.join(COMMON_EXTENSIONS)}))"
    rf"|(?<![:/A-Za-z0-9])(?P<known>(?:{'|'.join(KNOWN_DIRS)})/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+\.({'|'.join(COMMON_EXTENSIONS)}))"
)
DEBUG_KEYS = {
    "metadata", "pointers", "raw_pointer", "tool_use_id", "raw_tool_name",
    "raw_status", "raw_is_error", "raw_event_index", "fence_wrapper_stripped",
}
ERROR_STATUSES = {"error", "failed", "failure", "false"}
ERROR_PHRASE_RE = re.compile(
    r"(?i)(?<!no )(?<!without )(?:timeout|connection refused|failed|traceback|exception|"
    r"missing_argument|service unavailable|http\s*[45]\d\d)"
)
MOLECULE_KEYS = ("smiles", "ligand", "molecule", "candidate")
ARTIFACT_DIR_NAMES = {
    "protein_seq": "protein_sequence",
    "protein_structures": "protein_structure",
    "pdbfixer_result": "pdbfixer",
    "fpocket_result": "fpocket",
    "pocket_result": "pocket",
    "docking": "docking",
    "docking_results": "docking",
    "boltz_data": "boltz",
    "boltz_results": "boltz",
    "exp_data": "local",
    "outputs": "local",
    "result": "local",
    "results": "local",
}


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except Exception:
        return None


def _render_tag(tag: str, payload: dict[str, Any], tool_name: str = "") -> str:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if tag == "observation":
        return f'<observation tool_name="{tool_name}">{body}</observation>'
    return f"<{tag}>{body}</{tag}>"


def _artifact_from_relative(path: str) -> str:
    parts: list[str] = []
    for part in path.replace("\\", "/").split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            continue
        parts.append(part)
    if parts:
        parts[0] = ARTIFACT_DIR_NAMES.get(parts[0], parts[0])
        if len(parts) == 1:
            for prefix, artifact_dir in (
                ("pdbfixer_", "pdbfixer"),
                ("fpocket_", "fpocket"),
                ("boltz_", "boltz"),
                ("docking_", "docking"),
            ):
                if parts[0].startswith(prefix):
                    parts.insert(0, artifact_dir)
                    break
    return f"<artifact:{'/'.join(parts) or 'local/result'}>"


def sanitize_relative_paths(text: str, report: dict[str, Any]) -> str:
    if not text:
        return text
    protected: list[str] = []

    def protect(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"__FINAL_ARTIFACT_{len(protected) - 1}__"

    value = ARTIFACT_RE.sub(protect, text)

    def replace(match: re.Match[str]) -> str:
        raw = match.group("path") or match.group("known") or ""
        artifact = _artifact_from_relative(raw)
        report.setdefault("sanitized_relative_paths", []).append({"before": raw, "after": artifact})
        return artifact

    value = RELATIVE_PATH_RE.sub(replace, value)
    for idx, artifact in enumerate(protected):
        value = value.replace(f"__FINAL_ARTIFACT_{idx}__", artifact)
    return value


def _sanitize_structure(value: Any, report: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return sanitize_relative_paths(value, report)
    if isinstance(value, list):
        return [_sanitize_structure(item, report) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_structure(item, report) for key, item in value.items()}
    return value


def _detect_observation_status_conflict(obs: dict[str, Any], report: dict[str, Any], tool_name: str) -> None:
    content = obs.get("content")
    metadata = obs.get("metadata") if isinstance(obs.get("metadata"), dict) else {}
    error_value = content.get("error") if isinstance(content, dict) else None
    error_text = str(error_value or "").strip().lower()
    content_error = bool(error_value) and not (
        error_text.startswith("no error") or error_text.startswith("without error")
    )
    content_text = json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content or "")
    content_error_phrase = bool(ERROR_PHRASE_RE.search(content_text))
    content_status = content.get("status") if isinstance(content, dict) else None
    content_status_error = str(content_status).strip().lower() in ERROR_STATUSES if content_status is not None else False
    raw_is_error = metadata.get("raw_is_error") is True
    outer_error = obs.get("ok") is False or str(obs.get("status") or "").strip().lower() in ERROR_STATUSES
    success_wrapper = obs.get("ok") is True or str(obs.get("status") or "").strip().lower() in {"success", "partial_success"}
    if not success_wrapper or not (content_error or content_status_error or content_error_phrase or raw_is_error or outer_error):
        return
    if "observation_status_conflict_after_llm_clean" not in report.setdefault("errors", []):
        report["errors"].append("observation_status_conflict_after_llm_clean")
    report.setdefault("observation_status_conflicts", []).append(
        {
            "tool_name": tool_name,
            "ok": obs.get("ok"),
            "status": obs.get("status"),
            "content_status": content_status,
            "content_error_present": content_error,
            "content_error_phrase_present": content_error_phrase,
            "raw_is_error": raw_is_error,
        }
    )


def _strip_debug_keys(value: Any, removed: Counter[str]) -> None:
    if isinstance(value, dict):
        for key in list(value):
            if key in DEBUG_KEYS:
                removed[key] += 1
                value.pop(key, None)
            else:
                _strip_debug_keys(value[key], removed)
    elif isinstance(value, list):
        for item in value:
            _strip_debug_keys(item, removed)


def _clean_observation_message(content: str, message_idx: int, report: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        tool_name, inner = match.group(1), match.group(2)
        obs = _parse_json(inner)
        if not isinstance(obs, dict):
            report.setdefault("errors", []).append(f"message_{message_idx}_observation_json_invalid")
            return match.group(0)
        # Detect before stripping metadata because raw_is_error may be needed.
        _detect_observation_status_conflict(obs, report, tool_name)
        obs = _sanitize_structure(obs, report)
        removed: Counter[str] = Counter()
        _strip_debug_keys(obs, removed)
        if removed:
            report.setdefault("removed_observation_metadata", []).append(
                {"message_idx": message_idx, "tool_name": tool_name, "removed_keys": dict(removed)}
            )
        return _render_tag("observation", obs, tool_name)

    return OBS_RE.sub(replace, content)


def _extract_candidates(user_text: str) -> list[str]:
    match = re.search(r'"candidates"\s*:\s*', user_text)
    if not match:
        return []
    list_start = user_text.find("[", match.end())
    if list_start < 0:
        return []
    try:
        parsed, _ = json.JSONDecoder().raw_decode(user_text[list_start:])
    except Exception:
        parsed = None
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _molecule_from_arguments(arguments: Any) -> str | None:
    if not isinstance(arguments, dict):
        return None
    for key in MOLECULE_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip() and "artifact:" not in value:
            return value.strip()
    for value in arguments.values():
        found = _molecule_from_arguments(value) if isinstance(value, dict) else None
        if found:
            return found
    return None


def _docking_score(obs: dict[str, Any]) -> float | None:
    def walk(value: Any) -> float | None:
        if isinstance(value, dict):
            candidate = value.get("docking_affinity_value")
            if isinstance(candidate, (int, float)):
                return float(candidate)
            for nested in value.values():
                found = walk(nested)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for nested in value:
                found = walk(nested)
                if found is not None:
                    return found
        return None
    return walk(obs)


def _collect_vs_scores(messages: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    pending: deque[tuple[str, str | None, int]] = deque()
    scores: dict[str, float] = {}
    bindings: list[dict[str, Any]] = []
    call_order = 0
    for message in messages:
        content = str(message.get("content") or "")
        if message.get("role") == "assistant":
            for match in TOOL_RE.finditer(content):
                call = _parse_json(match.group(1))
                if not isinstance(call, dict):
                    continue
                tool_name = str(call.get("tool_name") or "")
                if not any(token in tool_name.lower() for token in ("docking", "quickvina")):
                    continue
                pending.append((tool_name, _molecule_from_arguments(call.get("arguments")), call_order))
                call_order += 1
        elif message.get("role") == "user":
            for match in OBS_RE.finditer(content):
                obs = _parse_json(match.group(2))
                if not isinstance(obs, dict) or not pending:
                    continue
                observed_tool = match.group(1)
                tool_name, smiles, order = pending[0]
                if observed_tool != tool_name:
                    continue
                pending.popleft()
                score = _docking_score(obs)
                if smiles and score is not None and obs.get("ok") is not False and str(obs.get("status") or "") != "error":
                    scores[smiles] = score
                    bindings.append({"smiles": smiles, "score": score, "call_order": order, "tool": tool_name})
    return scores, bindings


def _detect_vs_ranking_inconsistency(messages: list[dict[str, Any]], report: dict[str, Any]) -> None:
    user_text = next((str(m.get("content") or "") for m in messages if m.get("role") == "user"), "")
    candidates = _extract_candidates(user_text)
    final_message: dict[str, Any] | None = None
    final_obj: dict[str, Any] | None = None
    for message in messages:
        if message.get("role") != "assistant":
            continue
        matches = list(FINAL_RE.finditer(str(message.get("content") or "")))
        if matches:
            parsed = _parse_json(matches[-1].group(1))
            if isinstance(parsed, dict):
                final_message, final_obj = message, parsed
    task = str((final_obj or {}).get("task_type") or "").lower()
    if task != "vs" and "MolBench-VS" not in user_text and not isinstance((final_obj or {}).get("ranked_smiles"), list):
        return
    if not candidates or not isinstance(final_obj, dict):
        report.setdefault("warnings", []).append("vs_ranking_ambiguous_after_llm_clean")
        return
    scores, bindings = _collect_vs_scores(messages)
    if not scores:
        report.setdefault("warnings", []).append("vs_ranking_score_coverage_missing_after_llm_clean")
        return
    ranked = final_obj.get("ranked_smiles")
    if not isinstance(ranked, list):
        report.setdefault("errors", []).append("vs_ranking_inconsistent_after_llm_clean")
        return
    ranked = [str(item) for item in ranked]
    structural_error = (
        len(ranked) != len(candidates)
        or len(ranked) != len(set(ranked))
        or any(smiles not in set(candidates) for smiles in ranked)
    )
    seen_unscored = False
    previous_score: float | None = None
    non_monotonic = False
    inversions: list[dict[str, Any]] = []
    previous_smiles: str | None = None
    for smiles in ranked:
        score = scores.get(smiles)
        if score is None:
            seen_unscored = True
            continue
        if seen_unscored or (previous_score is not None and score < previous_score):
            non_monotonic = True
            inversions.append(
                {
                    "previous_smiles": previous_smiles,
                    "previous_score": previous_score,
                    "current_smiles": smiles,
                    "current_score": score,
                    "scored_after_unscored": seen_unscored,
                }
            )
        previous_score = score
        previous_smiles = smiles
    if structural_error or non_monotonic:
        if "vs_ranking_inconsistent_after_llm_clean" not in report.setdefault("errors", []):
            report["errors"].append("vs_ranking_inconsistent_after_llm_clean")
    if len(scores) < len(candidates):
        report.setdefault("warnings", []).append("vs_ranking_score_coverage_incomplete_after_llm_clean")
    report["vs_ranking_detection"] = {
        "ranked_smiles": ranked,
        "score_count": len(scores),
        "candidate_count": len(candidates),
        "structural_error": structural_error,
        "non_monotonic": non_monotonic,
        "inversions": inversions[:10],
        "bindings": bindings,
    }


def clean_sample(sample: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    report: dict[str, Any] = {
        "sample_id": str(sample.get("id") or ""),
        "errors": [],
        "warnings": [],
    }
    if set(sample) != {"schema_version", "id", "messages"} or not isinstance(sample.get("messages"), list):
        report["errors"].append("invalid_top_level_shape")
        return None, report
    cleaned = json.loads(json.dumps(sample))
    messages = cleaned["messages"]
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            report["errors"].append(f"message_{idx}_not_object")
            continue
        content = str(message.get("content") or "")
        if message.get("role") == "user" and "<observation " in content:
            content = _clean_observation_message(content, idx, report)
        content = sanitize_relative_paths(content, report)
        message["content"] = content
    _detect_vs_ranking_inconsistency(messages, report)
    if report["errors"]:
        return None, report
    return cleaned, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply deterministic final hard-clean rules to LLM-cleaned ReAct SFT samples.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--quarantine-dir")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    quarantine_dir = Path(args.quarantine_dir).expanduser().resolve() if args.quarantine_dir else report_dir / "quarantine"
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    for path in (output_dir, report_dir, quarantine_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    index: list[dict[str, Any]] = []
    for source in sorted(input_dir.glob("*.json")):
        try:
            sample = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            sample = {}
        cleaned, report = clean_sample(sample if isinstance(sample, dict) else {})
        report["source"] = str(source)
        if cleaned is None:
            status = "quarantined"
            shutil.copy2(source, quarantine_dir / source.name)
        else:
            status = "cleaned"
            (output_dir / source.name).write_text(json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["status"] = status
        (report_dir / f"{source.stem}.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        index.append(
            {
                "source": str(source),
                "status": status,
                "errors": report.get("errors", []),
                "warnings": report.get("warnings", []),
            }
        )
        counts[status] += 1
    summary = {"input_dir": str(input_dir), "output_dir": str(output_dir), "report_dir": str(report_dir), **counts, "files": index}
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
