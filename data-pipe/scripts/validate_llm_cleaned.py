#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

THOUGHT_RE = re.compile(r"<thought>([\s\S]*?)</thought>")
TOOL_RE = re.compile(r"<tool_call>([\s\S]*?)</tool_call>")
OBS_RE = re.compile(r'<observation\s+tool_name="([^"]+)">([\s\S]*?)</observation>')
FINAL_RE = re.compile(r"<final_answer>([\s\S]*?)</final_answer>")
LOCAL_PATH_RE = re.compile(r"/(?:root|home|tmp|mnt|workspace)/")
RELATIVE_PATH_RE = re.compile(
    r"(?<![:/A-Za-z0-9])(?:\.\.?/)+(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+\.(?:json|pdb|cif|mmcif|sdf|mol2|pdbqt|csv|tsv|txt|md|png|jpg|jpeg|svg|html|npy|npz|pt|pkl)"
    r"|(?<![:/A-Za-z0-9])(?:protein_seq|protein_structures|pdbfixer_result|fpocket_result|pocket_result|docking|docking_results|boltz_data|boltz_results|exp_data|outputs|result|results)/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+\.(?:json|pdb|cif|mmcif|sdf|mol2|pdbqt|csv|tsv|txt|md|png|jpg|jpeg|svg|html|npy|npz|pt|pkl)"
)
MARKDOWN_ARTIFACT_RE = re.compile(r"\[artifact:[^\]]+\]\(artifact:[^)]+\)")
MALFORMED_ARTIFACT_PATTERNS = (
    "<[artifact:",
    "skills[artifact:",
    "kcal[artifact:",
    "[artifact:local/C]",
    "[artifact:local/c3cc]",
    "[artifact:local/CN]",
    "[artifact:local/C2]",
)
ENGINEERING_PHRASES = (
    ".claude",
    "skills directory",
    "skill directory",
    "run_log",
    "todo list",
    "update todo",
    "phase 0",
    "task type triage",
    "type a",
    "methodology files",
    "workflow skills",
    "read the relevant skills",
    "read the methodology",
    "checking the directory",
    "current working directory",
)
ENGINEERING_LEVEL_RE = re.compile(
    r"(?i)\b(?:read(?:ing)?|load(?:ing)?|check(?:ing)?|follow(?:ing)?)\s+(?:the\s+)?l[123]\b"
    r"|\bl[123]\s+(?:workflow|methodology|skills?|protocol)\b"
)
UNSUPPORTED_METRIC_PATTERNS = (
    re.compile(r"(?i)\blog10\s*\(\s*(?:ic50|ki)\s*\)"),
    re.compile(r"(?i)\b(?:converted|corresponds)\s+to\s+(?:ic50|ki)\b"),
    re.compile(r"(?i)\b(?:ic50|ki)\s*(?:≈|~=|~|approximately|approx\.?)"),
    re.compile(r"(?i)\baffinity_pred_value\b[^\n]{0,120}\b(?:log10|ic50|ki)\b"),
)
PLAN_RULES = (
    ("will run docking", ("docking", "quickvina")),
    ("run quickvina", ("quickvina",)),
    ("rescore with equiscore", ("equiscore",)),
    ("cross-validation", ("cross", "validation")),
    ("supplemented by docking", ("docking", "quickvina")),
)
MOLECULAR_KEYS = {
    "smiles",
    "smiles_list",
    "ligand",
    "ligands",
    "molecule",
    "molecules",
    "candidate",
    "candidates",
    "answer_smiles",
    "selected_smiles",
    "ranked_smiles",
    "prediction",
    "predictions",
}
OBSERVATION_DEBUG_KEYS = {
    "metadata", "pointers", "raw_pointer", "tool_use_id", "raw_tool_name",
    "raw_status", "raw_is_error", "raw_event_index", "fence_wrapper_stripped",
}
ERROR_STATUSES = {"error", "failed", "failure", "false", "timeout"}
ERROR_PHRASE_RE = re.compile(
    r"(?i)(?<!no )(?<!without )(?:timeout|connection refused|failed|traceback|exception|"
    r"missing_argument|service unavailable|http\s*[45]\d\d)"
)


def _add_once(items: list[str], issue: str) -> None:
    if issue not in items:
        items.append(issue)


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except Exception:
        return None


def _task_from_sample(sample: dict[str, Any], final_obj: dict[str, Any] | None) -> str:
    if isinstance(final_obj, dict):
        task = str(final_obj.get("task_type") or "").strip().lower()
        if task:
            return task
    match = re.match(r"^mcp_sft_(vs|ac|pf|kg|e2e)_", str(sample.get("id") or ""))
    return match.group(1) if match else "unknown"


def _collect_keyed_strings(value: Any, active: bool = False) -> list[str]:
    out: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            out.extend(_collect_keyed_strings(nested, active or str(key).lower() in MOLECULAR_KEYS))
    elif isinstance(value, list):
        for nested in value:
            out.extend(_collect_keyed_strings(nested, active))
    elif active and isinstance(value, str) and value.strip():
        out.append(value.strip())
    return out


def _find_recursive_keys(value: Any, keys: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in keys:
                found.add(key)
            found.update(_find_recursive_keys(nested, keys))
    elif isinstance(value, list):
        for nested in value:
            found.update(_find_recursive_keys(nested, keys))
    return found


def _allowed_molecules(task: str, user_text: str) -> set[str]:
    if task == "ac":
        return {
            match.group(1).strip()
            for match in re.finditer(r"(?im)^\s*Molecule\s+[AB]\s*:\s*(\S+)\s*$", user_text)
        }
    if task == "pf":
        match = re.search(r"(?is)\bSMILES\s*:\s*(.*?)(?:\n\s*Constraints\s*:|$)", user_text)
        if not match:
            return set()
        return {line.strip(" -*\t") for line in match.group(1).splitlines() if line.strip(" -*\t")}
    if task == "vs":
        match = re.search(r'"candidates"\s*:\s*', user_text)
        if match:
            list_start = user_text.find("[", match.end())
            try:
                parsed, _ = json.JSONDecoder().raw_decode(user_text[list_start:]) if list_start >= 0 else (None, 0)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return {str(item).strip() for item in parsed if str(item).strip()}
    return set()


def _extract_messages(sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        return [], ["missing_messages"], warnings
    valid: list[dict[str, Any]] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            _add_once(errors, f"message_{idx}_not_object")
            continue
        role = str(message.get("role") or "")
        if role not in {"system", "user", "assistant"}:
            _add_once(errors, f"message_{idx}_invalid_role")
        valid.append(message)
    return valid, errors, warnings


def _fpocket_checks(observations: list[tuple[str, dict[str, Any]]], errors: list[str]) -> None:
    for tool_name, obs in observations:
        if "fpocket" not in tool_name.lower():
            continue
        content = obs.get("content") if isinstance(obs.get("content"), dict) else {}
        top = content.get("top_pocket") if isinstance(content.get("top_pocket"), dict) else {}
        center, size = top.get("center"), top.get("size")
        if isinstance(center, list) and isinstance(size, list) and center == size:
            _add_once(errors, "fpocket_size_equals_center")
        if isinstance(size, list) and any(isinstance(v, (int, float)) and v < 0 for v in size):
            _add_once(errors, "fpocket_size_negative")
        if MARKDOWN_ARTIFACT_RE.search(json.dumps(obs, ensure_ascii=False)):
            _add_once(errors, "fpocket_markdown_artifact")


def _docking_score(obs: dict[str, Any]) -> float | None:
    def walk(value: Any) -> float | None:
        if isinstance(value, dict):
            for key in ("docking_affinity_value", "affinity", "score"):
                candidate = value.get(key)
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
    return walk(obs.get("content"))


def _validate_file(path: Path, mode: str = "post-llm") -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        sample = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "path": str(path), "sample_id": "", "status": "invalid",
            "is_training_candidate": False, "errors": ["json_parse_failed"], "warnings": [],
        }

    if not isinstance(sample, dict):
        return {
            "path": str(path), "sample_id": "", "status": "invalid",
            "is_training_candidate": False, "errors": ["top_level_not_object"], "warnings": [],
        }
    sample_id = str(sample.get("id") or "")
    for field in ("schema_version", "id", "messages"):
        if field not in sample:
            _add_once(errors, f"missing_{field}")
    if any(field in sample for field in ("status", "cleaned_sample", "audit")):
        _add_once(errors, "top_level_wrapper_present")

    messages, message_errors, message_warnings = _extract_messages(sample)
    errors.extend(message_errors)
    warnings.extend(message_warnings)
    full_text = json.dumps(sample, ensure_ascii=False)
    if LOCAL_PATH_RE.search(full_text):
        _add_once(errors, "local_absolute_path")
    if RELATIVE_PATH_RE.search(full_text):
        _add_once(errors, "local_relative_path")
    if MARKDOWN_ARTIFACT_RE.search(full_text):
        _add_once(errors, "markdown_artifact")
    if any(pattern in full_text for pattern in MALFORMED_ARTIFACT_PATTERNS):
        _add_once(errors, "malformed_artifact")
    if re.search(r"https?://[^\s\"']*\[artifact:", full_text):
        _add_once(errors, "url_artifact_pollution")

    user_text = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")
    initial_user = next((str(m.get("content") or "") for m in messages if m.get("role") == "user"), "")
    assistant_text = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "assistant")
    thoughts = "\n".join(match.group(1) for match in THOUGHT_RE.finditer(assistant_text))
    for phrase in ENGINEERING_PHRASES:
        if phrase in thoughts.lower():
            _add_once(errors, f"engineering_chatter:{phrase}")
    if ENGINEERING_LEVEL_RE.search(thoughts):
        _add_once(errors, "engineering_chatter:level_workflow")

    tool_calls: list[dict[str, Any]] = []
    observations: list[tuple[str, dict[str, Any]]] = []
    final_obj: dict[str, Any] | None = None
    paired_actions: list[tuple[dict[str, Any], tuple[str, dict[str, Any]] | None]] = []
    pending_calls: list[dict[str, Any]] = []
    for message in messages:
        content = str(message.get("content") or "")
        if message.get("role") == "assistant":
            for match in TOOL_RE.finditer(content):
                parsed = _parse_json(match.group(1))
                if not isinstance(parsed, dict):
                    _add_once(errors, "tool_call_json_invalid")
                else:
                    tool_calls.append(parsed)
                    pending_calls.append(parsed)
            for match in FINAL_RE.finditer(content):
                parsed = _parse_json(match.group(1))
                if not isinstance(parsed, dict):
                    _add_once(errors, "final_answer_json_invalid")
                else:
                    final_obj = parsed
        elif message.get("role") == "user":
            for match in OBS_RE.finditer(content):
                parsed = _parse_json(match.group(2))
                if not isinstance(parsed, dict):
                    _add_once(errors, "observation_json_invalid")
                    continue
                observation = (match.group(1), parsed)
                observations.append(observation)
                debug_keys = _find_recursive_keys(parsed, OBSERVATION_DEBUG_KEYS)
                if debug_keys:
                    _add_once(errors, "observation_debug_metadata_present")
                obs_content = parsed.get("content") if isinstance(parsed.get("content"), dict) else {}
                error_text = str(obs_content.get("error") or "").strip().lower()
                content_error = bool(obs_content.get("error")) and not (
                    error_text.startswith("no error") or error_text.startswith("without error")
                )
                content_status = str(obs_content.get("status") or "").lower()
                content_text = json.dumps(obs_content, ensure_ascii=False)
                outer_status = str(parsed.get("status") or "").lower()
                if (content_error or content_status in ERROR_STATUSES or ERROR_PHRASE_RE.search(content_text)) and (
                    parsed.get("ok") is True or outer_status in {"success", "partial_success"}
                ):
                    _add_once(errors, "observation_status_conflict_after_llm_clean")
                call = pending_calls.pop(0) if pending_calls else {}
                paired_actions.append((call, observation))

    if not final_obj:
        _add_once(errors, "missing_final_answer")
    task = _task_from_sample(sample, final_obj)
    allowed = _allowed_molecules(task, initial_user)
    used_molecules: list[str] = []
    for call in tool_calls:
        used_molecules.extend(_collect_keyed_strings(call.get("arguments")))
    if isinstance(final_obj, dict):
        used_molecules.extend(_collect_keyed_strings(final_obj))
    if any("artifact:" in molecule for molecule in used_molecules):
        _add_once(errors, "artifact_inside_molecular_string")
    if task in {"ac", "pf", "vs"} and allowed:
        if any(molecule not in allowed for molecule in used_molecules):
            _add_once(errors, "non_exact_molecular_string")

    successful = [obs for _, obs in observations if obs.get("ok") is True or obs.get("status") in {"success", "partial_success"}]
    if task in {"ac", "pf", "vs"} and isinstance(final_obj, dict) and "evidence" in final_obj:
        evidence = final_obj.get("evidence")
        if not isinstance(evidence, list):
            _add_once(warnings, "evidence_schema_non_list")
        elif not evidence and successful:
            _add_once(warnings, "empty_evidence_with_success_observations")

    interpretation_text = f"{thoughts}\n{json.dumps(final_obj, ensure_ascii=False) if final_obj else ''}".lower()
    support_text = f"{user_text}\n{json.dumps(observations, ensure_ascii=False)}".lower()
    for pattern in UNSUPPORTED_METRIC_PATTERNS:
        interpretation_match = pattern.search(interpretation_text)
        if interpretation_match and not pattern.search(support_text):
            _add_once(warnings, "unsupported_metric_interpretation")

    tool_names = " ".join(str(call.get("tool_name") or "").lower() for call in tool_calls)
    for phrase, expected in PLAN_RULES:
        if phrase in thoughts.lower() and not any(token in tool_names for token in expected):
            _add_once(warnings, "possibly_unexecuted_plan")

    _fpocket_checks(observations, errors)

    if task == "vs" and isinstance(final_obj, dict):
        ranked = final_obj.get("ranked_smiles")
        if isinstance(ranked, list) and ranked:
            score_by_smiles: dict[str, float] = {}
            for call, observation in paired_actions:
                if not observation:
                    continue
                tool_name, obs = observation
                if not any(token in tool_name.lower() for token in ("docking", "quickvina")):
                    continue
                smiles = _collect_keyed_strings(call.get("arguments"))
                score = _docking_score(obs)
                if smiles and score is not None:
                    score_by_smiles[smiles[0]] = score
            ranked_strings = [str(v) for v in ranked]
            candidates = list(_allowed_molecules("vs", initial_user))
            if candidates and len(ranked_strings) != len(candidates):
                _add_once(errors, "vs_ranking_candidate_count_mismatch")
                _add_once(errors, "vs_ranking_inconsistent_after_llm_clean")
            if len(ranked_strings) != len(set(ranked_strings)):
                _add_once(errors, "vs_ranking_contains_duplicates")
                _add_once(errors, "vs_ranking_inconsistent_after_llm_clean")
            if candidates and any(smiles not in set(candidates) for smiles in ranked_strings):
                _add_once(errors, "vs_ranking_contains_non_candidate")
                _add_once(errors, "vs_ranking_inconsistent_after_llm_clean")
            seen_unscored = False
            previous_score: float | None = None
            for smiles in ranked_strings:
                score = score_by_smiles.get(smiles)
                if score is None:
                    seen_unscored = True
                    continue
                if seen_unscored:
                    _add_once(errors, "vs_scored_molecule_after_unscored")
                    _add_once(errors, "vs_ranking_inconsistent_after_llm_clean")
                if previous_score is not None and score < previous_score:
                    _add_once(errors, "vs_ranking_not_sorted_by_observed_docking_score")
                    _add_once(errors, "vs_ranking_inconsistent_after_llm_clean")
                previous_score = score
            evidence = final_obj.get("evidence")
            if isinstance(evidence, list):
                evidence_smiles = [
                    str(item.get("smiles"))
                    for item in evidence
                    if isinstance(item, dict) and item.get("smiles") is not None
                ]
                if evidence_smiles and evidence_smiles != ranked_strings[: len(evidence_smiles)]:
                    _add_once(errors, "vs_evidence_rank_mismatch")
                    _add_once(errors, "vs_ranking_inconsistent_after_llm_clean")
            if score_by_smiles and not all(smiles in score_by_smiles for smiles in ranked_strings):
                _add_once(warnings, "vs_ranking_score_coverage_incomplete")

    if mode == "pre-llm":
        repair_reasons: list[str] = []
        if "observation_status_conflict_after_llm_clean" in errors:
            repair_reasons.append("observation_status_conflict")
        if "vs_ranking_inconsistent_after_llm_clean" in errors:
            repair_reasons.append("vs_ranking_inconsistent")
        pre_name = {
            "observation_status_conflict_after_llm_clean": "observation_status_conflict",
            "vs_ranking_inconsistent_after_llm_clean": "vs_ranking_inconsistent",
        }
        findings: list[str] = []
        for issue in errors + warnings:
            _add_once(findings, pre_name.get(issue, issue))
        return {
            "path": str(path),
            "sample_id": sample_id,
            "status": "flagged" if repair_reasons else "clean",
            "errors": [],
            "warnings": findings,
            "needs_llm_semantic_repair": bool(repair_reasons),
            "repair_reasons": repair_reasons,
        }
    status = "invalid" if errors else ("warning" if warnings else "valid")
    return {
        "path": str(path),
        "sample_id": sample_id,
        "status": status,
        "is_training_candidate": status in {"valid", "warning"},
        "errors": errors,
        "warnings": warnings,
    }


def _build_report(input_dir: Path, files: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    counts = Counter(item["status"] for item in files)
    issues = Counter(issue for item in files for issue in item["errors"] + item["warnings"])
    return {
        "input_dir": str(input_dir),
        "total": len(files),
        "valid": counts["valid"],
        "warning": counts["warning"],
        "invalid": counts["invalid"],
        "candidate": counts["valid"] + counts["warning"],
        "excluded": counts["invalid"],
        "candidate_policy": "valid+warning included; invalid excluded",
        "flagged": counts["flagged"],
        "clean": counts["clean"],
        "mode": mode,
        "files": files,
        "issue_histogram": dict(sorted(issues.items())),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# LLM Clean Validation Report",
        "",
        f"- Input: `{report['input_dir']}`",
        f"- Total: {report['total']}",
        f"- Valid: {report['valid']}",
        f"- Warning: {report['warning']}",
        f"- Invalid: {report['invalid']}",
        f"- Candidate: {report['candidate']}",
        f"- Excluded: {report['excluded']}",
        f"- Candidate policy: `{report['candidate_policy']}`",
        f"- Pre-LLM flagged: {report.get('flagged', 0)}",
        f"- Pre-LLM clean: {report.get('clean', 0)}",
        "",
        "## Issue Histogram",
        "",
    ]
    histogram = report.get("issue_histogram") or {}
    if histogram:
        lines.extend(f"- `{key}`: {value}" for key, value in sorted(histogram.items()))
    else:
        lines.append("- None")
    lines.extend(["", "## Files", ""])
    for item in report["files"]:
        issues = item["errors"] + item["warnings"]
        lines.append(f"- `{item['status']}` `{item['path']}`: {', '.join(issues) if issues else 'ok'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LLM-cleaned ReAct SFT JSON files without modifying them.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--fail-on-invalid", action="store_true")
    parser.add_argument(
        "--quarantine-dir",
        help="In post-llm mode, move invalid files here after recording the validation report.",
    )
    parser.add_argument(
        "--mode",
        choices=("pre-llm", "post-llm"),
        default="post-llm",
        help="pre-llm only flags semantic repair candidates; post-llm is the final invalid gate.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    files = [_validate_file(path, mode=args.mode) for path in sorted(input_dir.glob("*.json"))]
    report = _build_report(input_dir, files, args.mode)
    if args.quarantine_dir:
        if args.mode != "post-llm":
            raise ValueError("--quarantine-dir is only valid with --mode post-llm")
        quarantine_dir = Path(args.quarantine_dir).expanduser().resolve()
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        for old in quarantine_dir.glob("*.json"):
            old.unlink()
        for item in files:
            if item["status"] != "invalid":
                continue
            source = Path(item["path"])
            target = quarantine_dir / source.name
            if source.is_file():
                shutil.move(str(source), str(target))
                item["quarantined_path"] = str(target)
        report["quarantine_dir"] = str(quarantine_dir)
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md).expanduser().resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(output_md, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "input_dir", "mode", "total", "valid", "warning", "invalid",
                    "candidate", "excluded", "flagged", "clean",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.fail_on_invalid and report["invalid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
