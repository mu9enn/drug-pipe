from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from pipeline.claude_agent.session_capture import run_stream_json, select_attempt
from pipeline.cleaning.acceptance_gate import decide_final_status
from pipeline.cleaning.artifacts import ABSOLUTE_PATH_RE, RELATIVE_PATH_RE
from pipeline.cleaning.invariants import (
    FINAL_RE,
    THOUGHT_RE,
    compare_immutable_facts,
    validate_final_record,
)
from pipeline.cleaning.io import base_manifest, read_jsonl, write_json, write_jsonl
from pipeline.cleaning.models import (
    EXAMPLE_DIR,
    PATCH_SCHEMA_VERSION,
    PROMPT_DIR,
    SCHEMA_DIR,
    patch_schema_findings,
    react_schema_findings,
)


PROTOCOL_TAG_RE = re.compile(r"</?(?:thought|tool_call|observation|final_answer)(?:\s[^>]*)?>", re.I)
PatchProvider = Callable[[dict[str, Any], dict[str, Any]], tuple[dict[str, Any] | None, dict[str, Any]]]


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"


def apply_restricted_patch(
    source: dict[str, Any],
    patch: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    findings = patch_schema_findings(patch)
    actions: list[dict[str, Any]] = []
    if findings:
        return copy.deepcopy(source), findings, actions
    if patch.get("sample_id") != source.get("id"):
        return copy.deepcopy(source), ["patch_sample_id_mismatch"], actions
    edits_by_message: dict[int, list[dict[str, Any]]] = {}
    seen: set[tuple[int, str, int]] = set()
    for edit in patch.get("edits") or []:
        target = (edit["message_index"], edit["segment_type"], edit["segment_index"])
        if target in seen:
            findings.append(f"duplicate_edit_target:{target}")
            continue
        seen.add(target)
        replacement = str(edit["replacement"]).strip()
        if edit["segment_type"] == "final_summary" and not replacement:
            findings.append(f"empty_final_summary_replacement:{target}")
            continue
        if PROTOCOL_TAG_RE.search(replacement):
            findings.append(f"replacement_contains_protocol_tag:{target}")
            continue
        if ABSOLUTE_PATH_RE.search(replacement) or RELATIVE_PATH_RE.search(replacement):
            findings.append(f"replacement_contains_path:{target}")
            continue
        edits_by_message.setdefault(edit["message_index"], []).append(edit)
    if findings:
        return copy.deepcopy(source), findings, actions

    candidate = copy.deepcopy(source)
    messages = candidate.get("messages") if isinstance(candidate.get("messages"), list) else []
    for message_index, edits in edits_by_message.items():
        if message_index < 0 or message_index >= len(messages):
            findings.append(f"message_index_out_of_range:{message_index}")
            continue
        message = messages[message_index]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            findings.append(f"edit_target_not_assistant:{message_index}")
            continue
        content = str(message.get("content") or "")
        thought_count = len(list(THOUGHT_RE.finditer(content)))
        for edit in edits:
            segment_index = int(edit["segment_index"])
            if edit["segment_type"] == "thought" and segment_index >= thought_count:
                findings.append(f"thought_index_out_of_range:{message_index}:{segment_index}")
            if edit["segment_type"] == "final_summary":
                matches = list(FINAL_RE.finditer(content))
                if segment_index != 0:
                    findings.append(f"invalid_final_summary_target:{message_index}")
                elif len(matches) != 1:
                    findings.append(f"final_answer_count_for_edit:{message_index}:{len(matches)}")
                else:
                    try:
                        payload = json.loads(matches[0].group(1))
                    except json.JSONDecodeError:
                        payload = None
                    if not isinstance(payload, dict) or not isinstance(payload.get("summary"), str):
                        findings.append(f"final_summary_not_editable:{message_index}")
    if findings:
        return copy.deepcopy(source), list(dict.fromkeys(findings)), []

    for message_index, edits in edits_by_message.items():
        message = messages[message_index]
        content = str(message.get("content") or "")
        thought_edits = {
            int(edit["segment_index"]): str(edit["replacement"]).strip()
            for edit in edits if edit["segment_type"] == "thought"
        }
        final_edits = [edit for edit in edits if edit["segment_type"] == "final_summary"]
        if thought_edits:
            current = -1

            def replace_thought(match: re.Match[str]) -> str:
                nonlocal current
                current += 1
                if current not in thought_edits:
                    return match.group(0)
                replacement = thought_edits[current]
                actions.append(
                    {
                        "message_index": message_index,
                        "segment_type": "thought",
                        "segment_index": current,
                        "operation": "delete" if not replacement else "replace",
                    }
                )
                return f"<thought>{replacement}</thought>" if replacement else ""

            content = THOUGHT_RE.sub(replace_thought, content)
            if not content.strip():
                findings.append(f"thought_deletion_would_empty_message:{message_index}")
        if final_edits:
            matches = list(FINAL_RE.finditer(content))
            payload = json.loads(matches[0].group(1))
            payload["summary"] = str(final_edits[0]["replacement"]).strip()
            content = FINAL_RE.sub(
                lambda _match: f"<final_answer>{_serialize(payload)}</final_answer>",
                content,
                count=1,
            )
            actions.append(
                {"message_index": message_index, "segment_type": "final_summary", "segment_index": 0}
            )
        message["content"] = content
    if findings:
        return copy.deepcopy(source), findings, []
    findings.extend(compare_immutable_facts(source, candidate))
    findings.extend(react_schema_findings(candidate))
    if findings:
        return copy.deepcopy(source), list(dict.fromkeys(findings)), []
    return candidate, findings, actions


def build_claude_patch_provider(
    *,
    claude_bin: str,
    debug_root: Path,
    timeout_sec: float,
) -> PatchProvider:
    prompt = (PROMPT_DIR / "llm_clean_v1.md").read_text(encoding="utf-8")

    def provide(source: dict[str, Any], cleaning_context: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        sample_dir = debug_root / _safe_name(str(source.get("id") or "sample"))
        sample_dir.mkdir(parents=True, exist_ok=True)
        write_json(sample_dir / "source_trajectory.json", source)
        write_json(sample_dir / "cleaning_context.json", cleaning_context)
        shutil.copy2(EXAMPLE_DIR / "react_trajectory_v1.example.json", sample_dir)
        shutil.copy2(EXAMPLE_DIR / "llm_clean_patch_v1.example.json", sample_dir)
        shutil.copy2(SCHEMA_DIR / "llm_clean_patch_v1.schema.json", sample_dir)
        patch_path = sample_dir / "llm_clean_patch.json"
        if patch_path.exists():
            patch_path.unlink()
        command = [
            claude_bin,
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--safe-mode",
            "--no-session-persistence",
            "--permission-mode",
            "bypassPermissions",
            "--tools",
            "Read,Write",
            "--allowedTools",
            "Read,Write",
            "--disable-slash-commands",
            "-p",
            prompt,
        ]
        metadata: dict[str, Any] = {
            "status": "failed",
            "debug_dir": str(sample_dir),
            "timeout_sec": timeout_sec,
            "findings": [],
        }
        attempt = run_stream_json(
            command,
            cwd=sample_dir,
            archive_root=sample_dir,
            timeout_sec=timeout_sec,
        )
        selected = select_attempt(attempt, sample_dir / "complete_session.jsonl")
        metadata.update(
            {
                "command": command,
                "return_code": attempt["return_code"],
                "session_file": str(sample_dir / "complete_session.jsonl"),
                "attempt_session_file": attempt["session_file"],
                "claude_attempts": [attempt],
                "selected_claude_attempt": attempt["attempt_index"],
                "session_byte_count": selected["byte_count"],
                "session_sha256": selected["sha256"],
                "parseable_event_count": selected["parseable_event_count"],
                "raw_session_valid": selected["raw_session_valid"],
            }
        )
        if attempt["timed_out"]:
            metadata["findings"].append("claude_timeout")
            return None, metadata
        if not selected["raw_session_valid"]:
            metadata["findings"].append("raw_session_invalid")
            return None, metadata
        if metadata.get("return_code") != 0:
            metadata["findings"].append(f"claude_exit_code:{metadata.get('return_code')}")
            return None, metadata
        if not patch_path.is_file():
            metadata["findings"].append("missing_llm_clean_patch_file")
            return None, metadata
        try:
            patch = json.loads(patch_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            metadata["findings"].append(f"patch_json_decode_error:{exc.msg}")
            return None, metadata
        if not isinstance(patch, dict):
            metadata["findings"].append("patch_not_object")
            return None, metadata
        metadata["status"] = "patch_received"
        metadata["patch_file"] = str(patch_path)
        return patch, metadata

    return provide


def clean_draft(
    source: dict[str, Any],
    python_audit: dict[str, Any],
    patch_provider: PatchProvider,
) -> dict[str, Any]:
    source_schema_findings = react_schema_findings(source)
    if source_schema_findings:
        raise ValueError(
            f"invalid Python-clean draft schema for {source.get('id')!r}: "
            + ", ".join(source_schema_findings)
        )
    llm_findings: list[str] = []
    cleaning_context = {
        "only_molclaw_tool": bool(
            (python_audit.get("trace_stats") or {}).get("only_molclaw_tool")
        )
    }
    patch, llm_report = patch_provider(source, cleaning_context)
    if patch is None:
        llm_findings.extend(llm_report.get("findings") or ["patch_provider_failed"])
        candidate = copy.deepcopy(source)
        actions = []
        llm_status = "failed_fallback"
    else:
        candidate, apply_findings, actions = apply_restricted_patch(source, patch)
        llm_findings.extend(apply_findings)
        if apply_findings:
            llm_status = "unsafe_patch_fallback"
        elif actions:
            llm_status = "cleaned"
        else:
            llm_status = "not_required"
    invariant_report = validate_final_record(candidate)
    immutable_findings = compare_immutable_facts(source, candidate)
    decision = decide_final_status(
        execution_valid=bool(python_audit.get("execution_valid")),
        task_answer_valid=bool(python_audit.get("task_answer_valid")),
        training_trace_valid=bool(python_audit.get("training_trace_valid")),
    )
    retained_python_audit = {
        key: value for key, value in python_audit.items() if key != "repair_hints"
    }
    return {
        "record": candidate,
        "audit": {
            **retained_python_audit,
            "final_status": decision["final_status"],
            "final_status_authority": decision["authority"],
            "final_status_reasons": decision["reasons"],
            "llm_clean": {
                **llm_report,
                "status": llm_status,
                "findings": llm_findings,
                "actions": actions,
                "cleaning_context": cleaning_context,
                "residual_prose_findings": invariant_report["prose_findings"],
                "residual_prose_finding_count": len(invariant_report["prose_findings"]),
                **({"patch": patch} if patch is not None else {}),
            },
            "final_invariants": invariant_report,
            "immutable_findings": immutable_findings,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finalize_python_rejection(audit: dict[str, Any]) -> dict[str, Any]:
    decision = decide_final_status(
        execution_valid=bool(audit.get("execution_valid")),
        task_answer_valid=bool(audit.get("task_answer_valid")),
        training_trace_valid=bool(audit.get("training_trace_valid")),
    )
    return {
        **audit,
        "final_status": decision["final_status"],
        "final_status_authority": decision["authority"],
        "final_status_reasons": decision["reasons"],
    }


def llm_clean(
    input_path: Path,
    output_root: Path,
    *,
    python_audit_path: Path | None = None,
    claude_bin: str = "claude",
    timeout_sec: float = 300.0,
    limit: int = 0,
    max_workers: int = 1,
    patch_provider: PatchProvider | None = None,
) -> dict[str, Any]:
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    input_path = input_path.resolve()
    output_root = output_root.resolve()
    python_audit_path = (python_audit_path or input_path.with_name("python_audit.jsonl")).resolve()
    drafts, input_rejected = read_jsonl(input_path)
    audit_rows, audit_parse_rejected = read_jsonl(python_audit_path)
    python_rejected_path = input_path.with_name("rejected.jsonl")
    if python_rejected_path.is_file():
        python_rejected, python_rejected_parse_errors = read_jsonl(python_rejected_path)
    else:
        python_rejected, python_rejected_parse_errors = [], []
    input_problems = [
        *(f"draft:{item}" for item in input_rejected),
        *(f"audit:{item}" for item in audit_parse_rejected),
        *(f"python_rejected:{item}" for item in python_rejected_parse_errors),
    ]
    if input_problems:
        raise ValueError("invalid cleaning input JSONL: " + "; ".join(input_problems))
    audits_by_id = {str(row.get("id") or ""): row for row in audit_rows}
    debug_root = output_root / "debug"
    provider = patch_provider or build_claude_patch_provider(
        claude_bin=claude_bin, debug_root=debug_root, timeout_sec=timeout_sec
    )
    processed: list[dict[str, Any]] = []
    finalized_python_rejected = [_finalize_python_rejection(row) for row in python_rejected]
    rejected: list[dict[str, Any]] = [*finalized_python_rejected]
    selected_drafts = drafts[:limit] if limit > 0 else drafts
    clean_inputs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for draft in selected_drafts:
        record_id = str(draft.get("id") or "")
        schema_errors = react_schema_findings(draft)
        python_audit = audits_by_id.get(record_id)
        if schema_errors:
            raise ValueError(
                f"invalid Python-clean draft schema for {record_id!r}: "
                + ", ".join(schema_errors)
            )
        if python_audit is None or python_audit.get("python_status") != "python_valid":
            raise ValueError(f"missing or invalid Python audit for draft {record_id!r}")
        clean_inputs.append((draft, python_audit))

    ordered_results: list[dict[str, Any] | None] = [None] * len(clean_inputs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(clean_draft, draft, python_audit, provider): index
            for index, (draft, python_audit) in enumerate(clean_inputs)
        }
        for future in as_completed(futures):
            ordered_results[futures[future]] = future.result()
    processed = [item for item in ordered_results if item is not None]

    accepted = [item for item in processed if item["audit"]["final_status"] == "accepted"]
    llm_status_hist = Counter(
        str((item["audit"].get("llm_clean") or {}).get("status") or "unknown")
        for item in processed
    )
    final_audits = [
        *finalized_python_rejected,
        *(item["audit"] for item in processed),
    ]
    outputs = {
        "react_trajectories": output_root / "react_trajectories.jsonl",
        "curation_audit": output_root / "curation_audit.jsonl",
        "rejected": output_root / "rejected.jsonl",
        "run_manifest": output_root / "run_manifest.json",
    }
    write_jsonl(outputs["react_trajectories"], [item["record"] for item in accepted])
    write_jsonl(outputs["curation_audit"], final_audits)
    write_jsonl(outputs["rejected"], rejected)
    repo_root = Path(__file__).resolve().parents[3]
    manifest = {
        **base_manifest(step="llm_clean", source=input_path, repo_root=repo_root),
        "input_sha256": _sha256(input_path),
        "patch_schema_version": PATCH_SCHEMA_VERSION,
        "input_count": len(drafts),
        "processed_count": len(processed),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "llm_clean_status_hist": dict(llm_status_hist),
        "llm_fallback_count": sum(
            llm_status_hist.get(status, 0)
            for status in ("failed_fallback", "unsafe_patch_fallback")
        ),
        "residual_prose_finding_count": sum(
            int((item["audit"].get("llm_clean") or {}).get("residual_prose_finding_count") or 0)
            for item in processed
        ),
        "claude_bin": claude_bin,
        "timeout_sec": timeout_sec,
        "max_workers": max_workers,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    write_json(outputs["run_manifest"], manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: apply restricted LLM prose patches and one final gate.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python-audit", default="")
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("MAX_WORKERS", "1") or 1))
    args = parser.parse_args()
    result = llm_clean(
        Path(args.input),
        Path(args.output_root),
        python_audit_path=Path(args.python_audit) if args.python_audit else None,
        claude_bin=args.claude_bin,
        timeout_sec=args.timeout_sec,
        limit=args.limit,
        max_workers=args.max_workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
