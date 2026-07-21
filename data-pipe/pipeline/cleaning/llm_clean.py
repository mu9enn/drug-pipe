from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

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


def _editable_targets(repair_hints: dict[str, Any]) -> set[tuple[int, str, int]]:
    targets: set[tuple[int, str, int]] = set()
    for item in repair_hints.get("editable_findings") or []:
        if not isinstance(item, dict):
            continue
        targets.add(
            (
                int(item.get("message_index", -1)),
                str(item.get("segment_type") or ""),
                int(item.get("segment_index", -1)),
            )
        )
    return targets


def apply_restricted_patch(
    source: dict[str, Any],
    patch: dict[str, Any],
    repair_hints: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    findings = patch_schema_findings(patch)
    actions: list[dict[str, Any]] = []
    if findings:
        return copy.deepcopy(source), findings, actions
    if patch.get("sample_id") != source.get("id"):
        return copy.deepcopy(source), ["patch_sample_id_mismatch"], actions
    allowed = _editable_targets(repair_hints)
    edits_by_message: dict[int, list[dict[str, Any]]] = {}
    seen: set[tuple[int, str, int]] = set()
    for edit in patch.get("edits") or []:
        target = (edit["message_index"], edit["segment_type"], edit["segment_index"])
        if target in seen:
            findings.append(f"duplicate_edit_target:{target}")
            continue
        seen.add(target)
        if target not in allowed:
            findings.append(f"edit_target_not_in_python_hints:{target}")
            continue
        replacement = str(edit["replacement"]).strip()
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
        thought_edits = {
            int(edit["segment_index"]): str(edit["replacement"]).strip()
            for edit in edits if edit["segment_type"] == "thought"
        }
        final_edits = [edit for edit in edits if edit["segment_type"] == "final_summary"]
        thought_count = len(list(THOUGHT_RE.finditer(content)))
        for segment_index in thought_edits:
            if segment_index >= thought_count:
                findings.append(f"thought_index_out_of_range:{message_index}:{segment_index}")
        if thought_edits and not findings:
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
                    }
                )
                return f"<thought>{replacement}</thought>"

            content = THOUGHT_RE.sub(replace_thought, content)
        if final_edits:
            if len(final_edits) != 1 or int(final_edits[0]["segment_index"]) != 0:
                findings.append(f"invalid_final_summary_target:{message_index}")
            matches = list(FINAL_RE.finditer(content))
            if len(matches) != 1:
                findings.append(f"final_answer_count_for_edit:{message_index}:{len(matches)}")
            else:
                try:
                    payload = json.loads(matches[0].group(1))
                except json.JSONDecodeError:
                    payload = None
                if not isinstance(payload, dict) or not isinstance(payload.get("summary"), str):
                    findings.append(f"final_summary_not_editable:{message_index}")
                elif not findings:
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
    return candidate, [], actions


def build_claude_patch_provider(
    *,
    claude_bin: str,
    debug_root: Path,
    timeout_sec: float,
) -> PatchProvider:
    prompt = (PROMPT_DIR / "llm_clean_v1.md").read_text(encoding="utf-8")

    def provide(source: dict[str, Any], repair_hints: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        sample_dir = debug_root / _safe_name(str(source.get("id") or "sample"))
        sample_dir.mkdir(parents=True, exist_ok=True)
        write_json(sample_dir / "source_trajectory.json", source)
        write_json(sample_dir / "repair_hints.json", repair_hints)
        shutil.copy2(EXAMPLE_DIR / "react_trajectory_v1.example.json", sample_dir)
        shutil.copy2(EXAMPLE_DIR / "llm_clean_patch_v1.example.json", sample_dir)
        shutil.copy2(SCHEMA_DIR / "llm_clean_patch_v1.schema.json", sample_dir)
        patch_path = sample_dir / "llm_clean_patch.json"
        if patch_path.exists():
            patch_path.unlink()
        command = [
            claude_bin,
            "--print",
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
        try:
            process = subprocess.run(
                command,
                cwd=sample_dir,
                check=False,
                text=True,
                capture_output=True,
                timeout=timeout_sec,
            )
            (sample_dir / "claude_stdout.txt").write_text(process.stdout or "", encoding="utf-8")
            (sample_dir / "claude_stderr.txt").write_text(process.stderr or "", encoding="utf-8")
            metadata["return_code"] = process.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            (sample_dir / "claude_stdout.txt").write_text(stdout, encoding="utf-8")
            (sample_dir / "claude_stderr.txt").write_text(stderr, encoding="utf-8")
            metadata["findings"].append("claude_timeout")
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
    llm_findings: list[str] = []
    patch, llm_report = patch_provider(source, python_audit.get("repair_hints") or {})
    if patch is None:
        llm_findings.extend(llm_report.get("findings") or ["patch_provider_failed"])
        candidate = copy.deepcopy(source)
        actions: list[dict[str, Any]] = []
        llm_status = "failed"
    else:
        candidate, apply_findings, actions = apply_restricted_patch(
            source, patch, python_audit.get("repair_hints") or {}
        )
        llm_findings.extend(apply_findings)
        llm_status = "unsafe_patch" if apply_findings else "cleaned"
    invariant_report = validate_final_record(candidate)
    immutable_findings = compare_immutable_facts(source, candidate)
    final_findings = list(dict.fromkeys(invariant_report["errors"] + immutable_findings))
    decision = decide_final_status(
        execution_valid=bool(python_audit.get("execution_valid")),
        task_answer_valid=bool(python_audit.get("task_answer_valid")),
        training_trace_valid=bool(python_audit.get("training_trace_valid")),
        llm_clean_status="cleaned" if llm_status == "cleaned" else "failed",
        llm_clean_findings=llm_findings,
        invariant_findings=final_findings,
        llm_clean_required=True,
    )
    return {
        "record": candidate,
        "audit": {
            **python_audit,
            "final_status": decision["final_status"],
            "final_status_authority": decision["authority"],
            "final_status_reasons": decision["reasons"],
            "llm_clean": {
                **llm_report,
                "status": llm_status,
                "findings": llm_findings,
                "actions": actions,
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
        llm_clean_status="not_run",
        llm_clean_findings=[],
        invariant_findings=[],
        llm_clean_required=True,
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
    patch_provider: PatchProvider | None = None,
) -> dict[str, Any]:
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
    audits_by_id = {str(row.get("id") or ""): row for row in audit_rows}
    debug_root = output_root / "debug"
    provider = patch_provider or build_claude_patch_provider(
        claude_bin=claude_bin, debug_root=debug_root, timeout_sec=timeout_sec
    )
    processed: list[dict[str, Any]] = []
    finalized_python_rejected = [_finalize_python_rejection(row) for row in python_rejected]
    draft_rejected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = [
        *finalized_python_rejected,
        *input_rejected,
        *audit_parse_rejected,
        *python_rejected_parse_errors,
    ]
    for draft in drafts:
        if limit > 0 and len(processed) >= limit:
            break
        record_id = str(draft.get("id") or "")
        schema_errors = react_schema_findings(draft)
        python_audit = audits_by_id.get(record_id)
        if schema_errors or python_audit is None or python_audit.get("python_status") != "python_valid":
            invalid_audit = {
                **(python_audit or {"id": record_id}),
                "final_status": "rejected",
                "final_status_authority": "final_acceptance_gate",
                "final_status_reasons": schema_errors or ["missing_or_invalid_python_audit"],
            }
            draft_rejected.append(invalid_audit)
            rejected.append(invalid_audit)
            continue
        processed.append(clean_draft(draft, python_audit, provider))

    accepted = [item for item in processed if item["audit"]["final_status"] == "accepted"]
    quarantine = [item["audit"] for item in processed if item["audit"]["final_status"] == "quarantine"]
    final_audits = [
        *finalized_python_rejected,
        *draft_rejected,
        *(item["audit"] for item in processed),
    ]
    outputs = {
        "react_trajectories": output_root / "react_trajectories.jsonl",
        "curation_audit": output_root / "curation_audit.jsonl",
        "rejected": output_root / "rejected.jsonl",
        "quarantine": output_root / "quarantine.jsonl",
        "run_manifest": output_root / "run_manifest.json",
    }
    write_jsonl(outputs["react_trajectories"], [item["record"] for item in accepted])
    write_jsonl(outputs["curation_audit"], final_audits)
    write_jsonl(outputs["rejected"], rejected)
    if quarantine:
        write_jsonl(outputs["quarantine"], quarantine)
    elif outputs["quarantine"].exists():
        outputs["quarantine"].unlink()
    repo_root = Path(__file__).resolve().parents[3]
    manifest = {
        **base_manifest(step="llm_clean", source=input_path, repo_root=repo_root),
        "input_sha256": _sha256(input_path),
        "patch_schema_version": PATCH_SCHEMA_VERSION,
        "input_count": len(drafts),
        "processed_count": len(processed),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "quarantine_count": len(quarantine),
        "claude_bin": claude_bin,
        "timeout_sec": timeout_sec,
        "outputs": {
            name: str(path) for name, path in outputs.items()
            if name != "quarantine" or quarantine
        },
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
    args = parser.parse_args()
    result = llm_clean(
        Path(args.input),
        Path(args.output_root),
        python_audit_path=Path(args.python_audit) if args.python_audit else None,
        claude_bin=args.claude_bin,
        timeout_sec=args.timeout_sec,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
