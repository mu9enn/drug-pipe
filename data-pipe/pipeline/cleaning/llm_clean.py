from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from pipeline.claude_agent.session_capture import (
    http_500_retry_delay,
    run_deepseek_harness,
    run_stream_json,
    select_attempt,
    session_has_retryable_http_500,
)
from pipeline.cleaning.invariants import compare_immutable_facts, validate_semantic_record
from pipeline.cleaning.io import base_manifest, read_jsonl, write_json, write_jsonl, write_pretty_json
from pipeline.cleaning.models import (
    LLM_CLEAN_SCENE_DIR,
    LLM_CLEAN_SYSTEM_PROMPT,
    LLM_CLEAN_USER_PROMPT,
    PATCH_SCHEMA_VERSION,
    patch_schema_findings,
    semantic_schema_findings,
)


PROTOCOL_TAG_RE = re.compile(r"</?(?:thought|tool_call|observation|final_answer)(?:\s[^>]*)?>", re.I)
FORBIDDEN_REASONING_RE = re.compile(
    r"(?:L2_workflows|L3_methodology|LR_research|auto-generated-skills|"
    r"CLAUDE\.md|\.claude/|execute-molclaw-trajectory|execution_protocol\.md|"
    r"teacher[- ](?:system|runtime)|runtime instructions?|system prompt|collector sidecar|"
    r"question\.json|prompt\.txt|system_prompt\.md|run_meta\.json|run_config\.json|"
    r"selected_attempt_artifacts\.json|complete_session(?:\.pretty)?\.jsonl)",
    re.IGNORECASE,
)
PatchProvider = Callable[[dict[str, Any], dict[str, Any]], tuple[dict[str, Any] | None, dict[str, Any]]]


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"


def _record_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(_serialize(value).encode()).hexdigest()


def _decisions(source: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in source.get("events") or [] if event.get("type") == "assistant_decision"]


def _editable_reasoning(source: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"decision_id": event["source_message_id"], "reasoning": event["reasoning"]}
        for event in _decisions(source)
    ]


def _reasoning_findings(candidate: dict[str, Any], first_id: str) -> list[str]:
    findings: list[str] = []
    decisions = _decisions(candidate)
    previous: str | None = None
    for index, event in enumerate(decisions):
        decision_id = str(event["source_message_id"])
        reasoning = str(event.get("reasoning") or "").strip()
        if FORBIDDEN_REASONING_RE.search(reasoning):
            findings.append(f"forbidden_teacher_narration:{decision_id}")
        marker_count = len(re.findall(r"high-level plan:", reasoning, flags=re.IGNORECASE))
        if index == 0:
            if decision_id != first_id or not reasoning.startswith("High-level plan:\n") or marker_count != 1:
                findings.append(f"invalid_first_high_level_plan:{decision_id}")
        elif marker_count:
            findings.append(f"high_level_plan_outside_first_decision:{decision_id}")

        paragraphs = [" ".join(part.split()) for part in re.split(r"\n\s*\n", reasoning) if part.strip()]
        if any(left == right for left, right in zip(paragraphs, paragraphs[1:])):
            findings.append(f"duplicate_consecutive_reasoning_paragraph:{decision_id}")
        normalized = " ".join(reasoning.split())
        if previous and normalized and normalized == previous:
            findings.append(f"duplicate_consecutive_decision_reasoning:{decision_id}")
        previous = normalized or previous
    return findings


def apply_reasoning_patch(
    source: dict[str, Any],
    patch: dict[str, Any],
    *,
    require_high_level_plan: bool,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    findings = patch_schema_findings(patch)
    actions: list[dict[str, Any]] = []
    if findings:
        return copy.deepcopy(source), findings, actions
    if patch.get("sample_id") != source.get("id"):
        return copy.deepcopy(source), ["patch_sample_id_mismatch"], actions
    source_decisions = _decisions(source)
    if not source_decisions:
        return copy.deepcopy(source), ["source_has_no_decisions"], actions
    by_id = {event["source_message_id"]: event for event in source_decisions}
    replacements: dict[str, str] = {}
    for edit in patch.get("reasoning_replacements") or []:
        decision_id = str(edit.get("decision_id") or "")
        replacement = str(edit.get("replacement") or "").strip()
        if decision_id not in by_id:
            findings.append(f"unknown_decision_id:{decision_id}")
        elif decision_id in replacements:
            findings.append(f"duplicate_reasoning_replacement:{decision_id}")
        elif PROTOCOL_TAG_RE.search(replacement):
            findings.append(f"reasoning_contains_legacy_protocol:{decision_id}")
        else:
            replacements[decision_id] = replacement

    plan = patch.get("high_level_plan")
    first_id = source_decisions[0]["source_message_id"]
    if require_high_level_plan:
        if not isinstance(plan, dict):
            findings.append("missing_high_level_plan")
        else:
            plan_id = str(plan.get("decision_id") or "")
            plan_text = str(plan.get("text") or "").strip()
            if plan_id != first_id:
                findings.append(f"high_level_plan_not_first_decision:{plan_id}:{first_id}")
            if PROTOCOL_TAG_RE.search(plan_text):
                findings.append("invalid_high_level_plan_text")
    elif plan is not None:
        findings.append("high_level_plan_disabled_but_present")
    if findings:
        return copy.deepcopy(source), list(dict.fromkeys(findings)), actions

    candidate = copy.deepcopy(source)
    candidate_by_id = {event["source_message_id"]: event for event in _decisions(candidate)}
    for decision_id, replacement in replacements.items():
        candidate_by_id[decision_id]["reasoning"] = replacement
        actions.append({"decision_id": decision_id, "operation": "replace_reasoning"})
    if require_high_level_plan:
        plan_text = str(plan["text"]).strip()
        existing = str(candidate_by_id[first_id].get("reasoning") or "").strip()
        candidate_by_id[first_id]["reasoning"] = (
            f"High-level plan:\n{plan_text}" + (f"\n\n{existing}" if existing else "")
        )
        actions.append({"decision_id": first_id, "operation": "prepend_high_level_plan"})
    findings.extend(compare_immutable_facts(source, candidate))
    if patch.get('answer_recovery'):
        from pipeline.cleaning.answer_recovery import apply_answer_patch
        try:
            candidate = apply_answer_patch(candidate, patch['answer_recovery'])
            actions.append({'operation': 'answer_recovery', **{k: v for k, v in patch['answer_recovery'].items() if k != 'answer'}})
        except (ValueError, KeyError) as exc:
            findings.append(f'invalid_answer_recovery:{exc}')
    findings.extend(validate_semantic_record(candidate)["errors"])
    actions.extend({'operation': 'prose_warning', 'finding': item}
                   for item in _reasoning_findings(candidate, first_id))
    if findings:
        return copy.deepcopy(source), list(dict.fromkeys(findings)), []
    return candidate, [], actions


def build_claude_patch_provider(
    *,
    claude_bin: str,
    debug_root: Path,
    timeout_sec: float,
    max_attempts: int = 3,
    require_high_level_plan: bool = True,
    harness: str = "claude",
    dsh_bin: str = "dsh",
    dsh_node_bin: str = "node",
    dsh_model: str = "deepseek-v4-flash",
    dsh_provider: str | None = None,
) -> PatchProvider:
    system_prompt = LLM_CLEAN_SYSTEM_PROMPT.read_text(encoding="utf-8").strip()
    user_prompt = LLM_CLEAN_USER_PROMPT.read_text(encoding="utf-8").strip()

    def provide(source: dict[str, Any], context: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        sample_dir = debug_root / _safe_name(str(source.get("id") or "sample"))
        sample_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(LLM_CLEAN_SCENE_DIR / ".claude", sample_dir / ".claude", dirs_exist_ok=True)
        if harness == "deepseek":
            shutil.copytree(
                LLM_CLEAN_SCENE_DIR / ".claude/skills",
                sample_dir / ".agents/skills",
                dirs_exist_ok=True,
            )
        binding_path = sample_dir / 'patch_input_sha256.txt'
        binding = hashlib.sha256(_serialize({'source': source, 'context': context,
            'system': system_prompt, 'user': user_prompt, 'model': dsh_model, 'harness': harness,
            'skill': (LLM_CLEAN_SCENE_DIR / '.claude/skills/clean-drug-trajectory/SKILL.md').read_text()}).encode()).hexdigest()
        if not binding_path.exists() or binding_path.read_text() != binding:
            (sample_dir / 'semantic_reasoning_patch.json').unlink(missing_ok=True)
        binding_path.write_text(binding)
        write_json(sample_dir / "source_trajectory.json", source)
        write_json(sample_dir / "cleaning_context.json", context)
        write_json(sample_dir / "editable_reasoning.json", _editable_reasoning(source))
        patch_path = sample_dir / "semantic_reasoning_patch.json"
        command = [
            claude_bin, "--print", "--verbose", "--output-format", "stream-json",
            "--no-session-persistence", "--permission-mode", "bypassPermissions",
            "--tools", "Read,Write,Skill", "--allowedTools", "Read,Write,Skill",
            "--system-prompt", system_prompt, "-p", user_prompt,
        ]
        report: dict[str, Any] = {
            "status": "failed", "debug_dir": str(sample_dir), "findings": [], "claude_attempts": []
        }
        if patch_path.is_file():
            try:
                cached = json.loads(patch_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cached = None
            if isinstance(cached, dict) and not apply_reasoning_patch(
                source, cached, require_high_level_plan=require_high_level_plan
            )[1]:
                report.update(status="cached_patch_received", patch_file=str(patch_path))
                return cached, report
        non_retryable_attempts = 0
        http_500_streak = 0
        while non_retryable_attempts < max_attempts:
            patch_path.unlink(missing_ok=True)
            if harness == "deepseek":
                attempt = run_deepseek_harness(
                    user_prompt,
                    system_prompt,
                    cwd=sample_dir,
                    archive_root=sample_dir,
                    dsh_bin=dsh_bin,
                    node_bin=dsh_node_bin,
                    model=dsh_model,
                    provider_id=dsh_provider,
                    timeout_sec=timeout_sec,
                )
            elif harness == "claude":
                attempt = run_stream_json(
                    command, cwd=sample_dir, archive_root=sample_dir, timeout_sec=timeout_sec
                )
            else:
                raise ValueError(f"unsupported harness: {harness}")
            selected = select_attempt(attempt, sample_dir / "complete_session.jsonl")
            report["claude_attempts"].append(attempt)
            findings: list[str] = []
            if attempt.get("timed_out"):
                findings.append("claude_timeout")
            if attempt.get("return_code") != 0:
                findings.append(f"claude_exit_code:{attempt.get('return_code')}")
            if not selected.get("raw_session_valid"):
                findings.append("raw_session_invalid")
            try:
                patch = json.loads(patch_path.read_text(encoding="utf-8")) if patch_path.is_file() else None
            except json.JSONDecodeError as exc:
                patch = None
                findings.append(f"patch_json_decode_error:{exc.msg}")
            if not isinstance(patch, dict):
                findings.append("missing_or_invalid_patch")
            else:
                findings.extend(
                    f"unsafe_patch:{item}"
                    for item in apply_reasoning_patch(
                        source, patch, require_high_level_plan=require_high_level_plan
                    )[1]
                )
            if not findings and isinstance(patch, dict):
                report.update(status="patch_received", patch_file=str(patch_path))
                return patch, report
            report["findings"].extend(findings)
            retryable = (
                attempt.get("return_code") != 0
                and not attempt.get("timed_out")
                and patch is None
                and session_has_retryable_http_500(Path(str(attempt["session_file"])))
            )
            if retryable:
                http_500_streak += 1
                non_retryable_attempts += 1
                time.sleep(http_500_retry_delay(http_500_streak))
                continue
            non_retryable_attempts += 1
            http_500_streak = 0
        return None, report

    return provide


def clean_semantic(
    source: dict[str, Any],
    patch_provider: PatchProvider,
    *,
    require_high_level_plan: bool,
) -> dict[str, Any]:
    schema_errors = semantic_schema_findings(source)
    if schema_errors:
        raise ValueError(f"invalid semantic source {source.get('id')}: {schema_errors}")
    from pipeline.cleaning.answer_recovery import recover_answer, final_event
    from pipeline.output_contracts import normalize_final_answer, task_constraints
    working, recovery = recover_answer(source)
    if recovery['status'] == 'quarantined':
        return {'record': None, 'source': source, 'audit': recovery}
    patch, report = patch_provider(working, {
        'require_high_level_plan': require_high_level_plan, 'patch_schema_version': PATCH_SCHEMA_VERSION,
        'answer_recovery_needed': recovery['status'] == 'pending',
        'causal_rule': 'Each decision may use only the question and preceding observations; the initial plan may not use later results.',
    })
    candidate, findings, actions = (working, ['no_patch'], [])
    if patch is not None:
        candidate, findings, actions = apply_reasoning_patch(working, patch, require_high_level_plan=require_high_level_plan)
    if findings:
        candidate = working
    task = candidate['metadata']['task_type']
    try:
        final_event(candidate)['final_answer'] = normalize_final_answer(final_event(candidate)['final_answer'], task,
            constraints=task_constraints(candidate['user_task'], task))
    except ValueError as exc:
        return {'record': None, 'source': source, 'audit': {**report, 'status': 'answer_pending', 'reason': str(exc)}}
    return {'record': candidate, 'source': source, 'audit': {
        **report, 'status': 'retained_with_warning' if findings else 'cleaned',
        'findings': findings, 'actions': actions, 'answer_recovery': recovery,
        'source_sha256': _record_sha256(source), 'cleaned_sha256': _record_sha256(candidate), 'patch': patch,
    }}


def llm_clean(
    input_path: Path,
    output_root: Path,
    *,
    claude_bin: str = "claude",
    timeout_sec: float = 300.0,
    limit: int = 0,
    max_workers: int = 1,
    max_attempts: int = 3,
    require_high_level_plan: bool = True,
    patch_provider: PatchProvider | None = None,
    harness: str = "claude",
    dsh_bin: str = "dsh",
    dsh_node_bin: str = "node",
    dsh_model: str = "deepseek-v4-flash",
    dsh_provider: str | None = None,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_root = output_root.resolve()
    records, parse_errors = read_jsonl(input_path)
    if parse_errors:
        raise ValueError(f"invalid semantic JSONL: {parse_errors}")
    selected = records[:limit] if limit > 0 else records
    provider = patch_provider or build_claude_patch_provider(
        claude_bin=claude_bin,
        debug_root=output_root / "debug",
        timeout_sec=timeout_sec,
        max_attempts=max_attempts,
        require_high_level_plan=require_high_level_plan,
        harness=harness,
        dsh_bin=dsh_bin,
        dsh_node_bin=dsh_node_bin,
        dsh_model=dsh_model,
        dsh_provider=dsh_provider,
    )
    ordered: list[dict[str, Any] | None] = [None] * len(selected)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                clean_semantic,
                source,
                provider,
                require_high_level_plan=require_high_level_plan,
            ): index
            for index, source in enumerate(selected)
        }
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()
    processed = [item for item in ordered if item is not None]
    accepted = [item for item in processed if item["record"] is not None]
    pending = [
        {"id": item["source"]["id"], "source": item["source"], "llm_clean": item["audit"]}
        for item in processed
        if item["record"] is None
    ]
    accepted_records = [item["record"] for item in accepted]
    write_jsonl(output_root / "semantic_trajectories.jsonl", accepted_records)
    write_pretty_json(output_root / "semantic_trajectories.pretty.json", accepted_records)
    write_jsonl(output_root / "llm_clean_audit.jsonl", [
        {"id": item["source"]["id"], "llm_clean": item["audit"]} for item in processed
    ])
    write_jsonl(output_root / "llm_pending.jsonl", pending)
    statuses = Counter(str(item["audit"].get("status") or "unknown") for item in processed)
    manifest = {
        **base_manifest(step="semantic_llm_clean", source=input_path, repo_root=Path(__file__).resolve().parents[3]),
        "patch_schema_version": PATCH_SCHEMA_VERSION,
        "input_count": len(records),
        "processed_count": len(processed),
        "cleaned_count": len(accepted),
        "pending_count": len(pending),
        "high_level_plan_enabled": require_high_level_plan,
        "harness": harness,
        "status_hist": dict(statuses),
        "source_mother_dataset_preserved": True,
    }
    write_json(output_root / "llm_clean_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean semantic reasoning through an immutable-fact patch.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--harness", choices=("claude", "deepseek"), default=os.environ.get("AGENT_HARNESS", "claude"))
    parser.add_argument("--dsh-bin", default=os.environ.get("DSH_BIN", "dsh"))
    parser.add_argument("--dsh-node-bin", default=os.environ.get("DSH_NODE_BIN", "node"))
    parser.add_argument("--dsh-model", default=os.environ.get("DSH_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--dsh-provider", default=os.environ.get("CC_SWITCH_PROVIDER", "dsv4flash"))
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("MAX_WORKERS", "1") or 1))
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    result = llm_clean(
        args.input,
        args.output_root,
        claude_bin=args.claude_bin,
        timeout_sec=args.timeout_sec,
        limit=args.limit,
        max_workers=args.max_workers,
        max_attempts=args.max_attempts,
        require_high_level_plan=True,
        harness=args.harness,
        dsh_bin=args.dsh_bin,
        dsh_node_bin=args.dsh_node_bin,
        dsh_model=args.dsh_model,
        dsh_provider=args.dsh_provider,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
