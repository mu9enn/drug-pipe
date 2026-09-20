"""Stage-2 LLM cleaning: give oversized reasoning one concise-rewrite pass."""
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
    http_500_retry_delay, run_deepseek_harness, run_stream_json, select_attempt,
    session_has_retryable_http_500,
)
from pipeline.cleaning.invariants import compare_immutable_facts, validate_semantic_record
from pipeline.cleaning.io import base_manifest, read_jsonl, write_json, write_jsonl, write_pretty_json
from pipeline.cleaning.models import (
    REASONING_SHORTEN_SCENE_DIR, REASONING_SHORTEN_SYSTEM_PROMPT,
    REASONING_SHORTEN_USER_PROMPT, SHORTEN_PATCH_SCHEMA_VERSION,
    semantic_schema_findings, shorten_patch_schema_findings,
)

TokenCounter = Callable[[str], int]
PatchProvider = Callable[[dict[str, Any], list[dict[str, Any]], dict[str, Any]], tuple[dict[str, Any] | None, dict[str, Any]]]
PROTOCOL_TAG_RE = re.compile(r"</?(?:thought|tool_call|observation|final_answer)(?:\s[^>]*)?>", re.I)


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"


def _decisions(source: dict[str, Any]) -> list[dict[str, Any]]:
    if 'messages' in source and 'events' not in source:
        return [
            {'source_message_id': f'm{i}_thought{j}', 'reasoning': match.group(1)}
            for i, message in enumerate(source['messages']) if message.get('role') == 'assistant'
            for j, match in enumerate(re.finditer(r'<thought>(.*?)</thought>', message.get('content', ''), re.S))
        ]
    return [event for event in source.get("events") or [] if event.get("type") == "assistant_decision"]


def oversized_targets(source: dict[str, Any], token_counter: TokenCounter, limit: int) -> list[dict[str, Any]]:
    """Select reasoning strictly above the configured threshold."""
    targets = []
    for event in _decisions(source):
        reasoning = str(event.get("reasoning") or "")
        count = token_counter(reasoning)
        if count > limit:
            targets.append({"decision_id": str(event["source_message_id"]), "reasoning": reasoning, "token_count": count})
    return targets


def apply_shorten_patch(source: dict[str, Any], patch: dict[str, Any], target_ids: set[str]) -> tuple[dict[str, Any], list[str]]:
    findings = shorten_patch_schema_findings(patch)
    if patch.get("sample_id") != source.get("id"):
        findings.append("patch_sample_id_mismatch")
    replacements: dict[str, str] = {}
    for edit in patch.get("reasoning_replacements") or []:
        decision_id = str(edit.get("decision_id") or "")
        replacement = str(edit.get("replacement") or "").strip()
        if decision_id in replacements:
            findings.append(f"duplicate_reasoning_replacement:{decision_id}")
        elif decision_id not in target_ids:
            findings.append(f"unrequested_reasoning_replacement:{decision_id}")
        elif PROTOCOL_TAG_RE.search(replacement):
            findings.append(f"reasoning_contains_legacy_protocol:{decision_id}")
        replacements[decision_id] = replacement
    missing = target_ids - set(replacements)
    if missing:
        findings.append(f"missing_reasoning_replacements:{sorted(missing)}")
    if findings:
        return copy.deepcopy(source), list(dict.fromkeys(findings))
    candidate = copy.deepcopy(source)
    if 'messages' in source and 'events' not in source:
        for i, message in enumerate(candidate['messages']):
            if message.get('role') != 'assistant':
                continue
            content = message.get('content', '')
            matches = list(re.finditer(r'<thought>(.*?)</thought>', content, re.S))
            for j, match in reversed(list(enumerate(matches))):
                decision_id = f'm{i}_thought{j}'
                if decision_id in replacements:
                    content = content[:match.start(1)] + replacements[decision_id] + content[match.end(1):]
            message['content'] = content
        # Exact span substitution preserves tool calls, observations, final answers,
        # masks, ordering and all other metadata byte-for-byte.
        return candidate, []
    for event in _decisions(candidate):
        decision_id = str(event["source_message_id"])
        if decision_id in replacements:
            event["reasoning"] = replacements[decision_id]
    findings.extend(compare_immutable_facts(source, candidate))
    findings.extend(validate_semantic_record(candidate)["errors"])
    return (copy.deepcopy(source), list(dict.fromkeys(findings))) if findings else (candidate, [])


def build_patch_provider(*, claude_bin: str, debug_root: Path, timeout_sec: float,
                         max_attempts: int = 3, harness: str = "claude", dsh_bin: str = "dsh",
                         dsh_node_bin: str = "node", dsh_model: str = "deepseek-v4-flash",
                         dsh_provider: str | None = None) -> PatchProvider:
    system_prompt = REASONING_SHORTEN_SYSTEM_PROMPT.read_text().strip()
    # Supply the complete reasoning on stdin. Read's per-line and per-response
    # truncation otherwise makes large JSON-string inputs silently incomplete.
    user_prompt = ('Compress each target reasoning in the JSON supplied on stdin once. '
        'Return ONLY a JSON object with schema_version="reasoning_shorten_patch_v1", '
        'sample_id, reasoning_replacements=[{decision_id,replacement}]. '
        'Include every target exactly once. Preserve scientific intent, evidence, '
        'parameters and uncertainty. Remove repetitive/circular reasoning. '
        'Do not call tools. Do not use Markdown fences.')
    if harness == 'deepseek':
        user_prompt = REASONING_SHORTEN_USER_PROMPT.read_text().strip()

    def provide(source: dict[str, Any], targets: list[dict[str, Any]], context: dict[str, Any]):
        round_index = int(context["round"])
        sample_dir = debug_root / _safe_name(str(source["id"])) / f"round_{round_index:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(REASONING_SHORTEN_SCENE_DIR / ".claude", sample_dir / ".claude", dirs_exist_ok=True)
        if harness == "deepseek":
            shutil.copytree(REASONING_SHORTEN_SCENE_DIR / ".claude/skills", sample_dir / ".agents/skills", dirs_exist_ok=True)
        payload = {"schema_version": SHORTEN_PATCH_SCHEMA_VERSION, "sample_id": source["id"],
                   "token_limit_exclusive": context["token_limit"],
                   "target_reasoning_tokens": context["target_tokens"], "targets": targets}
        write_json(sample_dir / "shortening_input.json", payload)
        patch_path = sample_dir / "reasoning_shorten_patch.json"
        binding = hashlib.sha256(_serialize({"payload": payload, "system": system_prompt, "user": user_prompt,
                                             "harness": harness, "model": dsh_model}).encode()).hexdigest()
        binding_path = sample_dir / "patch_input_sha256.txt"
        if not binding_path.exists() or binding_path.read_text() != binding:
            patch_path.unlink(missing_ok=True)
        binding_path.write_text(binding)
        command = [claude_bin, "--print", "--verbose", "--output-format", "stream-json",
                   "--no-session-persistence", "--permission-mode", "bypassPermissions",
                   "--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                   "--system-prompt", system_prompt, "-p", user_prompt]
        report: dict[str, Any] = {"status": "failed", "debug_dir": str(sample_dir), "findings": [], "claude_attempts": []}
        if patch_path.is_file():
            try:
                cached = json.loads(patch_path.read_text())
            except json.JSONDecodeError:
                cached = None
            if isinstance(cached, dict) and not apply_shorten_patch(
                source, cached, {target["decision_id"] for target in targets}
            )[1]:
                report.update(status="cached_patch_received", patch_file=str(patch_path))
                return cached, report
        for attempt_index in range(max_attempts):
            patch_path.unlink(missing_ok=True)
            if harness == "deepseek":
                attempt = run_deepseek_harness(user_prompt, system_prompt, cwd=sample_dir, archive_root=sample_dir,
                    dsh_bin=dsh_bin, node_bin=dsh_node_bin, model=dsh_model, provider_id=dsh_provider,
                    timeout_sec=timeout_sec)
            elif harness == "claude":
                attempt = run_stream_json(command, cwd=sample_dir, archive_root=sample_dir, timeout_sec=timeout_sec,
                                          input_text=json.dumps(payload, ensure_ascii=False))
            else:
                raise ValueError(f"unsupported harness: {harness}")
            selected = select_attempt(attempt, sample_dir / "complete_session.jsonl")
            if harness == 'claude' and attempt.get('return_code') == 0:
                for line in (sample_dir / 'complete_session.jsonl').read_text().splitlines():
                    try:
                        event = json.loads(line)
                        if event.get('type') == 'result' and not event.get('is_error'):
                            value = json.loads(event.get('result', ''))
                            write_json(patch_path, value)
                    except (json.JSONDecodeError, TypeError):
                        pass
            report["claude_attempts"].append(attempt)
            findings = []
            if attempt.get("timed_out"): findings.append("claude_timeout")
            if attempt.get("return_code") != 0: findings.append(f"claude_exit_code:{attempt.get('return_code')}")
            if not selected.get("raw_session_valid"): findings.append("raw_session_invalid")
            try:
                patch = json.loads(patch_path.read_text()) if patch_path.is_file() else None
            except json.JSONDecodeError as exc:
                patch = None; findings.append(f"patch_json_decode_error:{exc.msg}")
            if not isinstance(patch, dict): findings.append("missing_or_invalid_patch")
            else: findings.extend(apply_shorten_patch(source, patch, {t["decision_id"] for t in targets})[1])
            if not findings and isinstance(patch, dict):
                report.update(status="patch_received", patch_file=str(patch_path)); return patch, report
            report["findings"].extend(findings)
            retryable = (attempt.get("return_code") != 0 and not attempt.get("timed_out") and patch is None
                         and session_has_retryable_http_500(Path(str(attempt["session_file"]))))
            if retryable and attempt_index + 1 < max_attempts:
                time.sleep(http_500_retry_delay(attempt_index + 1))
        return None, report
    return provide


def shorten_semantic(source: dict[str, Any], provider: PatchProvider, token_counter: TokenCounter, *,
                     token_limit: int = 16384, target_tokens: int = 12000) -> dict[str, Any]:
    errors = [] if ('messages' in source and 'events' not in source) else semantic_schema_findings(source)
    if errors:
        raise ValueError(f"invalid semantic source {source.get('id')}: {errors}")
    working = copy.deepcopy(source)
    initial = oversized_targets(working, token_counter, token_limit)
    if not initial:
        return {"record": working, "source": source, "audit": {"status": "not_required",
                "source_max_reasoning_tokens": max((token_counter(str(e.get('reasoning') or '')) for e in _decisions(source)), default=0)}}
    patch, report = provider(working, initial, {"round": 1, "token_limit": token_limit, "target_tokens": target_tokens})
    if patch is None:
        return {"record": None, "source": source, "audit": {**report, "status": "pending",
                "source_max_reasoning_tokens": max(t["token_count"] for t in initial),
                "targeted_decision_ids": [t["decision_id"] for t in initial]}}
    candidate, findings = apply_shorten_patch(working, patch, {t["decision_id"] for t in initial})
    if findings:
        return {"record": None, "source": source, "audit": {**report, "status": "pending", "findings": findings,
                "source_max_reasoning_tokens": max(t["token_count"] for t in initial),
                "targeted_decision_ids": [t["decision_id"] for t in initial]}}
    audit = {**report, "status": "processed_once",
             "source_reasoning_tokens": {t["decision_id"]: t["token_count"] for t in initial},
             "targeted_decision_ids": [t["decision_id"] for t in initial],
             "post_llm_token_limit_not_enforced": True}
    return {"record": candidate, "source": source, "audit": audit}


def reasoning_shorten(input_path: Path, output_root: Path, *, tokenizer_path: Path | None = None,
                      token_counter: TokenCounter | None = None, patch_provider: PatchProvider | None = None,
                      token_limit: int = 16384, target_tokens: int = 12000,
                      max_workers: int = 1, limit: int = 0, claude_bin: str = "claude", timeout_sec: float = 300,
                      max_attempts: int = 3, harness: str = "claude", dsh_bin: str = "dsh", dsh_node_bin: str = "node",
                      dsh_model: str = "deepseek-v4-flash", dsh_provider: str | None = None) -> dict[str, Any]:
    if token_limit < 2:
        raise ValueError("token_limit must be at least 2")
    if not 0 < target_tokens < token_limit:
        raise ValueError("target_tokens must be positive and strictly below token_limit")
    records, parse_errors = read_jsonl(input_path.resolve())
    if parse_errors: raise ValueError(f"invalid semantic JSONL: {parse_errors}")
    legacy = bool(records and 'messages' in records[0] and 'events' not in records[0])
    if any(('messages' in r and 'events' not in r) != legacy for r in records):
        raise ValueError('mixed semantic/legacy formats')
    if token_counter is None:
        if tokenizer_path is None: raise ValueError("tokenizer_path is required")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=True)
        token_counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    selected = records[:limit] if limit > 0 else records
    provider = patch_provider or build_patch_provider(claude_bin=claude_bin, debug_root=output_root / "debug",
        timeout_sec=timeout_sec, max_attempts=max_attempts, harness=harness, dsh_bin=dsh_bin,
        dsh_node_bin=dsh_node_bin, dsh_model=dsh_model, dsh_provider=dsh_provider)
    ordered: list[dict[str, Any] | None] = [None] * len(selected)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(shorten_semantic, row, provider, token_counter, token_limit=token_limit,
                    target_tokens=target_tokens): index for index, row in enumerate(selected)}
        for future in as_completed(futures): ordered[futures[future]] = future.result()
    processed = [item for item in ordered if item is not None]
    accepted = [item for item in processed if item["record"] is not None]
    pending = [{"id": item["source"]["id"], "source": item["source"], "reasoning_shorten": item["audit"]}
               for item in processed if item["record"] is None]
    output_stem = 'react_trajectories' if legacy else 'semantic_trajectories'
    write_jsonl(output_root / f"{output_stem}.jsonl", [item["record"] for item in accepted])
    write_pretty_json(output_root / f"{output_stem}.pretty.json", [item["record"] for item in accepted])
    write_jsonl(output_root / "reasoning_shorten_audit.jsonl", [{"id": item["source"]["id"], "reasoning_shorten": item["audit"]} for item in processed])
    write_jsonl(output_root / "reasoning_shorten_pending.jsonl", pending)
    statuses = Counter(item["audit"]["status"] for item in processed)
    manifest = {**base_manifest(step="reasoning_shorten", source=input_path.resolve(), repo_root=Path(__file__).resolve().parents[3]),
        "schema_version": SHORTEN_PATCH_SCHEMA_VERSION, "input_count": len(records), "processed_count": len(processed),
        "accepted_count": len(accepted), "pending_count": len(pending), "processed_once_count": statuses.get("processed_once", 0),
        "not_required_count": statuses.get("not_required", 0), "token_limit_exclusive": token_limit,
        "target_reasoning_tokens": target_tokens, "post_llm_token_limit_enforced": False, "harness": harness,
        "tokenizer": str(tokenizer_path.resolve()) if tokenizer_path else "injected", "status_hist": dict(statuses)}
    write_json(output_root / "reasoning_shorten_manifest.json", manifest)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path); p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--tokenizer", required=True, type=Path); p.add_argument("--token-limit", type=int, default=16384)
    p.add_argument("--target-tokens", type=int, default=12000)
    p.add_argument("--claude-bin", default="claude"); p.add_argument("--timeout-sec", type=float, default=300)
    p.add_argument("--max-attempts", type=int, default=3); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-workers", type=int, default=int(os.environ.get("MAX_WORKERS", "1") or 1))
    p.add_argument("--harness", choices=("claude", "deepseek"), default=os.environ.get("AGENT_HARNESS", "claude"))
    p.add_argument("--dsh-bin", default=os.environ.get("DSH_BIN", "dsh")); p.add_argument("--dsh-node-bin", default=os.environ.get("DSH_NODE_BIN", "node"))
    p.add_argument("--dsh-model", default=os.environ.get("DSH_MODEL", "deepseek-v4-flash")); p.add_argument("--dsh-provider", default=os.environ.get("CC_SWITCH_PROVIDER", "dsv4flash"))
    a = p.parse_args()
    print(json.dumps(reasoning_shorten(a.input, a.output_root, tokenizer_path=a.tokenizer, token_limit=a.token_limit,
        target_tokens=a.target_tokens, max_workers=a.max_workers, limit=a.limit,
        claude_bin=a.claude_bin, timeout_sec=a.timeout_sec, max_attempts=a.max_attempts, harness=a.harness,
        dsh_bin=a.dsh_bin, dsh_node_bin=a.dsh_node_bin, dsh_model=a.dsh_model, dsh_provider=a.dsh_provider), indent=2))


if __name__ == "__main__": main()
