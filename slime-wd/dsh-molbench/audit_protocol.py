#!/usr/bin/env python3
"""Fail-closed audit of the first real DSH MolBench request."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


LOCAL_TOOLS = frozenset({"bash", "read", "write", "edit", "grep", "glob", "skill"})
MCP_PREFIX = "mcp__molclaw-scp__"


def digest_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def audit(
    run_dir: Path,
    system_prompt_file: Path,
    expected_mcp_tools: int,
    expected_skill_count: int,
) -> dict[str, Any]:
    transcripts = sorted(run_dir.glob("results/*/diagnostic_transcript.json"))
    if not transcripts:
        raise RuntimeError("no diagnostic transcript is available for request audit")
    selected_transcript: Path | None = None
    transcript: list[dict[str, Any]] = []
    headers: list[dict[str, Any]] = []
    for candidate in sorted(transcripts, key=lambda path: path.stat().st_mtime_ns):
        candidate_transcript = json.loads(candidate.read_text(encoding="utf-8"))
        candidate_headers = [
            entry for entry in candidate_transcript if entry.get("type") == "request/header"
        ]
        if candidate_headers:
            selected_transcript = candidate
            transcript = candidate_transcript
            headers = candidate_headers
            break
    if selected_transcript is None:
        raise RuntimeError("no completed transcript contains a request/header")
    header = headers[0].get("data", {}).get("header")
    if not isinstance(header, dict):
        raise RuntimeError("malformed request/header")

    expected_system = system_prompt_file.read_text(encoding="utf-8").rstrip()
    if header.get("system") != expected_system:
        raise RuntimeError("actual DSH system prompt differs from qwen35_system.md")
    tools = header.get("tools")
    if not isinstance(tools, list):
        raise RuntimeError("request/header has no tool catalog")
    names = [tool.get("name") for tool in tools if isinstance(tool, dict)]
    local = {name for name in names if isinstance(name, str) and not name.startswith(MCP_PREFIX)}
    mcp = [name for name in names if isinstance(name, str) and name.startswith(MCP_PREFIX)]
    if local != LOCAL_TOOLS:
        raise RuntimeError(f"unexpected local tools: expected {sorted(LOCAL_TOOLS)}, got {sorted(local)}")
    if len(mcp) != expected_mcp_tools or len(set(mcp)) != expected_mcp_tools:
        raise RuntimeError(
            f"unexpected MolClaw tool count: expected {expected_mcp_tools}, got {len(mcp)}"
        )
    config = header.get("config")
    if not isinstance(config, dict) or config.get("maxTokens") != 16384:
        raise RuntimeError(f"unexpected real request config: {config}")
    types = [entry.get("type") for entry in transcript]
    header_index = transcript.index(headers[0])
    pre_header = transcript[:header_index]
    task_messages = [
        entry for entry in pre_header
        if entry.get("type") == "user/message"
        and entry.get("data", {}).get("source", {}).get("kind") == "user"
    ]
    catalogs = [
        entry for entry in pre_header
        if entry.get("type") == "user/message"
        and entry.get("data", {}).get("source", {}).get("kind") == "skill-catalog"
    ]
    if len(task_messages) != 1 or len(catalogs) != 1:
        raise RuntimeError(
            "expected exactly one user task and one skill catalog before request assembly; "
            f"got task={len(task_messages)} catalog={len(catalogs)}"
        )
    if transcript.index(task_messages[0]) > transcript.index(catalogs[0]):
        raise RuntimeError("skill catalog precedes the user task")
    catalog_entries = catalogs[0].get("data", {}).get("source", {}).get("entries")
    if not isinstance(catalog_entries, list):
        raise RuntimeError("skill catalog event has no entries")
    skill_names = [
        entry.get("name") for entry in catalog_entries if isinstance(entry, dict)
    ]
    if len(skill_names) != expected_skill_count or len(set(skill_names)) != expected_skill_count:
        raise RuntimeError(
            f"unexpected skill count: expected {expected_skill_count}, got {len(skill_names)}"
        )
    task_content = task_messages[0].get("data", {}).get("content")
    if not isinstance(task_content, list) or len(task_content) != 1:
        raise RuntimeError("user task is not one content block")
    task_text = task_content[0].get("text") if isinstance(task_content[0], dict) else None
    if not isinstance(task_text, str) or not task_text:
        raise RuntimeError("user task has no text")
    task_hash = hashlib.sha256(task_text.encode()).hexdigest()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_prompts = manifest.get("task_prompt_sha256")
    task_id = selected_transcript.parent.name
    if not isinstance(expected_prompts, dict) or expected_prompts.get(task_id) != task_hash:
        raise RuntimeError(f"actual user task differs from the manifest prompt for {task_id}")

    result = {
        "schema_version": "dsh_molbench_protocol_audit_v1",
        "source_transcript": str(selected_transcript),
        "actual_request_config": config,
        "actual_system_sha256": hashlib.sha256(expected_system.encode()).hexdigest(),
        "actual_tool_catalog_sha256": digest_json(tools),
        "tool_count": len(names),
        "molclaw_tool_count": len(mcp),
        "local_tools": sorted(local),
        "actual_user_task_sha256": task_hash,
        "skill_catalog_count": len(skill_names),
        "skill_catalog_names": sorted(skill_names),
        "event_order": types,
        "passed": True,
    }
    atomic_json(run_dir / "protocol_audit.json", result)
    manifest["actual_first_request"] = result
    atomic_json(manifest_path, manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--system-prompt-file", type=Path, required=True)
    parser.add_argument("--expected-mcp-tools", type=int, default=81)
    parser.add_argument("--expected-skill-count", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(
        args.run_dir.resolve(),
        args.system_prompt_file.resolve(),
        args.expected_mcp_tools,
        args.expected_skill_count,
    ), indent=2))


if __name__ == "__main__":
    main()
