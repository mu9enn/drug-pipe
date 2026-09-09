#!/usr/bin/env python3
"""Capture one DeepSeek Harness tool round and compare its JSONL with Claude Code."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable
import uuid


DSML_RE = re.compile(r"(?:｜DSML｜|<\|DSML\||<\s*(?:tool_calls|invoke)\b)", re.IGNORECASE)
DEFAULT_MODEL = "deepseek-v4-flash"
PROBE_NAME = "dsh_probe.txt"


def _write_text(path: Path, text: str, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    records: list[dict[str, Any]] = []
    malformed: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed.append(line_number)
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                malformed.append(line_number)
    return records, malformed


def _content_blocks(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    yield block
        for child in value.values():
            yield from _content_blocks(child)
    elif isinstance(value, list):
        for child in value:
            yield from _content_blocks(child)


def detect_format(records: list[dict[str, Any]]) -> str:
    types = {str(record.get("type") or "") for record in records}
    if "turn/start" in types or "assistant/message" in types or "tool/call" in types:
        return "dsh-session-jsonl-v0"
    if types & {"assistant", "user", "result"}:
        return "claude-stream-json"
    return "unknown-jsonl"


def _assistant_texts(records: list[dict[str, Any]], trace_format: str) -> list[str]:
    texts: list[str] = []
    if trace_format == "dsh-session-jsonl-v0":
        for record in records:
            if record.get("type") != "assistant/message":
                continue
            message = (record.get("data") or {}).get("message")
            if not isinstance(message, dict):
                continue
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") in {"text", "reasoning"}:
                    text = block.get("text")
                    if isinstance(text, str):
                        texts.append(text)
    elif trace_format == "claude-stream-json":
        for record in records:
            if record.get("type") != "assistant":
                continue
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") in {"text", "thinking"}:
                    field = "thinking" if block.get("type") == "thinking" else "text"
                    text = block.get(field)
                    if isinstance(text, str):
                        texts.append(text)
    return texts


def summarize_trace(path: Path) -> dict[str, Any]:
    records, malformed = load_jsonl(path)
    trace_format = detect_format(records)
    event_types = Counter(str(record.get("type") or "<missing>") for record in records)
    block_types = Counter(
        str(block.get("type"))
        for record in records
        for block in _content_blocks(record)
        if isinstance(block.get("type"), str)
    )
    tool_calls: list[str] = []
    tool_result_count = 0
    terminal: dict[str, Any] | None = None
    if trace_format == "dsh-session-jsonl-v0":
        for record in records:
            if record.get("type") == "tool/call":
                data = record.get("data") or {}
                tool_calls.append(str(data.get("name") or ""))
            elif record.get("type") == "tool/result":
                tool_result_count += 1
            elif record.get("type") == "turn/end":
                terminal = (record.get("data") or {}).get("reason")
    elif trace_format == "claude-stream-json":
        for record in records:
            if record.get("type") == "assistant":
                message = record.get("message") or {}
                for block in message.get("content") or []:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_calls.append(str(block.get("name") or ""))
            elif record.get("type") == "user":
                message = record.get("message") or {}
                tool_result_count += sum(
                    1 for block in message.get("content") or []
                    if isinstance(block, dict) and block.get("type") == "tool_result"
                )
            elif record.get("type") == "result":
                terminal = {
                    "subtype": record.get("subtype"),
                    "is_error": record.get("is_error"),
                    "stop_reason": record.get("stop_reason"),
                }
    assistant_texts = _assistant_texts(records, trace_format)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
        "format": trace_format,
        "record_count": len(records),
        "malformed_line_count": len(malformed),
        "malformed_line_numbers": malformed[:20],
        "event_types": dict(sorted(event_types.items())),
        "content_block_types": dict(sorted(block_types.items())),
        "tool_call_count": len(tool_calls),
        "tool_result_count": tool_result_count,
        "tool_names": tool_calls,
        "terminal": terminal,
        "assistant_dsml_detected": any(DSML_RE.search(text) for text in assistant_texts),
    }


def compare_traces(dsh_session: Path, claude_session: Path) -> dict[str, Any]:
    dsh = summarize_trace(dsh_session)
    claude = summarize_trace(claude_session)
    return {
        "schema_version": "dsh_claude_trajectory_comparison_v1",
        "dsh": dsh,
        "claude": claude,
        "same_format": dsh["format"] == claude["format"],
        "event_type_intersection": sorted(set(dsh["event_types"]) & set(claude["event_types"])),
        "event_types_only_in_dsh": sorted(set(dsh["event_types"]) - set(claude["event_types"])),
        "event_types_only_in_claude": sorted(set(claude["event_types"]) - set(dsh["event_types"])),
    }


def comparison_markdown(report: dict[str, Any]) -> str:
    dsh = report["dsh"]
    claude = report["claude"]
    rows = [
        "# DSH and Claude trajectory comparison",
        "",
        "| Property | DeepSeek Harness | Claude Code |",
        "|---|---:|---:|",
        f"| Format | `{dsh['format']}` | `{claude['format']}` |",
        f"| JSON records | {dsh['record_count']} | {claude['record_count']} |",
        f"| Malformed lines | {dsh['malformed_line_count']} | {claude['malformed_line_count']} |",
        f"| Tool calls | {dsh['tool_call_count']} | {claude['tool_call_count']} |",
        f"| Tool results | {dsh['tool_result_count']} | {claude['tool_result_count']} |",
        f"| Assistant DSML detected | {str(dsh['assistant_dsml_detected']).lower()} | {str(claude['assistant_dsml_detected']).lower()} |",
        "",
        f"Same raw format: **{str(report['same_format']).lower()}**.",
        "",
        "The report intentionally excludes prompt and tool-result bodies.",
    ]
    return "\n".join(rows) + "\n"


def _node_version(node_bin: Path) -> tuple[str, tuple[int, int, int]]:
    process = subprocess.run(
        [str(node_bin), "--version"], text=True, capture_output=True, check=False, timeout=10,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Node version check failed with exit code {process.returncode}")
    raw = process.stdout.strip()
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", raw)
    if not match:
        raise RuntimeError(f"unrecognized Node version: {raw!r}")
    return raw, tuple(int(part) for part in match.groups())


def _resolve_executable(value: str, label: str) -> Path:
    resolved = shutil.which(value)
    if resolved is None and Path(value).is_file():
        resolved = str(Path(value).resolve())
    if resolved is None:
        raise FileNotFoundError(f"{label} executable not found: {value}")
    return Path(resolved).resolve()


def _ensure_empty_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"output directory must be absent or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _find_session(dsh_home: Path) -> Path:
    sessions = list((dsh_home / "sessions").glob("**/session.jsonl"))
    if len(sessions) != 1:
        raise RuntimeError(f"expected exactly one uncompressed DSH session, found {len(sessions)}")
    return sessions[0]


def _dsh_patch(model: str) -> str:
    return f"""- id: agent-default-model
  config:
    provider: deepseek-official
    model: {json.dumps(model)}

- id: system-prompt
  config:
    persona: >-
      You are a non-interactive harness validation agent. Follow the user's exact tool-use instruction.

- id: session-title-llm
  disabled: true

- id: session-persistence-jsonl
  config:
    root: !!js dshHomePath('sessions')
    compression: none
    packChunks: false
"""


def smoke_passed(validation: dict[str, Any]) -> bool:
    required_true = (
        "process_exit_zero",
        "session_found",
        "tool_call_recorded",
        "tool_result_recorded",
        "turn_completed",
        "probe_returned",
    )
    return (
        all(validation.get(key) is True for key in required_true)
        and validation.get("timed_out") is False
        and validation.get("assistant_dsml_detected") is False
    )


def run_smoke(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    _ensure_empty_output(output_dir)
    workspace = output_dir / "workspace"
    dsh_home = output_dir / "dsh_home"
    workspace.mkdir()
    dsh_home.mkdir()
    (workspace / ".git").mkdir()

    marker = f"DSH_PROBE_{uuid.uuid4().hex}"
    _write_text(workspace / PROBE_NAME, marker + "\n")
    prompt = (
        f"Use the native read tool exactly once to read {PROBE_NAME}. "
        "Do not use bash or any other tool. Then reply with only the file's exact content."
    )
    patch_path = output_dir / "smoke.patch.yml"
    _write_text(patch_path, _dsh_patch(args.model))

    dsh_bin = _resolve_executable(args.dsh_bin, "DSH")
    node_bin = _resolve_executable(args.node_bin, "Node")
    node_version, parsed_node = _node_version(node_bin)
    if parsed_node < (22, 19, 0):
        raise RuntimeError(f"DeepSeek Harness requires Node >=22.19; got {node_version}")
    if not os.environ.get("DEEPSEEK_BASE_URL"):
        raise RuntimeError("DEEPSEEK_BASE_URL is required")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required")

    command = [str(dsh_bin), "--profile", "headless", "--patch", str(patch_path), prompt]
    environment = os.environ.copy()
    environment.update({
        "DSH_HOME": str(dsh_home),
        "DSH_PERMISSION_MODE": "workspace-write",
        "DSH_TELEMETRY_DISABLED": "1",
        "PATH": str(node_bin.parent) + os.pathsep + environment.get("PATH", ""),
    })
    started = time.time()
    timed_out = False
    try:
        process = subprocess.run(
            command,
            cwd=workspace,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=args.timeout_sec,
        )
        return_code = int(process.returncode)
        stdout = process.stdout
        stderr = process.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        return_code = 124
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""

    _write_text(output_dir / "stdout.txt", stdout)
    _write_text(output_dir / "stderr.txt", stderr)
    session_source: Path | None = None
    validation: dict[str, Any] = {
        "process_exit_zero": return_code == 0,
        "timed_out": timed_out,
        "session_found": False,
        "tool_call_recorded": False,
        "tool_result_recorded": False,
        "turn_completed": False,
        "probe_returned": stdout.strip() == marker,
        "assistant_dsml_detected": None,
    }
    session_summary: dict[str, Any] | None = None
    failure: str | None = None
    try:
        session_source = _find_session(dsh_home)
        canonical = output_dir / "complete_session.jsonl"
        shutil.copyfile(session_source, canonical)
        if _sha256(session_source) != _sha256(canonical):
            raise RuntimeError("copied session checksum mismatch")
        session_summary = summarize_trace(canonical)
        records, _ = load_jsonl(canonical)
        validation.update({
            "session_found": True,
            "tool_call_recorded": session_summary["tool_names"] == ["read"],
            "tool_result_recorded": session_summary["tool_result_count"] == 1,
            "turn_completed": any(
                record.get("type") == "turn/end"
                and ((record.get("data") or {}).get("reason") or {}).get("kind") == "completed"
                for record in records
            ),
            "assistant_dsml_detected": session_summary["assistant_dsml_detected"],
        })
    except Exception as exc:  # Preserve subprocess diagnostics for a failed smoke.
        failure = f"{type(exc).__name__}: {exc}"

    passed = smoke_passed(validation)
    metadata = {
        "schema_version": "deepseek_harness_smoke_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "deepseek-official",
        "model": args.model,
        "base_url_configured": True,
        "api_key_configured": True,
        "dsh_bin": str(dsh_bin),
        "node_bin": str(node_bin),
        "node_version": node_version,
        "command": [str(dsh_bin), "--profile", "headless", "--patch", str(patch_path), "<probe-prompt>"],
        "return_code": return_code,
        "duration_sec": round(time.time() - started, 3),
        "failure": failure,
        "validation": validation,
        "session": session_summary,
        "passed": passed,
    }
    _write_json(output_dir / "run_meta.json", metadata)
    print(output_dir)
    return 0 if passed else 1


def run_compare(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = compare_traces(args.dsh_session.resolve(), args.claude_session.resolve())
    _write_json(output_dir / "trajectory_comparison.json", report)
    _write_text(output_dir / "trajectory_comparison.md", comparison_markdown(report))
    print(output_dir / "trajectory_comparison.md")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one real DSH read-tool smoke")
    run.add_argument("--dsh-bin", default=os.environ.get("DSH_BIN", "dsh"))
    run.add_argument("--node-bin", default=os.environ.get("DSH_NODE_BIN", "node"))
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--timeout-sec", type=int, default=300)
    run.set_defaults(func=run_smoke)

    compare = subparsers.add_parser("compare", help="compare DSH and Claude raw trajectories")
    compare.add_argument("--dsh-session", type=Path, required=True)
    compare.add_argument("--claude-session", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.set_defaults(func=run_compare)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
