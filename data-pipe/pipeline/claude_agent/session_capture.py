"""Immutable capture helpers for Claude Code and DeepSeek Harness invocations."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any
import uuid


CLAUDE_CODE_EXECUTION_ENV = {
    "CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY": "2",
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
}

HTTP_500_RE = re.compile(r"\b(?:code|status(?:_code)?)\b.{0,24}\b500\b", re.I | re.S)
DSML_RE = re.compile(r"(?:｜DSML｜|<\|DSML\||<\s*(?:tool_calls|invoke)\b)", re.I)


def claude_code_environment() -> dict[str, str]:
    """Return the controlled environment shared by all Data-Pipe Claude runs."""
    env = os.environ.copy()
    env.update(CLAUDE_CODE_EXECUTION_ENV)
    return env


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_session(path: Path) -> dict[str, Any]:
    byte_count = path.stat().st_size if path.is_file() else 0
    parseable_event_count = 0
    if path.is_file():
        with path.open("rb") as stream:
            for raw_line in stream:
                try:
                    value = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    parseable_event_count += 1
    return {
        "byte_count": byte_count,
        "sha256": _sha256(path) if path.is_file() else None,
        "parseable_event_count": parseable_event_count,
        "raw_session_valid": byte_count > 0 and parseable_event_count > 0,
    }


def session_format(path: Path) -> str:
    """Identify the two raw trajectory schemas accepted by Data-Pipe."""
    if not path.is_file():
        return "unknown"
    event_types: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                event_types.add(str(value.get("type") or ""))
    if event_types & {"turn/start", "assistant/message", "tool/call"}:
        return "dsh-session-jsonl-v0"
    if event_types & {"assistant", "user", "result"}:
        return "claude-stream-json"
    return "unknown"


def extract_assistant_text(path: Path, *, final_only: bool = False) -> str:
    """Extract committed assistant text from either supported raw schema."""
    trace_format = session_format(path)
    texts: list[str] = []
    last_id = None
    terminal_parts: list[str] = []
    terminal_has_tools = False
    if not path.is_file():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            message: Any = None
            if trace_format == "claude-stream-json" and event.get("type") == "assistant":
                message = event.get("message")
            elif trace_format == "dsh-session-jsonl-v0" and event.get("type") == "assistant/message":
                data = event.get("data")
                message = data.get("message") if isinstance(data, dict) else None
            if not isinstance(message, dict):
                continue
            if final_only:
                message_id = message.get("id") or object()
                if message_id != last_id:
                    last_id = message_id
                    terminal_parts = []
                    terminal_has_tools = False
                content = message.get("content")
                terminal_has_tools |= bool(message.get("tool_calls"))
                if isinstance(content, str):
                    terminal_parts.append(content)
                elif isinstance(content, list):
                    terminal_has_tools |= any(isinstance(b, dict) and b.get("type") in {"tool_use", "tool-call"} for b in content)
                    terminal_parts.extend(str(b.get("text") or "") for b in content if isinstance(b, dict) and b.get("type") == "text")
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                texts.append(content.strip())
                continue
            if not isinstance(content, list):
                continue
            committed = "".join(
                str(block.get("text") or "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip()
            if committed:
                texts.append(committed)
    if final_only:
        return "" if terminal_has_tools else "".join(terminal_parts).strip()
    return "\n".join(texts)


def session_has_dsml(path: Path) -> bool:
    return bool(DSML_RE.search(extract_assistant_text(path)))


def write_session_pretty(session_path: Path) -> Path:
    """Write a readable sidecar without changing the immutable raw stream."""
    pretty_path = session_path.with_suffix(".pretty.json")
    rows: list[Any] = []
    with session_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.rstrip("\r\n")
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                rows.append(
                    {
                        "type": "raw_stream_diagnostic",
                        "line_number": line_number,
                        "content": text,
                    }
                )
    pretty_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return pretty_path


def session_has_retryable_http_500(path: Path) -> bool:
    """Return true only for a terminal upstream HTTP-500 API failure."""
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "result" or not event.get("is_error"):
                continue
            text = json.dumps(
                {"result": event.get("result"), "error": event.get("error")},
                ensure_ascii=False,
                default=str,
            )
            if ("API Error" in text or "ChatCompletionStreamResponse" in text) and HTTP_500_RE.search(text):
                return True
    return False


def http_500_retry_delay(retry_count: int) -> int:
    """Exponential retry delay capped at five minutes."""
    if retry_count < 1:
        raise ValueError("retry_count must be >= 1")
    return min(300, 30 * (2 ** min(retry_count - 1, 4)))


def next_attempt_index(workdir: Path) -> int:
    attempts_root = workdir / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    indexes: list[int] = []
    for child in attempts_root.glob("attempt_*"):
        try:
            indexes.append(int(child.name.removeprefix("attempt_")))
        except ValueError:
            continue
    return max(indexes, default=0) + 1


def next_attempt(workdir: Path) -> tuple[int, Path]:
    attempts_root = workdir / "attempts"
    index = next_attempt_index(workdir)
    attempt_dir = attempts_root / f"attempt_{index:04d}"
    attempt_dir.mkdir(parents=False, exist_ok=False)
    session_path = attempt_dir / "complete_session.jsonl"
    session_path.touch()
    return index, session_path


def run_stream_json(
    command: list[str],
    *,
    cwd: Path,
    archive_root: Path,
    attempt_index: int | None = None,
    input_text: str | None = None,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    if "--verbose" not in command or "--output-format" not in command:
        raise ValueError("Claude command must request verbose stream-json output")
    output_index = command.index("--output-format") + 1
    if output_index >= len(command) or command[output_index] != "stream-json":
        raise ValueError("Claude command must request --output-format stream-json")

    if attempt_index is None:
        attempt_index, session_path = next_attempt(archive_root)
    else:
        if attempt_index < 1:
            raise ValueError("attempt_index must be >= 1")
        attempt_dir = archive_root / "attempts" / f"attempt_{attempt_index:04d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        session_path = attempt_dir / "complete_session.jsonl"
        if session_path.exists():
            raise FileExistsError(f"Claude attempt session already exists: {session_path}")
        session_path.touch()
    started = time.time()
    return_code = 1
    timed_out = False
    failure: str | None = None
    with session_path.open("wb") as session_stream:
        try:
            process = subprocess.run(
                command,
                cwd=str(cwd),
                env=claude_code_environment(),
                input=input_text.encode("utf-8") if input_text is not None else None,
                stdout=session_stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_sec,
            )
            return_code = int(process.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = 124
            failure = "timeout"
        except FileNotFoundError:
            return_code = 127
            failure = "executable_not_found"

    pretty_path = write_session_pretty(session_path)

    metadata = {
        "attempt_index": attempt_index,
        "workdir": str(cwd),
        "session_file": str(session_path),
        "return_code": return_code,
        "timed_out": timed_out,
        "timeout_sec": timeout_sec,
        "duration_sec": round(time.time() - started, 3),
        "failure": failure,
        "pretty_session_file": str(pretty_path),
    }
    metadata.update(inspect_session(session_path))
    return metadata


def _load_mcp_config(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    servers = payload.get("mcpServers") if isinstance(payload, dict) else None
    if not isinstance(servers, dict):
        raise ValueError("MCP config must contain an mcpServers object")
    result: list[dict[str, Any]] = []
    for name, raw in servers.items():
        if not isinstance(raw, dict):
            raise ValueError(f"invalid MCP server config: {name}")
        server_name = str(name).strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", server_name):
            raise ValueError(f"invalid DSH MCP server name: {name!r}")
        server_type = str(raw.get("type") or "stdio")
        timeout_ms = int(raw.get("timeout") or raw.get("toolCallTimeoutMs") or 60_000)
        if server_type in {"http", "sse", "streamable-http"}:
            result.append({
                "transport": "streamable-http",
                "serverName": server_name,
                "url": str(raw.get("url") or ""),
                "headers": dict(raw.get("headers") or {}),
                "toolCallTimeoutMs": timeout_ms,
                "failOnStartupError": True,
            })
        else:
            result.append({
                "transport": "stdio",
                "serverName": server_name,
                "command": str(raw.get("command") or ""),
                "args": list(raw.get("args") or []),
                "env": dict(raw.get("env") or {}),
                "cwd": str(raw.get("cwd") or "."),
                "toolCallTimeoutMs": timeout_ms,
                "failOnStartupError": True,
            })
    return result


def _dsh_patch(system_prompt: str, model: str, mcp_servers: list[dict[str, Any]], *, scientific_collection: bool = False) -> str:
    rows = [
        {"id": "agent-default-model", "config": {"provider": "deepseek-official", "model": model}},
        {"id": "system-prompt", "config": {"persona": system_prompt}},
        {"id": "session-title-llm", "disabled": True},
        {"id": "session-persistence-jsonl", "config": {
            "root": "__DSH_SESSION_ROOT__", "compression": "none", "packChunks": False,
        }},
    ]
    if scientific_collection:
        rows[1]["config"].update(includeHarnessIdentity=False, includeRuntimeContext=False)
        # Same seven local tool families as the evaluation preset on Linux.
        rows.extend({"id": name, "disabled": True} for name in (
            "tool-pwsh", "tool-jobs", "tool-subagent-control", "tool-subagent-list-agents",
            "tool-subagent", "tool-subagent-fork", "tool-subagent-report", "tool-workflow",
            "tool-todo", "tool-goal", "tool-ralph", "tool-str-replace-editor", "tool-web"))
        rows.append({"id": "tool-fs", "config": {"readImage": False}})
    # JSON is valid YAML. Replace only the one value that must remain a Cordis
    # expression; all user-controlled strings stay JSON-escaped.
    text = json.dumps(rows, ensure_ascii=False, indent=2).replace(
        '"__DSH_SESSION_ROOT__"', "!!js dshHomePath('sessions')"
    )
    if mcp_servers:
        inserted = [
            {
                "id": f"data-pipe-mcp-{index}",
                "name": "@deepseek-ai/dsh-mcp-client",
                "config": server,
            }
            for index, server in enumerate(mcp_servers, start=1)
        ]
        text = text[:-1] + ",\n  {\n    \"insert\": " + json.dumps(inserted, ensure_ascii=False, indent=2) + "\n  }\n]"
    return text + "\n"


def _find_dsh_session(dsh_home: Path) -> Path:
    sessions = list((dsh_home / "sessions").glob("**/session.jsonl"))
    if len(sessions) != 1:
        raise RuntimeError(f"expected exactly one DSH session, found {len(sessions)}")
    return sessions[0]


def _deepseek_environment(provider_id: str | None) -> dict[str, str]:
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "").strip()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if base_url and api_key:
        return {"DEEPSEEK_BASE_URL": base_url, "DEEPSEEK_API_KEY": api_key}
    if not provider_id:
        raise RuntimeError(
            "DeepSeek Harness requires DEEPSEEK_BASE_URL/DEEPSEEK_API_KEY or a cc-switch provider id"
        )
    database = Path(
        os.environ.get("CC_SWITCH_DB", str(Path.home() / ".cc-switch/cc-switch.db"))
    ).expanduser()
    if not database.is_file():
        raise RuntimeError(f"cc-switch database not found: {database}")
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT settings_config FROM providers WHERE id = ? AND app_type = 'claude'",
            (provider_id,),
        ).fetchone()
    if row is None:
        raise RuntimeError(f"cc-switch Claude provider not found: {provider_id}")
    settings = json.loads(str(row[0]))
    values = settings.get("env") if isinstance(settings, dict) else None
    values = values if isinstance(values, dict) else settings
    if not isinstance(values, dict):
        raise RuntimeError(f"invalid cc-switch settings for provider: {provider_id}")
    base_url = str(values.get("ANTHROPIC_BASE_URL") or "").strip().rstrip("/")
    api_key = str(values.get("ANTHROPIC_API_KEY") or "").strip()
    if not base_url or not api_key:
        raise RuntimeError(f"cc-switch provider lacks API URL/key: {provider_id}")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return {"DEEPSEEK_BASE_URL": base_url, "DEEPSEEK_API_KEY": api_key}


def _validated_node_version(node: str) -> str:
    """Return the Node version after enforcing DeepSeek Harness' engine range."""
    process = subprocess.run(
        [node, "--version"], text=True, capture_output=True, check=False, timeout=10,
    )
    version = process.stdout.strip()
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", version)
    if process.returncode != 0 or match is None:
        detail = process.stderr.strip() or version or f"exit {process.returncode}"
        raise RuntimeError(f"cannot determine Node version for DeepSeek Harness: {detail}")
    major, minor, _patch = (int(part) for part in match.groups())
    if not ((major == 22 and minor >= 19) or major >= 24):
        raise RuntimeError(
            f"DeepSeek Harness requires Node ^22.19.0 or >=24.0.0; found {version}"
        )
    return version


def run_deepseek_harness(
    prompt: str,
    system_prompt: str,
    *,
    cwd: Path,
    archive_root: Path,
    dsh_bin: str = "dsh",
    node_bin: str = "node",
    model: str = "deepseek-v4-flash",
    mcp_config_file: Path | None = None,
    provider_id: str | None = None,
    attempt_index: int | None = None,
    timeout_sec: float | None = None,
    scientific_collection: bool = False,
) -> dict[str, Any]:
    """Run one isolated DSH session and preserve its canonical durable JSONL."""
    resolved_dsh = shutil.which(dsh_bin) or (str(Path(dsh_bin).resolve()) if Path(dsh_bin).is_file() else "")
    resolved_node = shutil.which(node_bin) or (str(Path(node_bin).resolve()) if Path(node_bin).is_file() else "")
    if not resolved_dsh:
        raise FileNotFoundError(f"DSH executable not found: {dsh_bin}")
    if not resolved_node:
        raise FileNotFoundError(f"Node executable not found: {node_bin}")
    node_version = _validated_node_version(resolved_node)
    deepseek_env = _deepseek_environment(provider_id)
    if attempt_index is None:
        attempt_index, session_path = next_attempt(archive_root)
    else:
        if attempt_index < 1:
            raise ValueError("attempt_index must be >= 1")
        attempt_dir = archive_root / "attempts" / f"attempt_{attempt_index:04d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        session_path = attempt_dir / "complete_session.jsonl"
        if session_path.exists():
            raise FileExistsError(f"DSH attempt session already exists: {session_path}")
    attempt_dir = session_path.parent
    dsh_home = attempt_dir / "dsh_home"
    dsh_home.mkdir()
    patch_path = attempt_dir / f"dsh.{uuid.uuid4().hex}.patch.yml"
    mcp_servers = _load_mcp_config(mcp_config_file)
    try:
        patch_path.write_text(_dsh_patch(system_prompt, model, mcp_servers, scientific_collection=scientific_collection), encoding="utf-8")
        os.chmod(patch_path, 0o600)
    except Exception:
        patch_path.unlink(missing_ok=True)
        raise

    command = [resolved_dsh, "--profile", "headless", "--patch", str(patch_path), prompt]
    environment = os.environ.copy()
    environment.update({
        **deepseek_env,
        "DSH_HOME": str(dsh_home),
        "DSH_PERMISSION_MODE": "danger-full-access",
        "DSH_TELEMETRY_DISABLED": "1",
        # Node's fetch/undici does not honor HTTP(S)_PROXY unless this switch
        # is enabled. The lab MCP endpoint is reachable only through that proxy.
        "NODE_USE_ENV_PROXY": "1",
        "PATH": str(Path(resolved_node).parent) + os.pathsep + environment.get("PATH", ""),
    })
    started = time.time()
    return_code = 1
    timed_out = False
    failure: str | None = None
    stdout = ""
    stderr = ""
    try:
        process = subprocess.run(
            command, cwd=str(cwd), env=environment, text=True, capture_output=True,
            check=False, timeout=timeout_sec,
        )
        return_code = int(process.returncode)
        stdout, stderr = process.stdout, process.stderr
    except subprocess.TimeoutExpired as exc:
        return_code = 124
        timed_out = True
        failure = "timeout"
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    finally:
        # MCP headers may contain credentials. They must never become an artifact.
        patch_path.unlink(missing_ok=True)

    (attempt_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (attempt_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    try:
        source = _find_dsh_session(dsh_home)
        shutil.copyfile(source, session_path)
    except Exception as exc:
        if failure is None:
            failure = f"session_capture:{type(exc).__name__}:{exc}"
        session_path.touch(exist_ok=True)
    inspection = inspect_session(session_path)
    dsml_detected = session_has_dsml(session_path)
    if return_code == 0 and dsml_detected:
        return_code = 96
        failure = "dsml_tool_call_leaked_as_text"
    return {
        "attempt_index": attempt_index,
        "harness": "deepseek",
        "workdir": str(cwd),
        "session_file": str(session_path),
        "return_code": return_code,
        "timed_out": timed_out,
        "timeout_sec": timeout_sec,
        "duration_sec": round(time.time() - started, 3),
        "failure": failure,
        "pretty_session_file": str(write_session_pretty(session_path)),
        "stdout_file": str(attempt_dir / "stdout.txt"),
        "stderr_file": str(attempt_dir / "stderr.txt"),
        "command": [resolved_dsh, "--profile", "headless", "--patch", "<ephemeral-patch>", "<prompt>"],
        "node_version": node_version,
        "model": model,
        "mcp_startup_enforced": bool(mcp_servers),
        "assistant_text": extract_assistant_text(session_path),
        "result_text": extract_assistant_text(session_path, final_only=True),
        "dsml_detected": dsml_detected,
        **inspection,
    }


def select_attempt(attempt: dict[str, Any], canonical_path: Path) -> dict[str, Any]:
    source = Path(str(attempt["session_file"]))
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, canonical_path)
    pretty_path = write_session_pretty(canonical_path)
    selected = inspect_session(canonical_path)
    selected["pretty_session_file"] = str(pretty_path)
    if selected["sha256"] != attempt.get("sha256"):
        raise RuntimeError(
            f"selected Claude session checksum mismatch: {source} -> {canonical_path}"
        )
    return selected
