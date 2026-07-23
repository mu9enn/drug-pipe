from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..io_utils import sha256_file, sha256_text
from ..settings import ProjectConfig


_JSON_BLOCK_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)```")


def safe_name(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return s[:120] if s else "item"


def extract_stream_result(raw_stream: str) -> tuple[str, str]:
    assistant_chunks: list[str] = []
    result_text = ""
    for line in raw_stream.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == "assistant":
            message = obj.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        txt = item.get("text")
                        if isinstance(txt, str) and txt.strip():
                            assistant_chunks.append(txt)
            elif isinstance(content, str) and content.strip():
                assistant_chunks.append(content)
        elif obj.get("type") == "result":
            rt = obj.get("result")
            if isinstance(rt, str) and rt.strip():
                result_text = rt.strip()
    return ("\n".join(assistant_chunks).strip(), result_text.strip())


def extract_json_object(text: str) -> dict[str, Any] | None:
    candidates: list[str] = []
    for block in _JSON_BLOCK_RE.findall(text or ""):
        s = block.strip()
        if s.startswith("{") and s.endswith("}"):
            candidates.append(s)

    t = (text or "").strip()
    if t.startswith("{") and t.endswith("}"):
        candidates.append(t)
    else:
        first = t.find("{")
        last = t.rfind("}")
        if first >= 0 and last > first:
            candidates.append(t[first : last + 1].strip())

    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def inspect_raw_session(path: Path) -> dict[str, Any]:
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
        "sha256": sha256_file(path) if path.is_file() else None,
        "parseable_event_count": parseable_event_count,
        "raw_session_valid": byte_count > 0 and parseable_event_count > 0,
    }


def allocate_attempt(workdir: Path) -> tuple[int, Path]:
    attempts_root = workdir / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    indexes: list[int] = []
    for child in attempts_root.glob("attempt_*"):
        try:
            indexes.append(int(child.name.removeprefix("attempt_")))
        except ValueError:
            continue
    index = max(indexes, default=0) + 1
    attempt_dir = attempts_root / f"attempt_{index:04d}"
    attempt_dir.mkdir()
    session_path = attempt_dir / "complete_session.jsonl"
    session_path.touch()
    return index, session_path


def inspect_mcp_init(raw_stream: str, expected_server: str | list[str]) -> tuple[bool, str, dict[str, Any]]:
    init_obj: dict[str, Any] | None = None
    for line in raw_stream.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == "system" and obj.get("subtype") == "init":
            init_obj = obj

    if init_obj is None:
        return False, "missing_system_init_event", {}

    tools = init_obj.get("tools")
    tools = tools if isinstance(tools, list) else []
    mcp_tools = [t for t in tools if isinstance(t, str) and t.startswith("mcp__")]

    mcp_servers = init_obj.get("mcp_servers")
    mcp_servers = mcp_servers if isinstance(mcp_servers, list) else []
    status_by_name: dict[str, str] = {}
    for item in mcp_servers:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        status_by_name[name] = str(item.get("status") or "").strip().lower()

    snapshot = {
        "mcp_tools_count": len(mcp_tools),
        "mcp_servers": status_by_name,
    }
    expected_servers = [expected_server] if isinstance(expected_server, str) else list(expected_server)
    for name in expected_servers:
        if status_by_name.get(name) != "connected":
            got = status_by_name.get(name, "missing")
            return False, f"mcp_server_not_connected:{name}:{got}", snapshot
    if not mcp_tools:
        return False, "mcp_tools_missing_in_init", snapshot
    return True, "ok", snapshot


@dataclass
class ClaudeCodeRunResult:
    ok: bool
    return_code: int
    timed_out: bool
    latency_sec: float
    command: str
    provider: str
    provider_switch_ok: bool
    provider_switch_message: str
    prompt_sha256: str
    mcp_config_sha256: str | None
    mcp_server_name: str
    mcp_server_url: str
    workdir: str
    session_file: str
    attempt_session_files: list[str]
    claude_attempts: list[dict[str, Any]]
    selected_claude_attempt: int
    raw_stream: str
    assistant_text: str
    result_text: str


class ClaudeCodeRuntime:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.provider = os.getenv("MOLCLAW_AGENT_PROVIDER", os.getenv("CC_SWITCH_PROVIDER", "manual"))
        self.claude_bin = "claude"
        self.mcp_server_name = "molclaw-scp"
        self.mcp_server_url = (
            os.getenv("MOLCLAW_SCP_MCP_URL", "")
            or config.runtime.server_url
        )
        self.mcp_auth_header = os.getenv("MOLCLAW_SCP_MCP_AUTH_HEADER", "SCP-HUB-API-KEY")
        self.mcp_auth_token = (
            os.getenv("MOLCLAW_SCP_MCP_AUTH", "")
            or config.runtime.api_key
        )

    def switch_provider(self) -> tuple[bool, str]:
        # cmd = ["cc-switch", "provider", "switch", self.provider]
        # proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        # if proc.returncode != 0:
        #     return False, f"cc-switch failed: stdout={proc.stdout.strip()} stderr={proc.stderr.strip()}"
        # return True, proc.stdout.strip() or "ok"
        return True, "provider switch disabled (expect external cc-switch before run)"

    def _write_mcp_config(self, path: Path, mcp_servers: dict[str, dict[str, Any]] | None = None) -> None:
        if mcp_servers is None:
            if not self.mcp_server_url:
                raise RuntimeError("missing MCP server URL: MOLCLAW_SCP_MCP_URL")
            if not self.mcp_auth_token:
                raise RuntimeError("missing MCP auth token: MOLCLAW_SCP_MCP_AUTH")
            server: dict[str, Any] = {"type": "http", "url": self.mcp_server_url}
            if self.mcp_auth_header and self.mcp_auth_token:
                server["headers"] = {self.mcp_auth_header: self.mcp_auth_token}
            mcp_servers = {self.mcp_server_name: server}
        cfg = {"mcpServers": mcp_servers}
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def run_prompt(
        self,
        prompt: str,
        *,
        run_label: str,
        add_dirs: list[Path] | None = None,
        allowed_tools: str | None = None,
        workdir: Path | None = None,
        mcp_servers: dict[str, dict[str, Any]] | None = None,
        expected_mcp_servers: list[str] | None = None,
    ) -> ClaudeCodeRunResult:
        prompt_hash = sha256_text(prompt)
        # provider_ok, provider_msg = self.switch_provider()
        # if not provider_ok:
        #     return ClaudeCodeRunResult(
        #         ok=False,
        #         return_code=127,
        #         timed_out=False,
        #         latency_sec=0.0,
        #         command="",
        #         provider=self.provider,
        #         provider_switch_ok=False,
        #         provider_switch_message=provider_msg,
        #         prompt_sha256=prompt_hash,
        #         mcp_config_sha256=None,
        #         mcp_server_name=self.mcp_server_name,
        #         mcp_server_url=self.mcp_server_url,
        #         workdir=str((workdir or self.config.paths.root).resolve()),
        #         session_file="",
        #         raw_stream="",
        #         assistant_text="",
        #         result_text="",
        #     )
        provider_ok, provider_msg = True, "provider switch disabled (expect external cc-switch before run)"

        with tempfile.TemporaryDirectory(prefix=f"molclaw_{safe_name(run_label)}_") as tmpdir:
            td = Path(tmpdir)
            mcp_cfg = td / "mcp_config.json"
            self._write_mcp_config(mcp_cfg, mcp_servers=mcp_servers)
            mcp_cfg_hash = sha256_file(mcp_cfg)

            cmd = [
                self.claude_bin,
                "--dangerously-skip-permissions",
                "--verbose",
                "--output-format",
                "stream-json",
                "--mcp-config",
                str(mcp_cfg),
                "--strict-mcp-config",
            ]
            for p in add_dirs or []:
                cmd.extend(["--add-dir", str(p)])
            if allowed_tools:
                cmd.extend(["--allowedTools", allowed_tools])
            cmd.append("-p")

            actual_workdir = (workdir or self.config.paths.root).resolve()
            actual_workdir.mkdir(parents=True, exist_ok=True)
            session_path = actual_workdir / "complete_session.jsonl"
            max_ready_retries = max(0, int(os.getenv("CLAUDE_MCP_READY_RETRIES", "2")))
            ready_retry_wait_sec = max(0.0, float(os.getenv("CLAUDE_MCP_READY_RETRY_WAIT_SEC", "2")))
            timeout_value = float(os.getenv("CLAUDE_CODE_TIMEOUT_SEC", "0"))
            timeout_sec = timeout_value if timeout_value > 0 else None

            timed_out = False
            return_code = 1
            raw_stream = ""
            mcp_ready = False
            mcp_reason = "not_checked"
            mcp_attempts = 0
            claude_attempts: list[dict[str, Any]] = []
            t0 = time.time()

            while True:
                mcp_attempts += 1
                attempt_index, attempt_session_path = allocate_attempt(actual_workdir)
                attempt_started = time.time()
                attempt_failure: str | None = None
                try:
                    with attempt_session_path.open("wb") as session_f:
                        proc = subprocess.run(
                            cmd,
                            cwd=str(actual_workdir),
                            input=prompt.encode("utf-8"),
                            stdout=session_f,
                            stderr=subprocess.STDOUT,
                            check=False,
                            timeout=timeout_sec,
                        )
                        return_code = int(proc.returncode)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    return_code = 124
                    attempt_failure = "timeout"
                except FileNotFoundError:
                    return_code = 127
                    attempt_failure = "executable_not_found"

                inspection = inspect_raw_session(attempt_session_path)
                attempt_meta = {
                    "attempt_index": attempt_index,
                    "session_file": str(attempt_session_path),
                    "return_code": return_code,
                    "timed_out": timed_out,
                    "timeout_sec": timeout_sec,
                    "duration_sec": round(time.time() - attempt_started, 3),
                    "failure": attempt_failure,
                    **inspection,
                }
                claude_attempts.append(attempt_meta)
                raw_stream = attempt_session_path.read_text(
                    encoding="utf-8", errors="ignore"
                )
                mcp_ready, mcp_reason, _ = inspect_mcp_init(
                    raw_stream,
                    expected_mcp_servers or [self.mcp_server_name],
                )
                if mcp_ready:
                    break
                if timed_out or return_code in {124, 127}:
                    break
                if mcp_attempts > max_ready_retries:
                    break
                time.sleep(ready_retry_wait_sec)

            shutil.copyfile(attempt_session_path, session_path)
            if sha256_file(session_path) != inspection["sha256"]:
                raise RuntimeError(
                    f"selected Claude session checksum mismatch: {attempt_session_path}"
                )
            if not inspection["raw_session_valid"] and return_code == 0:
                return_code = 97
                claude_attempts[-1]["return_code"] = return_code
                claude_attempts[-1]["failure"] = "raw_session_invalid"
            if not mcp_ready and return_code == 0:
                return_code = 98
                claude_attempts[-1]["return_code"] = return_code
                claude_attempts[-1]["failure"] = f"mcp_not_ready:{mcp_reason}"

            latency = time.time() - t0

        assistant_text, result_text = extract_stream_result(raw_stream)
        return ClaudeCodeRunResult(
            ok=(return_code == 0),
            return_code=return_code,
            timed_out=timed_out,
            latency_sec=latency,
            command=f"{' '.join(shlex.quote(x) for x in cmd)} <stdin:prompt>",
            provider=self.provider,
            provider_switch_ok=True,
            provider_switch_message=provider_msg,
            prompt_sha256=prompt_hash,
            mcp_config_sha256=mcp_cfg_hash,
            mcp_server_name=self.mcp_server_name,
            mcp_server_url=self.mcp_server_url,
            workdir=str(actual_workdir),
            session_file=str(session_path),
            attempt_session_files=[
                str(attempt["session_file"]) for attempt in claude_attempts
            ],
            claude_attempts=claude_attempts,
            selected_claude_attempt=int(claude_attempts[-1]["attempt_index"]),
            raw_stream=raw_stream,
            assistant_text=assistant_text,
            result_text=result_text,
        )
