from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
import time
from typing import Any

from drug_agent.offline_guard import assert_tool_environment_allowed
from drug_agent.tools.mcp_client import MCPClient
from drug_agent.tools.tool_success import evaluate_tool_success
from drug_agent.utils import to_jsonable


class MCPToolExecutor:
    def __init__(
        self,
        connect_on_init: bool = True,
        request_timeout: float | None = None,
        connect_timeout: float | None = None,
        list_tools_timeout: float | None = None,
        execute_timeout: float | None = None,
        initialize_timeout: float = 30.0,
    ) -> None:
        assert_tool_environment_allowed("MCPToolExecutor initialization")
        self.request_timeout = self._resolve_timeout(request_timeout, "MOLCLAW_REQUEST_TIMEOUT_SEC")
        self.connect_timeout = self._resolve_timeout(connect_timeout, "MOLCLAW_CONNECT_TIMEOUT_SEC")
        self.list_tools_timeout = self._resolve_timeout(list_tools_timeout, "MOLCLAW_LIST_TOOLS_TIMEOUT_SEC")
        self.execute_timeout = self._resolve_timeout(execute_timeout, "MOLCLAW_TOOL_TIMEOUT_SEC")
        self.heartbeat_sec = self._resolve_timeout(None, "MOLCLAW_TOOL_HEARTBEAT_SEC")

        self._client = MCPClient(initialize_timeout=initialize_timeout)
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name="drug-agent-mcp-loop", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=2.0)
        self._closed = False
        self.unknown_semantic_as_failure = self._resolve_bool_env(
            "DRUG_AGENT_UNKNOWN_SEMANTIC_AS_FAILURE",
            default=True,
        )

        if connect_on_init:
            self._run(self._client.connect(), timeout=self._timeout_for("connect"), label="connect")

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def _run(self, coro: Any, timeout: float | None, label: str) -> Any:
        if self._closed:
            raise RuntimeError("MCPToolExecutor is closed")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        start = time.monotonic()

        while True:
            wait_timeout = timeout
            if wait_timeout is None and self.heartbeat_sec is not None:
                wait_timeout = self.heartbeat_sec
            if timeout is not None and self.heartbeat_sec is not None:
                elapsed = time.monotonic() - start
                remaining = timeout - elapsed
                if remaining <= 0:
                    future.cancel()
                    raise TimeoutError(f"MCP {label} timeout after {timeout}s")
                wait_timeout = min(self.heartbeat_sec, remaining)

            try:
                if wait_timeout is None:
                    return future.result()
                return future.result(timeout=wait_timeout)
            except concurrent.futures.TimeoutError as exc:
                elapsed = time.monotonic() - start
                if timeout is not None and elapsed >= timeout:
                    future.cancel()
                    raise TimeoutError(f"MCP {label} timeout after {timeout}s") from exc
                if self.heartbeat_sec is not None:
                    print(f"[MCPToolExecutor] waiting {label} elapsed={elapsed:.1f}s", flush=True)
                continue

    def _ensure_connected(self) -> None:
        self._run(self._client.connect(), timeout=self._timeout_for("connect"), label="connect")

    def list_tools(self) -> list[dict[str, Any]]:
        self._ensure_connected()
        raw = self._run(self._client.list_tools(), timeout=self._timeout_for("list_tools"), label="list_tools")
        return self._normalize_tool_list(raw)

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        if not isinstance(arguments, dict):
            tool_success = evaluate_tool_success(
                transport_ok=False,
                tool_schema_valid=False,
                parsed_payload=None,
                raw_payload=None,
                unknown_as_failure=self.unknown_semantic_as_failure,
            )
            return {
                "ok": False,
                "tool_name": tool_name,
                "result": None,
                "error": {"type": "ToolExecutionError", "message": "`arguments` must be an object"},
                "latency_sec": 0.0,
                "transport_ok": tool_success["transport_ok"],
                "tool_schema_valid": tool_success["tool_schema_valid"],
                "tool_execution_success": tool_success["tool_execution_success"],
                "tool_semantic_success": tool_success["tool_semantic_success"],
                "semantic_unknown": tool_success["semantic_unknown"],
                "metadata": {"tool_success": tool_success},
            }

        self._ensure_connected()
        try:
            raw = self._run(
                self._client.call_tool(tool_name, arguments),
                timeout=self._timeout_for("execute"),
                label=f"call_tool:{tool_name}",
            )
            parsed_payload = to_jsonable(raw.get("parsed"))
            raw_payload = to_jsonable(raw.get("raw"))
            tool_success = evaluate_tool_success(
                transport_ok=True,
                tool_schema_valid=True,
                parsed_payload=parsed_payload,
                raw_payload=raw_payload,
                unknown_as_failure=self.unknown_semantic_as_failure,
            )
            ok = bool(tool_success["tool_semantic_success"])
            error = None
            if not ok:
                error = {
                    "type": tool_success.get("semantic_error_type") or "ToolExecutionError",
                    "message": tool_success.get("semantic_error_message") or "tool execution failed",
                }
            return {
                "ok": ok,
                "tool_name": tool_name,
                "result": parsed_payload,
                "error": error,
                "latency_sec": round(time.monotonic() - started, 6),
                "transport_ok": tool_success["transport_ok"],
                "tool_schema_valid": tool_success["tool_schema_valid"],
                "tool_execution_success": tool_success["tool_execution_success"],
                "tool_semantic_success": tool_success["tool_semantic_success"],
                "semantic_unknown": tool_success["semantic_unknown"],
                "metadata": {
                    "raw": raw_payload,
                    "tool_success": tool_success,
                },
            }
        except Exception as exc:
            tool_success = evaluate_tool_success(
                transport_ok=False,
                tool_schema_valid=True,
                parsed_payload=None,
                raw_payload=None,
                unknown_as_failure=self.unknown_semantic_as_failure,
            )
            return {
                "ok": False,
                "tool_name": tool_name,
                "result": None,
                "error": {"type": "ToolExecutionError", "message": str(exc)},
                "latency_sec": round(time.monotonic() - started, 6),
                "transport_ok": tool_success["transport_ok"],
                "tool_schema_valid": tool_success["tool_schema_valid"],
                "tool_execution_success": tool_success["tool_execution_success"],
                "tool_semantic_success": tool_success["tool_semantic_success"],
                "semantic_unknown": tool_success["semantic_unknown"],
                "metadata": {"tool_success": tool_success},
            }

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._run(self._client.disconnect(), timeout=5.0, label="disconnect")
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
        self._closed = True

    @staticmethod
    def _resolve_timeout(value: float | None, env_key: str) -> float | None:
        if value is not None:
            try:
                v = float(value)
                return v if v > 0 else None
            except Exception:
                return None
        raw = os.environ.get(env_key)
        if raw is None:
            return None
        try:
            v = float(raw)
            return v if v > 0 else None
        except Exception:
            return None

    def _timeout_for(self, stage: str) -> float | None:
        if stage == "connect" and self.connect_timeout is not None:
            return self.connect_timeout
        if stage == "list_tools" and self.list_tools_timeout is not None:
            return self.list_tools_timeout
        if stage == "execute" and self.execute_timeout is not None:
            return self.execute_timeout
        return self.request_timeout

    @staticmethod
    def _resolve_bool_env(env_key: str, default: bool) -> bool:
        raw = os.environ.get(env_key)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _normalize_tool_list(raw_tools: Any) -> list[dict[str, Any]]:
        tool_items: list[Any]
        if isinstance(raw_tools, list):
            tool_items = raw_tools
        elif hasattr(raw_tools, "tools"):
            tool_items = list(getattr(raw_tools, "tools"))
        elif isinstance(raw_tools, dict) and isinstance(raw_tools.get("tools"), list):
            tool_items = raw_tools.get("tools")
        else:
            tool_items = []

        normalized = []
        for item in tool_items:
            if isinstance(item, dict):
                name = item.get("name")
                description = item.get("description") or ""
                input_schema = item.get("inputSchema") or item.get("input_schema") or {}
            else:
                name = getattr(item, "name", None)
                description = getattr(item, "description", "")
                input_schema = getattr(item, "inputSchema", None) or getattr(item, "input_schema", None) or {}

            normalized.append(
                {
                    "name": name,
                    "description": description,
                    "input_schema": to_jsonable(input_schema),
                }
            )
        return normalized
