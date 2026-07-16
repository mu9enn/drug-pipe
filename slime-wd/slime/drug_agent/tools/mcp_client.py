from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from drug_agent.offline_guard import assert_tool_environment_allowed
from drug_agent.utils import to_jsonable


class MCPClient:
    def __init__(
        self,
        server_url: str | None = None,
        api_key: str | None = None,
        initialize_timeout: float = 30.0,
    ) -> None:
        self.server_url = server_url or os.environ.get("MOLCLAW_SCP_SERVER_URL")
        self.api_key = api_key or os.environ.get("MOLCLAW_SCP_API_KEY")
        self.initialize_timeout = float(initialize_timeout)

        self._transport_ctx = None
        self._session_ctx = None
        self._session = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected and self._session is not None

    async def connect(self) -> bool:
        assert_tool_environment_allowed("MCP client connection")
        if self.connected:
            return True
        if not self.server_url:
            raise RuntimeError("MOLCLAW_SCP_SERVER_URL is missing")
        if not self.api_key:
            raise RuntimeError("MOLCLAW_SCP_API_KEY is missing")

        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except Exception as exc:
            raise RuntimeError("Cannot import MCP SDK. Install with: pip install -U mcp") from exc

        headers = {"SCP-HUB-API-KEY": self.api_key}
        self._transport_ctx = streamablehttp_client(url=self.server_url, headers=headers)

        try:
            read_stream, write_stream, _ = await self._transport_ctx.__aenter__()
            self._session_ctx = ClientSession(read_stream, write_stream)
            self._session = await self._session_ctx.__aenter__()
            await asyncio.wait_for(self._session.initialize(), timeout=self.initialize_timeout)
        except Exception as exc:
            await self.disconnect()
            raise RuntimeError(f"Failed to connect/initialize MCP session: {exc}") from exc

        self._connected = True
        return True

    async def disconnect(self) -> None:
        try:
            if self._session_ctx is not None:
                await self._session_ctx.__aexit__(None, None, None)
        finally:
            self._session_ctx = None
            self._session = None
            self._connected = False

        if self._transport_ctx is not None:
            await self._transport_ctx.__aexit__(None, None, None)
            self._transport_ctx = None

    async def list_tools(self) -> Any:
        await self._ensure_connected()
        return await self._session.list_tools()

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_connected()
        result = await self._session.call_tool(tool_name, arguments)
        return {"parsed": self.parse_result(result), "raw": to_jsonable(result)}

    def parse_result(self, result: Any) -> dict[str, Any]:
        try:
            content = getattr(result, "content", None)
            if isinstance(content, list) and content:
                first = content[0]
                text = getattr(first, "text", None)
                if isinstance(text, str):
                    try:
                        return json.loads(text)
                    except Exception:
                        return {"text": text}
                if isinstance(first, dict):
                    text_value = first.get("text")
                    if isinstance(text_value, str):
                        try:
                            return json.loads(text_value)
                        except Exception:
                            return {"text": text_value}
                    return to_jsonable(first)
                return to_jsonable(first)
            if isinstance(result, dict):
                return to_jsonable(result)
            return {"raw": str(result)}
        except Exception as exc:
            return {"error": f"parse error: {exc}", "raw": str(result)}

    async def _ensure_connected(self) -> None:
        if not self.connected:
            await self.connect()
