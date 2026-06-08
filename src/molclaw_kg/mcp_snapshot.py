from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

from .io_utils import write_json, write_jsonl
from .settings import ProjectConfig


async def _list_tools(server_url: str, api_key: str):
    transport = streamablehttp_client(url=server_url, headers={"SCP-HUB-API-KEY": api_key})
    read, write, _ = await transport.__aenter__()
    session_ctx = ClientSession(read, write)
    session = await session_ctx.__aenter__()
    try:
        await session.initialize()
        res = await session.list_tools()
        return res.tools
    finally:
        await session_ctx.__aexit__(None, None, None)
        await transport.__aexit__(None, None, None)


def _normalize_tool(tool_obj: Any) -> dict[str, Any]:
    name = getattr(tool_obj, "name", "")
    title = getattr(tool_obj, "title", "") or name
    desc = getattr(tool_obj, "description", "") or ""
    input_schema = getattr(tool_obj, "inputSchema", None) or getattr(tool_obj, "input_schema", None) or {}
    output_schema = getattr(tool_obj, "outputSchema", None) or getattr(tool_obj, "output_schema", None)
    annotations = getattr(tool_obj, "annotations", None)

    return {
        "tool_id": name,
        "title": title,
        "name": name,
        "description": desc,
        "inputSchema": input_schema,
        "outputSchema": output_schema,
        "annotations": annotations,
    }


def run_snapshot(config: ProjectConfig) -> dict[str, Any]:
    if not config.runtime.api_key:
        raise RuntimeError("MOLCLAW_SCP_API_KEY is required (or pass --api-key)")
    if not config.runtime.server_url:
        raise RuntimeError("MOLCLAW_SCP_MCP_URL is required (or pass --server-url)")

    tools = asyncio.run(_list_tools(config.runtime.server_url, config.runtime.api_key))
    rows = [_normalize_tool(t) for t in tools if getattr(t, "name", None)]
    rows.sort(key=lambda x: x["tool_id"])

    out_jsonl = config.paths.run_dir / "tool_snapshot.jsonl"
    out_meta = config.paths.run_dir / "tool_snapshot_meta.json"

    write_jsonl(out_jsonl, rows)
    write_json(
        out_meta,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "tool_count": len(rows),
            "server_url": config.runtime.server_url,
            "path": str(out_jsonl),
        },
    )

    return {
        "tool_count": len(rows),
        "snapshot_path": str(out_jsonl),
        "meta_path": str(out_meta),
    }
