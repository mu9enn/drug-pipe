from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


LOCAL_RAW_NAMES = {
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "bash": "Bash",
    "grep": "Grep",
    "glob": "Glob",
}


def public_mcp_name(server_name: str, raw_name: str) -> str:
    joined = f"mcp__{server_name}__{raw_name}"
    normalized = "".join(character if character.isalnum() or character in "_-" else "_" for character in joined)
    if normalized == joined and len(normalized) <= 64:
        return normalized
    digest = hashlib.sha256(f"{server_name}\0{raw_name}".encode()).hexdigest()[:12]
    return f"{normalized[:51]}_{digest}"


def request_tools(transcript: Path) -> list[dict[str, Any]]:
    events = json.loads(transcript.read_text(encoding="utf-8"))
    for event in events:
        if event.get("type") == "request/header":
            tools = ((event.get("data") or {}).get("header") or {}).get("tools")
            if isinstance(tools, list):
                return tools
    raise ValueError(f"request/header tools not found in {transcript}")


def build(
    transcript: Path,
    raw_catalog: Path,
    server_name: str,
) -> dict[str, Any]:
    raw_payload = json.loads(raw_catalog.read_text(encoding="utf-8"))
    raw_tools = raw_payload.get("tools") if isinstance(raw_payload, dict) else raw_payload
    public_catalog = any(str(tool.get('name', '')).startswith(f'mcp__{server_name}__') for tool in raw_tools)
    raw_by_public = {}
    for tool in raw_tools:
        name = str(tool.get('name') or '')
        if public_catalog and not name.startswith(f'mcp__{server_name}__'):
            continue
        raw = str(tool.get('raw_name') or name)
        if raw in set(LOCAL_RAW_NAMES) | set(LOCAL_RAW_NAMES.values()) | {'skill'}:
            continue
        raw_by_public[public_mcp_name(server_name, raw)] = raw
    selected = []
    for tool in request_tools(transcript):
        public = str(tool.get("name") or "")
        raw = raw_by_public.get(public) if public.startswith(f"mcp__{server_name}__") else None
        if public.startswith(f"mcp__{server_name}__") and raw is None:
            raise ValueError(f"cannot resolve MolClaw public tool name: {public}")
        raw = raw or LOCAL_RAW_NAMES.get(public) or public
        selected.append({
            "name": public,
            "raw_name": raw,
            "description": str(tool.get("description") or ""),
            "input_schema": tool.get("parameters") or {},
        })
    if len(raw_by_public) != 81 or not raw_by_public.keys() <= {row["name"] for row in selected}:
        raise ValueError("request header does not contain the complete 81-tool MolClaw catalog")
    names = [row['name'] for row in selected]
    local = {name for name in names if not name.startswith(f'mcp__{server_name}__')}
    if len(names) != 88 or len(set(names)) != 88 or local != set(LOCAL_RAW_NAMES) | {'skill'}:
        raise ValueError('export requires the dedicated 81 + 7 tool preset without filtering')
    return {
        "schema_version": "drug_agent_dsh_tool_set_v1",
        "server_name": server_name,
        "source_transcript": str(transcript.resolve()),
        "source_transcript_sha256": hashlib.sha256(transcript.read_bytes()).hexdigest(),
        "tools": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy the exact DSH deployment tool set from a request header.")
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--raw-catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--server-name", default="molclaw-scp")
    args = parser.parse_args()
    payload = build(args.transcript, args.raw_catalog, args.server_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "tool_count": len(payload["tools"])}, indent=2))


if __name__ == "__main__":
    main()
