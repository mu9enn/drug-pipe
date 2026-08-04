from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from drug_agent.tools.local_tools import LOCAL_TOOL_SPECS
from drug_agent.utils import write_json


def normalize_catalog(source: Path, output: Path) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("tools") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("tool catalog must contain a tools list")
    mcp_tools = [
        dict(row)
        for row in rows
        if isinstance(row, dict) and row.get("executor") != "local_sandbox"
    ]
    local_tools = [dict(spec, executor="local_sandbox") for spec in LOCAL_TOOL_SPECS]
    normalized = {
        "schema_version": "drug_agent_runtime_tool_catalog_v2",
        "source_catalog": str(source),
        "mcp_tool_count": len(mcp_tools),
        "local_tool_count": len(local_tools),
        "tool_count": len(mcp_tools) + len(local_tools),
        "tools": [*mcp_tools, *local_tools],
    }
    write_json(output, normalized)
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace captured local-tool rows with the current sandbox contract"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = normalize_catalog(Path(args.input), Path(args.output))
    print(json.dumps({
        key: result[key]
        for key in ("schema_version", "mcp_tool_count", "local_tool_count", "tool_count")
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
