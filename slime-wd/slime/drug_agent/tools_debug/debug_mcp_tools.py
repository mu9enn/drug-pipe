from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.tools.tool_executor import MCPToolExecutor
from drug_agent.tools.tool_registry import ToolRegistry, load_allowlist
from drug_agent.offline_guard import assert_tool_environment_allowed

REQUIRED_ENV_KEYS = (
    "MOLCLAW_SCP_SERVER_URL",
    "MOLCLAW_SCP_API_KEY",
)


def _error_payload(category: str, message: str) -> dict[str, Any]:
    return {
        "category": category,
        "message": message,
    }


def _base_output(started_at: float) -> dict[str, Any]:
    return {
        "ok": False,
        "error": None,
        "tool_count": 0,
        "latency_sec": round(time.monotonic() - started_at, 6),
        "result": {},
    }


def _missing_env() -> list[str]:
    missing = []
    for key in REQUIRED_ENV_KEYS:
        if not (os.environ.get(key) or "").strip():
            missing.append(key)
    return missing


def _safe_load_args(raw_args: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON in --args: {exc}"
    if not isinstance(payload, dict):
        return None, "--args must be a JSON object"
    return payload, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug MCP tools for drug_agent")
    parser.add_argument("--list-tools", action="store_true", help="List tools from MCP server")
    parser.add_argument("--tool", type=str, default=None, help="Tool name to call")
    parser.add_argument("--args", type=str, default="{}", help="Tool args as JSON object string")
    parser.add_argument("--allowlist", type=str, default=str(Path(__file__).resolve().parents[1] / "tools/allowlist_v0.json"))
    parser.add_argument("--allow-all", action="store_true")
    args = parser.parse_args()
    assert_tool_environment_allowed("debug_mcp_tools online tool execution")
    print("[ONLINE TOOL DEBUG] Real MolClaw/MCP calls are enabled.", flush=True)

    started = time.monotonic()
    output = _base_output(started)
    output["result"]["request"] = {
        "list_tools": bool(args.list_tools),
        "tool": args.tool,
        "allow_all": bool(args.allow_all),
        "allowlist": str(args.allowlist),
    }

    if (not args.list_tools) and (not args.tool):
        parser.print_help()
        return 2

    missing = _missing_env()
    if missing:
        output["error"] = _error_payload("missing_env", f"{missing[0]} is missing")
        output["result"]["missing_env"] = missing
        output["latency_sec"] = round(time.monotonic() - started, 6)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 1

    allowlist = load_allowlist(args.allowlist)
    executor = MCPToolExecutor(connect_on_init=False)
    registry = ToolRegistry(executor=executor, allowlist=allowlist, allow_all=args.allow_all)

    try:
        specs: list[dict[str, Any]] = []
        if args.list_tools or args.tool:
            specs = registry.list_tools(force_refresh=True)
        output["tool_count"] = len(specs)

        if args.list_tools:
            output["result"]["list_tools"] = {
                "tools": [
                    {
                        "name": s.get("name"),
                        "description": s.get("description", ""),
                    }
                    for s in specs
                ]
            }

        if args.tool:
            payload, arg_error = _safe_load_args(args.args)
            if arg_error:
                output["error"] = _error_payload("invalid_arguments", arg_error)
                output["result"]["tool_call"] = {
                    "tool_name": args.tool,
                    "arguments": args.args,
                }
                output["latency_sec"] = round(time.monotonic() - started, 6)
                print(json.dumps(output, ensure_ascii=False, indent=2))
                return 1

            ok_name, reason_name = registry.validate_tool_name(args.tool)
            ok_args, reason_args = registry.validate_arguments_basic(args.tool, payload)
            tool_call_result = {
                "tool_name": args.tool,
                "arguments": payload,
                "validation": {
                    "validate_tool_name": {"ok": ok_name, "reason": reason_name},
                    "validate_arguments": {"ok": ok_args, "reason": reason_args},
                },
            }
            output["result"]["tool_call"] = tool_call_result

            if not ok_name or not ok_args:
                output["error"] = _error_payload(
                    "validation_failed",
                    reason_name or reason_args or "tool validation failed",
                )
                output["latency_sec"] = round(time.monotonic() - started, 6)
                print(json.dumps(output, ensure_ascii=False, indent=2))
                return 1

            exec_result = executor.execute(args.tool, payload)
            output["result"]["tool_call"]["execution"] = exec_result
            if bool(exec_result.get("ok")):
                output["ok"] = True
            else:
                err = exec_result.get("error")
                message = "tool execution failed"
                if isinstance(err, dict) and isinstance(err.get("message"), str):
                    message = err["message"]
                output["error"] = _error_payload("tool_execution_error", message)

        if args.list_tools and not args.tool:
            output["ok"] = True
        if args.list_tools and args.tool and output["error"] is None:
            output["ok"] = True

        output["latency_sec"] = round(time.monotonic() - started, 6)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        category = "runtime_error"
        if "MOLCLAW_SCP_SERVER_URL is missing" in message or "MOLCLAW_SCP_API_KEY is missing" in message:
            category = "missing_env"
        output["error"] = _error_payload(category, message)
        output["latency_sec"] = round(time.monotonic() - started, 6)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 1
    finally:
        executor.close()


if __name__ == "__main__":
    raise SystemExit(main())
