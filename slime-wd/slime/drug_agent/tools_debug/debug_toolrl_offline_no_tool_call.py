from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.toolrl.molclaw_reward import reward_func
from drug_agent.toolrl.parse_tool_calls import parse_tool_calls


MCP_ENV_KEYS = (
    "MOLCLAW_SCP_SERVER_URL",
    "MOLCLAW_SCP_API_KEY",
    "MOLCLAW_CONNECT_TIMEOUT_SEC",
    "MOLCLAW_LIST_TOOLS_TIMEOUT_SEC",
    "MOLCLAW_TOOL_TIMEOUT_SEC",
    "MOLCLAW_TOOL_HEARTBEAT_SEC",
)


def _drug_agent_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assert_toolrl_script_is_offline(script_path: Path) -> dict[str, Any]:
    text = _read_text(script_path)
    forbidden = [
        "drug_agent.rollout.generate_with_drug_agent.generate",
        "--custom-generate-function-path",
        "MOLCLAW_SCP_SERVER_URL",
        "MOLCLAW_SCP_API_KEY",
        "MCPToolExecutor",
        "mcp_client",
        "tool_executor.execute",
        "call_tool",
    ]
    hits = [item for item in forbidden if item in text]
    uses_native_rollout = "slime.rollout.sglang_rollout.generate_rollout" in text
    return {
        "ok": not hits and uses_native_rollout,
        "script_path": str(script_path),
        "forbidden_hits": hits,
        "uses_native_sglang_rollout": uses_native_rollout,
    }


def _sample_from_response(response: str) -> SimpleNamespace:
    return SimpleNamespace(
        prompt=[{"role": "user", "content": "Choose the next MolClaw tool call."}],
        response=response,
        label={
            "target_tool_calls": [
                {
                    "tool_name": "fix_pdb",
                    "arguments": {
                        "input_path": "/tmp/protein.pdb",
                        "remove_water": True,
                    },
                }
            ]
        },
        metadata={
            "schema_version": "toolrl_step_v1",
            "allowed_tool_names": ["fix_pdb"],
            "target_tool_calls": [
                {
                    "tool_name": "fix_pdb",
                    "arguments": {
                        "input_path": "/tmp/protein.pdb",
                        "remove_water": True,
                    },
                }
            ],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify offline ToolRL reward does not call MCP/MolClaw tools")
    parser.add_argument(
        "--response",
        default='<thought>pick the cleanup tool</thought><tool_call>{"tool_name":"fix_pdb","arguments":{"input_path":"/tmp/protein.pdb","remove_water":true}}</tool_call>',
        help="Assistant response to parse and score",
    )
    args = parser.parse_args()

    old_env = {key: os.environ.get(key) for key in MCP_ENV_KEYS}
    for key in MCP_ENV_KEYS:
        os.environ.pop(key, None)

    try:
        parsed = parse_tool_calls(args.response, keep_non_molclaw=True)
        sample = _sample_from_response(args.response)
        reward = asyncio.run(reward_func(None, sample))
        script_check = _assert_toolrl_script_is_offline(
            _drug_agent_root() / "toolrl" / "scripts" / "run_toolrl_grpo.sh"
        )
        report = {
            "ok": bool(parsed.get("ok")) and reward.get("score", 0.0) > 0 and script_check["ok"],
            "mcp_env_present_after_unset": {key: os.environ.get(key) is not None for key in MCP_ENV_KEYS},
            "parsed": {
                "ok": parsed.get("ok"),
                "molclaw_tool_call_count": parsed.get("molclaw_tool_call_count"),
                "non_molclaw_tool_call_count": parsed.get("non_molclaw_tool_call_count"),
            },
            "reward": reward,
            "script_check": script_check,
            "evidence": {
                "reward_module": "drug_agent.toolrl.molclaw_reward",
                "offline_rollout_function": "slime.rollout.sglang_rollout.generate_rollout",
                "calls_mcp": False,
            },
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
