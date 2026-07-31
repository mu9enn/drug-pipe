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

from drug_agent.constants import CANONICAL_REACT_DATA, DRUG_AGENT_L1_SKILLS_ROOT, DRUG_AGENT_RUNS_ROOT
from drug_agent.offline_guard import assert_tool_environment_allowed
from drug_agent.protocol.react_protocol import parse_runtime_decision
from drug_agent.tools.artifact_registry import ArtifactRegistry
from drug_agent.tools.local_tools import LOCAL_TOOL_NAMES, LocalToolExecutor
from drug_agent.tools.runtime_env import load_molclaw_environment
from drug_agent.tools.tool_executor import MCPToolExecutor
from drug_agent.tools.tool_registry import ToolRegistry
from drug_agent.utils import append_jsonl, ensure_dir, normalize_tool_name, to_jsonable


REQUIRED_ENV_KEYS = (
    "MOLCLAW_SCP_SERVER_URL",
    "MOLCLAW_SCP_API_KEY",
)


def _error_payload(category: str, message: str) -> dict[str, Any]:
    return {"category": category, "message": message}


def _default_input_jsonl() -> Path:
    return CANONICAL_REACT_DATA


def _default_runs_root() -> Path:
    return DRUG_AGENT_RUNS_ROOT


def _load_row(input_jsonl: Path, index: int) -> dict[str, Any]:
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no != index:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"row at index={index} is not a JSON object")
            return obj
    raise IndexError(f"index {index} out of range for {input_jsonl}")


def _extract_tool_call_candidates(messages: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if not isinstance(messages, list):
        return candidates

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        raw = msg.get("content")
        if not isinstance(raw, str) or not raw.strip():
            continue

        parsed = parse_runtime_decision(raw)
        if parsed.get("ok") and parsed.get("decision_type") == "tool_call":
            for call_index, call in enumerate(parsed.get("tool_calls") or []):
                candidates.append(
                    {
                        "assistant_turn_index": i,
                        "call_index": call_index,
                        "raw_action": raw,
                        "parse_ok": True,
                        "parsed": {
                            "ok": True,
                            "decision_type": "tool_call",
                            "tool_name": call.get("tool_name"),
                            "arguments": call.get("arguments"),
                        },
                    }
                )
            continue

        # Preserve parse failures that look like intended tool calls.
        if "<tool_call>" in raw:
            candidates.append(
                {
                    "assistant_turn_index": i,
                    "raw_action": raw,
                    "parse_ok": False,
                    "parsed": parsed,
                }
            )
    return candidates


def _missing_env() -> list[str]:
    missing: list[str] = []
    for key in REQUIRED_ENV_KEYS:
        if not (os.environ.get(key) or "").strip():
            missing.append(key)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay assistant tool_call actions from SFT messages through MCP tools")
    parser.add_argument("--input-jsonl", type=str, default=str(_default_input_jsonl()))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--run-name", type=str, default=f"gate_replay_{int(time.time())}")
    parser.add_argument("--env-file", action="append", default=[])
    args = parser.parse_args()
    load_molclaw_environment(args.env_file)
    assert_tool_environment_allowed("debug_replay_trajectory online tool execution")
    print("[ONLINE TOOL DEBUG] Replaying actions against real MolClaw/MCP tools.", flush=True)

    started = time.monotonic()
    output: dict[str, Any] = {
        "ok": False,
        "error": None,
        "num_tool_calls": 0,
        "parse_success_rate": 0.0,
        "tool_success_rate": 0.0,
        "latency_sec": 0.0,
        "trace_path": None,
        "result": {
            "input_jsonl": args.input_jsonl,
            "index": args.index,
            "max_tool_calls": args.max_tool_calls,
            "run_name": args.run_name,
        },
    }

    executor: MCPToolExecutor | None = None
    try:
        input_jsonl = Path(args.input_jsonl)
        if not input_jsonl.exists():
            output["error"] = _error_payload("input_not_found", f"input jsonl not found: {input_jsonl}")
            output["latency_sec"] = round(time.monotonic() - started, 6)
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 1

        row = _load_row(input_jsonl, args.index)
        messages = row.get("messages")
        if not isinstance(messages, list):
            output["error"] = _error_payload("bad_sample", "messages is missing or not a list")
            output["latency_sec"] = round(time.monotonic() - started, 6)
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 1

        candidates = _extract_tool_call_candidates(messages)
        candidates = candidates[: max(0, args.max_tool_calls)]
        output["num_tool_calls"] = len(candidates)

        trace_root = ensure_dir(_default_runs_root() / args.run_name)
        trace_path = trace_root / "replay_trace.jsonl"
        output["trace_path"] = str(trace_path)
        local_executor = LocalToolExecutor(trace_root / "workspace", DRUG_AGENT_L1_SKILLS_ROOT)
        artifact_registry = ArtifactRegistry(trace_root / "workspace")

        parse_ok_count = 0
        tool_ok_count = 0
        tool_exec_count = 0
        trace_rows: list[dict[str, Any]] = []

        for replay_index, item in enumerate(candidates):
            parsed = item.get("parsed") if isinstance(item.get("parsed"), dict) else {}
            parse_ok = bool(item.get("parse_ok"))
            if parse_ok:
                parse_ok_count += 1

            step_row = {
                "replay_index": replay_index,
                "assistant_turn_index": item.get("assistant_turn_index"),
                "raw_action": item.get("raw_action"),
                "parsed_action": parsed,
                "parse_ok": parse_ok,
                "tool_result": None,
                "observation": None,
            }

            if parse_ok:
                tool_name = normalize_tool_name(parsed.get("tool_name"))
                tool_args = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {}

                missing = _missing_env()
                if missing:
                    step_row["tool_result"] = {
                        "ok": False,
                        "tool_name": tool_name,
                        "result": None,
                        "error": _error_payload("missing_env", f"{missing[0]} is missing"),
                        "latency_sec": 0.0,
                        "metadata": {"missing_env": missing},
                    }
                else:
                    if executor is None:
                        executor = MCPToolExecutor(connect_on_init=False)
                        registry = ToolRegistry(executor=executor, include_local_tools=True)
                        if not registry.list_tools(force_refresh=True):
                            raise RuntimeError("molclaw-scp list_tools returned an empty catalog")
                    ok_name, reason_name = registry.validate_tool_name(tool_name)
                    ok_args, reason_args = registry.validate_arguments_basic(tool_name, tool_args)
                    if not ok_name or not ok_args:
                        step_row["tool_result"] = {
                            "ok": False,
                            "tool_name": tool_name,
                            "result": None,
                            "error": _error_payload("validation_failed", reason_name or reason_args or "validation failed"),
                            "latency_sec": 0.0,
                            "metadata": {"tool_reason": reason_name, "args_reason": reason_args},
                        }
                    else:
                        raw_result = registry.execute(
                            tool_name,
                            artifact_registry.resolve(tool_args),
                            local_executor=local_executor,
                        )
                        metadata = to_jsonable(raw_result.get("metadata")) or {}
                        if isinstance(metadata, dict):
                            metadata.pop("raw", None)
                        step_row["tool_result"] = {
                            **{key: to_jsonable(value) for key, value in raw_result.items() if key not in {"result", "error", "metadata"}},
                            "result": artifact_registry.canonicalize(
                                to_jsonable(raw_result.get("result")),
                                local_result=tool_name in LOCAL_TOOL_NAMES,
                            ),
                            "error": artifact_registry.canonicalize(to_jsonable(raw_result.get("error"))),
                            "metadata": metadata,
                        }
                        tool_exec_count += 1
                        if bool(step_row["tool_result"].get("ok")):
                            tool_ok_count += 1

                step_row["observation"] = json.dumps(
                    {"observation": to_jsonable(step_row["tool_result"])},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

            trace_rows.append(to_jsonable(step_row))

        for row_item in trace_rows:
            append_jsonl(trace_path, row_item)
        append_jsonl(trace_path, {"artifact_audit": artifact_registry.audit_snapshot()})

        if output["num_tool_calls"] > 0:
            output["parse_success_rate"] = round(parse_ok_count / output["num_tool_calls"], 6)
        if tool_exec_count > 0:
            output["tool_success_rate"] = round(tool_ok_count / tool_exec_count, 6)

        output["ok"] = True
        output["result"]["task_id"] = (row.get("metadata") or {}).get("task_id")
        output["result"]["task_type"] = (row.get("metadata") or {}).get("task_type")
        output["result"]["trace_rows"] = len(trace_rows)

        # Keep structured warning if env is missing.
        missing = _missing_env()
        if missing:
            output["ok"] = False
            output["error"] = _error_payload("missing_env", f"{missing[0]} is missing")
            output["result"]["missing_env"] = missing

        output["latency_sec"] = round(time.monotonic() - started, 6)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0 if output["ok"] else 1
    except Exception as exc:
        output["error"] = _error_payload("runtime_error", f"{type(exc).__name__}: {exc}")
        output["latency_sec"] = round(time.monotonic() - started, 6)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 1
    finally:
        if executor is not None:
            executor.close()


if __name__ == "__main__":
    raise SystemExit(main())
