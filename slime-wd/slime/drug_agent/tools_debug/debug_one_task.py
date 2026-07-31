from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.constants import (
    CANONICAL_REACT_DATA,
    DATA_ROOT,
    DRUG_AGENT_L1_SKILLS_ROOT,
    DRUG_AGENT_RUNS_ROOT,
)
from drug_agent.offline_guard import assert_tool_environment_allowed
from drug_agent.protocol.react_protocol import final_answer_matches_task, parse_runtime_decision, project_final_answer
from drug_agent.protocol.prompts import format_final_contract, format_tool_catalog, fresh_task_messages
from drug_agent.tools.artifact_registry import ArtifactRegistry
from drug_agent.tools.tool_executor import MCPToolExecutor
from drug_agent.tools.local_tools import LOCAL_TOOL_NAMES, LocalToolExecutor
from drug_agent.tools.tool_registry import ToolRegistry
from drug_agent.tools.runtime_env import load_molclaw_environment, missing_molclaw_environment
from drug_agent.tools.tool_success import make_validation_failed_result
from drug_agent.tools_debug.sglang_launcher import detect_sglang_launch_command
from drug_agent.utils import append_jsonl, ensure_dir, normalize_tool_name, to_jsonable

REQUIRED_MCP_ENV = (
    "MOLCLAW_SCP_SERVER_URL",
    "MOLCLAW_SCP_API_KEY",
)

DEBUG_FORMAT_REMINDER = (
    "/no_think\n"
    "Use canonical ReAct XML. Put reasoning in <thought>...</thought>, followed by "
    "one or more <tool_call>{\"tool_name\":\"...\",\"arguments\":{...}}</tool_call> blocks, "
    "or one task-specific <final_answer>{...}</final_answer>. "
    "Do not mix tool calls and final answer in one generation."
)
DEBUG_LOCAL_TOOL_REMINDER = (
    "\nAvailable local tools: Read, Write, Edit, Bash, Grep, Glob, and Skill. "
    "They are confined to this debug task workspace; Skill exposes read-only L1 tools only."
)


def _error_payload(category: str, message: str) -> dict[str, Any]:
    return {"category": category, "message": message}


def _default_input_jsonl() -> Path:
    return CANONICAL_REACT_DATA


def _default_runs_root() -> Path:
    return DRUG_AGENT_RUNS_ROOT


def _resolve_default_debug_model_path() -> Path:
    data_root = Path(os.environ.get("DATA", "/root/slime_sxy/group-space/sunxiangyu/slime-wd/data"))
    candidate = Path(os.environ.get("DEBUG_MODEL_PATH", str(data_root / "Qwen3.5-122B-A10B")))
    if candidate.is_dir():
        return candidate
    return DATA_ROOT / "Qwen3.5-27B"


def _resolve_default_tokenizer_path() -> Path:
    if (DATA_ROOT / "Qwen3.5-27B").is_dir():
        return DATA_ROOT / "Qwen3.5-27B"
    return DATA_ROOT / "Qwen3.5-0.8B"


def _read_row(input_jsonl: Path, index: int) -> dict[str, Any]:
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no != index:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"row at index={index} is not a JSON object")
            return obj
    raise IndexError(f"index {index} out of range for {input_jsonl}")


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout_sec: float = 30.0,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, method=method.upper(), headers=headers, data=data)
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        text = resp.read().decode("utf-8")
    if not text.strip():
        return {}
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        return parsed
    return {"raw": parsed}


def _check_sglang_health(base_url: str, timeout_sec: float = 5.0) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url=f"{base_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8")
        return True, body
    except Exception as exc:
        return False, str(exc)


def _wait_for_health(base_url: str, timeout_sec: float = 300.0) -> tuple[bool, str]:
    start = time.monotonic()
    last_error = ""
    while (time.monotonic() - start) < timeout_sec:
        ok, text = _check_sglang_health(base_url, timeout_sec=5.0)
        if ok:
            return True, text
        last_error = text
        time.sleep(2.0)
    return False, last_error or "health check timeout"


def _extract_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        out: list[dict[str, str]] = []
        for item in prompt:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"system", "user", "assistant"} and isinstance(content, str):
                out.append({"role": role, "content": content})
        if out:
            return _fresh_messages(out)

    if isinstance(prompt, str) and prompt.strip():
        return [{"role": "user", "content": prompt}]

    messages = row.get("messages")
    if isinstance(messages, list):
        out = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"system", "user", "assistant"} and isinstance(content, str):
                out.append({"role": role, "content": content})
        if out:
            return _fresh_messages(out)
    return []


def _fresh_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only the fresh task boundary; never replay teacher decisions."""
    return fresh_task_messages(messages)


def _augment_messages_for_strict_json(
    messages: list[dict[str, str]],
    *,
    local_tools_enabled: bool,
    tool_catalog: str = "",
    final_contract: str = "",
) -> list[dict[str, str]]:
    out = [dict(m) for m in messages]
    if not out:
        return out

    # Reinforce strict JSON and disable reasoning mode for Qwen-family chat templates.
    reminder = DEBUG_FORMAT_REMINDER
    if local_tools_enabled:
        reminder += DEBUG_LOCAL_TOOL_REMINDER
    if tool_catalog:
        reminder += "\n" + tool_catalog
    if final_contract:
        reminder += "\n" + final_contract
    system_index = next((idx for idx, item in enumerate(out) if item.get("role") == "system"), None)
    target_index = system_index if system_index is not None else 0
    out[target_index]["content"] = (out[target_index].get("content") or "") + "\n\n" + reminder
    return out


def _sample_context(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    env_kwargs = metadata.get("env_kwargs")
    if not isinstance(env_kwargs, dict):
        env_kwargs = {}

    allowed_tools_raw = env_kwargs.get("allowed_tools")
    explicit_tool_policy = isinstance(allowed_tools_raw, list) and bool(allowed_tools_raw)
    if not isinstance(allowed_tools_raw, list):
        allowed_tools_raw = []
    allowed_tools = [normalize_tool_name(x) for x in allowed_tools_raw if isinstance(x, str) and x.strip()]
    local_tools_enabled = os.environ.get("DRUG_AGENT_ENABLE_LOCAL_TOOLS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if local_tools_enabled:
        for tool_name in LOCAL_TOOL_NAMES:
            if tool_name not in allowed_tools:
                allowed_tools.append(tool_name)

    return {
        "task_id": env_kwargs.get("task_id") or metadata.get("task_id"),
        "task_type": env_kwargs.get("task_type") or metadata.get("task_type"),
        "data_source": env_kwargs.get("data_source") or metadata.get("data_source"),
        "allowed_tools": allowed_tools,
        "explicit_tool_policy": explicit_tool_policy,
        "local_tools_enabled": local_tools_enabled,
        "max_steps": env_kwargs.get("max_steps") if isinstance(env_kwargs.get("max_steps"), int) else None,
    }


def _serialize_observations(payloads: list[dict[str, Any]]) -> str:
    return "\n".join(
        f'<observation tool_name="{str(payload.get("tool_name") or "runtime")}">'
        f'{json.dumps(to_jsonable(payload), ensure_ascii=False, separators=(",", ":"))}'
        "</observation>"
        for payload in payloads
    )


def _missing_mcp_env() -> list[str]:
    missing: list[str] = []
    for key in REQUIRED_MCP_ENV:
        if not (os.environ.get(key) or "").strip():
            missing.append(key)
    return missing


def _close_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _load_tokenizer(tokenizer_path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)


def _call_generate_endpoint(
    base_url: str,
    tokenizer,
    messages: list[dict[str, str]],
    temperature: float,
    max_new_tokens: int,
) -> tuple[str, str | None]:
    # Qwen3.x enables thinking mode by default; disable to reduce wrapper text for tool-calling JSON.
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        },
        "return_logprob": False,
    }
    response = _http_json("POST", f"{base_url}/generate", payload=payload, timeout_sec=180.0)
    text = response.get("text")
    if not isinstance(text, str):
        raise ValueError(f"/generate response missing text: {response}")
    finish_type = None
    meta = response.get("meta_info")
    if isinstance(meta, dict):
        finish = meta.get("finish_reason")
        if isinstance(finish, dict):
            ft = finish.get("type")
            if isinstance(ft, str):
                finish_type = ft
    return text, finish_type


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-task debug rollout for drug_agent with external/local SGLang")
    parser.add_argument("--input-jsonl", type=str, default=str(_default_input_jsonl()))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--sglang-base-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--run-name", type=str, default=f"debug_step1_{int(time.time())}")
    parser.add_argument("--disable-tool-call", action="store_true")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--env-file", action="append", default=[])

    # Optional local launch mode.
    parser.add_argument("--launch-local-server", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--mem-fraction-static", type=float, default=0.80)
    parser.add_argument("--health-timeout-sec", type=float, default=300.0)
    args = parser.parse_args()
    load_molclaw_environment(args.env_file)
    if not args.disable_tool_call:
        assert_tool_environment_allowed("debug_one_task online tool execution")
        print("[ONLINE TOOL DEBUG] Real MolClaw/MCP calls are enabled.", flush=True)

    started = time.monotonic()
    output: dict[str, Any] = {
        "ok": False,
        "error": None,
        "task_id": None,
        "task_type": None,
        "num_steps": 0,
        "action_valid_rate": 0.0,
        "tool_success_rate": 0.0,
        "final_success_rate": 0.0,
        "trace_path": None,
        "latency_sec": 0.0,
        "result": {
            "input_jsonl": args.input_jsonl,
            "index": args.index,
            "mode": "external_server",
            "sglang_base_url": args.sglang_base_url,
            "disable_tool_call": bool(args.disable_tool_call),
        },
    }

    local_server_proc: subprocess.Popen | None = None
    executor: MCPToolExecutor | None = None
    try:
        input_jsonl = Path(args.input_jsonl)
        if not input_jsonl.exists():
            output["error"] = _error_payload("input_not_found", f"input jsonl not found: {input_jsonl}")
            output["latency_sec"] = round(time.monotonic() - started, 6)
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 1

        row = _read_row(input_jsonl, args.index)
        context = _sample_context(row)
        task_id = context.get("task_id")
        task_type = context.get("task_type")
        allowed_tools = context.get("allowed_tools") if isinstance(context.get("allowed_tools"), list) else []
        output["task_id"] = task_id
        output["task_type"] = task_type

        messages = _extract_messages(row)
        if not messages:
            output["error"] = _error_payload("bad_sample", "cannot derive prompt/messages from sample")
            output["latency_sec"] = round(time.monotonic() - started, 6)
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 1
        base_url = args.sglang_base_url.rstrip("/")
        model_path_for_launch: Path | None = None
        if args.launch_local_server:
            output["result"]["mode"] = "local_launch"
            model_path_for_launch = Path(args.model_path) if args.model_path else _resolve_default_debug_model_path()
            if not model_path_for_launch.is_dir():
                output["error"] = _error_payload("model_not_found", f"model path not found: {model_path_for_launch}")
                output["latency_sec"] = round(time.monotonic() - started, 6)
                print(json.dumps(output, ensure_ascii=False, indent=2))
                return 1
            if (model_path_for_launch / "latest_checkpointed_iteration.txt").is_file():
                output["error"] = _error_payload(
                    "torch_dist_checkpoint_not_hf",
                    "A Slime torch-distributed checkpoint cannot be launched as an HF directory; "
                    "use scripts/run_molbench_eval.sh so actor weights are synchronized into SGLang.",
                )
                output["latency_sec"] = round(time.monotonic() - started, 6)
                print(json.dumps(output, ensure_ascii=False, indent=2))
                return 1

            detect = detect_sglang_launch_command(
                model_path=str(model_path_for_launch),
                port=args.port,
                tp_size=args.tp_size,
                host="0.0.0.0",
                context_length=args.context_length,
                mem_fraction_static=args.mem_fraction_static,
                trust_remote_code=True,
            )
            output["result"]["launcher_detection"] = detect
            if not detect.get("ok"):
                output["error"] = detect.get("error")
                output["latency_sec"] = round(time.monotonic() - started, 6)
                print(json.dumps(output, ensure_ascii=False, indent=2))
                return 1

            local_server_proc = subprocess.Popen(
                detect["cmd"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            base_url = f"http://127.0.0.1:{args.port}"
            output["result"]["sglang_base_url"] = base_url
            output["result"]["model_path"] = str(model_path_for_launch)
            ok, health_text = _wait_for_health(base_url=base_url, timeout_sec=args.health_timeout_sec)
            if not ok:
                output["error"] = _error_payload("sglang_launch_failed", health_text)
                output["latency_sec"] = round(time.monotonic() - started, 6)
                print(json.dumps(output, ensure_ascii=False, indent=2))
                return 1

        ok, health_text = _check_sglang_health(base_url=base_url, timeout_sec=8.0)
        if not ok:
            output["error"] = _error_payload("sglang_unreachable", health_text)
            output["latency_sec"] = round(time.monotonic() - started, 6)
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 1
        output["result"]["sglang_health"] = health_text

        try:
            models = _http_json("GET", f"{base_url}/v1/models", timeout_sec=20.0)
            output["result"]["models"] = models
        except Exception as exc:
            output["result"]["models_error"] = f"{type(exc).__name__}: {exc}"

        if not args.disable_tool_call:
            missing_env = missing_molclaw_environment()
            if missing_env:
                output["error"] = _error_payload("missing_env", f"{missing_env[0]} is missing")
                output["result"]["missing_env"] = missing_env
                output["latency_sec"] = round(time.monotonic() - started, 6)
                print(json.dumps(output, ensure_ascii=False, indent=2))
                return 1
            executor = MCPToolExecutor(connect_on_init=False)
            registry = ToolRegistry(
                executor=executor,
                include_local_tools=bool(context["local_tools_enabled"]),
            )
            tool_specs = registry.list_tools()
            if not tool_specs:
                raise RuntimeError("molclaw-scp list_tools returned an empty catalog")
            if context["explicit_tool_policy"]:
                permitted = set(context["allowed_tools"])
                tool_specs = [spec for spec in tool_specs if spec.get("name") in permitted]
            else:
                context["allowed_tools"] = [str(spec["name"]) for spec in tool_specs]
            tool_catalog = format_tool_catalog(tool_specs)
        else:
            registry = None
            tool_catalog = ""

        messages = _augment_messages_for_strict_json(
            messages,
            local_tools_enabled=bool(context["local_tools_enabled"]),
            tool_catalog=tool_catalog,
            final_contract=format_final_contract(str(task_type or "")),
        )

        tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else None
        if tokenizer_path is None:
            tokenizer_path = model_path_for_launch if args.launch_local_server else _resolve_default_tokenizer_path()
        if tokenizer_path is None or not tokenizer_path.is_dir():
            output["error"] = _error_payload(
                "tokenizer_not_found",
                f"tokenizer path not found: {tokenizer_path}",
            )
            output["latency_sec"] = round(time.monotonic() - started, 6)
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 1
        output["result"]["tokenizer_path"] = str(tokenizer_path)
        tokenizer = _load_tokenizer(tokenizer_path)

        trace_root = ensure_dir(_default_runs_root() / args.run_name)
        trace_path = trace_root / "debug_one_task_trace.jsonl"
        output["trace_path"] = str(trace_path)
        local_executor = (
            LocalToolExecutor(trace_root / "workspace", DRUG_AGENT_L1_SKILLS_ROOT)
            if context["local_tools_enabled"]
            else None
        )
        artifact_registry = ArtifactRegistry(trace_root / "workspace")
        output["result"]["workspace"] = (
            str(local_executor.workspace) if local_executor is not None else None
        )

        steps: list[dict[str, Any]] = []
        num_invalid = 0
        num_tool_success = 0
        num_tool_error = 0
        num_tool_schema_error = 0
        num_tool_execution_success = 0
        num_tool_semantic_error = 0
        num_tool_semantic_unknown = 0
        num_transport_error = 0
        done_reason: str | None = None
        final_answer: dict[str, Any] | None = None

        for step in range(max(1, args.max_steps)):
            raw_model_output, finish_type = _call_generate_endpoint(
                base_url=base_url,
                tokenizer=tokenizer,
                messages=messages,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )

            parsed = parse_runtime_decision(raw_model_output)
            parse_recovery = None
            normalized_model_output = raw_model_output
            parse_source = "canonical_react_strict"
            action_valid = bool(parsed.get("ok"))
            if not action_valid:
                num_invalid += 1

            tool_result: dict[str, Any] | None = None
            observation = None
            parse_error = None if action_valid else {
                "error_type": parsed.get("error_type"),
                "error_message": parsed.get("error_message"),
            }

            if not action_valid:
                observation_payload = {
                    "tool_name": "runtime",
                    "status": "error",
                    "is_error": True,
                    "content": parse_error,
                }
                observation = _serialize_observations([observation_payload])
                messages.append({"role": "assistant", "content": raw_model_output})
                messages.append({"role": "user", "content": observation})
            elif parsed.get("decision_type") == "tool_call":
                observation_payloads = []
                for parsed_call in parsed.get("tool_calls") or []:
                    tool_name = normalize_tool_name(parsed_call.get("tool_name"))
                    tool_args = parsed_call.get("arguments") if isinstance(parsed_call.get("arguments"), dict) else {}
                    if args.disable_tool_call:
                        current_result = {
                            "ok": False,
                            "tool_name": tool_name,
                            "result": None,
                            "error": _error_payload("tool_call_disabled", "tool call is disabled by --disable-tool-call"),
                            "latency_sec": 0.0,
                            "transport_ok": True,
                            "tool_schema_valid": True,
                            "tool_execution_success": False,
                            "tool_semantic_success": False,
                            "semantic_unknown": False,
                            "metadata": {},
                        }
                    else:
                        ok_name, reason_name = registry.validate_tool_name(tool_name, allowed_tools=allowed_tools)
                        ok_args, reason_args = registry.validate_arguments(tool_name, tool_args)
                        if not ok_name or not ok_args:
                            current_result = make_validation_failed_result(
                                tool_name=tool_name,
                                message=reason_name or reason_args or "validation failed",
                                tool_reason=reason_name,
                                args_reason=reason_args,
                            )
                        else:
                            current_result = registry.execute(
                                tool_name,
                                artifact_registry.resolve(tool_args),
                                local_executor=local_executor,
                            )
                    transport_ok = bool(current_result.get("transport_ok"))
                    tool_schema_valid = bool(current_result.get("tool_schema_valid"))
                    tool_execution_success = bool(current_result.get("tool_execution_success"))
                    tool_semantic_success = bool(current_result.get("tool_semantic_success"))
                    semantic_unknown = bool(current_result.get("semantic_unknown"))
                    num_tool_schema_error += int(not tool_schema_valid)
                    num_transport_error += int(not transport_ok)
                    num_tool_execution_success += int(tool_execution_success)
                    num_tool_success += int(tool_semantic_success)
                    num_tool_error += int(not tool_semantic_success)
                    num_tool_semantic_error += int(not tool_semantic_success)
                    num_tool_semantic_unknown += int(semantic_unknown)
                    metadata_payload = to_jsonable(current_result.get("metadata")) or {}
                    if isinstance(metadata_payload, dict):
                        metadata_payload.pop("raw", None)
                    observation_payloads.append(
                        {
                            "tool_name": tool_name,
                            "status": "success" if bool(current_result.get("ok")) else "error",
                            "is_error": not bool(current_result.get("ok")),
                            "content": {
                                "result": artifact_registry.canonicalize(
                                    to_jsonable(current_result.get("result")),
                                    local_result=tool_name in LOCAL_TOOL_NAMES,
                                ),
                                "error": artifact_registry.canonicalize(
                                    to_jsonable(current_result.get("error"))
                                ),
                            },
                            "metadata": metadata_payload,
                        }
                    )
                # Raw executor payloads may contain server paths. The model and
                # debug trace see the same canonical observations; raw path
                # mappings are emitted only in artifact_audit.
                observation = _serialize_observations(observation_payloads)
                tool_result = {"results": observation_payloads}
                messages.append({"role": "assistant", "content": raw_model_output})
                messages.append({"role": "user", "content": observation})
            elif parsed.get("decision_type") == "final_answer":
                payload_task_type = str((parsed.get("final_answer") or {}).get("task_type") or "").lower()
                if not final_answer_matches_task(parsed.get("final_answer"), str(task_type or "")):
                    num_invalid += 1
                    observation_payload = {
                        "tool_name": "runtime",
                        "status": "error",
                        "is_error": True,
                        "content": {
                            "error_type": "FinalTaskTypeMismatch",
                            "error_message": (
                                f"final_answer.task_type must be {task_type!r}, got {payload_task_type!r}"
                            ),
                        },
                    }
                    observation = _serialize_observations([observation_payload])
                    messages.append({"role": "assistant", "content": raw_model_output})
                    messages.append({"role": "user", "content": observation})
                    parsed = {**parsed, "ok": False, "error_type": "FinalTaskTypeMismatch"}
                    parse_error = {
                        "error_type": "FinalTaskTypeMismatch",
                        "error_message": observation_payload["content"]["error_message"],
                    }
                    action_valid = False
                else:
                    final_answer = artifact_registry.canonicalize(
                        parsed.get("final_answer") if isinstance(parsed.get("final_answer"), dict) else {},
                        register_unknown_paths=False,
                    )
                    messages.append({"role": "assistant", "content": raw_model_output})
                    done_reason = "final_answer"
            else:
                observation_payload = {
                    "type": "invalid_action",
                    "error_type": "unexpected_action_type",
                    "error_message": f"unsupported decision type: {parsed.get('decision_type')}",
                }
                observation_payload["tool_name"] = "runtime"
                observation_payload["status"] = "error"
                observation_payload["is_error"] = True
                observation_payload["content"] = {"message": observation_payload.pop("error_message")}
                observation = _serialize_observations([observation_payload])
                num_invalid += 1
                messages.append({"role": "assistant", "content": raw_model_output})
                messages.append({"role": "user", "content": observation})

            if done_reason is None and finish_type == "length":
                done_reason = "length"
            if done_reason is None and finish_type == "abort":
                done_reason = "abort"

            step_record = {
                "step": step,
                "event": "model_step",
                "raw_model_output": raw_model_output,
                "model_output": normalized_model_output,
                "parsed_action": to_jsonable(parsed),
                "action_valid": action_valid,
                "parse_error": parse_error,
                "parse_recovery": parse_recovery,
                "parse_source": parse_source,
                "tool_result": to_jsonable(tool_result),
                "observation": observation,
                "reward_components": {},
                "done_reason": done_reason,
            }
            steps.append(step_record)
            append_jsonl(trace_path, to_jsonable(step_record))

            if done_reason == "final_answer":
                break
            if done_reason in {"length", "abort"}:
                break

        if done_reason is None:
            done_reason = "max_steps"

        num_steps = len(steps)
        total_tools = num_tool_success + num_tool_error
        execution_attempt_count = total_tools - num_tool_schema_error
        action_valid_rate = (num_steps - num_invalid) / max(1, num_steps)
        tool_success_rate = num_tool_success / max(1, total_tools)
        tool_execution_success_rate = num_tool_execution_success / max(1, execution_attempt_count)
        final_success_rate = 1.0 if done_reason == "final_answer" else 0.0

        append_jsonl(
            trace_path,
            {
                "summary": {
                    "task_id": task_id,
                    "task_type": task_type,
                    "done_reason": done_reason,
                    "num_steps": num_steps,
                    "num_invalid": num_invalid,
                    "num_tool_success": num_tool_success,
                    "num_tool_error": num_tool_error,
                    "num_tool_schema_error": num_tool_schema_error,
                    "num_tool_execution_success": num_tool_execution_success,
                    "num_tool_semantic_error": num_tool_semantic_error,
                    "num_tool_semantic_unknown": num_tool_semantic_unknown,
                    "num_transport_error": num_transport_error,
                    "action_valid_rate": action_valid_rate,
                    "tool_success_rate": tool_success_rate,
                    "tool_execution_success_rate": tool_execution_success_rate,
                    "final_success_rate": final_success_rate,
                    "final_answer": to_jsonable(final_answer),
                    "projected_final_answer": (
                        project_final_answer(final_answer, str(task_type or ""))
                        if isinstance(final_answer, dict)
                        else None
                    ),
                    "artifact_audit": artifact_registry.audit_snapshot(),
                }
            },
        )

        output["ok"] = True
        output["num_steps"] = num_steps
        output["action_valid_rate"] = round(action_valid_rate, 6)
        output["tool_success_rate"] = round(tool_success_rate, 6)
        output["final_success_rate"] = round(final_success_rate, 6)
        output["result"]["done_reason"] = done_reason
        output["result"]["trace_rows"] = num_steps + 1
        output["latency_sec"] = round(time.monotonic() - started, 6)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except urllib.error.URLError as exc:
        output["error"] = _error_payload("network_error", str(exc))
        output["latency_sec"] = round(time.monotonic() - started, 6)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 1
    except Exception as exc:
        output["error"] = _error_payload("runtime_error", f"{type(exc).__name__}: {exc}")
        output["latency_sec"] = round(time.monotonic() - started, 6)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 1
    finally:
        if executor is not None:
            executor.close()
        _close_server(local_server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
