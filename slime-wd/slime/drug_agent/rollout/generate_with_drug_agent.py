from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from drug_agent.protocol.parse_policy import (
    parse_action_with_policy,
    resolve_rollout_controls,
)
from drug_agent.protocol.action_schema import ACTION_FINAL_ANSWER, ACTION_TOOL_CALL
from drug_agent.tools.tool_executor import MCPToolExecutor
from drug_agent.tools.tool_registry import ToolRegistry
from drug_agent.tools.tool_success import make_validation_failed_result
from drug_agent.utils import normalize_tool_name, to_jsonable

_RUNTIME_LOCK = threading.Lock()
_RUNTIME: dict[str, Any] | None = None

ROLLOUT_FORMAT_REMINDER = (
    "/no_think\n"
    "Output exactly one JSON object only, no extra text.\n"
    "Use schema: "
    '{"type":"tool_call","tool_name":"...","arguments":{...}} OR '
    '{"type":"final_answer","answer":{"summary":"...","evidence":[],"result":{},"ranked_molecules":[]}}'
)


def _get_runtime() -> dict[str, Any]:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            executor = MCPToolExecutor(connect_on_init=False)
            registry = ToolRegistry.from_env(executor=executor)
            _RUNTIME = {
                "executor": executor,
                "registry": registry,
            }
    return _RUNTIME


def _resolve_context(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    env_kwargs = metadata.get("env_kwargs") if isinstance(metadata.get("env_kwargs"), dict) else {}

    task_id = env_kwargs.get("task_id") or metadata.get("task_id") or f"sample_{sample.index}"
    task_type = env_kwargs.get("task_type") or metadata.get("task_type") or "unknown"
    data_source = env_kwargs.get("data_source") or metadata.get("data_source") or "drug_agent"

    allowed_tools_raw = env_kwargs.get("allowed_tools")
    if not isinstance(allowed_tools_raw, list):
        allowed_tools_raw = []
    allowed_tools = [normalize_tool_name(x) for x in allowed_tools_raw if isinstance(x, str) and x.strip()]

    max_steps = env_kwargs.get("max_steps")
    if not isinstance(max_steps, int) or max_steps <= 0:
        max_steps = int(os.environ.get("DRUG_AGENT_MAX_STEPS", "6"))

    return {
        "task_id": str(task_id),
        "task_type": str(task_type),
        "data_source": str(data_source),
        "allowed_tools": allowed_tools,
        "max_steps": max_steps,
        "env_kwargs": env_kwargs,
    }


def _augment_prompt_messages(prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(m) for m in prompt]
    if not out:
        return out

    out[0]["content"] = (out[0].get("content") or "") + "\n\n" + ROLLOUT_FORMAT_REMINDER
    for idx in range(len(out) - 1, -1, -1):
        if out[idx].get("role") == "user":
            out[idx]["content"] = (out[idx].get("content") or "") + "\n\n" + ROLLOUT_FORMAT_REMINDER
            break
    return out


def _to_prompt_text(state: GenerateState, prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        prompt = _augment_prompt_messages(prompt)
        try:
            return state.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return state.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
    return str(prompt)


def _serialize_observation(payload: dict[str, Any]) -> str:
    return "\n" + json.dumps({"observation": payload}, ensure_ascii=False, separators=(",", ":")) + "\n"


def _append_observation(
    state: GenerateState,
    obs_text: str,
    response_buffer: list[str],
    response_token_ids: list[int],
    loss_masks: list[int],
    rollout_log_probs: list[float],
) -> None:
    obs_token_ids = state.tokenizer(obs_text, add_special_tokens=False)["input_ids"]
    response_buffer.append(obs_text)
    response_token_ids.extend(obs_token_ids)
    loss_masks.extend([0] * len(obs_token_ids))
    rollout_log_probs.extend([0.0] * len(obs_token_ids))


async def _execute_tool(
    executor: MCPToolExecutor,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return await asyncio.to_thread(executor.execute, tool_name, arguments)


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for drug_agent custom generate."

    state = GenerateState(args)
    runtime = _get_runtime()
    executor: MCPToolExecutor = runtime["executor"]
    registry: ToolRegistry = runtime["registry"]

    context = _resolve_context(sample)
    task_id = context["task_id"]
    task_type = context["task_type"]
    data_source = context["data_source"]
    allowed_tools = context["allowed_tools"]
    max_steps = context["max_steps"]
    rollout_controls = resolve_rollout_controls()
    rollout_mode = rollout_controls["rollout_mode"]
    parse_recovery_enabled = bool(rollout_controls["parse_recovery_enabled"])
    allow_parse_recovery_override = bool(rollout_controls["allow_parse_recovery_override"])

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_text = _to_prompt_text(state, sample.prompt)
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    response_parts: list[str] = []
    response_token_ids: list[int] = []
    loss_masks: list[int] = []
    rollout_log_probs: list[float] = []

    actions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []

    done_reason = "max_steps"
    final_answer = None
    fatal_error = None
    num_invalid = 0
    num_tool_success = 0
    num_tool_error = 0
    num_tool_schema_error = 0
    num_tool_execution_success = 0
    num_tool_semantic_error = 0
    num_tool_semantic_unknown = 0
    num_transport_error = 0
    num_parse_recovery = 0
    strict_valid_count = 0
    recovered_valid_count = 0

    try:
        for step in range(max_steps):
            current_token_ids = prompt_token_ids + response_token_ids

            payload = {
                "input_ids": current_token_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }

            output = await post(url, payload)
            finish_type = output.get("meta_info", {}).get("finish_reason", {}).get("type")

            if finish_type == "abort":
                sample.status = Sample.Status.ABORTED
                done_reason = "abort"
                break

            cur_response = output.get("text", "")
            cur_token_ids: list[int]
            cur_log_probs: list[float]

            token_log_probs = output.get("meta_info", {}).get("output_token_logprobs")
            if isinstance(token_log_probs, list) and token_log_probs:
                cur_token_ids = []
                cur_log_probs = []
                for item in token_log_probs:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        cur_log_probs.append(float(item[0]))
                        cur_token_ids.append(int(item[1]))
                cur_response = state.tokenizer.decode(cur_token_ids)
            else:
                cur_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]
                cur_log_probs = [0.0] * len(cur_token_ids)

            response_parts.append(cur_response)
            response_token_ids.extend(cur_token_ids)
            loss_masks.extend([1] * len(cur_token_ids))
            rollout_log_probs.extend(cur_log_probs)

            action_record: dict[str, Any] = {
                "step": step,
                "raw_response": cur_response,
                "finish_type": finish_type,
            }

            if finish_type == "length":
                sample.status = Sample.Status.TRUNCATED
                done_reason = "length"
                actions.append(action_record)
                break

            parsed, parse_recovery, normalized_response, parse_source = parse_action_with_policy(
                cur_response,
                parse_recovery_enabled=parse_recovery_enabled,
            )
            action_record["parsed"] = parsed.to_dict()
            action_record["model_output"] = normalized_response
            action_record["parse_recovery"] = parse_recovery
            action_record["parse_source"] = parse_source
            actions.append(action_record)

            if not parsed.ok:
                num_invalid += 1
                obs_payload = {
                    "type": "invalid_action",
                    "error_type": parsed.error_type,
                    "error_message": parsed.error_message,
                    "raw_text": parsed.raw_text,
                }
                observations.append({"step": step, **obs_payload})
                _append_observation(
                    state,
                    _serialize_observation(obs_payload),
                    response_parts,
                    response_token_ids,
                    loss_masks,
                    rollout_log_probs,
                )
                continue
            if isinstance(parse_recovery, dict) and parse_recovery.get("recovered") is True:
                num_parse_recovery += 1
                recovered_valid_count += 1
            else:
                strict_valid_count += 1

            if parsed.action_type == ACTION_TOOL_CALL:
                tool_name = normalize_tool_name(parsed.tool_name)
                tool_args = parsed.arguments if isinstance(parsed.arguments, dict) else {}

                tool_ok, tool_reason = registry.validate_tool_name(tool_name, allowed_tools=allowed_tools)
                args_ok, args_reason = registry.validate_arguments_basic(tool_name, tool_args)

                if tool_ok and args_ok:
                    tool_result = await _execute_tool(executor, tool_name, tool_args)
                else:
                    err_message = tool_reason or args_reason or "tool validation failed"
                    tool_result = make_validation_failed_result(
                        tool_name=tool_name,
                        message=err_message,
                        tool_reason=tool_reason,
                        args_reason=args_reason,
                    )

                transport_ok = bool(tool_result.get("transport_ok"))
                tool_schema_valid = bool(tool_result.get("tool_schema_valid"))
                tool_execution_success = bool(tool_result.get("tool_execution_success"))
                tool_semantic_success = bool(tool_result.get("tool_semantic_success"))
                semantic_unknown = bool(tool_result.get("semantic_unknown"))

                if not tool_schema_valid:
                    num_tool_schema_error += 1
                if not transport_ok:
                    num_transport_error += 1
                if tool_execution_success:
                    num_tool_execution_success += 1
                if tool_semantic_success:
                    num_tool_success += 1
                else:
                    num_tool_error += 1
                    num_tool_semantic_error += 1
                if semantic_unknown:
                    num_tool_semantic_unknown += 1

                obs_payload = {
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "ok": bool(tool_result.get("ok")),
                    "result": to_jsonable(tool_result.get("result")),
                    "error": to_jsonable(tool_result.get("error")),
                    "latency_sec": tool_result.get("latency_sec"),
                    "transport_ok": transport_ok,
                    "tool_schema_valid": tool_schema_valid,
                    "tool_execution_success": tool_execution_success,
                    "tool_semantic_success": tool_semantic_success,
                    "semantic_unknown": semantic_unknown,
                    "metadata": to_jsonable(tool_result.get("metadata")),
                }
                observations.append({"step": step, **obs_payload})

                _append_observation(
                    state,
                    _serialize_observation(obs_payload),
                    response_parts,
                    response_token_ids,
                    loss_masks,
                    rollout_log_probs,
                )
                continue

            if parsed.action_type == ACTION_FINAL_ANSWER:
                final_answer = parsed.answer
                done_reason = "final_answer"
                sample.status = Sample.Status.COMPLETED
                break
    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        sample.status = Sample.Status.FAILED
        done_reason = "fatal_error"

    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED if done_reason == "final_answer" else Sample.Status.TRUNCATED

    sample.prompt = prompt_text
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response = "".join(response_parts)
    sample.response_length = len(response_token_ids)
    sample.loss_mask = loss_masks
    sample.rollout_log_probs = rollout_log_probs

    if len(sample.rollout_log_probs) != len(response_token_ids):
        fatal_error = (
            f"Token/logp mismatch: tokens={len(response_token_ids)} "
            f"logps={len(sample.rollout_log_probs)}"
        )
        sample.rollout_log_probs = [0.0] * len(response_token_ids)

    if not isinstance(sample.metadata, dict):
        sample.metadata = {}

    num_steps = len(actions)
    valid_count = strict_valid_count + recovered_valid_count
    action_valid_rate = valid_count / max(1, num_steps)
    strict_success_rate = strict_valid_count / max(1, num_steps)
    recovery_success_rate = recovered_valid_count / max(1, num_steps)
    total_tool_calls = num_tool_success + num_tool_error
    execution_attempt_count = total_tool_calls - num_tool_schema_error
    tool_success_rate = num_tool_success / max(1, total_tool_calls)
    tool_execution_success_rate = num_tool_execution_success / max(1, execution_attempt_count)

    trace = {
        "task_id": task_id,
        "task_type": task_type,
        "data_source": data_source,
        "evaluation": bool(evaluation),
        "allowed_tools": allowed_tools,
        "max_steps": max_steps,
        "rollout_mode": rollout_mode,
        "parse_recovery_enabled": parse_recovery_enabled,
        "allow_parse_recovery_override": allow_parse_recovery_override,
        "actions": actions,
        "observations": observations,
        "final_answer": final_answer,
        "done_reason": done_reason,
        "num_steps": num_steps,
        "num_invalid": num_invalid,
        "num_parse_recovery": num_parse_recovery,
        "strict_valid_count": strict_valid_count,
        "recovered_valid_count": recovered_valid_count,
        "num_tool_success": num_tool_success,
        "num_tool_error": num_tool_error,
        "num_tool_schema_error": num_tool_schema_error,
        "num_tool_execution_success": num_tool_execution_success,
        "num_tool_semantic_error": num_tool_semantic_error,
        "num_tool_semantic_unknown": num_tool_semantic_unknown,
        "num_transport_error": num_transport_error,
        "truncated": sample.status == Sample.Status.TRUNCATED,
        "error": fatal_error,
        "action_valid_rate": action_valid_rate,
        "strict_success_rate": strict_success_rate,
        "recovery_success_rate": recovery_success_rate,
        "tool_success_rate": tool_success_rate,
        "tool_execution_success_rate": tool_execution_success_rate,
    }
    sample.metadata["drug_agent_trace"] = trace

    return sample
