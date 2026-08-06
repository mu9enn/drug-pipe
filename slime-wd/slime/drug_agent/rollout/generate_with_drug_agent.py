from __future__ import annotations

import asyncio
import itertools
import json
import os
import re
import threading
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from drug_agent.evaluation.task_store import bind_task_identity, checkpoint_sample, restore_sample
from drug_agent.protocol.react_protocol import final_answer_matches_task, parse_runtime_decision, project_final_answer
from drug_agent.protocol.prompts import format_final_contract, format_tool_catalog, fresh_task_messages
from drug_agent.constants import DRUG_AGENT_L1_SKILLS_ROOT, DRUG_AGENT_WORKSPACES_ROOT
from drug_agent.tools.artifact_registry import ArtifactRegistry
from drug_agent.tools.local_tools import LOCAL_TOOL_NAMES, LocalToolExecutor
from drug_agent.tools.tool_executor import MCPToolExecutor
from drug_agent.tools.tool_registry import ToolRegistry, catalog_sha256
from drug_agent.tools.tool_success import make_validation_failed_result
from drug_agent.utils import normalize_tool_name, to_jsonable

_RUNTIME_LOCK = threading.Lock()
_RUNTIME: dict[str, Any] | None = None

ROLLOUT_FORMAT_REMINDER = (
    "/no_think\n"
    "Use canonical ReAct XML. Put reasoning in <thought>...</thought>, followed by "
    "one or more <tool_call>{\"tool_name\":\"...\",\"arguments\":{...}}</tool_call> blocks, "
    "or one task-specific <final_answer>{...}</final_answer> block. "
    "Never mix tool calls and final answer in one generation."
)
LOCAL_TOOL_REMINDER = (
    "\nAvailable local tools: Read, Write, Edit, Bash, Grep, and Glob. "
    "They operate only in this task's workspace; Read, Grep, and Glob may inspect read-only L1 skill documents."
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


def _verify_run_catalog(live_specs: list[dict[str, Any]]) -> str:
    actual = catalog_sha256(live_specs)
    expected_path = os.environ.get("DRUG_AGENT_EXPECTED_TOOL_CATALOG", "").strip()
    if not expected_path:
        return actual
    with open(expected_path, "r", encoding="utf-8") as handle:
        expected = json.load(handle)
    expected_hash = expected.get("sha256") if isinstance(expected, dict) else None
    if expected_hash != actual:
        raise RuntimeError(
            "Live molclaw-scp tool catalog changed after evaluation preflight: "
            f"expected={expected_hash}, actual={actual}"
        )
    return actual


def _resolve_context(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    env_kwargs = metadata.get("env_kwargs") if isinstance(metadata.get("env_kwargs"), dict) else {}

    task_id = env_kwargs.get("task_id") or metadata.get("task_id") or f"sample_{sample.index}"
    task_type = env_kwargs.get("task_type") or metadata.get("task_type") or "unknown"
    data_source = env_kwargs.get("data_source") or metadata.get("data_source") or "drug_agent"

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

    max_steps = env_kwargs.get("max_steps")
    if not isinstance(max_steps, int):
        max_steps = int(os.environ.get("DRUG_AGENT_MAX_STEPS", "0"))
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative (0 means unlimited)")

    return {
        "task_id": str(task_id),
        "task_type": str(task_type),
        "data_source": str(data_source),
        "allowed_tools": allowed_tools,
        "explicit_tool_policy": explicit_tool_policy,
        "max_steps": max_steps,
        "env_kwargs": env_kwargs,
        "local_tools_enabled": local_tools_enabled,
    }


def _augment_prompt_messages(
    prompt: list[dict[str, Any]],
    *,
    local_tools_enabled: bool,
    tool_catalog: str = "",
    final_contract: str = "",
) -> list[dict[str, Any]]:
    out = [dict(m) for m in prompt]
    if not out:
        return out

    reminder = ROLLOUT_FORMAT_REMINDER
    if local_tools_enabled:
        reminder += LOCAL_TOOL_REMINDER
    if tool_catalog:
        reminder += "\n" + tool_catalog
    if final_contract:
        reminder += "\n" + final_contract
    system_index = next((idx for idx, item in enumerate(out) if item.get("role") == "system"), None)
    target_index = system_index if system_index is not None else 0
    out[target_index]["content"] = (out[target_index].get("content") or "") + "\n\n" + reminder
    return out


def _to_prompt_text(
    state: GenerateState,
    prompt: Any,
    *,
    local_tools_enabled: bool,
    tool_catalog: str = "",
    final_contract: str = "",
) -> str:
    if isinstance(prompt, str):
        reminder = ROLLOUT_FORMAT_REMINDER
        if local_tools_enabled:
            reminder += LOCAL_TOOL_REMINDER
        if tool_catalog:
            reminder += "\n" + tool_catalog
        if final_contract:
            reminder += "\n" + final_contract
        return prompt + "\n\n" + reminder
    if isinstance(prompt, list):
        prompt = _augment_prompt_messages(
            prompt,
            local_tools_enabled=local_tools_enabled,
            tool_catalog=tool_catalog,
            final_contract=final_contract,
        )
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


def _serialize_observations(payloads: list[dict[str, Any]]) -> str:
    blocks = []
    for payload in payloads:
        tool_name = str(payload.get("tool_name") or "runtime")
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        blocks.append(f'<observation tool_name="{tool_name}">{body}</observation>')
    return "\n" + "\n".join(blocks) + "\n"


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
    registry: ToolRegistry,
    tool_name: str,
    arguments: dict[str, Any],
    local_executor: LocalToolExecutor | None,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        registry.execute,
        tool_name,
        arguments,
        local_executor=local_executor,
    )


def _close_task_executor(task_executor: MCPToolExecutor, sample: Sample) -> None:
    """Keep transport cleanup failures inside one task's diagnostics."""

    try:
        task_executor.close()
    except BaseException as exc:
        warning = f"{type(exc).__name__}: {exc}"
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        trace = sample.metadata.get("drug_agent_trace")
        if isinstance(trace, dict):
            warnings = trace.setdefault("cleanup_warnings", [])
            if isinstance(warnings, list):
                warnings.append(warning)
        print(f"[drug-agent eval] MCP cleanup warning for task: {warning}", flush=True)


def _workspace_name(task_id: str, sample_index: Any) -> str:
    safe_task = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("._") or "task"
    safe_index = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_index)).strip("._") or "sample"
    return f"{safe_task}__{safe_index}"


async def _generate_impl(
    args,
    sample: Sample,
    sampling_params,
    *,
    task_executor: MCPToolExecutor,
    evaluation: bool = False,
) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for drug_agent custom generate."

    state = GenerateState(args)
    runtime = _get_runtime()
    authority_registry: ToolRegistry = runtime["registry"]

    context = _resolve_context(sample)
    task_id = context["task_id"]
    task_type = context["task_type"]
    data_source = context["data_source"]
    live_tool_specs = await asyncio.to_thread(authority_registry.list_tools)
    if not live_tool_specs:
        raise RuntimeError("molclaw-scp list_tools returned an empty catalog")
    tool_catalog_hash = _verify_run_catalog(live_tool_specs)
    if context["explicit_tool_policy"]:
        allowed_tools = context["allowed_tools"]
    else:
        allowed_tools = [str(spec["name"]) for spec in live_tool_specs]
    allowed_specs = [spec for spec in live_tool_specs if spec.get("name") in set(allowed_tools)]
    tool_catalog = format_tool_catalog(allowed_specs)
    final_contract = format_final_contract(task_type)
    registry = ToolRegistry(
        executor=task_executor,
        include_local_tools=context["local_tools_enabled"],
    )
    registry.install_catalog(live_tool_specs)
    max_steps = context["max_steps"]
    local_executor = None
    workspace = None
    if context["local_tools_enabled"]:
        workspace = DRUG_AGENT_WORKSPACES_ROOT / _workspace_name(task_id, sample.index)
        local_executor = LocalToolExecutor(workspace, DRUG_AGENT_L1_SKILLS_ROOT)
    artifact_registry = ArtifactRegistry(workspace or (DRUG_AGENT_WORKSPACES_ROOT / _workspace_name(task_id, sample.index)))
    rollout_mode = "canonical_react_strict"
    parse_recovery_enabled = False
    allow_parse_recovery_override = False

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    source_prompt = sample.prompt
    if evaluation and isinstance(source_prompt, list):
        source_prompt = fresh_task_messages(source_prompt)
        if not any(item.get("role") == "user" for item in source_prompt):
            raise ValueError("evaluation sample has no fresh user question")
    prompt_text = _to_prompt_text(
        state,
        source_prompt,
        local_tools_enabled=context["local_tools_enabled"],
        tool_catalog=tool_catalog,
        final_contract=final_contract,
    )
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    response_parts: list[str] = []
    response_token_ids: list[int] = []
    loss_masks: list[int] = []
    rollout_log_probs: list[float] = []

    actions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata["_drug_agent_partial_trace"] = {
        "actions": actions,
        "observations": observations,
        "artifact_registry": artifact_registry,
    }

    done_reason = "max_steps" if max_steps > 0 else "running"
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
        step_iterator = range(max_steps) if max_steps > 0 else itertools.count()
        for step in step_iterator:
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

            parsed = parse_runtime_decision(cur_response)
            action_record["parsed"] = to_jsonable(parsed)
            action_record["model_output"] = cur_response
            action_record["parse_recovery"] = None
            action_record["parse_source"] = "canonical_react_strict"
            actions.append(action_record)

            if not parsed.get("ok"):
                num_invalid += 1
                obs_payload = {
                    "tool_name": "runtime",
                    "status": "error",
                    "is_error": True,
                    "content": {
                        "error_type": parsed.get("error_type"),
                        "error_message": parsed.get("error_message"),
                    },
                }
                observations.append({"step": step, **obs_payload})
                _append_observation(
                    state,
                    _serialize_observations([obs_payload]),
                    response_parts,
                    response_token_ids,
                    loss_masks,
                    rollout_log_probs,
                )
                continue
            strict_valid_count += 1

            if parsed.get("decision_type") == "tool_call":
                step_observations: list[dict[str, Any]] = []
                for parsed_call in parsed.get("tool_calls") or []:
                    tool_name = normalize_tool_name(parsed_call.get("tool_name"))
                    tool_args = parsed_call.get("arguments") if isinstance(parsed_call.get("arguments"), dict) else {}

                    tool_ok, tool_reason = registry.validate_tool_name(tool_name, allowed_tools=allowed_tools)
                    args_ok, args_reason = registry.validate_arguments(tool_name, tool_args)

                    if tool_ok and args_ok:
                        execution_args = artifact_registry.resolve(tool_args)
                        tool_result = await _execute_tool(registry, tool_name, execution_args, local_executor)
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

                    result_metadata = to_jsonable(tool_result.get("metadata"))
                    if not isinstance(result_metadata, dict):
                        result_metadata = {}
                    # Raw MCP payloads can contain server paths and belong only
                    # in the artifact audit, never in model-visible observations.
                    result_metadata.pop("raw", None)
                    model_result = artifact_registry.canonicalize(
                        to_jsonable(tool_result.get("result")),
                        local_result=tool_name in LOCAL_TOOL_NAMES,
                    )
                    model_error = artifact_registry.canonicalize(to_jsonable(tool_result.get("error")))
                    obs_payload = {
                        "tool_name": tool_name,
                        "status": "success" if bool(tool_result.get("ok")) else "error",
                        "is_error": not bool(tool_result.get("ok")),
                        "content": {
                            "result": model_result,
                            "error": model_error,
                        },
                        "metadata": {
                            "latency_sec": tool_result.get("latency_sec"),
                            "transport_ok": transport_ok,
                            "tool_schema_valid": tool_schema_valid,
                            "tool_execution_success": tool_execution_success,
                            "tool_semantic_success": tool_semantic_success,
                            "semantic_unknown": semantic_unknown,
                            **result_metadata,
                        },
                    }
                    observations.append({"step": step, **obs_payload})
                    step_observations.append(obs_payload)

                _append_observation(
                    state,
                    _serialize_observations(step_observations),
                    response_parts,
                    response_token_ids,
                    loss_masks,
                    rollout_log_probs,
                )
                continue

            if parsed.get("decision_type") == "final_answer":
                payload_task_type = str((parsed.get("final_answer") or {}).get("task_type") or "").lower()
                if not final_answer_matches_task(parsed.get("final_answer"), task_type):
                    num_invalid += 1
                    obs_payload = {
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
                    observations.append({"step": step, **obs_payload})
                    _append_observation(
                        state,
                        _serialize_observations([obs_payload]),
                        response_parts,
                        response_token_ids,
                        loss_masks,
                        rollout_log_probs,
                    )
                    continue
                final_answer = artifact_registry.canonicalize(
                    parsed.get("final_answer"),
                    register_unknown_paths=False,
                )
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
        "tool_catalog_sha256": tool_catalog_hash,
        "workspace": "<artifact:local/>" if workspace is not None else None,
        "max_steps": max_steps,
        "rollout_mode": rollout_mode,
        "parse_recovery_enabled": parse_recovery_enabled,
        "allow_parse_recovery_override": allow_parse_recovery_override,
        "actions": actions,
        "observations": observations,
        "final_answer": final_answer,
        "projected_final_answer": (
            project_final_answer(final_answer, task_type) if isinstance(final_answer, dict) else None
        ),
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
        "artifact_audit": artifact_registry.audit_snapshot(),
    }
    sample.metadata["drug_agent_trace"] = trace
    sample.metadata.pop("_drug_agent_partial_trace", None)

    return sample


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    """Bound complete agent tasks; SGLang token concurrency remains separate."""
    runtime = _get_runtime()
    loop = asyncio.get_running_loop()
    semaphore = runtime.get("task_semaphore")
    if semaphore is None or runtime.get("task_semaphore_loop") is not loop:
        max_workers = max(1, int(os.environ.get("DRUG_AGENT_MAX_WORKERS", "2")))
        semaphore = asyncio.Semaphore(max_workers)
        runtime["task_semaphore"] = semaphore
        runtime["task_semaphore_loop"] = loop
    async with semaphore:
        bind_task_identity(sample)
        restored = restore_sample(sample, evaluation=evaluation)
        if restored is not None:
            return restored

        # A task owns its MCP transport. This prevents session state and
        # connection recovery from crossing task/workspace boundaries.
        task_executor = MCPToolExecutor(connect_on_init=False)
        try:
            timeout = float(os.environ.get("DRUG_AGENT_TASK_TIMEOUT_SEC", "10800"))
            try:
                result = await asyncio.wait_for(
                    _generate_impl(
                        args,
                        sample,
                        sampling_params,
                        task_executor=task_executor,
                        evaluation=evaluation,
                    ),
                    timeout=timeout if timeout > 0 else None,
                )
            except asyncio.TimeoutError:
                context = _resolve_context(sample)
                sample.status = Sample.Status.FAILED
                if not isinstance(sample.metadata, dict):
                    sample.metadata = {}
                partial = sample.metadata.pop("_drug_agent_partial_trace", {})
                actions = partial.get("actions") if isinstance(partial, dict) else []
                observations = partial.get("observations") if isinstance(partial, dict) else []
                artifact_registry = partial.get("artifact_registry") if isinstance(partial, dict) else None
                sample.metadata["drug_agent_trace"] = {
                    "task_id": context["task_id"],
                    "task_type": context["task_type"],
                    "data_source": context["data_source"],
                    "evaluation": bool(evaluation),
                    "done_reason": "task_timeout",
                    "error": f"task timeout after {timeout}s",
                    "actions": to_jsonable(actions or []),
                    "observations": to_jsonable(observations or []),
                    "final_answer": None,
                    "projected_final_answer": None,
                    "artifact_audit": (
                        artifact_registry.audit_snapshot()
                        if isinstance(artifact_registry, ArtifactRegistry)
                        else {}
                    ),
                }
                result = sample
            # Save before teardown so even a cleanup failure cannot erase an
            # otherwise complete task.  The per-task file is idempotent and
            # can be restored by a later resume run.
            checkpoint_sample(result, evaluation=evaluation)
            return result
        finally:
            _close_task_executor(task_executor, sample)
