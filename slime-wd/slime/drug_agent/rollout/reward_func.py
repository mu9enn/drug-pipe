from __future__ import annotations

import json
from typing import Any

from slime.utils.types import Sample

from drug_agent.utils import clamp, to_jsonable


def _extract_trace(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    trace = metadata.get("drug_agent_trace")
    if isinstance(trace, dict):
        return trace

    return {
        "actions": [],
        "observations": [],
        "done_reason": "missing_trace",
        "num_steps": 0,
        "num_invalid": 0,
        "num_parse_recovery": 0,
        "strict_valid_count": 0,
        "recovered_valid_count": 0,
        "num_tool_success": 0,
        "num_tool_error": 0,
        "num_tool_schema_error": 0,
        "num_tool_execution_success": 0,
        "num_tool_semantic_error": 0,
        "num_tool_semantic_unknown": 0,
        "num_transport_error": 0,
        "truncated": sample.status == Sample.Status.TRUNCATED,
        "final_answer": None,
    }


def _label_dict(sample: Sample) -> dict[str, Any]:
    label = sample.label
    if isinstance(label, dict):
        return label
    return {}


def _actions(trace: dict[str, Any]) -> list[dict[str, Any]]:
    actions = trace.get("actions")
    if not isinstance(actions, list):
        return []
    out: list[dict[str, Any]] = []
    for action in actions:
        if isinstance(action, dict):
            out.append(action)
    return out


def _observations(trace: dict[str, Any]) -> list[dict[str, Any]]:
    observations = trace.get("observations")
    if not isinstance(observations, list):
        return []
    out: list[dict[str, Any]] = []
    for obs in observations:
        if isinstance(obs, dict):
            out.append(obs)
    return out


def _action_stats(trace: dict[str, Any]) -> dict[str, int]:
    actions = _actions(trace)
    derived_steps = len(actions)
    num_steps = int(trace.get("num_steps") or 0)
    if num_steps <= 0:
        num_steps = derived_steps
    num_steps = max(num_steps, derived_steps)

    strict_valid = int(trace.get("strict_valid_count") or 0)
    recovered_valid = int(trace.get("recovered_valid_count") or 0)
    num_invalid = int(trace.get("num_invalid") or 0)
    num_parse_recovery = int(trace.get("num_parse_recovery") or 0)

    if strict_valid == 0 and recovered_valid == 0 and actions:
        for action in actions:
            parsed = action.get("parsed")
            is_valid = isinstance(parsed, dict) and bool(parsed.get("ok"))
            recovered = isinstance(action.get("parse_recovery"), dict) and (
                action.get("parse_recovery") or {}
            ).get("recovered") is True
            if is_valid and recovered:
                recovered_valid += 1
            elif is_valid:
                strict_valid += 1
            else:
                num_invalid += 1
        if num_parse_recovery == 0:
            num_parse_recovery = recovered_valid
    else:
        if num_parse_recovery == 0 and recovered_valid > 0:
            num_parse_recovery = recovered_valid
        valid_total = strict_valid + recovered_valid
        if num_invalid <= 0 and num_steps >= valid_total:
            num_invalid = num_steps - valid_total

    valid_count = strict_valid + recovered_valid
    if num_steps < valid_count:
        num_steps = valid_count
    if num_invalid < 0:
        num_invalid = 0

    return {
        "num_steps": num_steps,
        "num_invalid": num_invalid,
        "num_parse_recovery": max(0, num_parse_recovery),
        "strict_valid_count": max(0, strict_valid),
        "recovered_valid_count": max(0, recovered_valid),
        "valid_count": max(0, valid_count),
    }


def _tool_stats(trace: dict[str, Any]) -> dict[str, int]:
    observations = _observations(trace)
    total_tool_results = 0
    schema_fail_count = 0
    transport_error_count = 0
    execution_attempt_count = 0
    execution_success_count = 0
    semantic_success_count = 0
    semantic_error_count = 0
    semantic_unknown_count = 0

    for obs in observations:
        if obs.get("type") != "tool_result":
            continue
        total_tool_results += 1
        transport_ok = bool(obs.get("transport_ok")) if "transport_ok" in obs else bool(obs.get("ok"))
        tool_schema_valid = bool(obs.get("tool_schema_valid")) if "tool_schema_valid" in obs else True
        tool_execution_success = (
            bool(obs.get("tool_execution_success")) if "tool_execution_success" in obs else bool(obs.get("ok"))
        )
        tool_semantic_success = (
            bool(obs.get("tool_semantic_success")) if "tool_semantic_success" in obs else bool(obs.get("ok"))
        )
        semantic_unknown = bool(obs.get("semantic_unknown"))
        error = obs.get("error") if isinstance(obs.get("error"), dict) else {}
        metadata = obs.get("metadata") if isinstance(obs.get("metadata"), dict) else {}
        tool_reason = metadata.get("tool_reason")
        args_reason = metadata.get("args_reason")
        is_schema_fail = (not tool_schema_valid) or bool(tool_reason) or bool(args_reason) or error.get(
            "type"
        ) == "ToolValidationError"
        if is_schema_fail:
            schema_fail_count += 1
            semantic_error_count += 1
            continue

        if not transport_ok:
            transport_error_count += 1
            semantic_error_count += 1
            continue

        execution_attempt_count += 1
        if tool_execution_success:
            execution_success_count += 1
        if tool_semantic_success:
            semantic_success_count += 1
        else:
            semantic_error_count += 1
        if semantic_unknown:
            semantic_unknown_count += 1

    return {
        "total_tool_results": total_tool_results,
        "schema_fail_count": schema_fail_count,
        "transport_error_count": transport_error_count,
        "execution_attempt_count": execution_attempt_count,
        "execution_success_count": execution_success_count,
        "semantic_success_count": semantic_success_count,
        "semantic_error_count": semantic_error_count,
        "semantic_unknown_count": semantic_unknown_count,
    }


def _compute_format_reward(action_stats: dict[str, int]) -> float:
    num_steps = int(action_stats.get("num_steps") or 0)
    num_invalid = int(action_stats.get("num_invalid") or 0)
    if num_steps <= 0:
        return -0.2
    invalid_ratio = num_invalid / max(1, num_steps)
    return clamp(0.2 - 0.45 * invalid_ratio, -0.4, 0.2)


def _compute_action_valid_reward(action_stats: dict[str, int]) -> float:
    num_steps = int(action_stats.get("num_steps") or 0)
    if num_steps <= 0:
        return -0.2
    valid_ratio = float(action_stats.get("valid_count") or 0) / max(1, num_steps)
    strict_ratio = float(action_stats.get("strict_valid_count") or 0) / max(1, num_steps)
    return clamp(valid_ratio * 0.2 + strict_ratio * 0.1 - 0.15, -0.3, 0.3)


def _compute_tool_schema_reward(tool_stats: dict[str, int]) -> float:
    total = int(tool_stats.get("total_tool_results") or 0)
    schema_fail_count = int(tool_stats.get("schema_fail_count") or 0)

    if total == 0:
        return -0.05

    schema_ok_ratio = (total - schema_fail_count) / total
    reward = schema_ok_ratio * 0.2 - (1.0 - schema_ok_ratio) * 0.15
    return clamp(reward, -0.25, 0.2)


def _compute_tool_execution_reward(tool_stats: dict[str, int]) -> float:
    execution_attempt_count = int(tool_stats.get("execution_attempt_count") or 0)
    execution_success_count = int(tool_stats.get("execution_success_count") or 0)
    transport_error_count = int(tool_stats.get("transport_error_count") or 0)
    if execution_attempt_count <= 0:
        return -0.05

    success_ratio = execution_success_count / execution_attempt_count
    transport_penalty = min(0.15, 0.03 * transport_error_count)
    reward = success_ratio * 0.2 - (1.0 - success_ratio) * 0.1 - transport_penalty
    return clamp(reward, -0.2, 0.2)


def _compute_tool_semantic_reward(tool_stats: dict[str, int]) -> float:
    execution_attempt_count = int(tool_stats.get("execution_attempt_count") or 0)
    semantic_success_count = int(tool_stats.get("semantic_success_count") or 0)
    semantic_unknown_count = int(tool_stats.get("semantic_unknown_count") or 0)
    if execution_attempt_count <= 0:
        return -0.05

    success_ratio = semantic_success_count / execution_attempt_count
    unknown_penalty = min(0.2, 0.04 * semantic_unknown_count)
    reward = success_ratio * 0.25 - (1.0 - success_ratio) * 0.15 - unknown_penalty
    return clamp(reward, -0.35, 0.25)


def _compute_parse_recovery_penalty(action_stats: dict[str, int]) -> float:
    num_parse_recovery = int(action_stats.get("num_parse_recovery") or 0)
    if num_parse_recovery <= 0:
        return 0.0
    return -min(0.4, 0.08 * num_parse_recovery)


def _compute_progress_reward(trace: dict[str, Any]) -> float:
    actions = _actions(trace)
    observations = _observations(trace)

    repeated_actions = 0
    last_raw = None
    for action in actions:
        if not isinstance(action, dict):
            continue
        raw = action.get("raw_response")
        if isinstance(raw, str) and raw == last_raw:
            repeated_actions += 1
        last_raw = raw if isinstance(raw, str) else None

    reward = 0.0
    if observations:
        reward += 0.05
    if repeated_actions > 0:
        reward -= min(0.2, repeated_actions * 0.05)

    return clamp(reward, -0.2, 0.1)


def _compute_final_reward(trace: dict[str, Any], label: dict[str, Any]) -> float:
    final_answer = trace.get("final_answer")
    if not isinstance(final_answer, dict):
        return -0.1

    reward = 0.2

    result = final_answer.get("result")
    ranked_local = final_answer.get("ranked_molecules")
    if isinstance(result, dict) and isinstance(result.get("ranked_molecules"), list) and result.get("ranked_molecules"):
        reward += 0.1
    if isinstance(ranked_local, list) and ranked_local:
        reward += 0.05

    ground_truth = label.get("ground_truth")
    if isinstance(ground_truth, str) and ground_truth.strip():
        flat_answer = json.dumps(final_answer, ensure_ascii=False)
        if ground_truth in flat_answer:
            reward += 0.25

    expected = label.get("expected") if isinstance(label.get("expected"), dict) else {}
    if expected.get("answer_hit_pass") is True:
        reward += 0.1

    return clamp(reward, -0.2, 0.5)


def _compute_efficiency_reward(trace: dict[str, Any], max_steps: int | None = None) -> float:
    done_reason = str(trace.get("done_reason") or "")
    num_steps = int(trace.get("num_steps") or 0)

    reward = 0.0
    if done_reason == "final_answer":
        reward += 0.1
    if done_reason in {"length", "max_steps", "fatal_error", "abort"}:
        reward -= 0.1

    if isinstance(max_steps, int) and max_steps > 0:
        over = max(0, num_steps - (max_steps // 2))
        reward -= min(0.15, over * 0.01)

    return clamp(reward, -0.2, 0.1)


async def _reward_one(args, sample: Sample, **kwargs) -> dict[str, Any]:
    if not isinstance(sample, Sample):
        raise TypeError("sample must be a Sample instance")

    trace = _extract_trace(sample)
    label = _label_dict(sample)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    env_kwargs = metadata.get("env_kwargs") if isinstance(metadata.get("env_kwargs"), dict) else {}

    max_steps = env_kwargs.get("max_steps") if isinstance(env_kwargs.get("max_steps"), int) else None
    action_stats = _action_stats(trace)
    tool_stats = _tool_stats(trace)

    components = {
        "format_reward": _compute_format_reward(action_stats),
        "action_valid_reward": _compute_action_valid_reward(action_stats),
        "tool_schema_reward": _compute_tool_schema_reward(tool_stats),
        "tool_execution_reward": _compute_tool_execution_reward(tool_stats),
        "tool_semantic_reward": _compute_tool_semantic_reward(tool_stats),
        "parse_recovery_penalty": _compute_parse_recovery_penalty(action_stats),
        "progress_reward": _compute_progress_reward(trace),
        "final_reward": _compute_final_reward(trace, label),
        "efficiency_reward": _compute_efficiency_reward(trace, max_steps=max_steps),
    }
    components["tool_reward"] = (
        components["tool_schema_reward"] + components["tool_execution_reward"] + components["tool_semantic_reward"]
    )

    score = (
        components["format_reward"]
        + components["action_valid_reward"]
        + components["tool_schema_reward"]
        + components["tool_execution_reward"]
        + components["tool_semantic_reward"]
        + components["parse_recovery_penalty"]
        + components["progress_reward"]
        + components["final_reward"]
        + components["efficiency_reward"]
    )
    score = clamp(score, -1.0, 1.0)

    diagnostics = {
        "num_steps": int(action_stats.get("num_steps") or 0),
        "num_invalid": int(action_stats.get("num_invalid") or 0),
        "num_parse_recovery": int(action_stats.get("num_parse_recovery") or 0),
        "strict_valid_count": int(action_stats.get("strict_valid_count") or 0),
        "recovered_valid_count": int(action_stats.get("recovered_valid_count") or 0),
        "tool_schema_fail_count": int(tool_stats.get("schema_fail_count") or 0),
        "tool_transport_error_count": int(tool_stats.get("transport_error_count") or 0),
        "tool_execution_attempt_count": int(tool_stats.get("execution_attempt_count") or 0),
        "tool_execution_success_count": int(tool_stats.get("execution_success_count") or 0),
        "tool_semantic_success_count": int(tool_stats.get("semantic_success_count") or 0),
        "tool_semantic_error_count": int(tool_stats.get("semantic_error_count") or 0),
        "tool_semantic_unknown_count": int(tool_stats.get("semantic_unknown_count") or 0),
        "num_tool_success": int(trace.get("num_tool_success") or 0),
        "num_tool_error": int(trace.get("num_tool_error") or 0),
        "num_tool_schema_error": int(trace.get("num_tool_schema_error") or tool_stats.get("schema_fail_count") or 0),
        "num_tool_execution_success": int(
            trace.get("num_tool_execution_success") or tool_stats.get("execution_success_count") or 0
        ),
        "num_tool_semantic_error": int(
            trace.get("num_tool_semantic_error") or tool_stats.get("semantic_error_count") or 0
        ),
        "num_tool_semantic_unknown": int(
            trace.get("num_tool_semantic_unknown") or tool_stats.get("semantic_unknown_count") or 0
        ),
        "num_transport_error": int(trace.get("num_transport_error") or tool_stats.get("transport_error_count") or 0),
        "num_final_answer": 1 if isinstance(trace.get("final_answer"), dict) else 0,
        "num_max_steps": 1 if trace.get("done_reason") == "max_steps" else 0,
        "done_reason": trace.get("done_reason"),
        "rollout_mode": trace.get("rollout_mode"),
        "parse_recovery_enabled": bool(trace.get("parse_recovery_enabled")),
        "status": sample.status.value,
    }

    out = {
        "score": score,
        "components": components,
        "diagnostics": diagnostics,
        "error_category": trace.get("done_reason") or "none",
    }

    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata["drug_agent_reward"] = to_jsonable(out)

    return out


async def reward_func(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        outputs = []
        for sample in sample_or_samples:
            outputs.append(await _reward_one(args, sample, **kwargs))
        return outputs

    return await _reward_one(args, sample_or_samples, **kwargs)
