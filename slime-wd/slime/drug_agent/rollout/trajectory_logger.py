from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

from drug_agent.constants import DEFAULT_RUN_NAME, SLIME_DRUG_RUNS_ROOT
from drug_agent.utils import append_jsonl, ensure_dir, to_jsonable, utc_now_iso


def _get_run_name(args) -> str:
    env_name = os.environ.get("DRUG_AGENT_RUN_NAME")
    if env_name:
        return env_name

    wandb_group = getattr(args, "wandb_group", None)
    if isinstance(wandb_group, str) and wandb_group.strip():
        return wandb_group.strip()

    save_dir = getattr(args, "save", None)
    if isinstance(save_dir, str) and save_dir.strip():
        return Path(save_dir).name

    return DEFAULT_RUN_NAME


def _reward_value(sample: Sample) -> float | None:
    reward = sample.reward
    if isinstance(reward, (int, float)):
        return float(reward)
    if isinstance(reward, dict):
        score = reward.get("score")
        if isinstance(score, (int, float)):
            return float(score)
    return None


def _reward_components(sample: Sample) -> dict[str, Any]:
    reward = sample.reward
    if isinstance(reward, dict) and isinstance(reward.get("components"), dict):
        return to_jsonable(reward["components"])

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    r = metadata.get("drug_agent_reward")
    if isinstance(r, dict) and isinstance(r.get("components"), dict):
        return to_jsonable(r["components"])

    return {}


def _trace_of(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    trace = metadata.get("drug_agent_trace")
    if isinstance(trace, dict):
        return trace
    return {}


def _build_row(sample: Sample, rollout_id: int) -> dict[str, Any]:
    trace = _trace_of(sample)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}

    return {
        "timestamp": utc_now_iso(),
        "rollout_id": rollout_id,
        "task_id": trace.get("task_id") or metadata.get("task_id"),
        "task_type": trace.get("task_type") or metadata.get("task_type"),
        "data_source": trace.get("data_source") or metadata.get("data_source"),
        "rollout_mode": trace.get("rollout_mode"),
        "parse_recovery_enabled": bool(trace.get("parse_recovery_enabled")),
        "prompt": to_jsonable(sample.prompt),
        "actions": to_jsonable(trace.get("actions") or []),
        "observations": to_jsonable(trace.get("observations") or []),
        "final_answer": to_jsonable(trace.get("final_answer")),
        "reward": _reward_value(sample),
        "reward_components": _reward_components(sample),
        "done_reason": trace.get("done_reason"),
        "num_steps": int(trace.get("num_steps") or 0),
        "num_invalid": int(trace.get("num_invalid") or 0),
        "num_parse_recovery": int(trace.get("num_parse_recovery") or 0),
        "strict_valid_count": int(trace.get("strict_valid_count") or 0),
        "recovered_valid_count": int(trace.get("recovered_valid_count") or 0),
        "num_tool_success": int(trace.get("num_tool_success") or 0),
        "num_tool_error": int(trace.get("num_tool_error") or 0),
        "num_tool_schema_error": int(trace.get("num_tool_schema_error") or 0),
        "num_tool_execution_success": int(trace.get("num_tool_execution_success") or 0),
        "num_tool_semantic_error": int(trace.get("num_tool_semantic_error") or 0),
        "num_tool_semantic_unknown": int(trace.get("num_tool_semantic_unknown") or 0),
        "num_transport_error": int(trace.get("num_transport_error") or 0),
        "strict_success_rate": float(trace.get("strict_success_rate") or 0.0),
        "recovery_success_rate": float(trace.get("recovery_success_rate") or 0.0),
        "tool_execution_success_rate": float(trace.get("tool_execution_success_rate") or 0.0),
        "truncated": bool(trace.get("truncated") or sample.status == Sample.Status.TRUNCATED),
        "error": trace.get("error"),
        "sample_status": sample.status.value,
    }


def _inject_metrics(rollout_extra_metrics: dict[str, Any] | None, rows: list[dict[str, Any]]) -> None:
    if rollout_extra_metrics is None:
        return
    if not rows:
        return

    total_actions = sum(int(r.get("num_steps") or 0) for r in rows)
    total_invalid = sum(int(r.get("num_invalid") or 0) for r in rows)
    total_strict_valid = sum(int(r.get("strict_valid_count") or 0) for r in rows)
    total_recovered_valid = sum(int(r.get("recovered_valid_count") or 0) for r in rows)
    total_tool_success = sum(int(r.get("num_tool_success") or 0) for r in rows)
    total_tool_error = sum(int(r.get("num_tool_error") or 0) for r in rows)
    total_tool_schema_error = sum(int(r.get("num_tool_schema_error") or 0) for r in rows)
    total_tool_execution_success = sum(int(r.get("num_tool_execution_success") or 0) for r in rows)
    total_tool_semantic_unknown = sum(int(r.get("num_tool_semantic_unknown") or 0) for r in rows)

    action_valid_rate = (total_actions - total_invalid) / max(1, total_actions)
    strict_success_rate = total_strict_valid / max(1, total_actions)
    recovery_success_rate = total_recovered_valid / max(1, total_actions)
    total_tool_calls = total_tool_success + total_tool_error
    execution_attempt_count = total_tool_calls - total_tool_schema_error
    tool_success_rate = total_tool_success / max(1, total_tool_calls)
    tool_execution_success_rate = total_tool_execution_success / max(1, execution_attempt_count)
    tool_semantic_unknown_rate = total_tool_semantic_unknown / max(1, execution_attempt_count)
    final_success_rate = sum(1 for r in rows if r.get("done_reason") == "final_answer") / max(1, len(rows))

    rollout_extra_metrics["action_valid_rate"] = action_valid_rate
    rollout_extra_metrics["strict_success_rate"] = strict_success_rate
    rollout_extra_metrics["recovery_success_rate"] = recovery_success_rate
    rollout_extra_metrics["tool_success_rate"] = tool_success_rate
    rollout_extra_metrics["tool_execution_success_rate"] = tool_execution_success_rate
    rollout_extra_metrics["tool_semantic_unknown_rate"] = tool_semantic_unknown_rate
    rollout_extra_metrics["final_success_rate"] = final_success_rate


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Custom rollout logger hook used by slime.

    Return False so slime keeps default logging in addition to this JSONL trace.
    """
    try:
        run_name = _get_run_name(args)
        out_dir = ensure_dir(SLIME_DRUG_RUNS_ROOT / run_name)
        out_path = out_dir / "trajectories.jsonl"

        rows = [_build_row(sample, rollout_id=rollout_id) for sample in samples]
        for row in rows:
            append_jsonl(out_path, row)

        _inject_metrics(rollout_extra_metrics, rows)
        if rollout_extra_metrics is not None:
            rollout_extra_metrics["trajectory_log_path"] = str(out_path)
            rollout_extra_metrics["trajectory_rollout_time_sec"] = float(rollout_time)

        return False
    except Exception as exc:
        if rollout_extra_metrics is not None:
            rollout_extra_metrics["trajectory_logger_error"] = f"{type(exc).__name__}: {exc}"
        return False
