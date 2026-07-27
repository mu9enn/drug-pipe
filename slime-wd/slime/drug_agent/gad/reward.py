from __future__ import annotations

import os
from typing import Any

from drug_agent.toolrl.parse_tool_calls import parse_tool_calls
from drug_agent.utils import clamp, to_jsonable


def _teacher_response(sample: Any) -> str:
    label = sample.label if isinstance(sample.label, dict) else {}
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return str(label.get("teacher_response") or metadata.get("teacher_response") or "")


def _state_messages(sample: Any) -> list[dict[str, Any]]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    label = sample.label if isinstance(sample.label, dict) else {}
    state = metadata.get("state_messages") or label.get("state_messages")
    if isinstance(state, list):
        return state
    if isinstance(sample.prompt, list):
        return sample.prompt
    return []


def _rule_components(args, sample: Any) -> tuple[float, float, dict[str, Any]]:
    """Compute strict format and decision-aware schema components."""
    parsed = parse_tool_calls(
        sample.response if isinstance(sample.response, str) else str(sample.response or ""),
        allowed_tool_names=None,
        keep_non_molclaw=True,
    )
    parse_ok = bool(parsed.get("ok"))
    label = sample.label if isinstance(sample.label, dict) else {}
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    decision_type = label.get("decision_type") or metadata.get("decision_type")
    format_score = 1.0 if parse_ok else 0.0
    if decision_type == "final_answer":
        schema_score = (
            1.0
            if parse_ok
            and parsed.get("has_final_answer")
            and int(parsed.get("tool_call_count") or 0) == 0
            else 0.0
        )
    else:
        schema_score = (
            1.0
            if parse_ok
            and int(parsed.get("molclaw_tool_call_count") or 0) > 0
            and not parsed.get("has_final_answer")
            and int(parsed.get("non_molclaw_tool_call_count") or 0) == 0
            else 0.0
        )
    rule = {
        "score": 0.5 * format_score + 0.5 * schema_score,
        "components": {"format": format_score, "decision_schema": schema_score},
        "diagnostics": {
            "parse_ok": parse_ok,
            "expected_decision_type": decision_type,
            "has_final_answer": bool(parsed.get("has_final_answer")),
            "molclaw_tool_call_count": int(parsed.get("molclaw_tool_call_count") or 0),
        },
    }
    return format_score, schema_score, rule


async def reward_func(args, sample_or_samples, **kwargs):
    samples = sample_or_samples if isinstance(sample_or_samples, list) else [sample_or_samples]
    items = []
    for sample in samples:
        teacher_response = _teacher_response(sample)
        if not teacher_response:
            raise ValueError("GAD sample is missing teacher_response in label/metadata")
        state_messages = _state_messages(sample)
        if not state_messages:
            raise ValueError("GAD sample metadata is missing the original state_messages list")
        items.append(
            {
                "sample_id": (sample.metadata or {}).get("sample_id"),
                "state_messages": state_messages,
                "teacher_response": teacher_response,
                "student_response": sample.response if isinstance(sample.response, str) else str(sample.response or ""),
                "student_weight_versions": sample.weight_versions,
            }
        )
    reward_mode = os.environ.get("GAD_REWARD_MODE", "pure").strip().lower()
    if reward_mode not in {"pure", "rule", "hybrid"}:
        raise ValueError(f"unsupported GAD_REWARD_MODE: {reward_mode}")
    result = {
        "normalized_scores": [0.0] * len(items),
        "raw_scores": [0.0] * len(items),
        "version_before": None,
        "version_after": None,
        "metrics": {},
    }
    if reward_mode in {"pure", "hybrid"}:
        raw_url = os.environ.get("GAD_DISCRIMINATOR_URL")
        if not raw_url:
            raise RuntimeError(f"GAD_DISCRIMINATOR_URL is required for {reward_mode} GAD reward")
        url = raw_url.rstrip("/")
        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=float(os.environ.get("GAD_RM_TIMEOUT_SEC", "600")))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{url}/score-and-update", json={"items": items}) as response:
                    response.raise_for_status()
                    result = await response.json()
        except Exception as exc:
            raise RuntimeError(f"GAD discriminator request failed: {type(exc).__name__}: {exc}") from exc

    gad_coef = float(os.environ.get("GAD_REWARD_COEF", "0.8"))
    format_coef = float(os.environ.get("GAD_FORMAT_REWARD_COEF", "0.1"))
    tool_coef = float(os.environ.get("GAD_TOOL_REWARD_COEF", "0.1"))
    final_clip = float(os.environ.get("GAD_FINAL_REWARD_CLIP", "2.0"))
    outputs = []
    for sample, gad_score, raw_score in zip(
        samples, result["normalized_scores"], result["raw_scores"], strict=True
    ):
        format_score, tool_score, rule = _rule_components(args, sample)
        if reward_mode == "pure":
            score = clamp(gad_score, -final_clip, final_clip)
        elif reward_mode == "rule":
            score = clamp(0.5 * format_score + 0.5 * tool_score, -final_clip, final_clip)
        else:
            score = clamp(gad_coef * gad_score + format_coef * format_score + tool_coef * tool_score, -final_clip, final_clip)
        out = {
            "score": score,
            "components": {
                "gad": gad_score,
                "gad_raw": raw_score,
                "format": format_score,
                "tool_schema": tool_score,
            },
            "diagnostics": {
                "reward_mode": reward_mode,
                "discriminator_version_before": result["version_before"],
                "discriminator_version_after": result["version_after"],
                "student_weight_versions": to_jsonable(sample.weight_versions),
                "discriminator_metrics": result.get("metrics") or {},
                "rule_reward": to_jsonable(rule),
            },
        }
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        sample.metadata["gad_reward"] = to_jsonable(out)
        outputs.append(out)
    return outputs if isinstance(sample_or_samples, list) else outputs[0]
