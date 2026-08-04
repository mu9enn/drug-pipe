from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from drug_agent.toolrl.normalization import (
    canonical_argument_map,
    canonical_tool_name,
    compare_values,
    load_tool_schema_config,
    normalize_value,
)
from drug_agent.toolrl.parse_tool_calls import parse_tool_calls
from drug_agent.utils import clamp, to_jsonable


@dataclass
class ToolCallScore:
    pred_index: int
    gold_index: int
    score: float
    tool_name_score: float
    param_name_score: float
    param_value_score: float
    matched_param_count: int
    pred_param_count: int
    gold_param_count: int
    pred_tool_name: str
    gold_tool_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pred_index": self.pred_index,
            "gold_index": self.gold_index,
            "score": self.score,
            "tool_name_score": self.tool_name_score,
            "param_name_score": self.param_name_score,
            "param_value_score": self.param_value_score,
            "matched_param_count": self.matched_param_count,
            "pred_param_count": self.pred_param_count,
            "gold_param_count": self.gold_param_count,
            "pred_tool_name": self.pred_tool_name,
            "gold_tool_name": self.gold_tool_name,
        }


def _label_dict(sample: Any) -> dict[str, Any]:
    if isinstance(sample.label, dict):
        return sample.label
    if isinstance(sample.label, str):
        try:
            payload = json.loads(sample.label)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    label = metadata.get("label")
    if isinstance(label, dict):
        return label
    return {}


def _extract_gold_tool_calls(sample: Any) -> list[dict[str, Any]]:
    label = _label_dict(sample)
    candidates = [
        label.get("target_tool_calls"),
        label.get("tool_calls"),
        label.get("expected_tool_calls"),
    ]
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    candidates.extend(
        [
            metadata.get("target_tool_calls"),
            metadata.get("tool_calls"),
            metadata.get("expected_tool_calls"),
        ]
    )
    for candidate in candidates:
        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except Exception:
                continue
        if isinstance(candidate, list):
            out = [item for item in candidate if isinstance(item, dict)]
            if out:
                return out
    return []


def _extract_response_text(sample: Any) -> str:
    response = sample.response
    if isinstance(response, str):
        return response
    return str(response or "")


def _decision_type(sample: Any) -> str:
    label = _label_dict(sample)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return str(label.get("decision_type") or metadata.get("decision_type") or "tool_call")


def _target_final_answer(sample: Any) -> Any:
    label = _label_dict(sample)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return label.get("target_final_answer", metadata.get("target_final_answer"))


def _without_summary(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_summary(item) for key, item in value.items() if key != "summary"}
    if isinstance(value, list):
        return [_without_summary(item) for item in value]
    return value


def _official_match_score(left: list[Any], right: list[Any]) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    left_count = Counter(json.dumps(item, sort_keys=True, ensure_ascii=False) for item in left)
    right_count = Counter(json.dumps(item, sort_keys=True, ensure_ascii=False) for item in right)
    intersection = sum(min(left_count[key], right_count[key]) for key in left_count.keys() & right_count.keys())
    union = len(left) + len(right) - intersection
    return intersection / union if union else 0.0


def _official_tool_correctness(pred_calls: list[dict[str, Any]], gold_calls: list[dict[str, Any]]) -> float:
    """Faithful adaptation of ToolRL's compute_tool_call_reward to canonical call keys."""
    def normalized(call: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": str(call.get("tool_name") or call.get("name") or ""),
            "parameters": call.get("arguments") if isinstance(call.get("arguments"), dict) else (
                call.get("parameters") if isinstance(call.get("parameters"), dict) else {}
            ),
        }

    gold = [normalized(call) for call in gold_calls]
    pred = [normalized(call) for call in pred_calls]
    if gold == pred:
        return 3.0
    score = _official_match_score([item["name"] for item in gold], [item["name"] for item in pred])
    local_max = 1.0
    used_pred: set[int] = set()
    for gold_call in gold:
        gold_name = gold_call["name"]
        gold_params = gold_call["parameters"]
        local_max += 1.0 + len(gold_params)
        best_score = 0.0
        best_index = -1
        for index, pred_call in enumerate(pred):
            if index in used_pred or pred_call["name"] != gold_name:
                continue
            pred_params = pred_call["parameters"]
            param_score = _official_match_score(list(gold_params), list(pred_params))
            value_score = sum(
                1.0 for key, value in gold_params.items()
                if key in pred_params and pred_params[key] == value
            )
            candidate = param_score + value_score
            if candidate > best_score:
                best_score = candidate
                best_index = index
        if best_index >= 0:
            used_pred.add(best_index)
            score += best_score
    return 6.0 * score / local_max - 3.0


def _official_reward(sample: Any, parsed: dict[str, Any], pred_calls: list[dict[str, Any]], gold_calls: list[dict[str, Any]]) -> dict[str, Any]:
    expected = _decision_type(sample)
    predicted = "final_answer" if parsed.get("has_final_answer") and not parsed.get("has_tool_call") else (
        "tool_call" if parsed.get("has_tool_call") and not parsed.get("has_final_answer") else "invalid"
    )
    format_score = 1.0 if parsed.get("ok") and predicted == expected else 0.0
    if expected == "final_answer":
        gold_final = _without_summary(_target_final_answer(sample))
        pred_final = _without_summary(parsed.get("final_answer"))
        correctness = 3.0 if format_score and pred_final == gold_final else -3.0
        extension = "drug_pipe_terminal_decision_extension"
    else:
        correctness = _official_tool_correctness(pred_calls, gold_calls) if format_score else -3.0
        extension = None
    return {
        "score": format_score + correctness,
        "format": format_score,
        "components": {"format": format_score, "correctness": correctness},
        "diagnostics": {
            "reward_mode": "official",
            "expected_decision_type": expected,
            "predicted_decision_type": predicted,
            "official_toolrl_extension": extension,
            "parse_ok": bool(parsed.get("ok")),
            "pred_call_count": len(pred_calls),
            "gold_call_count": len(gold_calls),
        },
        "errors": [] if format_score else [{"type": "DecisionFormatMismatch", "message": f"expected {expected}, got {predicted}"}],
        "warnings": [],
    }


def _format_reward(parsed: dict[str, Any]) -> float:
    if not parsed.get("ok"):
        return -0.3
    tool_calls = parsed.get("molclaw_tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        if parsed.get("has_final_answer"):
            return 0.35
        return 0.2
    if parsed.get("non_molclaw_tool_calls"):
        return 0.85
    return 1.0


def _molclaw_final_answer_reward(sample: Any, parsed: dict[str, Any]) -> dict[str, Any]:
    """Score terminal decisions without treating an empty call set as correct.

    The dense tool-call scorer intentionally gives full tool-component credit
    when both predicted and gold call lists are empty.  That is correct for a
    *valid* no-call decision, but terminal rows carry their supervision in
    ``target_final_answer`` instead.  Applying the tool scorer to those rows
    lets malformed text with no final-answer block receive a positive reward.
    Keep terminal decisions on the same [-0.5, 1.0] MolClaw scale while
    requiring both canonical ReAct format and exact structured output (with
    the duplicated human-readable summary ignored, as in official ToolRL).
    """
    has_only_final = bool(
        parsed.get("ok")
        and parsed.get("has_final_answer")
        and not parsed.get("has_tool_call")
    )
    target = _without_summary(_target_final_answer(sample))
    predicted = _without_summary(parsed.get("final_answer"))
    exact = bool(has_only_final and predicted == target)
    score = 1.0 if exact else -0.5
    format_reward = 1.0 if has_only_final else -0.3
    correctness = 1.0 if exact else 0.0
    error_type = None
    error_message = None
    if not has_only_final:
        error_type = parsed.get("error_type") or "TerminalDecisionFormatMismatch"
        error_message = parsed.get("error_message") or "expected one valid final_answer decision"
    elif not exact:
        error_type = "FinalAnswerMismatch"
        error_message = "predicted final answer does not match target_final_answer"

    return {
        "score": score,
        "format": format_reward,
        "tool_name": 0.0,
        "param_name": 0.0,
        "param_value": 0.0,
        "matched_calls": 0.0,
        "components": {
            "format": format_reward,
            "terminal_correctness": correctness,
            "tool_name": 0.0,
            "param_name": 0.0,
            "param_value": 0.0,
            "matched_calls": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tool_call_score": 0.0,
        },
        "diagnostics": {
            "reward_mode": "molclaw",
            "expected_decision_type": "final_answer",
            "predicted_decision_type": "final_answer" if has_only_final else "invalid",
            "parse_ok": bool(parsed.get("ok")),
            "has_final_answer": bool(parsed.get("has_final_answer")),
            "terminal_exact_match": exact,
            "pred_call_count": 0,
            "gold_call_count": 0,
            "matched_calls": 0,
            "unmatched_pred_count": 0,
            "unmatched_gold_count": 0,
        },
        "errors": [] if exact else [{"type": error_type, "message": error_message}],
        "warnings": [],
    }


def _pair_tool_calls(pred: list[dict[str, Any]], gold: list[dict[str, Any]], config: dict[str, Any]) -> list[ToolCallScore]:
    if not pred or not gold:
        return []

    score_matrix: list[list[float]] = []
    detail_matrix: list[list[dict[str, Any]]] = []
    for pred_idx, pred_call in enumerate(pred):
        pred_name = canonical_tool_name(pred_call.get("tool_name"), config)
        pred_args = canonical_argument_map(pred_call.get("arguments") or {}, tool_name=pred_name, config=config)
        row = []
        details_row = []
        for gold_idx, gold_call in enumerate(gold):
            gold_name = canonical_tool_name(gold_call.get("tool_name"), config)
            gold_args = canonical_argument_map(gold_call.get("arguments") or {}, tool_name=gold_name, config=config)
            name_score = 1.0 if pred_name == gold_name else 0.0
            pred_keys = set(pred_args.keys())
            gold_keys = set(gold_args.keys())
            if pred_keys or gold_keys:
                matched_keys = pred_keys & gold_keys
                key_precision = len(matched_keys) / max(1, len(pred_keys))
                key_recall = len(matched_keys) / max(1, len(gold_keys))
                if key_precision + key_recall == 0:
                    param_name_score = 0.0
                else:
                    param_name_score = 2 * key_precision * key_recall / (key_precision + key_recall)
                value_scores = []
                for key in matched_keys:
                    value_scores.append(
                        compare_values(
                            pred_args.get(key),
                            gold_args.get(key),
                            tool_name=pred_name,
                            param_name=key,
                            config=config,
                        )["score"]
                    )
                param_value_score = sum(value_scores) / len(value_scores) if value_scores else (1.0 if not pred_keys and not gold_keys else 0.0)
            else:
                param_name_score = 1.0
                param_value_score = 1.0

            score = 0.45 * name_score + 0.25 * param_name_score + 0.30 * param_value_score
            row.append(score)
            details_row.append(
                {
                    "tool_name_score": name_score,
                    "param_name_score": param_name_score,
                    "param_value_score": param_value_score,
                    "pred_tool_name": pred_name,
                    "gold_tool_name": gold_name,
                    "pred_param_count": len(pred_keys),
                    "gold_param_count": len(gold_keys),
                    "matched_param_count": len(pred_keys & gold_keys),
                }
            )
        score_matrix.append(row)
        detail_matrix.append(details_row)

    assignments = _best_assignment(score_matrix)
    scores: list[ToolCallScore] = []
    for pred_idx, gold_idx in assignments:
        details = detail_matrix[pred_idx][gold_idx]
        # Parameter overlap must never turn a different tool name into a
        # matched call. Runtime aliases are intentionally absent, so the
        # reward boundary must preserve the same exact-name contract.
        if details["tool_name_score"] == 0.0:
            continue
        scores.append(
            ToolCallScore(
                pred_index=pred_idx,
                gold_index=gold_idx,
                score=score_matrix[pred_idx][gold_idx],
                tool_name_score=details["tool_name_score"],
                param_name_score=details["param_name_score"],
                param_value_score=details["param_value_score"],
                matched_param_count=details["matched_param_count"],
                pred_param_count=details["pred_param_count"],
                gold_param_count=details["gold_param_count"],
                pred_tool_name=details["pred_tool_name"],
                gold_tool_name=details["gold_tool_name"],
            )
        )
    return scores


def _best_assignment(score_matrix: list[list[float]]) -> list[tuple[int, int]]:
    if not score_matrix or not score_matrix[0]:
        return []

    n = len(score_matrix)
    m = len(score_matrix[0])
    if n == 0 or m == 0:
        return []

    if max(n, m) > 8:
        used_gold = set()
        greedy: list[tuple[int, int]] = []
        candidates = []
        for i, row in enumerate(score_matrix):
            for j, score in enumerate(row):
                candidates.append((score, i, j))
        for score, i, j in sorted(candidates, reverse=True):
            if i in {x for x, _ in greedy} or j in used_gold:
                continue
            greedy.append((i, j))
            used_gold.add(j)
            if len(greedy) == min(n, m):
                break
        return greedy

    if n > m:
        transposed = [[score_matrix[i][j] for i in range(n)] for j in range(m)]
        pairs = _best_assignment(transposed)
        return [(j, i) for i, j in pairs]

    # n <= m: assign each prediction to a unique gold index, maximizing total score.
    best_total = float("-inf")
    best_pairs: list[tuple[int, int]] = []

    def backtrack(i: int, used_mask: int, total: float, pairs: list[tuple[int, int]]) -> None:
        nonlocal best_total, best_pairs
        if i == n:
            if total > best_total:
                best_total = total
                best_pairs = pairs.copy()
            return

        # Allow skipping a prediction so the final F1 can penalize unmatched calls.
        backtrack(i + 1, used_mask, total, pairs)

        for j in range(m):
            if used_mask & (1 << j):
                continue
            pairs.append((i, j))
            backtrack(i + 1, used_mask | (1 << j), total + score_matrix[i][j], pairs)
            pairs.pop()

    backtrack(0, 0, 0.0, [])
    return best_pairs


def _compute_pair_metrics(
    pred_calls: list[dict[str, Any]],
    gold_calls: list[dict[str, Any]],
    pair_scores: list[ToolCallScore],
) -> dict[str, Any]:
    matched_pred = {pair.pred_index for pair in pair_scores}
    matched_gold = {pair.gold_index for pair in pair_scores}

    precision = len(matched_pred) / max(1, len(pred_calls))
    recall = len(matched_gold) / max(1, len(gold_calls))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    avg_tool_name = sum(pair.tool_name_score for pair in pair_scores) / len(pair_scores) if pair_scores else 0.0
    avg_param_name = sum(pair.param_name_score for pair in pair_scores) / len(pair_scores) if pair_scores else 0.0
    avg_param_value = sum(pair.param_value_score for pair in pair_scores) / len(pair_scores) if pair_scores else 0.0
    avg_pair_score = sum(pair.score for pair in pair_scores) / len(pair_scores) if pair_scores else 0.0
    unmatched_pred = [idx for idx in range(len(pred_calls)) if idx not in matched_pred]
    unmatched_gold = [idx for idx in range(len(gold_calls)) if idx not in matched_gold]

    return {
        "matched_calls": len(pair_scores),
        "pred_call_count": len(pred_calls),
        "gold_call_count": len(gold_calls),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_tool_name_score": avg_tool_name,
        "avg_param_name_score": avg_param_name,
        "avg_param_value_score": avg_param_value,
        "avg_pair_score": avg_pair_score,
        "unmatched_pred_indices": unmatched_pred,
        "unmatched_gold_indices": unmatched_gold,
    }


def _tool_reward_components(pred_calls: list[dict[str, Any]], gold_calls: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    pair_scores = _pair_tool_calls(pred_calls, gold_calls, config=config)
    metrics = _compute_pair_metrics(pred_calls, gold_calls, pair_scores)

    if not pred_calls and not gold_calls:
        metrics.update(
            {
                "tool_name_score": 1.0,
                "param_name_score": 1.0,
                "param_value_score": 1.0,
                "tool_call_score": 1.0,
            }
        )
        return metrics

    precision = metrics["precision"]
    recall = metrics["recall"]
    pair_quality = metrics["avg_pair_score"]
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = metrics["f1"]

    tool_name_score = metrics["avg_tool_name_score"] * f1
    param_name_score = metrics["avg_param_name_score"] * f1
    param_value_score = metrics["avg_param_value_score"] * f1
    tool_call_score = 0.2 * f1 + 0.35 * metrics["avg_tool_name_score"] + 0.2 * metrics["avg_param_name_score"] + 0.25 * metrics["avg_param_value_score"]
    metrics.update(
        {
            "tool_name_score": clamp(tool_name_score, 0.0, 1.0),
            "param_name_score": clamp(param_name_score, 0.0, 1.0),
            "param_value_score": clamp(param_value_score, 0.0, 1.0),
            "tool_call_score": clamp(tool_call_score, 0.0, 1.0),
            "pair_quality": pair_quality,
        }
    )
    return metrics


def _reward_one(args, sample: Any, **kwargs) -> dict[str, Any]:
    config = load_tool_schema_config()
    response_text = _extract_response_text(sample)
    parsed = parse_tool_calls(response_text, allowed_tool_names=None, keep_non_molclaw=True)
    gold_calls = _extract_gold_tool_calls(sample)

    pred_calls = parsed.get("molclaw_tool_calls") if isinstance(parsed.get("molclaw_tool_calls"), list) else []
    pred_calls = [item for item in pred_calls if isinstance(item, dict)]
    gold_calls = [item for item in gold_calls if isinstance(item, dict)]

    reward_mode = os.environ.get("TOOLRL_REWARD_MODE", "official").strip().lower()
    if reward_mode not in {"official", "molclaw"}:
        raise ValueError(f"unsupported TOOLRL_REWARD_MODE: {reward_mode}")
    if reward_mode == "official":
        out = _official_reward(sample, parsed, pred_calls, gold_calls)
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        sample.metadata["toolrl_reward"] = to_jsonable(out)
        return out

    # Terminal rows have no gold tool calls by construction.  They must be
    # scored against target_final_answer before the dense empty-call shortcut,
    # otherwise malformed/no-output generations receive positive reward.
    if _decision_type(sample) == "final_answer":
        out = _molclaw_final_answer_reward(sample, parsed)
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        sample.metadata["toolrl_reward"] = to_jsonable(out)
        return out

    format_reward = _format_reward(parsed)
    tool_metrics = _tool_reward_components(pred_calls, gold_calls, config=config)

    num_pred = len(pred_calls)
    num_gold = len(gold_calls)
    matched = int(tool_metrics.get("matched_calls") or 0)
    precision = float(tool_metrics.get("precision") or 0.0)
    recall = float(tool_metrics.get("recall") or 0.0)
    f1 = float(tool_metrics.get("f1") or 0.0)

    # Offline step-level ToolRL uses the reference tool calls as the target.
    # If the model emits the right call structure, pair quality drives most of
    # the reward; unmatched calls are still penalized via precision / recall.
    score = (
        0.15 * format_reward
        + 0.25 * f1
        + 0.20 * float(tool_metrics.get("tool_name_score") or 0.0)
        + 0.20 * float(tool_metrics.get("param_name_score") or 0.0)
        + 0.20 * float(tool_metrics.get("param_value_score") or 0.0)
    )
    if num_pred == 0 and num_gold > 0:
        score -= 0.15
    if num_pred > num_gold:
        score -= min(0.15, 0.03 * (num_pred - num_gold))
    if num_gold > num_pred:
        score -= min(0.15, 0.03 * (num_gold - num_pred))
    if parsed.get("non_molclaw_tool_calls"):
        score -= min(0.10, 0.02 * len(parsed.get("non_molclaw_tool_calls") or []))
    if not parsed.get("ok"):
        # Dense partial-call credit is useful only inside a valid ReAct
        # envelope.  Extra free text or malformed tags must never turn into a
        # positive training target even if a recoverable call happened to
        # match the gold call.
        score = min(score, -0.3)
    score = clamp(score, -0.5, 1.0)

    components = {
        "format": format_reward,
        "tool_name": float(tool_metrics.get("tool_name_score") or 0.0),
        "param_name": float(tool_metrics.get("param_name_score") or 0.0),
        "param_value": float(tool_metrics.get("param_value_score") or 0.0),
        "matched_calls": float(matched),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tool_call_score": float(tool_metrics.get("tool_call_score") or 0.0),
    }

    diagnostics = {
        "reward_mode": "molclaw",
        "pred_call_count": num_pred,
        "gold_call_count": num_gold,
        "matched_calls": matched,
        "unmatched_pred_count": max(0, num_pred - matched),
        "unmatched_gold_count": max(0, num_gold - matched),
        "parse_ok": bool(parsed.get("ok")),
        "parse_error_type": parsed.get("error_type"),
        "parse_error_message": parsed.get("error_message"),
        "has_final_answer": bool(parsed.get("has_final_answer")),
        "molclaw_tool_call_count": int(parsed.get("molclaw_tool_call_count") or 0),
        "non_molclaw_tool_call_count": int(parsed.get("non_molclaw_tool_call_count") or 0),
        "fence_wrappers_stripped": int(parsed.get("fence_wrappers_stripped") or 0),
        "fence_inner_content_preserved": int(parsed.get("fence_inner_content_preserved") or 0),
    }

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not parsed.get("ok"):
        errors.append({"type": parsed.get("error_type"), "message": parsed.get("error_message")})
    elif num_pred == 0 and num_gold > 0:
        warnings.append({"type": "NoPredictedToolCall", "message": "response did not contain any MolClaw tool calls"})
    if parsed.get("non_molclaw_tool_calls"):
        warnings.append(
            {
                "type": "FilteredNonMolclawToolCalls",
                "count": len(parsed.get("non_molclaw_tool_calls") or []),
            }
        )

    out = {
        "score": score,
        "format": format_reward,
        "tool_name": components["tool_name"],
        "param_name": components["param_name"],
        "param_value": components["param_value"],
        "matched_calls": components["matched_calls"],
        "components": components,
        "diagnostics": diagnostics,
        "errors": errors,
        "warnings": warnings,
    }

    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata["toolrl_reward"] = to_jsonable(out)
    return out


async def reward_func(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        outputs = []
        for sample in sample_or_samples:
            outputs.append(_reward_one(args, sample, **kwargs))
        return outputs
    return _reward_one(args, sample_or_samples, **kwargs)
