from __future__ import annotations

import asyncio
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from drug_agent.toolrl.normalization import (
    canonical_argument_map,
    canonical_param_name,
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


def _decision_role(sample: Any) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return str(metadata.get("decision_role") or _decision_type(sample))


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


def _official_8cee13e_reward(
    sample: Any,
    response_text: str,
    parsed: dict[str, Any],
    pred_calls: list[dict[str, Any]],
    gold_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """ToolRL commit 8cee13e reward, with final_answer as a response tag rename.

    The official response branch scores format only; it does not compare the
    response text with the teacher. Tool-call matching is invocation-level,
    order-insensitive, and multiplicity-preserving via ``used_pred``.
    """
    expected = _decision_type(sample)
    official_pred_calls: list[dict[str, Any]] = []
    if expected == "final_answer":
        match = re.fullmatch(r"<thought>.*?</thought>\n<final_answer>.*?</final_answer>", response_text, re.DOTALL)
        valid = bool(match and response_text.count("<final_answer>") == response_text.count("</final_answer>") == 1)
        predicted = "final_answer" if valid else "invalid"
    else:
        match = re.fullmatch(r"<thought>.*?</thought>\n<tool_call>\n(.*?)\n</tool_call>", response_text, re.DOTALL)
        valid = bool(match and response_text.count("<tool_call>") == response_text.count("</tool_call>") == 1)
        if valid:
            try:
                payloads = [json.loads(line) for line in match.group(1).splitlines() if line.strip()]
                if not payloads or not all(isinstance(item, dict) for item in payloads):
                    raise ValueError("empty/non-object tool payload")
                official_pred_calls = [
                    {
                        "tool_name": item.get("tool_name", item.get("name")),
                        "arguments": item.get("arguments", item.get("parameters", {})),
                    }
                    for item in payloads
                ]
                valid = all(call["tool_name"] and isinstance(call["arguments"], dict) for call in official_pred_calls)
            except Exception:
                valid = False
        predicted = "tool_call" if valid else "invalid"
    format_score = 1.0 if valid else 0.0
    if expected == "final_answer":
        correctness = 0.0
    else:
        correctness = _official_tool_correctness(official_pred_calls, gold_calls) if format_score else -3.0
    return {
        "score": format_score + correctness,
        "format": format_score,
        "components": {"format": format_score, "correctness": correctness},
        "diagnostics": {
            "reward_mode": "toolrl_official_8cee13e",
            "canonical_commit": "8cee13ec0ca72f0461da372a93a6fd8140dbb840",
            "expected_decision_type": expected,
            "predicted_decision_type": predicted,
            "final_answer_is_response_tag_rename": expected == "final_answer",
            "thought_tag_required": True,
            "pred_call_count": len(official_pred_calls),
            "gold_call_count": len(gold_calls),
        },
        "errors": [] if format_score else [{"type": "OfficialFormatMismatch", "message": f"expected thought + {expected}"}],
        "warnings": [],
    }


def _format_reward(parsed: dict[str, Any]) -> float:
    if not parsed.get("ok"):
        return -0.3
    tool_calls = [
        item
        for item in (parsed.get("tool_calls") or [])
        if isinstance(item, dict)
        and item not in (parsed.get("unsupported_tool_calls") or [])
    ]
    if not isinstance(tool_calls, list) or not tool_calls:
        if parsed.get("has_final_answer"):
            return 0.35
        return 0.2
    if parsed.get("unsupported_tool_calls"):
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


def _drug_pipe_final_reward(sample: Any, parsed: dict[str, Any]) -> dict[str, Any]:
    """Apply the structured-final extension only when explicitly enabled.

    Turning it off preserves the Drug-Pipe envelope while reducing the final
    objective to format-only response semantics.  The exact ToolRL baseline
    still goes through ``_official_8cee13e_reward`` and is unaffected here.
    """
    exact_enabled = os.environ.get("TOOLRL_STRUCTURED_FINAL_EXACT", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }
    out = _molclaw_final_answer_reward(sample, parsed)
    out["diagnostics"]["structured_final_exact_enabled"] = exact_enabled
    if exact_enabled:
        return out
    valid = bool(
        parsed.get("ok")
        and parsed.get("has_final_answer")
        and not parsed.get("has_tool_call")
    )
    out["score"] = 1.0 if valid else -0.5
    out["components"]["terminal_correctness"] = 0.0
    out["diagnostics"]["terminal_exact_match"] = None
    if valid:
        out["errors"] = []
    return out


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


def _decision_aware_tool_reward(
    sample: Any,
    parsed: dict[str, Any],
    pred_calls: list[dict[str, Any]],
    gold_calls: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Tool-name-first dense reward on the common [-0.5, 1.0] scale."""
    valid_envelope = bool(
        parsed.get("ok")
        and parsed.get("has_tool_call")
        and not parsed.get("has_final_answer")
    )
    tool_metrics = _tool_reward_components(pred_calls, gold_calls, config=config)
    matched = int(tool_metrics.get("matched_calls") or 0)
    name_f1 = float(tool_metrics.get("f1") or 0.0)
    pred_count = len(pred_calls)
    gold_count = len(gold_calls)
    count_completeness = (
        1.0 - abs(pred_count - gold_count) / max(pred_count, gold_count)
        if pred_count or gold_count
        else 1.0
    )
    param_name = float(tool_metrics.get("param_name_score") or 0.0)
    param_value = float(tool_metrics.get("param_value_score") or 0.0)
    format_score = 1.0 if valid_envelope else 0.0
    quality = (
        0.10 * format_score
        + 0.55 * name_f1
        + 0.10 * count_completeness
        + 0.10 * param_name
        + 0.15 * param_value
    )
    score = clamp(1.5 * quality - 0.5, -0.5, 1.0)
    gate_reason = None
    if not valid_envelope:
        score = -0.5
        gate_reason = "invalid_react_tool_envelope"
    elif matched == 0:
        score = -0.5
        gate_reason = "no_correct_tool_name"

    components = {
        "format": format_score,
        "tool_name_f1": name_f1,
        "call_completeness": count_completeness,
        "param_name": param_name,
        "param_value": param_value,
        "weighted_quality": quality,
        "matched_calls": float(matched),
    }
    errors = []
    if gate_reason is not None:
        errors.append({"type": "DecisionAwareRewardGate", "message": gate_reason})
    return {
        "score": score,
        "format": format_score,
        "tool_name": name_f1,
        "param_name": param_name,
        "param_value": param_value,
        "matched_calls": float(matched),
        "components": components,
        "diagnostics": {
            "reward_mode": "decision_aware",
            "decision_role": _decision_role(sample),
            "expected_decision_type": "tool_call",
            "predicted_decision_type": "tool_call" if valid_envelope else "invalid",
            "parse_ok": bool(parsed.get("ok")),
            "pred_call_count": pred_count,
            "gold_call_count": gold_count,
            "matched_calls": matched,
            "unmatched_pred_count": max(0, pred_count - matched),
            "unmatched_gold_count": max(0, gold_count - matched),
            "gate_reason": gate_reason,
        },
        "errors": errors,
        "warnings": [],
    }


def _decision_aware_final_reward(sample: Any, parsed: dict[str, Any]) -> dict[str, Any]:
    out = _molclaw_final_answer_reward(sample, parsed)
    out["diagnostics"]["reward_mode"] = "decision_aware"
    out["diagnostics"]["decision_role"] = _decision_role(sample)
    return out


_CRITICAL_ARGUMENT_PATTERN = re.compile(
    r"(^|_)(id|ids|name|target|gene|protein|ligand|receptor|sequence|smiles|mutation|chain|"
    r"artifact|file|path|input|structure|complex|pdb|cif|sdf|mol2)(_|$)",
    re.IGNORECASE,
)


@lru_cache(maxsize=4)
def _tool_catalog_schemas(path: str) -> dict[str, dict[str, Any]]:
    if not path or not Path(path).is_file():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tools = payload.get("tools") if isinstance(payload, dict) else None
    if not isinstance(tools, list):
        return {}
    return {
        canonical_tool_name(str(tool.get("name") or "")): tool.get("input_schema")
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("input_schema"), dict)
    }


def _argument_schema(tool_name: str) -> dict[str, Any]:
    path = os.environ.get("DRUG_AGENT_TOOL_CATALOG", "")
    return _tool_catalog_schemas(path).get(canonical_tool_name(tool_name), {})


def _schema_value_valid(value: Any, schema: dict[str, Any]) -> bool:
    """Small, deterministic JSON-schema subset used by the reward boundary."""
    if not schema:
        return True
    expected = schema.get("type")
    type_ok = {
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }.get(str(expected), True)
    if not type_ok:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return False
        if "maximum" in schema and value > schema["maximum"]:
            return False
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            return False
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            return False
    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            return False
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            return False
    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            return False
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            return False
        item_schema = schema.get("items")
        if isinstance(item_schema, dict) and not all(_schema_value_valid(item, item_schema) for item in value):
            return False
    return True


def _is_critical_argument(name: str, schema: dict[str, Any]) -> bool:
    extension = schema.get("x-toolrl-importance")
    if extension in {"identity", "critical"}:
        return True
    if extension in {"configurable", "configuration"}:
        return False
    return bool(_CRITICAL_ARGUMENT_PATTERN.search(name))


def _hierarchical_argument_metrics(
    pred_call: dict[str, Any], gold_call: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    tool_name = canonical_tool_name(gold_call.get("tool_name") or gold_call.get("name"), config)
    pred_args = canonical_argument_map(pred_call.get("arguments") or {}, tool_name=tool_name, config=config)
    gold_args = canonical_argument_map(gold_call.get("arguments") or {}, tool_name=tool_name, config=config)
    schema = _argument_schema(tool_name)
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = {
        canonical_param_name(str(name), config)
        for name in (schema.get("required") or [])
        if isinstance(name, str)
    }
    # A catalog-less local tool falls back to the teacher's keys.  This keeps
    # the gate useful without pretending all teacher values are mandatory.
    if not properties:
        required = set(gold_args)
    required_coverage = len(required & set(pred_args)) / max(1, len(required)) if required else 1.0

    critical_keys = {
        key for key in gold_args
        if _is_critical_argument(key, properties.get(key) if isinstance(properties.get(key), dict) else {})
    }
    critical_scores = [
        compare_values(pred_args.get(key), gold_args[key], tool_name=tool_name, param_name=key, config=config)["score"]
        if key in pred_args else 0.0
        for key in critical_keys
    ]
    critical_exact = sum(score >= 1.0 for score in critical_scores) / len(critical_scores) if critical_scores else 1.0

    configurable_keys = set(pred_args) - critical_keys
    validity = []
    for key in configurable_keys:
        param_schema = properties.get(key) if isinstance(properties.get(key), dict) else {}
        validity.append(bool(key in properties or not properties) and _schema_value_valid(pred_args[key], param_schema))
    configurable_validity = sum(validity) / len(validity) if validity else 1.0
    all_keys_exact = set(pred_args) == set(gold_args) and all(
        compare_values(pred_args[key], gold_args[key], tool_name=tool_name, param_name=key, config=config)["score"] >= 1.0
        for key in gold_args
    )
    return {
        "required_coverage": required_coverage,
        "critical_exact": critical_exact,
        "configurable_validity": configurable_validity,
        "teacher_exact": bool(all_keys_exact),
        "required_arguments": sorted(required),
        "critical_arguments": sorted(critical_keys),
        "configurable_arguments": sorted(configurable_keys),
    }


def _hierarchical_tool_reward(
    sample: Any,
    parsed: dict[str, Any],
    pred_calls: list[dict[str, Any]],
    gold_calls: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Stage-gated reward: envelope -> tool set -> required args -> validity."""
    valid_envelope = bool(parsed.get("ok") and parsed.get("has_tool_call") and not parsed.get("has_final_answer"))
    pairs = _pair_tool_calls(pred_calls, gold_calls, config=config) if valid_envelope else []
    metrics = _compute_pair_metrics(pred_calls, gold_calls, pairs)
    tool_f1 = float(metrics["f1"])
    tool_set_exact = len(pairs) == len(pred_calls) == len(gold_calls)
    gate = "format"
    argument_metrics: list[dict[str, Any]] = []

    if not valid_envelope:
        score = -0.5
        gate = "invalid_react_tool_envelope"
    elif not pairs:
        score = -0.4
        gate = "wrong_tool"
    elif not tool_set_exact:
        score = -0.05 + 0.35 * tool_f1
        gate = "partial_tool_set"
    else:
        for pair in pairs:
            argument_metrics.append(_hierarchical_argument_metrics(pred_calls[pair.pred_index], gold_calls[pair.gold_index], config))
        required_coverage = min(item["required_coverage"] for item in argument_metrics) if argument_metrics else 1.0
        critical_exact = min(item["critical_exact"] for item in argument_metrics) if argument_metrics else 1.0
        configurable_validity = min(item["configurable_validity"] for item in argument_metrics) if argument_metrics else 1.0
        teacher_exact = all(item["teacher_exact"] for item in argument_metrics)
        if required_coverage < 1.0:
            score = 0.30 + 0.20 * required_coverage
            gate = "missing_required_arguments"
        elif critical_exact < 1.0:
            score = 0.55 + 0.15 * critical_exact
            gate = "critical_argument_mismatch"
        elif configurable_validity < 1.0:
            score = 0.72 + 0.13 * configurable_validity
            gate = "invalid_configurable_arguments"
        elif teacher_exact:
            score = 1.0
            gate = "teacher_equivalent"
        else:
            score = 0.90
            gate = "valid_alternative_configuration"

    required_coverage = min((item["required_coverage"] for item in argument_metrics), default=0.0)
    critical_exact = min((item["critical_exact"] for item in argument_metrics), default=0.0)
    configurable_validity = min((item["configurable_validity"] for item in argument_metrics), default=0.0)
    out = {
        "score": clamp(score, -0.5, 1.0),
        "format": 1.0 if valid_envelope else 0.0,
        "tool_name": tool_f1,
        "param_name": required_coverage,
        "param_value": critical_exact,
        "matched_calls": float(len(pairs)),
        "components": {
            "tool_name_f1": tool_f1,
            "tool_set_exact": float(tool_set_exact),
            "required_argument_coverage": required_coverage,
            "critical_argument_exact": critical_exact,
            "configurable_argument_validity": configurable_validity,
        },
        "diagnostics": {
            "reward_mode": "hierarchical",
            "decision_role": _decision_role(sample),
            "is_initial_step": bool((sample.metadata or {}).get("is_initial_step")) if isinstance(sample.metadata, dict) else False,
            "reward_stage": gate,
            "parse_ok": bool(parsed.get("ok")),
            "pred_call_count": len(pred_calls),
            "gold_call_count": len(gold_calls),
            "matched_calls": len(pairs),
            "argument_policy": argument_metrics,
        },
        "errors": [] if score >= 0 else [{"type": "HierarchicalRewardGate", "message": gate}],
        "warnings": [],
    }
    return out


def _sample_was_truncated(sample: Any) -> bool:
    """Accept Slime's enum status without importing its torch-backed types."""
    status = getattr(sample, "status", None)
    status_value = getattr(status, "value", status)
    return str(status_value or "").strip().lower() in {"truncated", "status.truncated"}


def _apply_truncation_guard(sample: Any, out: dict[str, Any]) -> dict[str, Any]:
    diagnostics = out.setdefault("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {"original_diagnostics": to_jsonable(diagnostics)}
        out["diagnostics"] = diagnostics
    truncated = _sample_was_truncated(sample)
    diagnostics["response_truncated"] = truncated
    diagnostics["truncation_guard_applied"] = False
    if truncated:
        original_score = float(out.get("score") or 0.0)
        diagnostics["score_before_truncation_guard"] = original_score
        out["score"] = min(original_score, 0.0)
        diagnostics["truncation_guard_applied"] = original_score > 0.0
        warnings = out.setdefault("warnings", [])
        if isinstance(warnings, list):
            warnings.append(
                {
                    "type": "TruncatedResponse",
                    "message": "response hit the rollout cap; positive reward was suppressed",
                }
            )
    return out


def _attach_protocol_diagnostics(out: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    diagnostics = out.setdefault("diagnostics", {})
    blocks = parsed.get("blocks") if isinstance(parsed.get("blocks"), list) else []
    thoughts = [str(block.get("body") or "") for block in blocks if block.get("kind") == "thought"]
    diagnostics.update(
        {
            "thought_present": bool(thoughts),
            "thought_block_count": len(thoughts),
            "thought_char_count": sum(len(item) for item in thoughts),
            "tool_call_container_count": sum(block.get("kind") == "tool_call" for block in blocks),
        }
    )
    return out


def _reward_one(args, sample: Any, **kwargs) -> dict[str, Any]:
    config = load_tool_schema_config()
    response_text = _extract_response_text(sample)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    strict_toolrl_turn = str(metadata.get("protocol") or "") == "toolrl_turn_v1"
    parsed = parse_tool_calls(
        response_text,
        allowed_tool_names=None,
        keep_non_molclaw=True,
        strict_toolrl_turn=strict_toolrl_turn,
    )
    gold_calls = _extract_gold_tool_calls(sample)

    pred_calls = parsed.get("tool_calls") if isinstance(parsed.get("tool_calls"), list) else []
    pred_calls = [item for item in pred_calls if isinstance(item, dict)]
    gold_calls = [item for item in gold_calls if isinstance(item, dict)]

    reward_mode = os.environ.get("TOOLRL_REWARD_MODE", "official").strip().lower()
    if reward_mode not in {"toolrl_official_8cee13e", "official", "molclaw", "decision_aware", "hierarchical"}:
        raise ValueError(f"unsupported TOOLRL_REWARD_MODE: {reward_mode}")
    if reward_mode == "toolrl_official_8cee13e":
        out = _official_8cee13e_reward(sample, response_text, parsed, pred_calls, gold_calls)
        out = _attach_protocol_diagnostics(out, parsed)
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        sample.metadata["toolrl_reward"] = to_jsonable(out)
        return out
    if reward_mode == "official":
        out = _official_reward(sample, parsed, pred_calls, gold_calls)
        out = _attach_protocol_diagnostics(out, parsed)
        out = _apply_truncation_guard(sample, out)
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        sample.metadata["toolrl_reward"] = to_jsonable(out)
        return out

    if reward_mode == "decision_aware":
        if _decision_type(sample) == "final_answer":
            out = _decision_aware_final_reward(sample, parsed)
        else:
            out = _decision_aware_tool_reward(sample, parsed, pred_calls, gold_calls, config)
        out = _attach_protocol_diagnostics(out, parsed)
        out = _apply_truncation_guard(sample, out)
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        sample.metadata["toolrl_reward"] = to_jsonable(out)
        return out

    if reward_mode == "hierarchical":
        if _decision_type(sample) == "final_answer":
            out = _drug_pipe_final_reward(sample, parsed)
            out["diagnostics"]["reward_mode"] = "hierarchical"
            out["diagnostics"]["decision_role"] = _decision_role(sample)
        else:
            out = _hierarchical_tool_reward(sample, parsed, pred_calls, gold_calls, config)
        out = _attach_protocol_diagnostics(out, parsed)
        out = _apply_truncation_guard(sample, out)
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        sample.metadata["toolrl_reward"] = to_jsonable(out)
        return out

    # Terminal rows have no gold tool calls by construction.  They must be
    # scored against target_final_answer before the dense empty-call shortcut,
    # otherwise malformed/no-output generations receive positive reward.
    if _decision_type(sample) == "final_answer":
        out = _molclaw_final_answer_reward(sample, parsed)
        out = _attach_protocol_diagnostics(out, parsed)
        out = _apply_truncation_guard(sample, out)
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
    if parsed.get("unsupported_tool_calls"):
        score -= min(0.10, 0.02 * len(parsed.get("unsupported_tool_calls") or []))
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
    out = _attach_protocol_diagnostics(out, parsed)
    out = _apply_truncation_guard(sample, out)

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
