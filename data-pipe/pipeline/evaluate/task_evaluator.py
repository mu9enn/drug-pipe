from __future__ import annotations

from typing import Any


SUPPORTED_TASKS = {"vs", "ac", "pf", "kg", "e2e"}
MOLBENCH_TASKS = {"vs", "ac", "pf"}
RDKIT_REQUIRED_MESSAGE = "RDKit is required for MolBench chemical evaluation."
PROCESS_ONLY_ANSWERS = {
    "analysis complete",
    "analysis completed",
    "completed",
    "completed successfully",
    "done",
    "task complete",
    "task completed",
}
REFUSAL_PREFIXES = (
    "i cannot ",
    "i can't ",
    "unable to ",
    "no result",
)


def as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    text = value.strip()
    try:
        import json

        parsed = json.loads(text)
    except Exception:
        return [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if item is not None and str(item).strip()]
    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]
    return []


def load_chemistry_module() -> tuple[Any | None, str | None]:
    try:
        from rdkit import Chem  # type: ignore

        return Chem, None
    except Exception as exc:  # pragma: no cover - environment-dependent
        return None, str(exc)


def _require_chemistry(chemistry: Any | None) -> Any:
    if chemistry is None:
        raise RuntimeError(RDKIT_REQUIRED_MESSAGE)
    return chemistry


def _canonicalize(values: list[str], chemistry: Any) -> tuple[list[str], list[dict[str, Any]]]:
    canonical: list[str] = []
    errors: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        molecule = chemistry.MolFromSmiles(value)
        if molecule is None:
            errors.append({"index": index, "value": value, "reason": "invalid_smiles"})
            continue
        canonical.append(chemistry.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    return canonical, errors


def _set_metrics(prediction: list[str], ground_truth: list[str]) -> dict[str, Any]:
    predicted = set(prediction)
    expected = set(ground_truth)
    if not predicted and not expected:
        precision = recall = f1 = 1.0
    elif not predicted or not expected:
        precision = recall = f1 = 0.0
    else:
        true_positive = len(predicted & expected)
        precision = true_positive / len(predicted)
        recall = true_positive / len(expected)
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact_set_match": predicted == expected,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "acc": float(predicted == expected),
    }


def _base_result(
    task: str,
    *,
    reasons: list[str],
    metrics: dict[str, Any],
    prediction: list[str],
    ground_truth: list[str],
    candidates: list[str],
    audit: dict[str, Any],
) -> dict[str, Any]:
    task_answer_reasons = [reason for reason in reasons if reason == "parse_error"]
    task_answer_valid = not task_answer_reasons
    aggregate_eligible = not reasons
    return {
        "task": task,
        "task_answer_valid": task_answer_valid,
        "task_answer_invalid_reasons": task_answer_reasons,
        "aggregate_eligible": aggregate_eligible,
        "aggregate_invalid_reasons": reasons,
        # Compatibility alias for evaluator consumers. These are metric
        # eligibility reasons, not Data-Pipe curation rejection reasons.
        "invalid_reasons": reasons,
        "metrics": metrics,
        "canonical": {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "candidates": candidates,
        },
        "audit": {
            **audit,
            "task_answer_valid": task_answer_valid,
            "aggregate_eligible": aggregate_eligible,
        },
    }


def evaluate_vs(
    *,
    prediction: Any,
    ground_truth: Any,
    candidates: Any,
    chemistry: Any,
    parse_error: Any = None,
) -> dict[str, Any]:
    ranking_raw = as_string_list(prediction)
    ground_truth_raw = as_string_list(ground_truth)
    candidates_raw = as_string_list(candidates)
    ranking, ranking_errors = _canonicalize(ranking_raw, chemistry)
    expected, expected_errors = _canonicalize(ground_truth_raw, chemistry)
    candidate_list, candidate_errors = _canonicalize(candidates_raw, chemistry)

    reasons: list[str] = []
    if parse_error:
        reasons.append("parse_error")
    if expected_errors:
        reasons.append(f"invalid_ground_truth_smiles:{len(expected_errors)}")
    if candidate_errors:
        reasons.append(f"invalid_candidate_smiles:{len(candidate_errors)}")
    if ranking_errors:
        reasons.append(f"invalid_prediction_smiles:{len(ranking_errors)}")
    if not candidate_list:
        reasons.append("empty_candidate_set")
    if not ranking:
        reasons.append("empty_prediction")
    if len(ranking) != len(candidate_list):
        reasons.append(f"length_mismatch:{len(ranking)}!={len(candidate_list)}")
    duplicate_count = len(ranking) - len(set(ranking))
    if duplicate_count:
        reasons.append(f"duplicate_predictions:{duplicate_count}")
    candidate_set = set(candidate_list)
    outside_count = sum(value not in candidate_set for value in ranking)
    if outside_count:
        reasons.append(f"outside_candidate_set:{outside_count}")

    expected_set = set(expected)
    return _base_result(
        "vs",
        reasons=reasons,
        metrics={
            "top3_hit_num": float(sum(value in expected_set for value in ranking[:3])),
            "top10_hit_num": float(sum(value in expected_set for value in ranking[:10])),
        },
        prediction=ranking,
        ground_truth=expected,
        candidates=candidate_list,
        audit={
            "prediction_size": len(ranking),
            "ground_truth_size": len(expected),
            "candidate_size": len(candidate_list),
            "duplicate_prediction_count": duplicate_count,
            "outside_candidate_count": outside_count,
            "chemistry_canonicalization": True,
        },
    )


def evaluate_ac(
    *,
    prediction: Any,
    ground_truth: Any,
    chemistry: Any,
    parse_error: Any = None,
) -> dict[str, Any]:
    prediction_raw = as_string_list(prediction)
    ground_truth_raw = as_string_list(ground_truth)
    predicted, prediction_errors = _canonicalize(prediction_raw, chemistry)
    expected, expected_errors = _canonicalize(ground_truth_raw, chemistry)
    reasons: list[str] = []
    if parse_error:
        reasons.append("parse_error")
    if len(prediction_raw) != 1:
        reasons.append("empty_prediction" if not prediction_raw else f"invalid_prediction_count:{len(prediction_raw)}")
    if len(ground_truth_raw) != 1:
        reasons.append(
            "empty_ground_truth" if not ground_truth_raw else f"invalid_ground_truth_count:{len(ground_truth_raw)}"
        )
    if prediction_errors:
        reasons.append(f"invalid_prediction_smiles:{len(prediction_errors)}")
    if expected_errors:
        reasons.append(f"invalid_ground_truth_smiles:{len(expected_errors)}")
    correct = bool(predicted and expected and predicted[0] == expected[0])
    return _base_result(
        "ac",
        reasons=reasons,
        metrics={"acc": float(correct), "is_correct": correct},
        prediction=predicted,
        ground_truth=expected,
        candidates=[],
        audit={
            "prediction_size": len(predicted),
            "ground_truth_size": len(expected),
            "candidate_size": 0,
            "chemistry_canonicalization": True,
        },
    )


def evaluate_pf(
    *,
    prediction: Any,
    ground_truth: Any,
    chemistry: Any,
    parse_error: Any = None,
) -> dict[str, Any]:
    prediction_raw = as_string_list(prediction)
    ground_truth_raw = as_string_list(ground_truth)
    predicted, prediction_errors = _canonicalize(prediction_raw, chemistry)
    expected, expected_errors = _canonicalize(ground_truth_raw, chemistry)
    reasons: list[str] = []
    if parse_error:
        reasons.append("parse_error")
    if not prediction_raw:
        reasons.append("empty_prediction")
    if not ground_truth_raw:
        reasons.append("empty_ground_truth")
    if prediction_errors:
        reasons.append(f"invalid_prediction_smiles:{len(prediction_errors)}")
    if expected_errors:
        reasons.append(f"invalid_ground_truth_smiles:{len(expected_errors)}")
    return _base_result(
        "pf",
        reasons=reasons,
        metrics=_set_metrics(predicted, expected),
        prediction=predicted,
        ground_truth=expected,
        candidates=[],
        audit={
            "prediction_size": len(predicted),
            "ground_truth_size": len(expected),
            "candidate_size": 0,
            "chemistry_canonicalization": True,
        },
    )


def _has_answer(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_has_answer(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_answer(item) for item in value)
    return bool(str(value or "").strip())


def _has_result_content(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_has_result_content(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_result_content(item) for item in value)
    if value is None:
        return False
    if isinstance(value, (int, float, bool)):
        return True
    text = " ".join(str(value).strip().lower().split())
    if not text or text in PROCESS_ONLY_ANSWERS:
        return False
    return not text.startswith(REFUSAL_PREFIXES)


def evaluate_exploratory(
    task: str,
    *,
    prediction: Any,
    parse_error: Any = None,
    task_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if parse_error:
        reasons.append("parse_error")
    answer_present = _has_answer(prediction)
    if not answer_present:
        reasons.append("empty_prediction")
    result_content_present = _has_result_content(prediction)
    if answer_present and not result_content_present:
        reasons.append("missing_result_content")

    contract = task_contract or {}
    required_fields = contract.get("required_final_fields")
    if isinstance(required_fields, list) and required_fields:
        if not isinstance(prediction, dict):
            reasons.append("required_final_structure_missing")
        else:
            missing = [str(field) for field in required_fields if not _has_answer(prediction.get(str(field)))]
            if missing:
                reasons.append(f"missing_required_final_fields:{','.join(missing)}")

    return _base_result(
        task,
        reasons=reasons,
        metrics={
            "answer_present": answer_present,
            "result_content_present": result_content_present,
        },
        prediction=[],
        ground_truth=[],
        candidates=[],
        audit={
            "prediction_size": int(answer_present),
            "ground_truth_size": 0,
            "candidate_size": 0,
            "chemistry_canonicalization": False,
            "required_final_fields": required_fields or [],
        },
    )


def evaluate_task_answer(
    task: str,
    *,
    prediction: Any,
    ground_truth: Any,
    candidates: Any = None,
    chemistry: Any | None = None,
    parse_error: Any = None,
    task_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_name = str(task).strip().lower()
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported task: {task!r}")

    if task_name == "vs":
        return evaluate_vs(
            prediction=prediction,
            ground_truth=ground_truth,
            candidates=candidates,
            chemistry=_require_chemistry(chemistry),
            parse_error=parse_error,
        )
    if task_name == "ac":
        return evaluate_ac(
            prediction=prediction,
            ground_truth=ground_truth,
            chemistry=_require_chemistry(chemistry),
            parse_error=parse_error,
        )
    if task_name == "pf":
        return evaluate_pf(
            prediction=prediction,
            ground_truth=ground_truth,
            chemistry=_require_chemistry(chemistry),
            parse_error=parse_error,
        )
    return evaluate_exploratory(
        task_name,
        prediction=prediction,
        parse_error=parse_error,
        task_contract=task_contract,
    )
