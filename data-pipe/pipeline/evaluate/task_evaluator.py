from __future__ import annotations

from typing import Any


SUPPORTED_TASKS = {"vs", "ac", "pf", "kg", "e2e"}


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


def _canonicalize(values: list[str], chemistry: Any | None) -> tuple[list[str], list[dict[str, Any]]]:
    if chemistry is None:
        return list(values), []
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


def evaluate_task_answer(
    task: str,
    *,
    prediction: Any,
    ground_truth: Any,
    candidates: Any = None,
    chemistry: Any | None = None,
    parse_error: Any = None,
) -> dict[str, Any]:
    task_name = str(task).strip().lower()
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported task: {task!r}")

    prediction_raw = as_string_list(prediction)
    ground_truth_raw = as_string_list(ground_truth)
    candidates_raw = as_string_list(candidates)
    if task_name in {"kg", "e2e"}:
        reasons = ["parse_error"] if parse_error else []
        answer_present = bool(prediction) if isinstance(prediction, (dict, list)) else bool(str(prediction or "").strip())
        if not answer_present:
            reasons.append("empty_prediction")
        return {
            "task": task_name,
            "task_answer_valid": not reasons,
            "invalid_reasons": reasons,
            "metrics": {"answer_present": answer_present},
            "canonical": {
                "prediction": prediction_raw,
                "ground_truth": ground_truth_raw,
                "candidates": candidates_raw,
            },
            "audit": {
                "prediction_size": len(prediction_raw) if prediction_raw else int(answer_present),
                "ground_truth_size": len(ground_truth_raw),
                "candidate_size": len(candidates_raw),
                "chemistry_canonicalization": False,
            },
        }

    prediction_canonical, prediction_errors = _canonicalize(prediction_raw, chemistry)
    ground_truth_canonical, ground_truth_errors = _canonicalize(ground_truth_raw, chemistry)
    candidates_canonical, candidate_errors = _canonicalize(candidates_raw, chemistry)
    reasons: list[str] = []
    metrics: dict[str, Any] = {}

    if parse_error:
        reasons.append("parse_error")
    if ground_truth_errors:
        reasons.append(f"invalid_ground_truth_smiles:{len(ground_truth_errors)}")
    if prediction_errors:
        reasons.append(f"invalid_prediction_smiles:{len(prediction_errors)}")

    if task_name == "vs":
        if candidate_errors:
            reasons.append(f"invalid_candidate_smiles:{len(candidate_errors)}")
        if not candidates_canonical:
            reasons.append("empty_candidate_set")
        if len(prediction_canonical) != len(candidates_canonical):
            reasons.append(f"length_mismatch:{len(prediction_canonical)}!={len(candidates_canonical)}")
        if len(set(prediction_canonical)) != len(prediction_canonical):
            reasons.append("duplicate_predictions")
        outside = sum(value not in set(candidates_canonical) for value in prediction_canonical)
        if outside:
            reasons.append(f"outside_candidate_set:{outside}")
        expected = set(ground_truth_canonical)
        metrics = {
            "top3_hit_num": float(sum(value in expected for value in prediction_canonical[:3])),
            "top10_hit_num": float(sum(value in expected for value in prediction_canonical[:10])),
        }
    elif task_name == "ac":
        if len(prediction_raw) != 1:
            reasons.append("empty_prediction" if not prediction_raw else f"invalid_prediction_count:{len(prediction_raw)}")
        predicted = prediction_canonical[0] if prediction_canonical else ""
        expected = ground_truth_canonical[0] if ground_truth_canonical else ""
        correct = bool(predicted and expected and predicted == expected)
        metrics = {"acc": float(correct), "is_correct": correct}
    elif task_name == "pf":
        if not prediction_raw:
            reasons.append("empty_prediction")
        metrics = _set_metrics(prediction_canonical, ground_truth_canonical)
    return {
        "task": task_name,
        "task_answer_valid": not reasons,
        "invalid_reasons": reasons,
        "metrics": metrics,
        "canonical": {
            "prediction": prediction_canonical,
            "ground_truth": ground_truth_canonical,
            "candidates": candidates_canonical,
        },
        "audit": {
            "prediction_size": len(prediction_canonical),
            "ground_truth_size": len(ground_truth_canonical),
            "candidate_size": len(candidates_canonical),
            "chemistry_canonicalization": chemistry is not None,
        },
    }
