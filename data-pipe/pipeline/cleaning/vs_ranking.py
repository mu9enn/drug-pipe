from __future__ import annotations

import json
from typing import Any

from pipeline.cleaning.artifacts import inspect_observation_status


QUICKVINA_TOOL = "molecule_docking_quickvina_fullprocess"
MOLECULE_KEYS = ("smiles", "ligand_smiles", "ligand", "molecule", "compound", "candidate")


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _values_for_key(value: Any, key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(value, dict):
        for item_key, item in value.items():
            if str(item_key) == key:
                values.append(item)
            values.extend(_values_for_key(item, key))
    elif isinstance(value, list):
        for item in value:
            values.extend(_values_for_key(item, key))
    return values


def molecule_from_arguments(arguments: Any) -> str | None:
    if not isinstance(arguments, dict):
        return None
    for key in MOLECULE_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip() and "<artifact:" not in value:
            return value.strip()
    return None


def quickvina_context(arguments: dict[str, Any]) -> str:
    return _serialize(
        {
            key: value
            for key, value in arguments.items()
            if key not in MOLECULE_KEYS
        }
    )


def successful_quickvina_result(
    tool_name: str,
    arguments: Any,
    observation_payload: Any,
    *,
    tool_use_id: str | None = None,
) -> dict[str, Any] | None:
    if tool_name != QUICKVINA_TOOL or inspect_observation_status(observation_payload)["is_error"]:
        return None
    molecule = molecule_from_arguments(arguments)
    values = _values_for_key(observation_payload, "docking_affinity_value")
    try:
        score = float(values[-1]) if values and values[-1] is not None else None
    except (TypeError, ValueError):
        score = None
    if not molecule or score is None:
        return None
    return {
        "tool_use_id": tool_use_id,
        "smiles": molecule,
        "score": score,
        "context": quickvina_context(arguments),
    }


def rank_by_best_quickvina_score(
    ranking: list[str],
    results: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    """Rank scored molecules by their best pose; keep missing-score entries last."""
    records_by_smiles: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        smiles = result.get("smiles")
        score = result.get("score")
        if isinstance(smiles, str) and smiles and isinstance(score, (int, float)):
            records_by_smiles.setdefault(smiles, []).append(result)

    best_scores = {
        smiles: min(float(record["score"]) for record in records)
        for smiles, records in records_by_smiles.items()
    }
    repaired = sorted(
        ranking,
        key=lambda smiles: (
            str(smiles) not in best_scores,
            best_scores.get(str(smiles), 0.0),
        ),
    )
    relevant = {
        smiles: {
            "scores": [float(record["score"]) for record in records_by_smiles.get(smiles, [])],
            "best_score": best_scores.get(smiles),
            "contexts": sorted(
                {str(record.get("context") or "") for record in records_by_smiles.get(smiles, [])}
            ),
        }
        for smiles in dict.fromkeys(str(item) for item in ranking)
    }
    return repaired, {
        "aggregation": "minimum_successful_docking_affinity",
        "molecules": relevant,
        "missing_scores": [smiles for smiles in dict.fromkeys(ranking) if smiles not in best_scores],
    }
