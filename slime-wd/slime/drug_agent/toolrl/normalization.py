from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from drug_agent.utils import bool_from_any, clamp, normalize_tool_name


DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("tool_schema_config.yaml")


@lru_cache(maxsize=8)
def load_tool_schema_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    text = config_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - fallback path
        try:
            import yaml  # type: ignore
        except Exception as yaml_exc:  # pragma: no cover - fallback path
            raise RuntimeError(
                f"Failed to parse {config_path} as JSON and PyYAML is unavailable"
            ) from yaml_exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a mapping")
    payload.setdefault("defaults", {})
    payload.setdefault("tools", {})
    return payload


def _canonical_name(text: str) -> str:
    return text.strip().lower()


def canonical_tool_name(name: str | None, config: dict[str, Any] | None = None) -> str:
    raw = normalize_tool_name(name)
    return _canonical_name(raw)


def canonical_param_name(name: str | None, config: dict[str, Any] | None = None) -> str:
    if not isinstance(name, str):
        return ""
    return _canonical_name(name)


def _artifact_payload(value: Any, *, suffix_len: int = 2) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None

    lower = stripped.lower()
    placeholder_tokens = {"<artifact>", "<file>", "<path>"}
    if any(token in lower for token in placeholder_tokens) or lower.startswith(("artifact://", "file://", "path://")):
        return {"kind": "artifact", "placeholder": True, "token": lower}

    normalized = stripped.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    basename = parts[-1] if parts else normalized
    tail_parts = parts[-suffix_len:] if suffix_len > 0 else parts
    return {
        "kind": "artifact",
        "placeholder": False,
        "path": normalized,
        "basename": basename,
        "tail": "/".join(tail_parts),
        "suffix": Path(basename).suffix.lower(),
    }


def _infer_kind(tool_name: str, param_name: str, value: Any, config: dict[str, Any] | None = None) -> str:
    tool_rules = {}
    if config and isinstance(config.get("tools"), dict):
        tool_rules = config["tools"].get(canonical_tool_name(tool_name, config), {}) or {}
    param_types = tool_rules.get("param_types") if isinstance(tool_rules.get("param_types"), dict) else {}
    canonical_param = canonical_param_name(param_name, config)
    kind = param_types.get(canonical_param)
    if isinstance(kind, str):
        return kind

    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if bool_from_any(value) is not None:
            return "bool"
        try:
            float(lowered)
        except Exception:
            pass
        else:
            return "number"
        if "smiles" in canonical_param:
            return "smiles"
        if any(token in canonical_param for token in {"path", "file", "dir", "folder", "artifact"}):
            return "artifact"
        return "string"
    if isinstance(value, list):
        if "smiles" in canonical_param:
            return "smiles_list"
        if any(token in canonical_param for token in {"path", "file", "artifact"}):
            return "artifact_list"
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "string"


def normalize_scalar(value: Any, kind: str = "string", *, config: dict[str, Any] | None = None) -> Any:
    defaults = config.get("defaults") if config and isinstance(config.get("defaults"), dict) else {}
    suffix_len = int(defaults.get("artifact_path_suffix_len") or 2)

    if kind == "bool":
        parsed = bool_from_any(value)
        if parsed is not None:
            return parsed
        return bool(value)

    if kind == "number":
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            try:
                num = float(stripped)
            except Exception:
                return stripped
            return round(num, 6)
        return value

    if kind == "smiles":
        if not isinstance(value, str):
            return value
        return re.sub(r"\s+", "", value.strip())

    if kind == "smiles_list":
        if not isinstance(value, list):
            return [normalize_scalar(value, "smiles", config=config)]
        normalized = [normalize_scalar(item, "smiles", config=config) for item in value]
        return sorted(normalized, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))

    if kind == "artifact":
        payload = _artifact_payload(value, suffix_len=suffix_len)
        return payload if payload is not None else value

    if kind == "artifact_list":
        if not isinstance(value, list):
            return [normalize_scalar(value, "artifact", config=config)]
        return sorted(
            [normalize_scalar(item, "artifact", config=config) for item in value],
            key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False),
        )

    if kind == "dict":
        if not isinstance(value, dict):
            return value
        return normalize_mapping(value, config=config)

    if kind == "list":
        if not isinstance(value, list):
            return [normalize_value(value, config=config)]
        return [normalize_value(item, config=config) for item in value]

    if not isinstance(value, str):
        return value
    return re.sub(r"\s+", " ", value.strip())


def normalize_value(value: Any, *, tool_name: str | None = None, param_name: str | None = None, config: dict[str, Any] | None = None) -> Any:
    if isinstance(value, dict):
        return normalize_mapping(value, tool_name=tool_name, config=config)
    if isinstance(value, list):
        kind = _infer_kind(tool_name or "", param_name or "", value, config=config)
        if kind == "artifact_list":
            return normalize_scalar(value, "artifact_list", config=config)
        if kind == "smiles_list":
            return normalize_scalar(value, "smiles_list", config=config)
        return [normalize_value(item, tool_name=tool_name, param_name=param_name, config=config) for item in value]

    kind = _infer_kind(tool_name or "", param_name or "", value, config=config)
    return normalize_scalar(value, kind, config=config)


def normalize_mapping(
    mapping: dict[str, Any],
    *,
    tool_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in mapping.items():
        canonical_key = canonical_param_name(key, config)
        normalized[canonical_key] = normalize_value(value, tool_name=tool_name, param_name=canonical_key, config=config)
    return normalized


def _score_scalar(pred: Any, gold: Any, *, config: dict[str, Any] | None = None) -> float:
    defaults = config.get("defaults") if config and isinstance(config.get("defaults"), dict) else {}
    tol = float(defaults.get("numeric_tolerance") or 1e-6)

    if pred == gold:
        return 1.0

    if isinstance(pred, dict) and isinstance(gold, dict):
        if pred.get("kind") == "artifact" and gold.get("kind") == "artifact":
            if pred.get("placeholder") or gold.get("placeholder"):
                return 0.75 if pred.get("basename") or gold.get("basename") else 0.5
            if pred.get("basename") and pred.get("basename") == gold.get("basename") and pred.get("suffix") == gold.get("suffix"):
                return 1.0
            if pred.get("tail") and pred.get("tail") == gold.get("tail"):
                return 0.85
            return 0.0
        return _score_mapping(pred, gold, config=config)

    if isinstance(pred, list) and isinstance(gold, list):
        if not pred and not gold:
            return 1.0
        if all(isinstance(x, (str, int, float, bool, dict, list)) for x in pred + gold):
            pred_list = [json.dumps(x, sort_keys=True, ensure_ascii=False) if isinstance(x, (dict, list)) else x for x in pred]
            gold_list = [json.dumps(x, sort_keys=True, ensure_ascii=False) if isinstance(x, (dict, list)) else x for x in gold]
            pred_set = list(pred_list)
            gold_set = list(gold_list)
            matched = 0
            used = [False] * len(gold_set)
            for item in pred_set:
                for idx, other in enumerate(gold_set):
                    if used[idx]:
                        continue
                    if item == other:
                        used[idx] = True
                        matched += 1
                        break
            precision = matched / max(1, len(pred_set))
            recall = matched / max(1, len(gold_set))
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)
        return 0.0

    if isinstance(pred, (int, float)) and isinstance(gold, (int, float)):
        if abs(float(pred) - float(gold)) <= tol:
            return 1.0
        scale = max(abs(float(pred)), abs(float(gold)), 1.0)
        gap = abs(float(pred) - float(gold)) / scale
        return clamp(1.0 - gap, 0.0, 1.0)

    if isinstance(pred, bool) and isinstance(gold, bool):
        return 1.0 if pred == gold else 0.0

    if isinstance(pred, str) and isinstance(gold, str):
        p = re.sub(r"\s+", " ", pred.strip())
        g = re.sub(r"\s+", " ", gold.strip())
        if p.lower() == g.lower():
            return 1.0
        if p == g:
            return 1.0
        return 0.0

    if pred is None and gold is None:
        return 1.0
    return 0.0


def _score_mapping(pred: dict[str, Any], gold: dict[str, Any], *, config: dict[str, Any] | None = None) -> float:
    if not isinstance(pred, dict) or not isinstance(gold, dict):
        return 0.0
    if not pred and not gold:
        return 1.0
    pred_keys = set(pred.keys())
    gold_keys = set(gold.keys())
    if not pred_keys and not gold_keys:
        return 1.0

    matched_keys = pred_keys & gold_keys
    key_precision = len(matched_keys) / max(1, len(pred_keys))
    key_recall = len(matched_keys) / max(1, len(gold_keys))
    key_f1 = 0.0 if key_precision + key_recall == 0 else 2 * key_precision * key_recall / (key_precision + key_recall)

    value_scores: list[float] = []
    for key in matched_keys:
        value_scores.append(_score_scalar(pred[key], gold[key], config=config))
    value_score = sum(value_scores) / len(value_scores) if value_scores else 0.0
    return 0.35 * key_f1 + 0.65 * value_score


def compare_values(
    pred: Any,
    gold: Any,
    *,
    tool_name: str | None = None,
    param_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kind = _infer_kind(tool_name or "", param_name or "", gold, config=config)
    if isinstance(gold, list) and kind not in {"smiles_list", "artifact_list"}:
        kind = "list"
    if isinstance(gold, dict):
        kind = "dict"
    normalized_pred = normalize_value(pred, tool_name=tool_name, param_name=param_name, config=config)
    normalized_gold = normalize_value(gold, tool_name=tool_name, param_name=param_name, config=config)
    score = _score_scalar(normalized_pred, normalized_gold, config=config)
    return {
        "kind": kind,
        "score": clamp(score, 0.0, 1.0),
        "pred": normalized_pred,
        "gold": normalized_gold,
    }


def canonical_tool_rule(tool_name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_tool_schema_config()
    tools = config.get("tools") if isinstance(config.get("tools"), dict) else {}
    canonical_name = canonical_tool_name(tool_name, config)
    rule = tools.get(canonical_name)
    return rule if isinstance(rule, dict) else {}


def canonical_argument_map(arguments: dict[str, Any], *, tool_name: str | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_tool_schema_config()
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        canonical_key = canonical_param_name(key, config)
        out[canonical_key] = value
    return out
