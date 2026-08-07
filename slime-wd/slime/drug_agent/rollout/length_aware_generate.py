"""Use a larger generation budget only for declared long-output decisions.

The default Slime rollout path applies one ``max_new_tokens`` value to every
sample.  Drug-agent tool decisions are usually short, while terminal virtual
screening or property-filtering tasks can require a long ranked JSON result.
This hook keeps the ordinary budget small and raises it only from auditable
metadata; it never inspects the teacher response or its token length.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any


def _metadata(sample: Any) -> dict[str, Any]:
    value = getattr(sample, "metadata", None)
    return value if isinstance(value, dict) else {}


def _decision_type(sample: Any, metadata: dict[str, Any]) -> str:
    value = metadata.get("decision_type")
    if value:
        return str(value)
    label = getattr(sample, "label", None)
    if isinstance(label, dict) and label.get("decision_type"):
        return str(label["decision_type"])
    return ""


def resolve_response_cap(args: Namespace, sample: Any, sampling_params: dict[str, Any]) -> tuple[int, str]:
    """Return ``(max_new_tokens, tier)`` without using gold response content."""

    default_cap = int(sampling_params["max_new_tokens"])
    long_cap_raw = getattr(args, "rollout_long_response_len", None)
    long_cap = default_cap if long_cap_raw is None else int(long_cap_raw)
    if default_cap <= 0:
        raise ValueError(f"rollout max response length must be positive, got {default_cap}")
    if long_cap < default_cap:
        raise ValueError(
            "rollout long response length must be >= the default: "
            f"long={long_cap} default={default_cap}"
        )

    metadata = _metadata(sample)
    explicit = metadata.get("rollout_max_response_len")
    if explicit is not None:
        try:
            requested = int(explicit)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid metadata rollout_max_response_len={explicit!r}") from exc
        if requested <= 0 or requested > long_cap:
            raise ValueError(
                "metadata rollout_max_response_len must be in "
                f"[1, {long_cap}], got {requested}"
            )
        return max(default_cap, requested), "metadata"

    long_task_types = {str(item) for item in (getattr(args, "rollout_long_task_types", None) or [])}
    task_type = str(metadata.get("task_type") or "")
    if (
        long_cap > default_cap
        and task_type in long_task_types
        and _decision_type(sample, metadata) == "final_answer"
    ):
        return long_cap, "long"
    return default_cap, "default"


async def generate(args: Namespace, sample: Any, sampling_params: dict[str, Any], evaluation: bool = False):
    """Delegate to Slime's generator with a metadata-selected token budget."""

    del evaluation  # The cap policy is identical for train and eval calls.
    from slime.rollout.sglang_rollout import generate as sglang_generate

    params = sampling_params.copy()
    cap, tier = resolve_response_cap(args, sample, params)
    params["max_new_tokens"] = cap
    metadata = _metadata(sample)
    metadata["resolved_rollout_max_response_len"] = cap
    metadata["rollout_response_length_tier"] = tier
    sample.metadata = metadata
    return await sglang_generate(args, sample, params)
