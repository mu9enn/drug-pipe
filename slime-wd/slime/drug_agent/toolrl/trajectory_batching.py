from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, TypeVar


T = TypeVar("T")


def validate_trajectory_order(
    items: list[T],
    *,
    metadata_of: Callable[[T], dict[str, Any]],
) -> list[T]:
    """Validate canonical trajectory order while allowing selection holes."""
    completed_sources: set[str] = set()
    current_source: str | None = None
    current_trajectory_index = -1
    previous_ordinal = -1
    for item in items:
        metadata = metadata_of(item)
        source_id = str(metadata.get("source_id") or "").strip()
        if not source_id:
            raise ValueError("ToolRL decision is missing metadata.source_id")
        trajectory_index = int(metadata.get("trajectory_index", -1))
        ordinal = int(metadata.get("decision_ordinal", -1))
        if trajectory_index < 0 or ordinal < 0:
            raise ValueError(f"ToolRL decision {source_id} has invalid canonical order metadata")
        if source_id != current_source:
            if current_source is not None:
                completed_sources.add(current_source)
            if source_id in completed_sources:
                raise ValueError(f"trajectory {source_id} re-enters the decision stream")
            if trajectory_index <= current_trajectory_index:
                raise ValueError("trajectory_index is not strictly increasing")
            current_source = source_id
            current_trajectory_index = trajectory_index
            previous_ordinal = -1
        if ordinal <= previous_ordinal:
            raise ValueError(f"trajectory {source_id} decisions are not in original ordinal order")
        previous_ordinal = ordinal
    return items


def group_selected_trajectories(
    items: list[T],
    *,
    metadata_of: Callable[[T], dict[str, Any]],
) -> list[tuple[str, list[T]]]:
    validate_trajectory_order(items, metadata_of=metadata_of)
    grouped: OrderedDict[str, list[T]] = OrderedDict()
    for item in items:
        source_id = str(metadata_of(item)["source_id"])
        grouped.setdefault(source_id, []).append(item)
    return list(grouped.items())


def contiguous_batches(items: list[T], batch_size: int) -> list[list[T]]:
    """Cut the canonical stream directly; trajectories may cross boundaries."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def validate_contiguous_decision_batches(
    items: list[T],
    *,
    metadata_of: Callable[[T], dict[str, Any]],
    rollout_batch_size: int,
) -> list[list[T]]:
    validate_trajectory_order(items, metadata_of=metadata_of)
    return contiguous_batches(items, rollout_batch_size)
