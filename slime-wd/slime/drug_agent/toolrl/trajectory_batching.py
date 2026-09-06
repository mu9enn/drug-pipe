from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, TypeVar


T = TypeVar("T")


def group_complete_trajectories(
    items: list[T],
    *,
    metadata_of: Callable[[T], dict[str, Any]],
) -> list[tuple[str, list[T]]]:
    """Group decision items and require every source trajectory to be complete."""
    grouped: OrderedDict[str, list[T]] = OrderedDict()
    for item in items:
        metadata = metadata_of(item)
        source_id = str(metadata.get("source_id") or "").strip()
        if not source_id:
            raise ValueError("ToolRL decision is missing metadata.source_id")
        grouped.setdefault(source_id, []).append(item)

    trajectories: list[tuple[str, list[T]]] = []
    for source_id, decisions in grouped.items():
        declared_counts = {
            int(metadata_of(item).get("trajectory_decision_count", -1))
            for item in decisions
        }
        if len(declared_counts) != 1:
            raise ValueError(f"trajectory {source_id} has inconsistent trajectory_decision_count")
        declared_count = declared_counts.pop()
        ordinals = [int(metadata_of(item).get("decision_ordinal", -1)) for item in decisions]
        if declared_count < 1 or sorted(ordinals) != list(range(declared_count)):
            raise ValueError(
                f"trajectory {source_id} is incomplete: "
                f"declared={declared_count} ordinals={sorted(ordinals)}"
            )
        ordered = [item for _, item in sorted(zip(ordinals, decisions), key=lambda pair: pair[0])]
        trajectories.append((source_id, ordered))
    return trajectories


def pack_complete_trajectories(
    trajectories: list[tuple[str, list[T]]],
    rollout_batch_size: int,
) -> list[list[tuple[str, list[T]]]]:
    """Pack whole trajectories into exact fixed-size decision batches.

    The first trajectory that fits the remaining capacity is selected. A bad
    batch size fails closed instead of splitting or duplicating a trajectory.
    """
    if rollout_batch_size < 1:
        raise ValueError("rollout_batch_size must be positive")
    oversized = [(source_id, len(items)) for source_id, items in trajectories if len(items) > rollout_batch_size]
    if oversized:
        raise ValueError(
            f"trajectory exceeds rollout_batch_size={rollout_batch_size}: {oversized[:10]}"
        )
    total = sum(len(items) for _, items in trajectories)
    if total % rollout_batch_size:
        raise ValueError(
            f"complete trajectories contain {total} decisions, not divisible by "
            f"rollout_batch_size={rollout_batch_size}"
        )

    pending = [
        trajectory
        for _, trajectory in sorted(
            enumerate(trajectories),
            key=lambda pair: (-len(pair[1][1]), pair[0]),
        )
    ]
    batches: list[list[tuple[str, list[T]]]] = []
    while pending:
        remaining = rollout_batch_size
        batch: list[tuple[str, list[T]]] = []
        while remaining:
            index = next(
                (index for index, (_, items) in enumerate(pending) if len(items) <= remaining),
                None,
            )
            if index is None:
                sizes = [len(items) for _, items in pending]
                raise ValueError(
                    "cannot form an exact trajectory-atomic rollout batch: "
                    f"remaining_capacity={remaining} pending_trajectory_sizes={sizes[:30]}"
                )
            source_id, items = pending.pop(index)
            batch.append((source_id, items))
            remaining -= len(items)
        batches.append(batch)
    return batches


def flatten_batch(batch: list[tuple[str, list[T]]]) -> list[T]:
    return [item for _, trajectory in batch for item in trajectory]


def validate_packed_decision_batches(
    items: list[T],
    *,
    metadata_of: Callable[[T], dict[str, Any]],
    rollout_batch_size: int,
) -> list[list[T]]:
    """Load materialized batches and prove no trajectory crosses a boundary."""
    batch_groups: OrderedDict[int, list[T]] = OrderedDict()
    source_batches: dict[str, set[int]] = {}
    for item in items:
        metadata = metadata_of(item)
        batch_id = int(metadata.get("trajectory_batch_id", -1))
        position = int(metadata.get("trajectory_batch_position", -1))
        declared_size = int(metadata.get("trajectory_batch_decision_count", -1))
        source_id = str(metadata.get("source_id") or "").strip()
        if batch_id < 0 or position < 0 or declared_size != rollout_batch_size:
            raise ValueError("ToolRL decision has invalid trajectory batch metadata")
        batch_groups.setdefault(batch_id, []).append(item)
        source_batches.setdefault(source_id, set()).add(batch_id)

    crossing = {source_id: sorted(ids) for source_id, ids in source_batches.items() if len(ids) != 1}
    if crossing:
        raise ValueError(f"trajectories cross rollout batch boundaries: {crossing}")
    group_complete_trajectories(items, metadata_of=metadata_of)

    batches: list[list[T]] = []
    for batch_id in sorted(batch_groups):
        batch = batch_groups[batch_id]
        if len(batch) != rollout_batch_size:
            raise ValueError(
                f"trajectory batch {batch_id} contains {len(batch)} decisions; "
                f"expected {rollout_batch_size}"
            )
        positions = [int(metadata_of(item)["trajectory_batch_position"]) for item in batch]
        if sorted(positions) != list(range(rollout_batch_size)):
            raise ValueError(f"trajectory batch {batch_id} has invalid positions: {sorted(positions)}")
        ordered = [item for _, item in sorted(zip(positions, batch), key=lambda pair: pair[0])]
        completed_sources: set[str] = set()
        current_source: str | None = None
        current_ordinals: list[int] = []
        for item in ordered:
            metadata = metadata_of(item)
            source_id = str(metadata.get("source_id") or "")
            ordinal = int(metadata.get("decision_ordinal", -1))
            if source_id != current_source:
                if current_source is not None:
                    completed_sources.add(current_source)
                if source_id in completed_sources:
                    raise ValueError(
                        f"trajectory {source_id} is interleaved inside rollout batch {batch_id}"
                    )
                current_source = source_id
                current_ordinals = []
            current_ordinals.append(ordinal)
            if current_ordinals != list(range(len(current_ordinals))):
                raise ValueError(
                    f"trajectory {source_id} is out of decision order inside rollout batch {batch_id}"
                )
        batches.append(ordered)
    return batches
