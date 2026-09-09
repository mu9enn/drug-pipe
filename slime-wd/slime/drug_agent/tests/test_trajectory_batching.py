from __future__ import annotations

from types import SimpleNamespace

import pytest

from drug_agent.toolrl.trajectory_batching import (
    contiguous_batches,
    group_selected_trajectories,
    validate_contiguous_decision_batches,
    validate_trajectory_order,
)
from drug_agent.toolrl.trajectory_data_source import TrajectoryBatchDataSource


def _item(source_id: str, trajectory_index: int, ordinal: int):
    return SimpleNamespace(
        metadata={
            "source_id": source_id,
            "trajectory_index": trajectory_index,
            "decision_ordinal": ordinal,
        }
    )


def _metadata(item):
    return item.metadata


def test_selected_trajectory_order_allows_ordinal_holes() -> None:
    items = [_item("a", 0, 0), _item("a", 0, 3), _item("a", 0, 6), _item("b", 1, 2), _item("b", 1, 5)]
    assert validate_trajectory_order(items, metadata_of=_metadata) == items
    assert [(source, len(rows)) for source, rows in group_selected_trajectories(items, metadata_of=_metadata)] == [
        ("a", 3),
        ("b", 2),
    ]


def test_batches_cut_across_trajectory_boundary_without_reordering() -> None:
    items = [
        *[_item("a", 0, ordinal) for ordinal in (0, 2, 5, 6, 9, 11)],
        *[_item("b", 1, ordinal) for ordinal in (1, 4, 8, 10, 12)],
        _item("c", 2, 3),
    ]
    batches = validate_contiguous_decision_batches(items, metadata_of=_metadata, rollout_batch_size=4)
    assert [[(x.metadata["source_id"], x.metadata["decision_ordinal"]) for x in batch] for batch in batches] == [
        [("a", 0), ("a", 2), ("a", 5), ("a", 6)],
        [("a", 9), ("a", 11), ("b", 1), ("b", 4)],
        [("b", 8), ("b", 10), ("b", 12), ("c", 3)],
    ]


def test_tail_batch_is_retained() -> None:
    items = [_item("a", 0, ordinal) for ordinal in range(5)]
    assert [len(batch) for batch in contiguous_batches(items, 4)] == [4, 1]


def test_reentry_and_reordering_fail_closed() -> None:
    with pytest.raises(ValueError, match="re-enters"):
        validate_trajectory_order(
            [_item("a", 0, 0), _item("b", 1, 0), _item("a", 0, 1)], metadata_of=_metadata
        )
    with pytest.raises(ValueError, match="original ordinal order"):
        validate_trajectory_order([_item("a", 0, 3), _item("a", 0, 2)], metadata_of=_metadata)


def test_data_source_reads_continuously_and_keeps_grpo_groups_separate() -> None:
    source = TrajectoryBatchDataSource.__new__(TrajectoryBatchDataSource)
    source.args = SimpleNamespace(n_samples_per_prompt=4, rollout_batch_size=4)
    source.samples = [
        *[_item("a", 0, ordinal) for ordinal in (0, 3, 6)],
        *[_item("b", 1, ordinal) for ordinal in (2, 5)],
    ]
    source.cursor = 0
    source.epoch_id = 0
    source.sample_group_index = 0
    source.sample_index = 0
    sampled = source.get_samples(4)
    assert [group[0].metadata["source_id"] for group in sampled] == ["a", "a", "a", "b"]
    assert [group[0].metadata["decision_ordinal"] for group in sampled] == [0, 3, 6, 2]
    assert all(len(group) == 4 for group in sampled)
    assert len({group[0].group_index for group in sampled}) == 4
    wrapped = source.get_samples(4)
    assert [group[0].metadata["source_id"] for group in wrapped] == ["b", "a", "a", "a"]
    assert source.epoch_id == 1
    assert len(source) == 8
