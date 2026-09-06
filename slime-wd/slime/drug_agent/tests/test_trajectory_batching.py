from __future__ import annotations

from types import SimpleNamespace
import unittest

from drug_agent.toolrl.trajectory_batching import (
    flatten_batch,
    group_complete_trajectories,
    pack_complete_trajectories,
    validate_packed_decision_batches,
)
from drug_agent.toolrl.trajectory_data_source import TrajectoryBatchDataSource


def _item(source_id: str, ordinal: int, count: int, batch_id: int = -1, position: int = -1):
    return SimpleNamespace(
        metadata={
            "source_id": source_id,
            "decision_ordinal": ordinal,
            "trajectory_decision_count": count,
            "trajectory_batch_id": batch_id,
            "trajectory_batch_position": position,
            "trajectory_batch_decision_count": 4,
        }
    )


def _metadata(item):
    return item.metadata


class TrajectoryBatchingTest(unittest.TestCase):
    def test_multiple_complete_trajectories_share_batch_without_interleaving(self) -> None:
        items = [_item("a", 1, 2), _item("b", 0, 2), _item("a", 0, 2), _item("b", 1, 2)]
        trajectories = group_complete_trajectories(items, metadata_of=_metadata)
        batches = pack_complete_trajectories(trajectories, 4)
        flattened = flatten_batch(batches[0])
        self.assertEqual(
            [(item.metadata["source_id"], item.metadata["decision_ordinal"]) for item in flattened],
            [("a", 0), ("a", 1), ("b", 0), ("b", 1)],
        )

    def test_packed_batch_rejects_trajectory_crossing_boundary(self) -> None:
        items = [
            _item("a", 0, 2, 0, 0),
            _item("b", 0, 2, 0, 1),
            _item("b", 1, 2, 0, 2),
            _item("c", 0, 1, 0, 3),
            _item("a", 1, 2, 1, 0),
            _item("d", 0, 3, 1, 1),
            _item("d", 1, 3, 1, 2),
            _item("d", 2, 3, 1, 3),
        ]
        with self.assertRaisesRegex(ValueError, "cross rollout batch boundaries"):
            validate_packed_decision_batches(items, metadata_of=_metadata, rollout_batch_size=4)

    def test_packed_batch_rejects_interleaved_trajectories(self) -> None:
        items = [
            _item("a", 0, 2, 0, 0),
            _item("b", 0, 2, 0, 1),
            _item("a", 1, 2, 0, 2),
            _item("b", 1, 2, 0, 3),
        ]
        with self.assertRaisesRegex(ValueError, "interleaved inside rollout batch"):
            validate_packed_decision_batches(items, metadata_of=_metadata, rollout_batch_size=4)

    def test_no_exact_whole_trajectory_pack_fails_closed(self) -> None:
        trajectories = [
            ("a", [_item("a", index, 3) for index in range(3)]),
            ("b", [_item("b", index, 3) for index in range(3)]),
            ("c", [_item("c", index, 2) for index in range(2)]),
        ]
        with self.assertRaisesRegex(ValueError, "cannot form an exact trajectory-atomic"):
            pack_complete_trajectories(trajectories, 4)

    def test_packer_places_long_trajectories_before_short_fillers(self) -> None:
        trajectories = [
            ("one-a", [_item("one-a", 0, 1)]),
            ("one-b", [_item("one-b", 0, 1)]),
            ("three-a", [_item("three-a", index, 3) for index in range(3)]),
            ("three-b", [_item("three-b", index, 3) for index in range(3)]),
        ]
        batches = pack_complete_trajectories(trajectories, 4)
        self.assertEqual(
            [[source_id for source_id, _ in batch] for batch in batches],
            [["three-a", "one-a"], ["three-b", "one-b"]],
        )

    def test_data_source_returns_one_prepacked_batch_without_decision_shuffle(self) -> None:
        source = TrajectoryBatchDataSource.__new__(TrajectoryBatchDataSource)
        source.args = SimpleNamespace(rollout_batch_size=4, n_samples_per_prompt=2, rollout_seed=42)
        source.shuffle_batches = False
        source.batches = [
            [_item("a", 0, 2), _item("a", 1, 2), _item("b", 0, 2), _item("b", 1, 2)],
            [_item("c", index, 4) for index in range(4)],
        ]
        source.sample_group_index = 0
        source.sample_index = 0
        source._set_epoch(0)
        sampled = source.get_samples(4)
        self.assertEqual(len(sampled), 4)
        self.assertTrue(all(len(group) == 2 for group in sampled))
        self.assertEqual([group[0].metadata["source_id"] for group in sampled], ["a", "a", "b", "b"])
        self.assertEqual([group[0].metadata["decision_ordinal"] for group in sampled], [0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
