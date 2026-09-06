from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from drug_agent.scripts.materialize_toolrl_training_view import materialize_toolrl_training_view
from drug_agent.scripts.validate_trajectory_toolrl_batches import validate_file


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return " ".join(str(message.get("content") or "") for message in messages)

    def __call__(self, texts, **kwargs):
        return {"input_ids": [text.split() for text in texts]}


def _row(source_id: str, ordinal: int, count: int, prompt_words: int, target_words: int) -> dict:
    target = " ".join(["target"] * target_words)
    return {
        "prompt": [{"role": "user", "content": " ".join(["prompt"] * prompt_words)}],
        "tools": [{"type": "function", "function": {"name": "Read", "parameters": {}}}],
        "label": {
            "decision_type": "tool_call",
            "target_assistant": {"role": "assistant", "content": target},
        },
        "metadata": {
            "source_id": source_id,
            "decision_ordinal": ordinal,
            "trajectory_decision_count": count,
            "decision_type": "tool_call",
            "task_type": "kg",
            "tool_names": ["Read"],
        },
    }


class MaterializeTrajectoryToolRLTest(unittest.TestCase):
    def test_capacity_filter_rejects_whole_trajectory_and_never_pads_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                _row("a", 0, 2, 2, 2),
                _row("a", 1, 2, 3, 2),
                _row("b", 0, 2, 2, 2),
                _row("b", 1, 2, 4, 2),
                _row("too-long", 0, 2, 8, 2),
                _row("too-long", 1, 2, 2, 2),
            ]
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = root / "view.jsonl"
            manifest = materialize_toolrl_training_view(
                input_path=source,
                output_path=output,
                manifest_path=root / "manifest.json",
                tokenizer=_Tokenizer(),
                model_name="fake",
                max_prompt_tokens=5,
                max_target_tokens=5,
                rollout_batch_size=4,
            )
            output_rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(manifest["schema_version"], "toolrl_trajectory_batched_view_v1")
            self.assertEqual(manifest["accepted_trajectories"], 2)
            self.assertEqual(manifest["accepted_records"], 4)
            self.assertEqual(manifest["rejected_trajectories"], 1)
            self.assertEqual(manifest["rejected_records"], 2)
            self.assertNotIn("padding_records", manifest)
            self.assertEqual([row["metadata"]["source_id"] for row in output_rows], ["a", "a", "b", "b"])
            self.assertEqual(
                [row["metadata"]["trajectory_batch_position"] for row in output_rows],
                [0, 1, 2, 3],
            )
            self.assertEqual({row["metadata"]["trajectory_batch_id"] for row in output_rows}, {0})
            pretty_rows = json.loads(output.with_suffix(".pretty.json").read_text())
            self.assertEqual(pretty_rows, output_rows)
            validation = validate_file(output, 4)
            self.assertEqual(validation["trajectory_batches"], 1)
            self.assertEqual(validation["trajectories"], 2)

    def test_incomplete_trajectory_is_rejected_before_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source.write_text(json.dumps(_row("broken", 1, 2, 2, 2)) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trajectory broken is incomplete"):
                materialize_toolrl_training_view(
                    input_path=source,
                    output_path=root / "view.jsonl",
                    manifest_path=root / "manifest.json",
                    tokenizer=_Tokenizer(),
                    model_name="fake",
                    max_prompt_tokens=5,
                    max_target_tokens=5,
                    rollout_batch_size=2,
                )


if __name__ == "__main__":
    unittest.main()
