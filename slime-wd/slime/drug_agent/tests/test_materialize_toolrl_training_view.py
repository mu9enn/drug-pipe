from __future__ import annotations

import json
from pathlib import Path

from drug_agent.scripts.materialize_toolrl_training_view import materialize_toolrl_training_view
from drug_agent.scripts.validate_trajectory_toolrl_batches import validate_file


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("add_generation_prompt"):
            return " ".join(str(message.get("content") or "") for message in messages) + " GEN"
        if messages and messages[-1].get("role") == "assistant":
            return " ".join(str(message.get("content") or "") for message in messages[:-1]) + " GEN " + str(messages[-1].get("content") or "")
        return " ".join(str(message.get("content") or "") for message in messages)

    def encode(self, text, **kwargs):
        return text.split()


def _row(source_id: str, trajectory_index: int, ordinal: int, prompt_words: int, target_words: int) -> dict:
    return {
        "id": f"{source_id}:{ordinal}",
        "prompt": [{"role": "user", "content": " ".join(["prompt"] * prompt_words)}],
        "tools": [{"type": "function", "function": {"name": "Read", "parameters": {}}}],
        "label": {
            "decision_type": "final_answer",
            "target_assistant": {"role": "assistant", "content": " ".join(["target"] * target_words)},
        },
        "metadata": {
            "source_id": source_id,
            "trajectory_index": trajectory_index,
            "decision_ordinal": ordinal,
            "decision_type": "final_answer",
        },
    }


def test_length_findings_do_not_delete_rows_and_tail_is_retained(tmp_path: Path) -> None:
    rows = [
        _row("a", 0, 0, 2, 2),
        _row("a", 0, 3, 8, 2),
        _row("b", 1, 1, 2, 8),
    ]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "view.jsonl"
    manifest = materialize_toolrl_training_view(
        input_path=source,
        output_path=output,
        manifest_path=tmp_path / "manifest.json",
        tokenizer=_Tokenizer(),
        model_name="fake",
        max_prompt_tokens=5,
        max_target_tokens=5,
        rollout_batch_size=2,
    )
    output_rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert manifest["schema_version"] == "toolrl_v8_training_stream_v1"
    assert len(manifest["length_findings"]) == 2
    assert output_rows == rows
    assert json.loads(output.with_suffix(".pretty.json").read_text()) == rows
    validation = validate_file(output, 2)
    assert validation["rollout_batches_including_tail"] == 2
    assert validation["tail_batch_size"] == 1


def test_out_of_order_stream_fails(tmp_path: Path) -> None:
    rows = [_row("a", 0, 3, 2, 2), _row("a", 0, 1, 2, 2)]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    try:
        materialize_toolrl_training_view(
            input_path=source,
            output_path=tmp_path / "view.jsonl",
            manifest_path=tmp_path / "manifest.json",
            tokenizer=_Tokenizer(), model_name="fake", max_prompt_tokens=5,
            max_target_tokens=5, rollout_batch_size=2,
        )
    except ValueError as error:
        assert "original ordinal order" in str(error)
    else:
        raise AssertionError("out-of-order stream was accepted")
