from __future__ import annotations

import json
from pathlib import Path

from drug_agent.scripts.materialize_toolrl_training_view import materialize_toolrl_training_view


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return " ".join(str(message.get("content") or "") for message in messages)

    def __call__(self, texts, **kwargs):
        return {"input_ids": [text.split() for text in texts]}


def _row(index: int, prompt_words: int, target_words: int) -> dict:
    target = " ".join(["target"] * target_words)
    return {
        "prompt": [{"role": "user", "content": " ".join(["prompt"] * prompt_words)}],
        "label": {"assistant_content": target, "decision_type": "tool_call"},
        "target_assistant": {"content": target},
        "metadata": {
            "source_id": f"sample-{index}",
            "assistant_index": index,
            "decision_type": "tool_call",
            "task_type": "kg",
            "tool_names": ["Read"],
        },
    }


def test_capacity_filter_and_shortest_padding_are_audited(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in [_row(0, 2, 2), _row(1, 4, 2), _row(2, 8, 2), _row(3, 2, 9)]),
        encoding="utf-8",
    )
    output = tmp_path / "view.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = materialize_toolrl_training_view(
        input_path=source,
        output_path=output,
        manifest_path=manifest_path,
        tokenizer=_Tokenizer(),
        model_name="fake",
        max_prompt_tokens=5,
        max_target_tokens=5,
        multiple=4,
        batch_size=2,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert manifest["accepted_records"] == 2
    assert manifest["rejected_records"] == 2
    assert manifest["padding_records"] == 2
    assert manifest["output"]["records"] == 4
    assert [row["metadata"]["source_id"] for row in rows] == ["sample-0", "sample-1", "sample-0", "sample-1"]
    assert manifest["rejection_reason_counts"] == {
        "prompt_exceeds_max_tokens": 1,
        "target_exceeds_max_tokens": 1,
    }
