from __future__ import annotations

import json
from pathlib import Path

from drug_agent.scripts.materialize_decision_aware_toolrl_view import materialize_decision_aware_view


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        text = " ".join(str(message.get("content") or "") for message in messages)
        return text.split() if kwargs.get("tokenize") else text

    def encode(self, text, **kwargs):
        return str(text).split()


def _row(
    source_id: str,
    assistant_index: int,
    role: str,
    tool: str | None,
    *,
    task_type: str = "kg",
    repeated_trajectory: bool = False,
    repeated: bool = False,
    prompt_words: int = 4,
) -> dict:
    decision_type = "final_answer" if role == "final" else "tool_call"
    calls = [] if tool is None else [{"tool_name": tool, "arguments": {"value": assistant_index}}]
    final = {"task_type": task_type, "result": source_id, "evidence": []} if role == "final" else None
    return {
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": " ".join(["prompt"] * prompt_words)},
        ],
        "label": {
            "decision_type": decision_type,
            "target_tool_calls": calls,
            "target_final_answer": final,
        },
        "metadata": {
            "source_id": source_id,
            "task_id": source_id,
            "task_type": task_type,
            "assistant_index": assistant_index,
            "decision_type": decision_type,
            "decision_role": role,
            "decision_ordinal": assistant_index,
            "trajectory_decision_count": 5,
            "tool_names": [tool] if tool else [],
            "target_tool_calls": calls,
            "target_final_answer": final,
            "trajectory_has_repeated_tool_call": repeated_trajectory,
            "is_repeated_tool_call": repeated,
        },
    }


def test_curated_view_is_deterministic_weighted_and_audited(tmp_path: Path):
    rows = [
        _row("a", 0, "planning", "Rare", task_type="kg", repeated_trajectory=True),
        _row("a", 1, "tool_step", "Common", repeated_trajectory=True),
        _row("a", 2, "tool_step", "Common", repeated_trajectory=True, repeated=True),
        _row("a", 3, "final", None, task_type="kg", repeated_trajectory=True),
        _row("b", 0, "initial_tool_step", "Common", task_type="ac"),
        _row("b", 1, "tool_step", "Rare", task_type="ac"),
        _row("b", 2, "tool_step", "Common", task_type="ac"),
        _row("b", 3, "final", None, task_type="ac"),
    ]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    outputs = []
    manifests = []
    for index in range(2):
        output = tmp_path / f"view-{index}.jsonl"
        manifest_path = tmp_path / f"manifest-{index}.json"
        manifest = materialize_decision_aware_view(
            input_path=source,
            output_path=output,
            manifest_path=manifest_path,
            tokenizer=_Tokenizer(),
            model_name="fake",
            max_prompt_tokens=100,
            max_response_tokens=100,
            summary_max_tokens=20,
            intermediate_budget=2,
            min_per_tool=1,
            max_per_trajectory=1,
            multiple=4,
            seed=42,
        )
        outputs.append(output.read_text(encoding="utf-8"))
        manifests.append(manifest)
    assert outputs[0] == outputs[1]
    manifest = manifests[0]
    assert manifest["unique_records"] == 6
    assert manifest["intermediate_selected"] == 2
    assert manifest["effective_records"] % 4 == 0
    assert manifest["coverage"]["missing_tools"] == []
    assert len(manifest["selection"]) == len(rows)
    assert any(
        not record["selected"] and record["exclusion_reason"]
        for record in manifest["selection"]
    )
    output_rows = [json.loads(line) for line in outputs[0].splitlines()]
    planning = [row for row in output_rows if row["metadata"]["decision_role"] == "planning"]
    repeated_final = [
        row
        for row in output_rows
        if row["metadata"]["decision_role"] == "final"
        and row["metadata"]["trajectory_has_repeated_tool_call"]
    ]
    assert sum("planning_2x" in row["metadata"]["sampling_reason"] for row in planning) == 1
    assert sum(
        "repeated_trajectory_final_3x" in row["metadata"]["sampling_reason"]
        for row in repeated_final
    ) == 1
    assert all("sampling_reason" in row["metadata"] for row in output_rows)
    assert all("sampling_copy_index" in row["metadata"] for row in output_rows)
