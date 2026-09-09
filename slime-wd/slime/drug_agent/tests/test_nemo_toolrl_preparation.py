import json

from drug_agent.scripts.prepare_nemo_toolrl import encoding_text
from drug_agent.scripts.materialize_toolrl_v8_length_tiers import materialize


def test_full_document_preserves_history_parameters_and_target_boundaries():
    large = "before" + "x" * 100000 + "middle parameter" + "y" * 100000 + "after"
    calls = [{"function": {"name": name, "arguments": {"content": large}}} for name in ("write", "read")]
    row = {"id": "not-content", "prompt": [{"role": "user", "content": large}],
           "tools": [{"type": "function", "function": {"name": "write"}}],
           "label": {"target_assistant": {"role": "assistant", "reasoning_content": large, "tool_calls": calls}},
           "metadata": {"selection_description": "do not embed this", "future": "excluded"}}
    doc = json.loads(encoding_text(row))
    assert doc["history_before_current_decision"] == row["prompt"]
    assert doc["available_tool_schemas"] == row["tools"]
    assert doc["current_teacher_response"] == row["label"]["target_assistant"]
    assert set(doc) == {"history_before_current_decision", "available_tool_schemas", "current_teacher_response"}


def test_length_partition_can_omit_pretty_without_changing_rows(tmp_path):
    rows = [{"id": str(i), "prompt": [{"role": "user", "content": "original"}], "tools": [], "label": {"x": i}} for i in range(3)]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    details = tmp_path / "details.jsonl"
    fields = ["reasoning_tokens", "structured_action_or_final_tokens", "native_action_or_final_tokens", "prompt_plus_reference_tokens"]
    details.write_text("".join(json.dumps({"decision_id": str(i), "prompt_tokens": 10,
                                          "full_reference_tokens": n, **dict.fromkeys(fields, 1)}) + "\n"
                               for i, n in enumerate([100, 40000, 70000])))
    root = tmp_path / "output"
    result = materialize(source, details, root, emit_pretty=False)
    for i, tier in enumerate(("standard_32k", "long_64k_candidate", "oversize_review")):
        assert json.loads((root / f"{tier}.jsonl").read_text()) == rows[i]
        assert result["outputs"][tier]["records"] == 1
        assert "pretty_json" not in result["outputs"][tier]
    assert not list(root.glob("*.pretty.json"))
