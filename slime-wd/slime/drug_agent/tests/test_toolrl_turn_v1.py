from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from drug_agent.data.validate_sft_messages import audit_react_actions
from drug_agent.protocol.react_protocol import parse_runtime_decision
from drug_agent.protocol.toolrl_turn import normalize_trajectory, serialize_decision, split_assistant_segments
from drug_agent.scripts.audit_runtime_parser_compatibility import audit as audit_runtime_parser
from drug_agent.scripts.audit_sft_toolrl_serializer_parity import audit as audit_serializer_parity
from drug_agent.scripts.materialize_toolrl_turn_v6 import _sft_records_for_trajectory
from drug_agent.scripts.select_toolrl_decisions import _canonical_target
from drug_agent.scripts.validate_fixed_toolrl_traversal import validate as validate_fixed_traversal
from drug_agent.toolrl.convert_react_to_toolrl_steps import convert_react_to_toolrl_steps
from drug_agent.toolrl.molclaw_reward import reward_func
from drug_agent.toolrl.official_grpo import compute_official_8cee13e_advantages
from drug_agent.toolrl.prompt_strategy import apply_prompt_strategy


def _sample(response: str, calls: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        prompt=[{"role": "user", "content": "task"}],
        response=response,
        label={"decision_type": "tool_call", "target_tool_calls": calls},
        metadata={"protocol": "toolrl_turn_v1", "decision_role": "tool_step"},
    )


def _reward(sample: SimpleNamespace) -> dict:
    import os

    previous = os.environ.get("TOOLRL_REWARD_MODE")
    os.environ["TOOLRL_REWARD_MODE"] = "hierarchical"
    try:
        return asyncio.run(reward_func(None, sample))
    finally:
        if previous is None:
            os.environ.pop("TOOLRL_REWARD_MODE", None)
        else:
            os.environ["TOOLRL_REWARD_MODE"] = previous


def _reward_mode(sample: SimpleNamespace, mode: str) -> dict:
    import os

    previous = os.environ.get("TOOLRL_REWARD_MODE")
    os.environ["TOOLRL_REWARD_MODE"] = mode
    try:
        return asyncio.run(reward_func(None, sample))
    finally:
        if previous is None:
            os.environ.pop("TOOLRL_REWARD_MODE", None)
        else:
            os.environ["TOOLRL_REWARD_MODE"] = previous


def test_serializer_groups_calls_and_merges_thoughts_without_reordering():
    source = {
        "id": "react_kg_turn",
        "messages": [
            {"role": "system", "content": "Use ReAct."},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": (
                    "<thought>first</thought><thought>second</thought>"
                    '<tool_call>{"tool_name":"Read","arguments":{"file_path":"a"}}</tool_call>'
                    '<tool_call>{"tool_name":"Glob","arguments":{"pattern":"*.pdb"}}</tool_call>'
                ),
            },
            {"role": "user", "content": '<observation tool_name="Read">{"status":"success"}</observation>'},
        ],
    }
    normalized, audit = normalize_trajectory(source)
    content = normalized["messages"][2]["content"]
    assert content.count("<thought>") == 1
    assert "first\n\nsecond" in content
    assert content.count("<tool_call>") == 1
    assert content.index('"tool_name":"Read"') < content.index('"tool_name":"Glob"')
    assert normalized["messages"][3] == source["messages"][3]
    assert audit["multi_call_turns"] == 1
    parsed = parse_runtime_decision(content, strict_toolrl_turn=True)
    assert [call["tool_name"] for call in parsed["tool_calls"]] == ["Read", "Glob"]


def test_serializer_does_not_invent_missing_thought():
    content = serialize_decision(tool_calls=[{"tool_name": "Read", "arguments": {"file_path": "a"}}])
    assert "<thought>" not in content
    assert parse_runtime_decision(content, strict_toolrl_turn=True)["ok"] is True


def test_thought_after_action_becomes_prefix_conditioned_decisions(tmp_path: Path):
    source_content = (
        "<thought>thought 1</thought>"
        '<tool_call>{"tool_name":"Read","arguments":{"file_path":"a"}}</tool_call>'
        "<thought>thought 2</thought>"
        '<tool_call>{"tool_name":"Glob","arguments":{"pattern":"*.pdb"}}</tool_call>'
        '<tool_call>{"tool_name":"Bash","arguments":{"command":"pwd"}}</tool_call>'
    )
    source = {
        "id": "react_kg_subturn",
        "messages": [
            {"role": "system", "content": "Use ReAct."},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": source_content},
        ],
    }
    normalized, audit = normalize_trajectory(source)
    content = normalized["messages"][2]["content"]
    segments = split_assistant_segments(content)
    assert len(segments) == 2
    assert [call["tool_name"] for call in segments[0]["tool_calls"]] == ["Read"]
    assert [call["tool_name"] for call in segments[1]["tool_calls"]] == ["Glob", "Bash"]
    assert segments[0]["thoughts"] == ["thought 1"]
    assert segments[1]["thoughts"] == ["thought 2"]
    assert audit["assistant_turns"] == 1
    assert audit["expanded_decisions"] == 2
    assert audit["multi_segment_turns"] == 1
    assert audit["causal_interleaving_quarantines"] == 0
    assert normalized["messages"][2]["step_loss_mask"] == 0

    input_path = tmp_path / "sft.jsonl"
    output_path = tmp_path / "steps.jsonl"
    input_path.write_text(json.dumps(normalized, ensure_ascii=False) + "\n", encoding="utf-8")
    report = convert_react_to_toolrl_steps(input_path, output_path)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert report["kept_rows"] == 2
    assert len(rows) == 2
    assert rows[0]["label"]["assistant_content"] == segments[0]["content"]
    assert rows[0]["metadata"]["assistant_prefix"] == ""
    assert rows[1]["label"]["assistant_content"] == segments[1]["content"]
    assert rows[1]["metadata"]["assistant_prefix"] == segments[0]["content"] + "\n"


def test_thought_only_incomplete_tail_is_not_a_decision(tmp_path: Path):
    source = {
        "id": "tail",
        "messages": [
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": '<tool_call>{"tool_name":"Read","arguments":{}}</tool_call>'
                '<thought>unfinished next rationale</thought>',
            },
        ],
    }
    normalized, audit = normalize_trajectory(source)
    assert audit["expanded_decisions"] == 1
    assert audit["thought_only_incomplete_tails"] == 1
    assert audit["multi_segment_turns"] == 0
    assert audit["interleaved_segment_turns"] == 1
    assert normalized["messages"][1]["step_loss_mask"] == 0
    input_path = tmp_path / "sft.jsonl"
    output_path = tmp_path / "steps.jsonl"
    input_path.write_text(json.dumps(normalized) + "\n", encoding="utf-8")
    convert_react_to_toolrl_steps(input_path, output_path)
    assert len(output_path.read_text().splitlines()) == 1


def test_standalone_thought_only_message_is_preserved_but_not_supervised(tmp_path: Path):
    source = {
        "id": "standalone-tail",
        "messages": [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "<thought>unfinished rationale</thought>"},
            {"role": "user", "content": "<observation>async result</observation>"},
        ],
    }
    normalized, audit = normalize_trajectory(source)
    assert audit["expanded_decisions"] == 0
    assert audit["thought_only_incomplete_tails"] == 1
    assert normalized["messages"][1]["content"] == "<thought>unfinished rationale</thought>"
    assert normalized["messages"][1]["step_loss_mask"] == 0

    input_path = tmp_path / "sft.jsonl"
    output_path = tmp_path / "steps.jsonl"
    input_path.write_text(json.dumps(normalized) + "\n", encoding="utf-8")
    report = convert_react_to_toolrl_steps(input_path, output_path)
    assert report["kept_rows"] == 0
    assert output_path.read_text() == ""


def test_prefix_conditioned_sft_targets_match_toolrl_gold(tmp_path: Path):
    source = {
        "id": "parity",
        "messages": [
            {"role": "system", "content": "Use ReAct."},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": (
                    "<thought>T1</thought>"
                    '<tool_call>{"tool_name":"Read","arguments":{"file_path":"a"}}</tool_call>'
                    "<thought>T2</thought>"
                    '<tool_call>{"tool_name":"Glob","arguments":{"pattern":"*.pdb"}}</tool_call>'
                    '<tool_call>{"tool_name":"Bash","arguments":{"command":"pwd"}}</tool_call>'
                ),
                "step_loss_mask": 1,
            },
        ],
    }
    normalized, _ = normalize_trajectory(source)
    sft_records = _sft_records_for_trajectory(normalized)
    assert len(sft_records) == 3
    assert sft_records[0]["messages"][-1]["step_loss_mask"] == 0
    assert sft_records[1]["messages"][-1]["loss_char_start"] == 0
    assert sft_records[2]["messages"][-1]["loss_char_start"] > 0

    sft_path = tmp_path / "sft.jsonl"
    canonical_path = tmp_path / "canonical.jsonl"
    rl_path = tmp_path / "rl.jsonl"
    sft_path.write_text("".join(json.dumps(row) + "\n" for row in sft_records), encoding="utf-8")
    canonical_path.write_text(json.dumps(normalized) + "\n", encoding="utf-8")
    convert_react_to_toolrl_steps(canonical_path, rl_path)
    report = audit_serializer_parity(sft_path, rl_path)
    assert report["ok"] is True
    assert report["supplemental_prefix_conditioned_targets"] == 2
    assert report["toolrl_gold_actions_compared"] == 2


@pytest.mark.parametrize(
    "response",
    [
        '<tool_call>{"tool_name":"Read","arguments":{}},{"tool_name":"Glob","arguments":{}}</tool_call>',
        '<tool_call>[{"tool_name":"Read","arguments":{}}]</tool_call>',
        '<tool_call>{"tool_name":"Read","arguments":{}}</tool_call><tool_call>{"tool_name":"Glob","arguments":{}}</tool_call>',
        '<tool_call>{"tool_name":"Read","arguments":{}}</tool_call> trailing',
    ],
)
def test_strict_turn_reward_rejects_noncanonical_envelopes(response: str):
    out = _reward(_sample(response, [{"tool_name": "Read", "arguments": {}}]))
    assert out["score"] == -0.5
    assert out["diagnostics"]["reward_stage"] == "invalid_react_tool_envelope"


def test_multi_call_reward_is_order_insensitive_inside_one_container():
    calls = [
        {"tool_name": "Read", "arguments": {"file_path": "a"}},
        {"tool_name": "Glob", "arguments": {"pattern": "*.pdb"}},
    ]
    response = serialize_decision(tool_calls=list(reversed(calls)))
    out = _reward(_sample(response, calls))
    assert out["score"] == 1.0
    assert out["diagnostics"]["matched_calls"] == 2


def test_v6_canonical_target_keeps_thought_and_single_container():
    content = serialize_decision(
        thoughts=["reason"],
        tool_calls=[{"tool_name": "Read", "arguments": {"file_path": "a"}}],
    )
    row = {
        "label": {"protocol": "toolrl_turn_v1", "assistant_content": content},
        "metadata": {"protocol": "toolrl_turn_v1"},
    }
    assert _canonical_target(row) == content


def test_official_reward_preserves_repeated_same_tool_multiplicity():
    gold = [
        {"tool_name": "Read", "arguments": {"file_path": "a"}},
        {"tool_name": "Read", "arguments": {"file_path": "b"}},
    ]
    one = serialize_decision(thoughts=["inspect"], tool_calls=gold[:1])
    both_reordered = serialize_decision(thoughts=["inspect"], tool_calls=list(reversed(gold)))
    one_score = _reward_mode(_sample(one, gold), "toolrl_official_8cee13e")["score"]
    both_score = _reward_mode(_sample(both_reordered, gold), "toolrl_official_8cee13e")["score"]
    assert one_score < 4.0
    assert both_score == 4.0


def test_official_final_is_format_only_tag_rename():
    sample = SimpleNamespace(
        prompt=[{"role": "user", "content": "task"}],
        response='<thought>done</thought>\n<final_answer>different free-form response</final_answer>',
        label={"decision_type": "final_answer", "target_final_answer": {"task_type": "kg", "result": "teacher", "evidence": []}},
        metadata={"protocol": "toolrl_turn_v1", "decision_role": "final"},
    )
    out = _reward_mode(sample, "toolrl_official_8cee13e")
    assert out["score"] == 1.0
    assert out["components"]["correctness"] == 0.0


def test_structured_final_exact_is_an_independent_extension(monkeypatch):
    sample = SimpleNamespace(
        prompt=[{"role": "user", "content": "task"}],
        response=(
            '<thought>done</thought>\n<final_answer>'
            '{"task_type":"kg","result":"different","evidence":[]}'
            '</final_answer>'
        ),
        label={
            "decision_type": "final_answer",
            "target_final_answer": {"task_type": "kg", "result": "teacher", "evidence": []},
        },
        metadata={"protocol": "toolrl_turn_v1", "decision_role": "final"},
    )
    monkeypatch.setenv("TOOLRL_STRUCTURED_FINAL_EXACT", "1")
    assert _reward_mode(sample, "hierarchical")["score"] == -0.5
    monkeypatch.setenv("TOOLRL_STRUCTURED_FINAL_EXACT", "0")
    out = _reward_mode(sample, "hierarchical")
    assert out["score"] == 1.0
    assert out["diagnostics"]["structured_final_exact_enabled"] is False


def test_official_kl_is_applied_before_group_normalization():
    data = {
        "raw_reward": [-3.0] * 4,
        "kl": [torch.tensor([value, value]) for value in (0.1, 0.2, 0.3, 0.4)],
        "loss_masks": [torch.ones(2) for _ in range(4)],
    }
    args = SimpleNamespace(n_samples_per_prompt=4, kl_coef=0.001)
    compute_official_8cee13e_advantages(args, data)
    values = torch.stack([item[0] for item in data["advantages"]])
    assert torch.isfinite(values).all()
    assert values.std() > 0
    assert abs(float(values.mean())) < 1e-3


def test_prompt_strategy_separates_official_catalog_from_skill_discovery():
    row = {"prompt": [{"role": "system", "content": "skills only"}, {"role": "user", "content": "task"}], "metadata": {}}
    catalog = {"tools": [{"name": "Read", "description": "read file", "input_schema": {"type": "object"}}]}
    baseline = apply_prompt_strategy(row, strategy="official_catalog", catalog=catalog)
    production = apply_prompt_strategy(row, strategy="drug_pipe_skill_discovery", catalog=catalog)
    assert "Available Tools" in baseline["prompt"][0]["content"]
    assert "Name: Read" in baseline["prompt"][0]["content"]
    assert production["prompt"][0]["content"] == "skills only"


def test_runtime_and_reward_parser_share_strict_multi_call_grammar():
    report = audit_runtime_parser()
    assert report["ok"] is True
    assert report["cases"]["whitespace_separated_objects"]["runtime_invocation_count"] == 3
    assert report["cases"]["comma_separated_objects"]["runtime_valid"] is False
    assert report["cases"]["json_array"]["runtime_valid"] is False
    assert report["cases"]["multiple_containers"]["runtime_valid"] is False


def test_sft_validator_tracks_all_invocations_in_one_container():
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": serialize_decision(
                tool_calls=[
                    {"tool_name": "dock", "arguments": {"ligand": "a"}},
                    {"tool_name": "dock", "arguments": {"ligand": "b"}},
                    {"tool_name": "score", "arguments": {"path": "x"}},
                ]
            ),
        },
        {
            "role": "user",
            "content": (
                '<observation tool_name="score">{"status":"success"}</observation>'
                '<observation tool_name="dock">{"status":"success"}</observation>'
                '<observation tool_name="dock">{"status":"success"}</observation>'
            ),
        },
    ]
    counts, issues = audit_react_actions(messages)
    assert counts["assistant_tool_call_total"] == 3
    assert counts["out_of_order_tool_results"] == 1
    assert counts["react_json_parse_failed"] == 0
    assert issues == []


def test_fixed_traversal_audit_checks_every_decision_once(tmp_path: Path):
    dataset = tmp_path / "steps.jsonl"
    audit = tmp_path / "audit.jsonl"
    rows = []
    audit_rows = []
    for index in range(4):
        metadata = {
            "source_id": f"source-{index}",
            "assistant_index": 2,
            "assistant_subturn_index": 0,
            "decision_type": "tool_call",
        }
        rows.append({"metadata": metadata})
        audit_rows.append(
            {
                "decision_key": f"source-{index}:2:0:tool_call",
                "group_index": index,
                "dataset_cursor": index,
                "dataset_epoch": 0,
                "accepted_for_update": True,
            }
        )
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    audit.write_text("".join(json.dumps(row) + "\n" for row in audit_rows), encoding="utf-8")
    report = validate_fixed_traversal(dataset, audit)
    assert report["ok"] is True
    assert report["consumed_decisions"] == 4

    audit.write_text("".join(json.dumps(row) + "\n" for row in audit_rows[:-1]), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_fixed_traversal(dataset, audit)
