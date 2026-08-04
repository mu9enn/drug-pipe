from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

from drug_agent.toolrl.molclaw_reward import reward_func


def _sample(response: str, label: dict, metadata: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        prompt=[{"role": "user", "content": "prompt"}],
        response=response,
        label=label,
        metadata=metadata or {},
    )


def _reward(sample, mode="molclaw"):
    previous = os.environ.get("TOOLRL_REWARD_MODE")
    os.environ["TOOLRL_REWARD_MODE"] = mode
    try:
        return asyncio.run(reward_func(None, sample))
    finally:
        if previous is None:
            os.environ.pop("TOOLRL_REWARD_MODE", None)
        else:
            os.environ["TOOLRL_REWARD_MODE"] = previous


def test_reward_perfect_match_near_one():
    sample = _sample(
        '<thought>t</thought><tool_call>{"tool_name":"fix_pdb","arguments":{"input_path":"/tmp/a.pdb","remove_water":true}}</tool_call>',
        {
            "target_tool_calls": [
                {"tool_name": "fix_pdb", "arguments": {"input_path": "/tmp/a.pdb", "remove_water": True}}
            ]
        },
    )
    out = _reward(sample)
    assert out["score"] > 0.95
    assert out["matched_calls"] == 1
    assert out["tool_name"] > 0.9
    assert out["param_name"] > 0.9
    assert out["param_value"] > 0.9


def test_reward_order_insensitive_multi_tool_calls():
    sample = _sample(
        (
            '<tool_call>{"tool_name":"is_valid_smiles","arguments":{"smiles_list":["CCO","CCN"]}}</tool_call>'
            '<tool_call>{"tool_name":"mcp__molclaw-scp__fix_pdb","arguments":{"input_path":"<artifact>","remove_water":"true"}}</tool_call>'
        ),
        {
            "target_tool_calls": [
                {"tool_name": "fix_pdb", "arguments": {"input_path": "/tmp/a.pdb", "remove_water": True}},
                {"tool_name": "is_valid_smiles", "arguments": {"smiles_list": ["CCO", "CCN"]}},
            ]
        },
    )
    out = _reward(sample)
    assert out["score"] > 0.7
    assert out["matched_calls"] == 2


def test_reward_hyphen_tool_name_no_longer_matches_underscore():
    sample = _sample(
        '<tool_call>{"tool_name":"mcp__molclaw-scp__fix-pdb","arguments":{"input_path":"/tmp/a.pdb","remove_water":true}}</tool_call>',
        {
            "target_tool_calls": [
                {"tool_name": "fix_pdb", "arguments": {"input_path": "/tmp/a.pdb", "remove_water": True}}
            ]
        },
    )
    out = _reward(sample)
    assert out["matched_calls"] == 0
    assert out["tool_name"] == 0.0
    assert out["score"] < 0.4


def test_reward_parameter_alias_no_longer_matches():
    sample = _sample(
        '<tool_call>{"tool_name":"pred_binding_affinity_boltz2","arguments":{"protein_path":"/tmp/p.pdb","smiles":"CCO"}}</tool_call>',
        {
            "target_tool_calls": [
                {"tool_name": "pred_binding_affinity_boltz2", "arguments": {"protein_path": "/tmp/p.pdb", "ligand_smiles": "CCO"}}
            ]
        },
    )
    out = _reward(sample)
    assert out["matched_calls"] == 1
    assert out["param_name"] < 1.0


def test_reward_missing_and_extra_params_penalized():
    sample = _sample(
        '<tool_call>{"tool_name":"fix_pdb","arguments":{"input_path":"/tmp/a.pdb","extra":1}}</tool_call>',
        {
            "target_tool_calls": [
                {"tool_name": "fix_pdb", "arguments": {"input_path": "/tmp/a.pdb", "remove_water": True}}
            ]
        },
    )
    out = _reward(sample)
    assert out["score"] < 0.95
    assert out["param_name"] < 1.0


def test_reward_bool_number_smiles_and_artifact_matching():
    sample = _sample(
        '<tool_call>{"tool_name":"pred_pocket_prank","arguments":{"input_path":"<artifact>","top_n":"5","radius":"1.5"}}</tool_call>',
        {
            "target_tool_calls": [
                {"tool_name": "pred_pocket_prank", "arguments": {"input_path": "/tmp/complex.pdb", "top_n": 5, "radius": 1.5}}
            ]
        },
    )
    out = _reward(sample)
    assert out["score"] > 0.7
    assert out["matched_calls"] == 1


def test_official_reward_exact_tool_match_uses_official_range():
    sample = _sample(
        '<thought>t</thought><tool_call>{"tool_name":"fix_pdb","arguments":{"input_path":"x"}}</tool_call>',
        {
            "decision_type": "tool_call",
            "target_tool_calls": [{"tool_name": "fix_pdb", "arguments": {"input_path": "x"}}],
        },
    )
    out = _reward(sample, mode="official")
    assert out["score"] == 4.0
    assert out["components"]["correctness"] == 3.0


def test_official_final_answer_extension_scores_structured_result_without_summary():
    final = {"task_type": "kg", "result": "artifact", "evidence": []}
    sample = _sample(
        '<thought>done</thought><final_answer>' + __import__("json").dumps(final) + '</final_answer>',
        {"decision_type": "final_answer", "target_final_answer": {**final, "summary": "duplicate"}},
    )
    out = _reward(sample, mode="official")
    assert out["score"] == 4.0
    assert out["diagnostics"]["official_toolrl_extension"] == "drug_pipe_terminal_decision_extension"


def test_molclaw_terminal_exact_answer_scores_one_without_summary():
    final = {"task_type": "kg", "result": "artifact", "evidence": []}
    sample = _sample(
        '<thought>done</thought><final_answer>' + __import__("json").dumps(final) + '</final_answer>',
        {"decision_type": "final_answer", "target_final_answer": {**final, "summary": "duplicate"}},
    )
    out = _reward(sample)
    assert out["score"] == 1.0
    assert out["diagnostics"]["terminal_exact_match"] is True


def test_molclaw_terminal_malformed_empty_call_set_is_negative():
    sample = _sample(
        "unstructured output with no final answer",
        {"decision_type": "final_answer", "target_final_answer": {"result": "expected"}},
    )
    out = _reward(sample)
    assert out["score"] == -0.5
    assert out["diagnostics"]["terminal_exact_match"] is False
    assert out["components"]["tool_call_score"] == 0.0


def test_molclaw_terminal_wrong_valid_answer_is_negative():
    predicted = {"task_type": "kg", "result": "wrong", "evidence": []}
    expected = {"task_type": "kg", "result": "expected", "evidence": []}
    sample = _sample(
        '<thought>done</thought><final_answer>' + __import__("json").dumps(predicted) + '</final_answer>',
        {"decision_type": "final_answer", "target_final_answer": expected},
    )
    out = _reward(sample)
    assert out["score"] == -0.5
    assert out["errors"][0]["type"] == "FinalAnswerMismatch"


def test_molclaw_malformed_tool_response_cannot_receive_positive_dense_credit():
    sample = _sample(
        '<tool_call>{"tool_name":"fix_pdb","arguments":{"input_path":"x"}}</tool_call> garbage',
        {
            "decision_type": "tool_call",
            "target_tool_calls": [{"tool_name": "fix_pdb", "arguments": {"input_path": "x"}}],
        },
    )
    out = _reward(sample)
    assert out["diagnostics"]["parse_ok"] is False
    assert out["score"] <= -0.3
