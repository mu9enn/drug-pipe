from __future__ import annotations

import asyncio
from types import SimpleNamespace

from drug_agent.toolrl.molclaw_reward import reward_func


def _sample(response: str, label: dict, metadata: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        prompt=[{"role": "user", "content": "prompt"}],
        response=response,
        label=label,
        metadata=metadata or {},
    )


def test_reward_perfect_match_near_one():
    sample = _sample(
        '<thought>t</thought><tool_call>{"tool_name":"fix_pdb","arguments":{"input_path":"/tmp/a.pdb","remove_water":true}}</tool_call>',
        {
            "target_tool_calls": [
                {"tool_name": "fix_pdb", "arguments": {"input_path": "/tmp/a.pdb", "remove_water": True}}
            ]
        },
    )
    out = asyncio.run(reward_func(None, sample))
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
    out = asyncio.run(reward_func(None, sample))
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
    out = asyncio.run(reward_func(None, sample))
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
    out = asyncio.run(reward_func(None, sample))
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
    out = asyncio.run(reward_func(None, sample))
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
    out = asyncio.run(reward_func(None, sample))
    assert out["score"] > 0.7
    assert out["matched_calls"] == 1
