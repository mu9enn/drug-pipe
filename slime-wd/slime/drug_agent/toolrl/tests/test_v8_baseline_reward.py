from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

from drug_agent.protocol.toolrl_turn import serialize_decision
from drug_agent.toolrl.molclaw_reward import reward_func


def _score(response: str, target: dict) -> dict:
    sample = SimpleNamespace(
        prompt=[{"role": "user", "content": "task"}],
        response=response,
        label={"decision_type": "final_answer", "target_final_answer": target},
        metadata={"protocol": "toolrl_turn_v1", "decision_role": "final"},
    )
    previous = os.environ.get("TOOLRL_REWARD_MODE")
    os.environ["TOOLRL_REWARD_MODE"] = "v8_baseline"
    try:
        return asyncio.run(reward_func(None, sample))
    finally:
        if previous is None:
            os.environ.pop("TOOLRL_REWARD_MODE", None)
        else:
            os.environ["TOOLRL_REWARD_MODE"] = previous


def test_v8_final_reward_ignores_evidence_but_requires_task_answer():
    target = {
        "task_type": "pf",
        "selected_smiles": ["CC"],
        "evidence": [{"tool_name": "calculate_mol_basic_info"}],
        "summary": "teacher prose",
    }
    response = serialize_decision(
        thoughts=["computed"],
        final_answer={
            "task_type": "pf",
            "selected_smiles": ["CC"],
            "evidence": [],
            "summary": "different prose",
        },
    )
    out = _score(response, target)
    assert out["score"] == 1.0
    assert out["components"]["terminal_correctness"] == 1.0


def test_v8_final_reward_rejects_format_correct_wrong_answer():
    target = {"task_type": "pf", "selected_smiles": ["CC"]}
    response = serialize_decision(
        thoughts=["guessed"],
        final_answer={"task_type": "pf", "selected_smiles": ["CO"]},
    )
    out = _score(response, target)
    assert out["score"] == -0.5
    assert out["components"]["terminal_correctness"] == 0.0
