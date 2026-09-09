from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from drug_agent.scripts.select_toolrl_v8_trials import select as select_trials
from drug_agent.toolrl.molclaw_reward import _teacher_order_bonus
from drug_agent.toolrl.v8_dataset import (
    build_selection_description,
    cluster_and_select,
    materialize_decisions,
    quota_for_cluster,
)


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def test_materializer_uses_one_semantic_assistant_response_per_decision_and_no_future_leakage(tmp_path: Path) -> None:
    semantic = {
        "id": "trajectory-a",
        "user_task": "task",
        "metadata": {"task_type": "kg"},
        "events": [
            {"type": "assistant_decision", "source_message_id": "m1", "reasoning": "r1", "tool_calls": [
                {"name": "a", "arguments": {"x": 1}, "source_tool_use_id": "c1"},
                {"name": "b", "arguments": {"y": 2}, "source_tool_use_id": "c2"},
            ], "final_answer": None},
            {"type": "tool_observation", "name": "a", "source_tool_use_id": "c1", "content": "OBS_A"},
            {"type": "tool_observation", "name": "b", "source_tool_use_id": "c2", "content": "OBS_B"},
            {"type": "assistant_decision", "source_message_id": "m2", "reasoning": "r2", "tool_calls": [], "final_answer": '{"result":"ok"}'},
        ],
    }
    first_assistant = {"role": "assistant", "reasoning_content": "r1", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "a", "arguments": {"x": 1}}},
        {"id": "c2", "type": "function", "function": {"name": "b", "arguments": {"y": 2}}},
    ], "step_loss_mask": 1}
    qwen = {
        "id": "trajectory-a", "tools": [_tool("a"), _tool("b")], "messages": [
            {"role": "system", "content": "system", "step_loss_mask": 0},
            {"role": "user", "content": "task", "step_loss_mask": 0},
            first_assistant,
            {"role": "tool", "name": "a", "tool_call_id": "c1", "content": "OBS_A", "step_loss_mask": 0},
            {"role": "tool", "name": "b", "tool_call_id": "c2", "content": "OBS_B", "step_loss_mask": 0},
            {"role": "assistant", "reasoning_content": "r2", "content": '{"result":"ok"}', "step_loss_mask": 1},
        ],
    }
    semantic_path = tmp_path / "semantic.jsonl"
    qwen_path = tmp_path / "qwen.jsonl"
    semantic_path.write_text(json.dumps(semantic) + "\n")
    qwen_path.write_text(json.dumps(qwen) + "\n")
    rows, manifest = materialize_decisions(semantic_path, qwen_path)
    assert len(rows) == 2 and manifest["decision_records"] == 2
    assert [call["name"] for call in rows[0]["label"]["target_tool_calls"]] == ["a", "b"]
    assert "OBS_A" not in json.dumps(rows[0]["prompt"])
    assert "r1" not in json.dumps(rows[0]["prompt"])
    assert "OBS_A" in json.dumps(rows[1]["prompt"])
    assert rows[1]["label"]["target_final_answer"] == {"result": "ok"}


def test_cluster_quota_and_selection_are_reproducible_and_restore_source_order() -> None:
    records = []
    vectors = []
    for index in range(100):
        records.append({"id": f"d{index:03d}", "metadata": {"comparison_scope": "same", "source_id": f"t{index % 7}"}})
        angle = index / 1000
        vectors.append([1.0, angle])
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    selected_a, report_a, assignments_a = cluster_and_select(records, matrix, distance_threshold=0.05)
    selected_b, report_b, assignments_b = cluster_and_select(records, matrix, distance_threshold=0.05)
    assert quota_for_cluster(100) == 10
    assert selected_a == sorted(selected_a)
    assert selected_a == selected_b
    assert report_a == report_b
    assert assignments_a == assignments_b
    assert len(selected_a) == 10


def test_budget_can_preserve_a_minimum_number_of_final_decisions() -> None:
    records = []
    vectors = []
    for index in range(12):
        decision_type = "final_answer" if index >= 8 else "tool_call"
        records.append({
            "id": f"d{index:02d}",
            "metadata": {
                "comparison_scope": decision_type,
                "decision_type": decision_type,
                "source_id": f"t{index}",
            },
        })
        vectors.append([1.0, index / 1000])
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    selected, report, _ = cluster_and_select(
        records,
        matrix,
        distance_threshold=0.1,
        total_budget=5,
        min_final_records=3,
    )
    assert len(selected) == 5
    assert sum(records[index]["metadata"]["decision_type"] == "final_answer" for index in selected) >= 3
    assert report["minimum_final_records"] == 3


def test_lossy_action_guard_hashes_complete_action_not_shared_head_and_tail() -> None:
    prefix = "A" * 3500
    suffix = "Z" * 3500
    scopes = []
    for middle in ("ligand-one", "ligand-two"):
        target = {
            "role": "assistant",
            "tool_calls": [{
                "id": middle,
                "type": "function",
                "function": {
                    "name": "mcp__molclaw-scp__pred_binding_affinity_boltz2",
                    "arguments": {"payload": prefix + middle + suffix},
                },
            }],
        }
        _, scope, guarded = build_selection_description(
            user_task="task", prompt=[], target=target, task_type="kg"
        )
        assert guarded is True
        scopes.append(scope)
    assert scopes[0] != scopes[1]


def test_order_bonus_is_zero_for_single_or_indistinguishable_calls_and_positive_for_teacher_order() -> None:
    config = {}
    calls = [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}]
    pairs = [SimpleNamespace(pred_index=0, gold_index=0), SimpleNamespace(pred_index=1, gold_index=1)]
    result = _teacher_order_bonus(calls, calls, pairs, content_correctness=1.0, weight=0.1, config=config)
    assert result["order_agreement"] == 1.0
    assert result["order_bonus"] == 0.1
    reversed_pairs = [SimpleNamespace(pred_index=0, gold_index=1), SimpleNamespace(pred_index=1, gold_index=0)]
    result = _teacher_order_bonus(calls, calls, reversed_pairs, content_correctness=1.0, weight=0.1, config=config)
    assert result["order_agreement"] == 0.0 and result["order_bonus"] == 0.0
    duplicate = [{"name": "a", "arguments": {"x": 1}}, {"name": "a", "arguments": {"x": 1}}]
    result = _teacher_order_bonus(duplicate, duplicate, pairs, content_correctness=1.0, weight=0.1, config=config)
    assert result["order_comparable_pairs"] == 0 and result["order_bonus"] == 0.0


def test_optional_trial_selector_uses_two_scores_and_restores_original_order(tmp_path: Path) -> None:
    input_path = tmp_path / "decisions.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    rows = [
        {
            "id": f"d{index}",
            "prompt": [{"role": "user", "content": "task"}],
            "tools": [],
            "label": {},
            "metadata": {"source_id": f"t{index // 2}", "trajectory_index": index // 2, "decision_ordinal": index % 2},
        }
        for index in range(10)
    ]
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    scores_path.write_text("".join(
        json.dumps({"decision_id": row["id"], "content_scores": [index / 10, index / 10]}) + "\n"
        for index, row in enumerate(rows)
    ))
    report = select_trials(
        input_path, scores_path, tmp_path / "selected", budget=5, priority_fraction=0.8, seed=7
    )
    selected = [json.loads(line) for line in (tmp_path / "selected/selected_decisions.jsonl").read_text().splitlines()]
    assert report["selection"]["priority_records"] == 4
    assert len(selected) == 5
    assert [int(row["id"][1:]) for row in selected] == sorted(int(row["id"][1:]) for row in selected)
    assert {"d0", "d1", "d2", "d3"}.issubset({row["id"] for row in selected})
