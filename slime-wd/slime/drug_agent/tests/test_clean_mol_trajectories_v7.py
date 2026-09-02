from __future__ import annotations

from drug_agent.scripts.clean_mol_trajectories_v7 import clean_trajectory


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content, "step_loss_mask": 1}


def _error_observation(tool_name: str) -> dict:
    return {
        "role": "user",
        "content": f'<observation tool_name="{tool_name}">{{"status":"error","is_error":true}}</observation>',
        "step_loss_mask": 1,
    }


def test_cleanup_normalizes_final_deduplicates_evidence_and_masks_observation():
    duplicate = {"tool_name": "dock", "status": "success"}
    record = {
        "id": "vs-1",
        "messages": [
            {"role": "user", "content": "task"},
            _assistant(
                '<final_answer>{"task_type":"vs","ranked_smiles":["CCO"],'
                '"selected_smiles":"CCO","evidence":['
                '{"tool_name":"dock","status":"success"},'
                '{"tool_name":"dock","status":"success"}]}</final_answer>'
            ),
        ],
    }
    cleaned, stats, unresolved = clean_trajectory(record)
    assert unresolved == []
    assert cleaned is not None
    assert '"selected_smiles":["CCO"]' in cleaned["messages"][-1]["content"]
    assert cleaned["messages"][-1]["content"].count('"tool_name":"dock"') == 1
    assert stats["evidence_duplicates_removed"] == 1


def test_cleanup_keeps_only_one_identical_failed_retry_pair():
    call = '<tool_call>{"tool_name":"Read","arguments":{"file_path":"missing"}}</tool_call>'
    messages = [{"role": "user", "content": "task"}]
    for _ in range(4):
        messages.extend([_assistant(call), _error_observation("Read")])
    cleaned, stats, unresolved = clean_trajectory({"id": "retry", "messages": messages})
    assert unresolved == []
    assert cleaned is not None
    assert sum(message.get("role") == "assistant" for message in cleaned["messages"]) == 2
    assert stats["identical_failed_retry_pairs_removed"] == 2
    observations = [message for message in cleaned["messages"] if message.get("content", "").startswith("<observation")]
    assert all(message["step_loss_mask"] == 0 for message in observations)


def test_cleanup_unwraps_unprovenanced_assistant_artifact_reference():
    record = {
        "id": "artifact",
        "messages": [
            {"role": "user", "content": "task"},
            _assistant('<tool_call>{"tool_name":"fix_pdb","arguments":{"input_path":"<artifact:structure/fake.pdb>"}}</tool_call>'),
        ],
    }
    cleaned, stats, unresolved = clean_trajectory(record)
    assert unresolved == []
    assert cleaned is not None
    assert "<artifact:structure/" not in cleaned["messages"][-1]["content"]
    assert '"input_path":"fake.pdb"' in cleaned["messages"][-1]["content"]
    assert stats["unresolved_artifact_refs_unwrapped_from_actions"] == 1


def test_cleanup_unwraps_local_workspace_paths_in_assistant_actions():
    record = {
        "id": "local-artifact",
        "messages": [
            {"role": "user", "content": "task"},
            _assistant('<tool_call>{"tool_name":"Write","arguments":{"file_path":"<artifact:local/result.md>","content":"ok"}}</tool_call>'),
        ],
    }
    cleaned, stats, unresolved = clean_trajectory(record)
    assert unresolved == []
    assert cleaned is not None
    assert "<artifact:local/" not in cleaned["messages"][-1]["content"]
    assert '"file_path":"result.md"' in cleaned["messages"][-1]["content"]
    assert stats["local_artifact_refs_unwrapped"] == 1
