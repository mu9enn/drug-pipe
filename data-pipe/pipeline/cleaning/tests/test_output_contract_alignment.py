from __future__ import annotations

import json
import unittest

from pipeline.cleaning.output_contract_alignment import align_record


def record(task_type: str, task: str, payload: dict) -> dict:
    return {
        "schema_version": "drug_agent_semantic_trajectory_v1",
        "id": "sample",
        "user_task": task,
        "events": [
            {"type": "assistant_decision", "source_message_id": "d0", "reasoning": "check", "tool_calls": [
                {"name": "is_valid_smiles", "arguments": {"smiles": "CCO"}, "source_tool_use_id": "c0"}
            ], "final_answer": None},
            {"type": "tool_observation", "name": "is_valid_smiles", "source_tool_use_id": "c0",
             "status": "success", "is_error": False, "content": "ok"},
            {"type": "assistant_decision", "source_message_id": "d1", "reasoning": "done",
             "tool_calls": [], "final_answer": json.dumps(payload)},
        ],
        "metadata": {"task_type": task_type},
    }


class OutputContractAlignmentTest(unittest.TestCase):
    def test_ac_rewrites_prompt_and_keeps_only_answer_and_evidence(self) -> None:
        result = align_record(record(
            "ac",
            "For the target X, you are given:\nMolecule A: CCO\nMolecule B: CCN\nOnly output the corresponding SMILES.",
            {"task_type": "ac", "answer_smiles": "CCO", "evidence": [], "summary": "extra"},
        ))
        self.assertIn('"answer_smiles"', result["user_task"])
        self.assertEqual(json.loads(result["events"][-1]["final_answer"]), {
            "answer_smiles": "CCO", "evidence": [],
        })

    def test_pf_rewrites_plain_lines_as_json_contract(self) -> None:
        result = align_record(record(
            "pf",
            "SMILES:\nCCO\nCCN\nConstraints:\nTask.\nOutput format:\nPrint each satisfying SMILES on its own line, and nothing else.",
            {"task_type": "pf", "selected_smiles": ["CCO"], "evidence": [], "summary": "extra"},
        ))
        self.assertIn('"selected_smiles"', result["user_task"])
        self.assertEqual(set(json.loads(result["events"][-1]["final_answer"])), {"selected_smiles", "evidence"})

    def test_vs_removes_redundant_selected_smiles(self) -> None:
        task = json.dumps({"task": "rank", "candidates": ["CCO"], "output_format": "Return a JSON array of exactly 60 SMILES strings."})
        result = align_record(record(
            "vs", task,
            {"task_type": "vs", "ranked_smiles": ["CCO"], "selected_smiles": ["CCO"], "evidence": []},
        ))
        self.assertIn('"ranked_smiles"', json.loads(result["user_task"])["output_format"])
        self.assertEqual(json.loads(result["events"][-1]["final_answer"]), {
            "ranked_smiles": ["CCO"], "evidence": [],
        })

    def test_open_ended_contract_uses_result_and_evidence(self) -> None:
        result = align_record(record(
            "kg", "Do the work.",
            {"task_type": "kg", "result": ["artifact"], "evidence": [], "summary": "extra"},
        ))
        self.assertIn('"result"', result["user_task"])
        self.assertEqual(set(json.loads(result["events"][-1]["final_answer"])), {"result", "evidence"})


if __name__ == "__main__":
    unittest.main()
