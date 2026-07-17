from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pipeline.cleaning.hard_cleaner import hard_clean  # noqa: E402
from pipeline.cleaning.llm_cleaner import clean_with_llm  # noqa: E402
from pipeline.cleaning.acceptance_gate import decide_final_status  # noqa: E402


def sample_record(*middle: dict, final: dict | None = None) -> dict:
    messages = [
        {"role": "system", "content": "protocol", "step_loss_mask": 0},
        {"role": "user", "content": "task", "step_loss_mask": 0},
        *middle,
        {
            "role": "assistant",
            "content": f"<final_answer>{json.dumps(final or {'task_type': 'kg', 'result': 'done'})}</final_answer>",
            "step_loss_mask": 1,
        },
    ]
    return {"schema_version": "drug_agent_sft_react_json_v1", "id": "sample-1", "messages": messages}


class LlmCleanerTest(unittest.TestCase):
    def test_allows_reasoning_rewrite_without_changing_protocol(self) -> None:
        sample = sample_record(
            {
                "role": "assistant",
                "content": '<thought>Claude Code chatter</thought>\n<tool_call>{"tool_name":"x","arguments":{}}</tool_call>',
                "step_loss_mask": 1,
            },
            {
                "role": "user",
                "content": '<observation tool_name="x">{"status":"success","value":1}</observation>',
                "step_loss_mask": 0,
            },
        )

        def rewrite(value):
            value["messages"][2]["content"] = value["messages"][2]["content"].replace(
                "Claude Code chatter", "Use the recorded measurement."
            )
            return value

        cleaned, report = clean_with_llm(sample, rewrite)
        self.assertEqual(report["status"], "cleaned")
        self.assertIn("recorded measurement", cleaned["messages"][2]["content"])

    def test_rejects_changed_observation_and_returns_source(self) -> None:
        sample = sample_record(
            {
                "role": "assistant",
                "content": '<tool_call>{"tool_name":"x","arguments":{}}</tool_call>',
                "step_loss_mask": 1,
            },
            {
                "role": "user",
                "content": '<observation tool_name="x">{"status":"success","value":1}</observation>',
                "step_loss_mask": 0,
            },
        )

        def rewrite(value):
            value["messages"][3]["content"] = value["messages"][3]["content"].replace('"value":1', '"value":2')
            return value

        cleaned, report = clean_with_llm(sample, rewrite)
        self.assertEqual(report["status"], "unsafe_rewrite")
        self.assertIn("llm_changed_observations_or_order", report["findings"])
        self.assertEqual(cleaned, sample)


class HardCleanerTest(unittest.TestCase):
    def test_reports_status_conflict_and_does_not_assign_acceptance(self) -> None:
        sample = sample_record(
            {
                "role": "assistant",
                "content": '<tool_call>{"tool_name":"x","arguments":{}}</tool_call>',
                "step_loss_mask": 1,
            },
            {
                "role": "user",
                "content": (
                    '<observation tool_name="x">{"status":"success","is_error":false,'
                    '"content":{"status":"error","message":"failed"}}</observation>'
                ),
                "step_loss_mask": 0,
            },
            final={"task_type": "kg", "result": "failed"},
        )
        _, report = hard_clean(sample)
        self.assertIn("observation_status_conflict", report["errors"])
        self.assertNotIn("accepted", report)
        self.assertNotIn("status", report)

    def test_reports_final_artifact_and_numeric_conflicts(self) -> None:
        sample = sample_record(
            {
                "role": "assistant",
                "content": '<tool_call>{"tool_name":"dock","arguments":{"smiles":"CCO"}}</tool_call>',
                "step_loss_mask": 1,
            },
            {
                "role": "user",
                "content": (
                    '<observation tool_name="dock">{"status":"success","is_error":false,'
                    '"content":{"score":-7.1,"artifact":"<artifact:docking/real.sdf>"}}</observation>'
                ),
                "step_loss_mask": 0,
            },
            final={
                "task_type": "e2e",
                "result": "done",
                "evidence": [{"score": -8.5, "artifact": "<artifact:docking/invented.sdf>"}],
            },
        )
        _, report = hard_clean(sample)
        self.assertIn("final_references_unknown_artifact", report["errors"])
        self.assertIn("final_numeric_evidence_not_in_observations", report["errors"])

    def test_reports_vs_ranking_inconsistent_with_scores(self) -> None:
        sample = sample_record(
            {
                "role": "assistant",
                "content": (
                    '<tool_call>{"tool_name":"dock","arguments":{"smiles":"A"}}</tool_call>\n'
                    '<tool_call>{"tool_name":"dock","arguments":{"smiles":"B"}}</tool_call>'
                ),
                "step_loss_mask": 1,
            },
            {
                "role": "user",
                "content": (
                    '<observation tool_name="dock">{"status":"success","content":{"score":-5.0}}</observation>\n'
                    '<observation tool_name="dock">{"status":"success","content":{"score":-7.0}}</observation>'
                ),
                "step_loss_mask": 0,
            },
            final={"task_type": "vs", "ranked_smiles": ["A", "B"]},
        )
        _, report = hard_clean(sample)
        self.assertIn("vs_ranking_inconsistent_with_tool_scores", report["errors"])

    def test_sanitizes_relative_path_without_changing_protocol_order(self) -> None:
        sample = sample_record(
            {
                "role": "assistant",
                "content": '<thought>Read ../outputs/result.pdb</thought><tool_call>{"tool_name":"x","arguments":{}}</tool_call>',
                "step_loss_mask": 1,
            },
            {
                "role": "user",
                "content": '<observation tool_name="x">{"status":"success","content":{}}</observation>',
                "step_loss_mask": 0,
            },
        )
        cleaned, report = hard_clean(copy.deepcopy(sample))
        self.assertIn("<artifact:local/result.pdb>", cleaned["messages"][2]["content"])
        self.assertEqual(report["counts"]["tool_calls"], 1)
        self.assertEqual(report["counts"]["observations"], 1)


class AcceptanceGateTest(unittest.TestCase):
    def test_gate_is_the_only_stage_that_assigns_final_status(self) -> None:
        decision = decide_final_status(
            execution_valid=True,
            task_answer_valid=True,
            training_trace_valid=True,
            llm_clean_status="cleaned",
            llm_clean_findings=[],
            hard_clean_findings=[],
        )
        self.assertEqual(decision["final_status"], "accepted")
        self.assertEqual(decision["authority"], "final_acceptance_gate")

    def test_gate_quarantines_unsafe_cleaning_without_relabeling_base_facts(self) -> None:
        decision = decide_final_status(
            execution_valid=True,
            task_answer_valid=True,
            training_trace_valid=True,
            llm_clean_status="unsafe_rewrite",
            llm_clean_findings=["llm_changed_task_prediction"],
            hard_clean_findings=[],
        )
        self.assertEqual(decision["final_status"], "quarantine")
        self.assertIn("llm_clean_unsafe_rewrite", decision["reasons"])


if __name__ == "__main__":
    unittest.main()
