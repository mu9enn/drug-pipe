from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_DIR))

from cleaning.python_clean import clean_sample, python_clean  # noqa: E402
from cleaning.trace_parser import RolloutSample, terminal_execution_findings  # noqa: E402


class PythonCleanTest(unittest.TestCase):
    @staticmethod
    def _paired_molclaw_events() -> list[dict]:
        return [
            {
                "type": "assistant",
                "message": {"content": [{
                    "type": "tool_use", "id": "m1",
                    "name": "mcp__molclaw-scp__is_valid_smiles",
                    "input": {"smiles": "CCO"},
                }]},
            },
            {
                "type": "user",
                "message": {"content": [{
                    "type": "tool_result", "tool_use_id": "m1",
                    "content": {"status": "success", "valid": True},
                }]},
            },
        ]

    def test_python_step_emits_draft_without_final_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            sample = run / "row0001_idx0"
            sample.mkdir(parents=True)
            (run / "run_config.json").write_text('{"task":"kg"}', encoding="utf-8")
            (sample / "question.json").write_text(
                '{"task":"kg","question":"Run the grounded task","answer":[]}', encoding="utf-8"
            )
            (sample / "parsed_answer.json").write_text(
                '{"answer":{"result":"grounded result"}}', encoding="utf-8"
            )
            (sample / "run_meta.json").write_text('{"return_code":0}', encoding="utf-8")
            events = [
                {
                    "type": "assistant",
                    "message": {"content": [{
                        "type": "tool_use", "id": "c1",
                        "name": "mcp__molclaw-scp__x", "input": {},
                    }]},
                },
                {
                    "type": "user",
                    "message": {"content": [{
                        "type": "tool_result", "tool_use_id": "c1",
                        "content": {"status": "success", "result": "done"},
                    }]},
                },
            ]
            (sample / "complete_session.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            output = root / "python"
            manifest = python_clean(run, output)
            audit = json.loads((output / "python_audit.jsonl").read_text(encoding="utf-8"))
            draft = json.loads((output / "python_drafts.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(manifest["python_valid_count"], 1)
            self.assertEqual(audit["python_status"], "python_valid")
            self.assertNotIn("final_status", audit)
            self.assertEqual(set(draft), {"schema_version", "id", "messages"})
            self.assertNotIn("ground_truth", json.dumps(draft))

    def test_missing_parsed_answer_is_not_a_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            sample = run / "row0001_idx0"
            sample.mkdir(parents=True)
            (run / "run_config.json").write_text('{"task":"kg"}', encoding="utf-8")
            (sample / "question.json").write_text(
                '{"task":"kg","question":"Run the grounded task"}', encoding="utf-8"
            )
            (sample / "run_meta.json").write_text('{"return_code":0}', encoding="utf-8")
            events = [
                {
                    "type": "assistant",
                    "message": {"content": [{
                        "type": "tool_use", "id": "c1",
                        "name": "mcp__molclaw-scp__is_valid_smiles",
                        "input": {"smiles": "CCO"},
                    }]},
                },
                {
                    "type": "user",
                    "message": {"content": [{
                        "type": "tool_result", "tool_use_id": "c1",
                        "content": {"status": "success", "valid": True},
                    }]},
                },
            ]
            (sample / "complete_session.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            output = root / "python"
            manifest = python_clean(run, output)
            audit = json.loads((output / "python_audit.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(manifest["input_count"], 1)
            self.assertEqual(manifest["python_valid_count"], 1)
            self.assertEqual(audit["execution_invalid_reasons"], [])
            self.assertFalse(audit["parsed_answer_present"])

    def test_execution_gate_distinguishes_terminal_api_failure_from_tool_failure(self) -> None:
        def events(*, terminal_error: bool) -> list[dict]:
            rows = [
                {
                    "type": "assistant",
                    "message": {"content": [{
                        "type": "tool_use", "id": "c1",
                        "name": "mcp__molclaw-scp__is_valid_smiles",
                        "input": {"smiles": "bad"},
                    }]},
                },
                {
                    "type": "user",
                    "message": {"content": [{
                        "type": "tool_result", "tool_use_id": "c1",
                        "is_error": True,
                        "content": {"status": "error", "message": "ordinary tool failure"},
                    }]},
                },
            ]
            rows.append(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": terminal_error,
                    **({"api_error_status": 400} if terminal_error else {}),
                }
            )
            return rows

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            (run / "run_config.json").write_text('{"task":"kg"}', encoding="utf-8")
            for index, terminal_error in enumerate((False, True), start=1):
                sample = run / f"row{index:04d}_idx{index}"
                sample.mkdir()
                (sample / "question.json").write_text(
                    '{"task":"kg","question":"Run"}', encoding="utf-8"
                )
                (sample / "parsed_answer.json").write_text(
                    '{"answer":"done"}', encoding="utf-8"
                )
                (sample / "run_meta.json").write_text(
                    '{"return_code":0}', encoding="utf-8"
                )
                (sample / "complete_session.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in events(terminal_error=terminal_error)),
                    encoding="utf-8",
                )
            output = root / "python"
            manifest = python_clean(run, output)
            audits = [
                json.loads(line)
                for line in (output / "python_audit.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(manifest["python_valid_count"], 1)
            self.assertEqual(manifest["rejected_count"], 1)
            self.assertTrue(audits[0]["execution_valid"])
            self.assertFalse(audits[1]["execution_valid"])
            self.assertIn("terminal_result_error", audits[1]["execution_invalid_reasons"])
            self.assertIn("terminal_api_error:400", audits[1]["execution_invalid_reasons"])

    def test_terminal_gate_uses_only_the_last_top_level_result(self) -> None:
        events = [
            {"type": "result", "is_error": True, "api_error_status": 500},
            {
                "type": "user",
                "message": {"content": [{
                    "type": "tool_result",
                    "api_error_status": 400,
                    "content": {"status": "error"},
                }]},
            },
            {"type": "result", "subtype": "success", "is_error": False},
        ]
        self.assertEqual(terminal_execution_findings(events), [])
        events[-1] = {
            "type": "result",
            "subtype": "error_during_execution",
            "terminal_reason": "aborted_streaming",
            "is_error": False,
        }
        self.assertEqual(
            terminal_execution_findings(events),
            [
                "terminal_subtype:error_during_execution",
                "terminal_reason:aborted_streaming",
            ],
        )

    def test_a_b_c_are_the_only_filter_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def evaluate_case(
                name: str,
                *,
                events: list[dict] | None,
                run_meta: dict,
                parsed: dict,
                runner_error: bool = False,
            ) -> dict:
                sample_dir = root / name
                sample_dir.mkdir()
                (sample_dir / "question.json").write_text(
                    '{"task":"kg","question":"Run"}', encoding="utf-8"
                )
                (sample_dir / "parsed_answer.json").write_text(
                    json.dumps(parsed), encoding="utf-8"
                )
                (sample_dir / "run_meta.json").write_text(
                    json.dumps(run_meta), encoding="utf-8"
                )
                if events is not None:
                    text = "".join(json.dumps(event) + "\n" for event in events)
                    if runner_error:
                        text += "[runner-error] agent terminated\n"
                    (sample_dir / "complete_session.jsonl").write_text(
                        text, encoding="utf-8"
                    )
                rollout = RolloutSample(
                    root,
                    sample_dir,
                    sample_dir,
                    1,
                    "1",
                    1,
                )
                return clean_sample(
                    rollout,
                    default_task="kg",
                    chemistry=None,
                )["audit"]

            paired = self._paired_molclaw_events()
            rc = evaluate_case(
                "rc", events=paired, run_meta={"return_code": 1}, parsed={"answer": "x"}
            )
            timeout = evaluate_case(
                "timeout",
                events=paired,
                run_meta={"return_code": 0, "timed_out": True},
                parsed={"answer": "x"},
            )
            runner = evaluate_case(
                "runner",
                events=paired,
                run_meta={"return_code": 0},
                parsed={"answer": "x"},
                runner_error=True,
            )
            missing = evaluate_case(
                "missing",
                events=None,
                run_meta={"return_code": 0},
                parsed={"answer": "x"},
            )
            parse = evaluate_case(
                "parse",
                events=paired,
                run_meta={"return_code": 0},
                parsed={"parse_error": "missing answer tag"},
            )
            no_mol = evaluate_case(
                "no_mol",
                events=[],
                run_meta={"return_code": 0},
                parsed={"answer": "x"},
            )

            self.assertEqual(rc["execution_invalid_reasons"], ["runner_nonzero_rc:1"])
            self.assertEqual(timeout["execution_invalid_reasons"], ["timeout"])
            self.assertEqual(runner["execution_invalid_reasons"], ["runner_error_last_line"])
            self.assertIn("missing_session", missing["execution_invalid_reasons"])
            self.assertEqual(parse["task_answer_invalid_reasons"], ["parse_error"])
            self.assertEqual(no_mol["training_trace_invalid_reasons"], ["missing_molclaw_usage"])
            self.assertEqual(parse["python_status_reasons"], ["task_answer_invalid"])
            self.assertEqual(no_mol["python_status_reasons"], ["training_trace_invalid"])

    def test_only_molclaw_mode_has_gate_parity_and_drops_local_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            sample = run / "row0001_idx0"
            sample.mkdir(parents=True)
            (run / "run_config.json").write_text('{"task":"kg"}', encoding="utf-8")
            (sample / "question.json").write_text(
                '{"task":"kg","question":"Validate CCO"}', encoding="utf-8"
            )
            (sample / "parsed_answer.json").write_text(
                '{"answer":"valid"}', encoding="utf-8"
            )
            (sample / "run_meta.json").write_text('{"return_code":0}', encoding="utf-8")
            events = [
                {
                    "type": "assistant",
                    "message": {"content": [
                        {
                            "type": "tool_use", "id": "r1", "name": "Read",
                            "input": {"file_path": "run_log.md"},
                        },
                        {
                            "type": "tool_use", "id": "m1",
                            "name": "mcp__molclaw-scp__is_valid_smiles",
                            "input": {"smiles": "CCO"},
                        },
                    ]},
                },
                {
                    "type": "user",
                    "message": {"content": [
                        {"type": "tool_result", "tool_use_id": "r1", "content": "started"},
                        {
                            "type": "tool_result", "tool_use_id": "m1",
                            "content": {"status": "success", "valid": True},
                        },
                    ]},
                },
            ]
            (sample / "complete_session.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            default_manifest = python_clean(run, root / "default")
            only_manifest = python_clean(
                run, root / "only", only_molclaw_tool=True
            )
            self.assertEqual(
                (
                    default_manifest["python_valid_count"],
                    default_manifest["rejected_count"],
                ),
                (
                    only_manifest["python_valid_count"],
                    only_manifest["rejected_count"],
                ),
            )
            default_record = json.loads(
                (root / "default/python_drafts.jsonl").read_text(encoding="utf-8")
            )
            only_record = json.loads(
                (root / "only/python_drafts.jsonl").read_text(encoding="utf-8")
            )
            default_text = "\n".join(message["content"] for message in default_record["messages"])
            only_text = "\n".join(message["content"] for message in only_record["messages"])
            self.assertIn('"tool_name":"Read"', default_text)
            self.assertNotIn('"tool_name":"Read"', only_text)
            self.assertEqual(default_manifest["retained_local_tool_call_count"], 1)
            self.assertEqual(only_manifest["retained_local_tool_call_count"], 0)

    def test_structuring_schema_regression_fails_the_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            sample = run / "row0001_idx0"
            sample.mkdir(parents=True)
            (run / "run_config.json").write_text('{"task":"kg"}', encoding="utf-8")
            (sample / "question.json").write_text(
                '{"task":"kg","question":"Run"}', encoding="utf-8"
            )
            (sample / "parsed_answer.json").write_text(
                '{"answer":"done"}', encoding="utf-8"
            )
            (sample / "run_meta.json").write_text('{"return_code":0}', encoding="utf-8")
            (sample / "complete_session.jsonl").write_text(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {"content": [{
                            "type": "tool_use", "id": "m1",
                            "name": "mcp__molclaw-scp__x", "input": {},
                        }]},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "type": "user",
                        "message": {"content": [{
                            "type": "tool_result", "tool_use_id": "m1",
                            "content": {"status": "success"},
                        }]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            invalid_messages = [{"role": "assistant", "content": "not ReAct"}]
            stats = {"molclaw_usage_count": 1}
            with mock.patch(
                "cleaning.python_clean.reconstruct_react_messages",
                return_value=(invalid_messages, stats),
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid ReAct"):
                    python_clean(run, root / "python")


if __name__ == "__main__":
    unittest.main()
