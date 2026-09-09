from __future__ import annotations

import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pipeline.cleaning.semantic_builder import _unwrap_final, build_semantic_trajectory, compact_observation
from pipeline.cleaning.path_sanitizer import TrajectoryPathNormalizer
from pipeline.cleaning.trace_parser import load_session_events, question_text, safe_load_json


def assistant(message_id: str, *content: dict, line: int) -> dict:
    return {"type": "assistant", "_line_no": line, "message": {"id": message_id, "content": list(content)}}


def result(call_id: str, content, *, line: int) -> dict:
    return {"type": "user", "_line_no": line, "message": {"content": [{"type": "tool_result", "tool_use_id": call_id, "content": content}]}}


class SemanticBuilderTest(unittest.TestCase):
    def test_normalizes_dsh_messages_calls_results_and_terminal(self) -> None:
        with TemporaryDirectory() as directory:
            session = Path(directory) / "complete_session.jsonl"
            rows = [
                {"type": "turn/start", "seq": 0, "data": {"turn": 1}},
                {"type": "assistant/message", "seq": 1, "data": {"message": {
                    "id": "d1", "content": [
                        {"type": "reasoning", "text": "plan"},
                        {"type": "tool-call", "id": "c1", "name": "mcp__server__tool", "arguments": '{"x":1}'},
                    ],
                }}},
                {"type": "tool/result", "seq": 2, "data": {"message": {
                    "source": {"callId": "c1"}, "content": [{
                        "type": "tool-result", "toolCallId": "c1",
                        "content": [{"type": "text", "text": '{"status":"success"}'}],
                        "isError": False,
                    }],
                }}},
                {"type": "assistant/message", "seq": 3, "data": {"message": {
                    "id": "d2", "content": [{"type": "text", "text": "```json\n{\"result\":\"ok\"}\n```"}],
                }}},
                {"type": "turn/end", "seq": 4, "data": {"reason": {"kind": "completed"}}},
            ]
            session.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            events, malformed, runner_error = load_session_events(session)

        self.assertEqual(malformed, 0)
        self.assertFalse(runner_error)
        self.assertEqual([event["type"] for event in events], ["assistant", "user", "assistant", "result"])
        call = events[0]["message"]["content"][1]
        self.assertEqual(call["input"], {"x": 1})
        self.assertEqual(events[1]["message"]["content"][0]["tool_use_id"], "c1")
        self.assertFalse(events[-1]["is_error"])
        self.assertEqual(_unwrap_final(events[2]["message"]["content"][0]["text"]), '{"result":"ok"}')

    def test_path_normalizer_does_not_rewrite_prefix_collision(self) -> None:
        normalizer = TrajectoryPathNormalizer(
            workspace=Path("/tmp/attempt/workdir"),
            source_id="sample",
        )
        self.assertEqual(
            normalizer.normalize_text("/tmp/attempt/workdir-extra/server.pdb"),
            "/tmp/attempt/workdir-extra/server.pdb",
        )
        self.assertEqual(
            normalizer.normalize_text("/tmp/attempt/workdir/outputs/result.json"),
            "outputs/result.json",
        )

    def test_ignores_claude_runtime_diagnostic_but_counts_malformed_data(self) -> None:
        with TemporaryDirectory() as directory:
            session = Path(directory) / "complete_session.jsonl"
            session.write_text(
                '{"type":"system","subtype":"init"}\n'
                '[claude-code:unrecognized_model] {"model":"fixture"}\n'
                'not-json\n',
                encoding="utf-8",
            )
            events, malformed, runner_error = load_session_events(session)

        self.assertEqual(len(events), 1)
        self.assertEqual(malformed, 1)
        self.assertFalse(runner_error)

    def test_groups_one_message_id_and_orders_results_by_call(self) -> None:
        events = [
            assistant("d1", {"type": "thinking", "thinking": "plan"}, line=1),
            assistant("d1", {"type": "tool_use", "id": "a", "name": "a", "input": {"x": 1}}, line=2),
            result("a", {"status": "success", "value": 1}, line=3),
            assistant("d1", {"type": "tool_use", "id": "b", "name": "b", "input": {}}, line=4),
            assistant("d1", {"type": "tool_use", "id": "c", "name": "c", "input": {}}, line=5),
            result("c", {"status": "success", "value": 3}, line=6),
            result("b", {"status": "success", "value": 2}, line=7),
            assistant("runtime", {"type": "thinking", "thinking": "read L2"}, {"type": "tool_use", "id": "r", "name": "Read", "input": {"file_path": ".claude/skills/L2_workflows/x.md"}}, line=8),
            result("r", "teacher material", line=9),
            assistant("final", {"type": "thinking", "thinking": "done"}, {"type": "text", "text": "<answer>final response</answer>"}, line=10),
        ]
        record, audit = build_semantic_trajectory(
            events,
            record_id="sample",
            user_task="task",
            workspace=Path.cwd(),
            task_type="kg",
            source_session_sha256="fixture-sha",
        )
        decisions = [event for event in record["events"] if event["type"] == "assistant_decision"]
        self.assertEqual([event["source_message_id"] for event in decisions], ["d1", "final"])
        self.assertEqual([call["source_tool_use_id"] for call in decisions[0]["tool_calls"]], ["a", "b", "c"])
        observations = [event for event in record["events"] if event["type"] == "tool_observation"]
        self.assertEqual([event["source_tool_use_id"] for event in observations], ["a", "b", "c"])
        self.assertEqual(audit["retained_tool_call_groups"], [3])
        self.assertEqual(decisions[-1]["final_answer"], "final response")

    def test_projects_mixed_l1_l2_bash_to_native_reads(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        workspace = Path(temporary.name)
        first = workspace / ".claude/skills/L1_tools/tool/SKILL.md"
        first.parent.mkdir(parents=True)
        first.write_text("tool instructions", encoding="utf-8")
        events = [
            assistant(
                "runtime",
                {
                    "type": "tool_use",
                    "id": "runtime-call",
                    "name": "Bash",
                    "input": {"command": "cd .claude/skills; ls L2_workflows/; cat L1_tools/tool/SKILL.md"},
                },
                line=1,
            ),
            result("runtime-call", "L2 listing\ntool instructions\n", line=2),
            assistant(
                "science",
                {"type": "tool_use", "id": "science-call", "name": "science", "input": {}},
                line=3,
            ),
            result("science-call", {"status": "success"}, line=4),
            assistant("final", {"type": "text", "text": "done"}, line=5),
        ]
        record, audit = build_semantic_trajectory(
            events,
            record_id="sample",
            user_task="task",
            workspace=workspace,
            task_type="kg",
            source_session_sha256="fixture-sha",
        )

        decisions = [event for event in record["events"] if event["type"] == "assistant_decision"]
        self.assertEqual([event["source_message_id"] for event in decisions], ["runtime", "science", "final"])
        self.assertEqual(decisions[0]["tool_calls"][0]["name"], "Read")
        self.assertEqual(decisions[0]["tool_calls"][0]["arguments"]["file_path"], ".agents/skills/tool/SKILL.md")
        observations = [event for event in record["events"] if event["type"] == "tool_observation"]
        self.assertEqual(observations[0]["content"], "tool instructions")
        self.assertEqual(audit["derived_calls"][0]["source_tool_use_id"], "runtime-call")

    def test_drops_broad_claude_skill_discovery_bash(self) -> None:
        events = [
            assistant(
                "runtime",
                {
                    "type": "tool_use",
                    "id": "runtime-call",
                    "name": "Bash",
                    "input": {"command": "find /workspace/.claude -maxdepth 3; ls /workspace/L1_tools/*"},
                },
                line=1,
            ),
            result("runtime-call", "L1_tools\nL2_workflows\nL3_methodology", line=2),
            assistant(
                "science",
                {"type": "tool_use", "id": "science-call", "name": "science", "input": {}},
                line=3,
            ),
            result("science-call", {"status": "success"}, line=4),
            assistant("final", {"type": "text", "text": "done"}, line=5),
        ]
        record, _ = build_semantic_trajectory(
            events,
            record_id="sample",
            user_task="task",
            workspace=Path.cwd(),
            task_type="kg",
            source_session_sha256="fixture-sha",
        )

        decisions = [event for event in record["events"] if event["type"] == "assistant_decision"]
        self.assertEqual([event["source_message_id"] for event in decisions], ["science", "final"])

    def test_compacts_blobs_and_long_arrays_deterministically(self) -> None:
        value = {"status": "success", "blob": "QUJD" * 300, "values": list(range(5000))}
        compacted, audit = compact_observation(value, 300)
        self.assertEqual(compacted["status"], "success")
        self.assertEqual(compacted["values"], value["values"])
        self.assertGreaterEqual(audit["blob_count"], 1)

    def test_large_retained_result_is_compacted_recursively(self) -> None:
        compacted, audit = compact_observation(
            {"status": "success", "result": {"scores": list(range(10000))}},
            500,
        )
        self.assertEqual(compacted["status"], "success")
        self.assertEqual(compacted["result"]["scores"], list(range(10000)))

    def test_compaction_pins_evidence_reused_downstream(self) -> None:
        workspace = Path.cwd()
        server_path = "/server/runs/unique/output.pdb"
        events = [
            assistant("d1", {"type": "tool_use", "id": "c1", "name": "produce", "input": {}}, line=1),
            result("c1", {"status": "success", "payload": "x" * 5000, "output_file": server_path}, line=2),
            assistant("d2", {"type": "tool_use", "id": "c2", "name": "consume", "input": {"input_path": server_path}}, line=3),
            result("c2", {"status": "success"}, line=4),
            assistant("final", {"type": "text", "text": "done"}, line=5),
        ]
        record, _ = build_semantic_trajectory(
            events, record_id="evidence", user_task="task", workspace=workspace,
            task_type="kg", source_session_sha256="fixture-sha", max_observation_chars=300,
        )
        first_observation = next(event for event in record["events"] if event["type"] == "tool_observation")
        self.assertIn(server_path, json.dumps(first_observation["content"]))

    def test_real_pwd_uses_trajectory_specific_absolute_root(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        workspace = Path(temporary.name).resolve()
        local_path = str(workspace / "outputs/result.json")
        events = [
            assistant("pwd", {"type": "tool_use", "id": "p1", "name": "Bash", "input": {"command": "pwd"}}, line=1),
            result("p1", str(workspace), line=2),
            assistant("write", {"type": "tool_use", "id": "w1", "name": "Write", "input": {"file_path": local_path, "content": "ok"}}, line=3),
            result("w1", f"created {local_path}", line=4),
            assistant("final", {"type": "text", "text": f"saved {local_path}"}, line=5),
        ]
        record, audit = build_semantic_trajectory(
            events, record_id="pwd-fixture", user_task="task", workspace=workspace,
            task_type="kg", source_session_sha256="fixture-sha",
        )
        expected_root = "/" + hashlib.sha256(b"pwd-fixture").hexdigest()[:20]
        rendered = json.dumps(record)
        self.assertNotIn(str(workspace), rendered)
        self.assertGreaterEqual(rendered.count(expected_root), 3)
        self.assertEqual(audit["path_normalization"]["pwd_root"], expected_root)

    def test_egfr_parallel_regression_is_three_three(self) -> None:
        root = Path("/home/sunxiangyu/slime_sxy/claude-run/protein_repair_case/01_raw_capture/molbench_kg_boyue-ds_run_20260903_170222/row0001_idxegfr_retrieve_repair_001")
        if not root.is_dir():
            self.skipTest("EGFR fixture is not mounted")
        events, malformed, _ = load_session_events(root / "complete_session.jsonl")
        self.assertEqual(malformed, 0)
        record, audit = build_semantic_trajectory(
            events,
            record_id="egfr",
            user_task=question_text(safe_load_json(root / "question.json")),
            workspace=root / "attempts/attempt_0001/workdir",
            task_type="kg",
            source_session_sha256="fixture-sha",
        )
        self.assertEqual([size for size in audit["retained_tool_call_groups"] if size > 1], [3, 3])
        ids = [event["source_message_id"] for event in record["events"] if event["type"] == "assistant_decision"]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("<final_answer>", json.dumps(record, ensure_ascii=False))

    def test_short_case_l1_and_path_contract_regression(self) -> None:
        root = Path("/home/sunxiangyu/slime_sxy/claude-run/protein_repair_short_case/01_raw_capture/molbench_kg_manual_run_20260904_125503/row0001_idxprotein_repair_short_001")
        if not root.is_dir():
            self.skipTest("protein-repair short fixture is not mounted")
        events, malformed, _ = load_session_events(root / "complete_session.jsonl")
        self.assertEqual(malformed, 0)
        record, audit = build_semantic_trajectory(
            events,
            record_id="short",
            user_task=question_text(safe_load_json(root / "question.json")),
            workspace=root / "attempts/attempt_0001/workdir",
            task_type="kg",
            source_session_sha256="fixture-sha",
        )
        decisions = [event for event in record["events"] if event["type"] == "assistant_decision"]
        self.assertIn(["Read", "Read"], [[call["name"] for call in decision["tool_calls"]] for decision in decisions])
        observations = [event for event in record["events"] if event["type"] == "tool_observation"]
        self.assertTrue(all("L2_workflows" not in str(item["content"]) for item in observations[:2]))
        rendered = json.dumps(record, ensure_ascii=False)
        self.assertIn('"file_path": "run_log.md"', rendered)
        self.assertIn("/data/lwj/wll/code/DrugAgentTools/sxy_sum/source_tree/protein_structures/3G5Y.pdb", rendered)
        self.assertNotIn("resource://", rendered)
        self.assertNotIn("<artifact:", rendered)
        self.assertNotIn("protein_repair_short_case", rendered)
        self.assertNotIn("deployment_tool_set_sha256", record["metadata"])
        self.assertEqual(len(audit["derived_calls"]), 1)
