from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from drug_agent.scripts.normalize_v7_path_contract import normalize_record, parse_tool_calls
from drug_agent.tools.artifact_registry import ArtifactReferenceContractError, ArtifactRegistry
from drug_agent.tools.local_tools import LocalToolExecutor
from drug_agent.toolrl.convert_react_to_toolrl_steps import convert_react_to_toolrl_steps


class V7PathContractDataTest(unittest.TestCase):
    def test_generated_tool_calls_are_valid_json_and_preserve_local_paths(self) -> None:
        record = {
            "messages": [
                {"role": "user", "content": "<observation><artifact:local/a.pdb></observation>", "step_loss_mask": 0},
                {"role": "assistant", "content": '<tool_call>{"tool_name":"Read","arguments":{"file_path":"<artifact:local/a.pdb>"}}</tool_call>'},
                {"role": "user", "content": "<observation>ok</observation>", "step_loss_mask": 0},
                {"role": "assistant", "content": '<tool_call>{"tool_name":"Bash","arguments":{"command":"ls <artifact:local/a.pdb>"}}</tool_call>'},
            ]
        }
        result = normalize_record(record, Counter())
        self.assertIn("workspace/a.pdb", result["messages"][0]["content"])
        for index in (1, 3):
            payload = result["messages"][index]["content"].split("<tool_call>\n", 1)[1].split("\n</tool_call>", 1)[0]
            json.loads(payload)
        self.assertEqual(parse_tool_calls(result["messages"][1]["content"])[0]["arguments"]["file_path"], "workspace/a.pdb")
        self.assertEqual(parse_tool_calls(result["messages"][3]["content"])[0]["arguments"]["command"], "ls workspace/a.pdb")

    def test_masks_resource_used_as_local_path_and_preserves_history(self) -> None:
        counters: Counter[str] = Counter()
        record = {
            "messages": [
                {"role": "assistant", "content": '<tool_call>{"tool_name":"Read","arguments":{"file_path":"<artifact:structure/a.pdb>"}}</tool_call>'},
                {"role": "user", "content": "<observation>permission denied</observation>", "step_loss_mask": 0},
                {"role": "assistant", "content": "<thought>stopped</thought>"},
            ]
        }
        result = normalize_record(record, counters)
        self.assertEqual(len(result["messages"]), 3)
        self.assertEqual(result["messages"][0]["step_loss_mask"], 0)
        self.assertTrue(result["messages"][0]["path_contract_supervision_masked"])
        self.assertIn("resource://structure/a.pdb", result["messages"][0]["content"])
        self.assertEqual(counters["masked_assistant_actions"], 1)
        self.assertEqual(counters["preserved_paired_observations"], 1)

    def test_toolrl_skips_masked_target_but_keeps_later_decision(self) -> None:
        record = {
            "id": "path-contract-test",
            "messages": [
                {"role": "system", "content": "system", "step_loss_mask": 0},
                {"role": "user", "content": "task", "step_loss_mask": 0},
                {
                    "role": "assistant",
                    "content": '<tool_call>{"tool_name":"Read","arguments":{"file_path":"resource://structure/a.pdb"}}</tool_call>',
                    "step_loss_mask": 0,
                    "path_contract_supervision_masked": True,
                },
                {"role": "user", "content": '<observation tool_name="Read">failed</observation>', "step_loss_mask": 0},
                {"role": "assistant", "content": '<tool_call>{"tool_name":"Read","arguments":{"file_path":"workspace/a.pdb"}}</tool_call>', "step_loss_mask": 1},
            ],
        }
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source.jsonl"
            output = Path(root) / "steps.jsonl"
            skipped = Path(root) / "skipped.jsonl"
            source.write_text(json.dumps(record) + "\n")
            report = convert_react_to_toolrl_steps(source, output, skipped_report_path=skipped)
            self.assertEqual(report["counts"]["skip_path_contract_masked_action"], 1)
            self.assertEqual(report["kept_rows"], 1)
            row = json.loads(output.read_text().strip())
            self.assertEqual(row["label"]["target_tool_calls"][0]["arguments"]["file_path"], "workspace/a.pdb")


class V7PathContractRuntimeTest(unittest.TestCase):
    def test_workspace_path_is_valid_for_read_and_bash(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            workspace = root_path / "workspace"
            skills = root_path / "skills"
            workspace.mkdir()
            skills.mkdir()
            (workspace / "a.pdb").write_text("ATOM\n")
            executor = LocalToolExecutor(workspace, skills)
            read = executor.execute("Read", {"file_path": "workspace/a.pdb"})
            listing = executor.execute("Bash", {"command": "ls workspace/a.pdb"})
            self.assertTrue(read["ok"])
            self.assertEqual(read["result"]["path"], "workspace/a.pdb")
            self.assertTrue(listing["ok"])

    def test_resource_and_legacy_handles_fail_before_filesystem_execution(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            registry = ArtifactRegistry(root)
            with self.assertRaises(ArtifactReferenceContractError):
                registry.resolve_for_execution({"command": "ls resource://structure/a.pdb"}, filesystem_only=True)
            with self.assertRaises(ArtifactReferenceContractError):
                registry.resolve_for_execution({"file_path": "<artifact:local/a.pdb>"}, filesystem_only=True)

    def test_registered_resource_resolves_for_resource_tool(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            registry = ArtifactRegistry(root)
            reference = registry.register("/server/job/a.pdb", namespace="structure")
            self.assertEqual(reference, "resource://structure/a.pdb")
            self.assertEqual(registry.resolve_for_execution({"pdb_file_path": reference}), {"pdb_file_path": "/server/job/a.pdb"})


if __name__ == "__main__":
    unittest.main()
