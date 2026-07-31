from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

from drug_agent.tools.local_tools import LOCAL_TOOL_NAMES, LocalToolExecutor
from drug_agent.tools.tool_registry import ToolRegistry
from drug_agent.tools_debug.debug_one_task import (
    _augment_messages_for_strict_json,
    _fresh_messages,
    _sample_context,
)


class _FakeMCPExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self) -> list[dict]:
        return [
            {
                "name": "mcp__molclaw-scp__is_valid_smiles",
                "description": "validate",
                "input_schema": {
                    "type": "object",
                    "properties": {"smiles": {"type": "string"}},
                    "required": ["smiles"],
                },
            }
        ]

    def execute(self, tool_name: str, arguments: dict) -> dict:
        self.calls.append((tool_name, arguments))
        return {"ok": True, "tool_name": tool_name, "result": {"valid": True}}


class LocalToolExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.workspace = root / "workspace"
        self.skills = root / "skills"
        (self.skills / "molclaw-test").mkdir(parents=True)
        (self.skills / "molclaw-test" / "SKILL.md").write_text(
            "# Test skill\nUse the scientific tool carefully.\n", encoding="utf-8"
        )
        self.executor = LocalToolExecutor(self.workspace, self.skills)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_read_write_edit_grep_and_glob(self) -> None:
        written = self.executor.execute(
            "Write", {"file_path": "run_log.md", "content": "step: retrieve\n"}
        )
        self.assertTrue(written["ok"])
        edited = self.executor.execute(
            "Edit",
            {
                "file_path": "run_log.md",
                "old_string": "retrieve",
                "new_string": "retrieve structure",
            },
        )
        self.assertTrue(edited["ok"])
        read = self.executor.execute("Read", {"file_path": "run_log.md"})
        self.assertEqual(read["result"]["content"], "step: retrieve structure\n")
        grep = self.executor.execute("Grep", {"pattern": "structure", "path": "."})
        self.assertEqual(grep["result"]["matches"][0]["path"], "run_log.md")
        glob = self.executor.execute("Glob", {"pattern": "*.md"})
        self.assertEqual(glob["result"]["matches"], ["run_log.md"])

    def test_l1_skill_is_read_only_and_l2_is_unavailable(self) -> None:
        result = self.executor.execute("Skill", {"skill": "molclaw-test"})
        self.assertTrue(result["ok"])
        self.assertIn("scientific tool", result["result"]["content"])
        denied_write = self.executor.execute(
            "Write",
            {
                "file_path": "skills/L1_tools/molclaw-test/SKILL.md",
                "content": "changed",
            },
        )
        self.assertFalse(denied_write["ok"])
        denied_l2 = self.executor.execute(
            "Read", {"file_path": "skills/L2_workflows/anything.md"}
        )
        self.assertFalse(denied_l2["ok"])

    def test_path_traversal_and_symlink_escape_are_rejected(self) -> None:
        outside = Path(self.tmp.name) / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "escape").symlink_to(outside)
        for path in ("../outside.txt", str(outside), "escape"):
            with self.subTest(path=path):
                result = self.executor.execute("Read", {"file_path": path})
                self.assertFalse(result["ok"])

    def test_restricted_bash_allows_file_commands(self) -> None:
        write = self.executor.execute(
            "Bash", {"command": "echo first > run_log.md"}
        )
        self.assertTrue(write["ok"])
        append = self.executor.execute(
            "Bash", {"command": "echo second >> run_log.md"}
        )
        self.assertTrue(append["ok"])
        read = self.executor.execute("Bash", {"command": "cat run_log.md | wc -l"})
        self.assertTrue(read["ok"])
        self.assertEqual(read["result"]["stdout"].strip(), "2")

    def test_restricted_bash_rejects_dangerous_operations(self) -> None:
        commands = [
            "curl https://example.com",
            "python -c 'print(1)'",
            "rm -rf .",
            "cat $(pwd)/result.md",
            "find . -exec cat {} ;",
            "find . /etc",
            "find . ../outside",
            "cat /etc/passwd",
            "test -f /etc/passwd",
            "echo bad > ../outside.txt",
        ]
        for command in commands:
            with self.subTest(command=command):
                result = self.executor.execute("Bash", {"command": command})
                self.assertFalse(result["ok"])


class ToolRegistryDispatchTest(unittest.TestCase):
    def test_registry_dispatches_local_and_mcp_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skills = root / "skills"
            skills.mkdir()
            local = LocalToolExecutor(root / "workspace", skills)
            mcp = _FakeMCPExecutor()
            registry = ToolRegistry(
                executor=mcp,
                include_local_tools=True,
            )

            local_ok, local_reason = registry.validate_tool_name(
                "Write", allowed_tools=["Write", "is_valid_smiles"]
            )
            self.assertTrue(local_ok, local_reason)
            local_result = registry.execute(
                "Write",
                {"file_path": "result.md", "content": "done"},
                local_executor=local,
            )
            self.assertTrue(local_result["ok"])

            mcp_result = registry.execute("is_valid_smiles", {"smiles": "CCO"})
            self.assertTrue(mcp_result["ok"])
            self.assertEqual(mcp.calls, [("is_valid_smiles", {"smiles": "CCO"})])

            for tool_name in LOCAL_TOOL_NAMES:
                self.assertIsNotNone(registry.load_tool_schema(tool_name))

    def test_debug_context_and_prompt_advertise_local_tools_only_when_enabled(self) -> None:
        row = {"metadata": {"env_kwargs": {"allowed_tools": ["is_valid_smiles"]}}}
        messages = [{"role": "user", "content": "Run the task"}]
        with mock.patch.dict("os.environ", {"DRUG_AGENT_ENABLE_LOCAL_TOOLS": "1"}):
            context = _sample_context(row)
        self.assertTrue(context["local_tools_enabled"])
        self.assertTrue(set(LOCAL_TOOL_NAMES).issubset(context["allowed_tools"]))
        prompted = _augment_messages_for_strict_json(
            messages,
            local_tools_enabled=True,
        )
        self.assertIn("Available local tools", prompted[0]["content"])

        with mock.patch.dict("os.environ", {"DRUG_AGENT_ENABLE_LOCAL_TOOLS": "0"}):
            disabled = _sample_context(row)
        self.assertFalse(disabled["local_tools_enabled"])
        self.assertFalse(set(LOCAL_TOOL_NAMES) & set(disabled["allowed_tools"]))
        no_local_prompt = _augment_messages_for_strict_json(
            messages,
            local_tools_enabled=False,
        )
        self.assertNotIn("Available local tools", no_local_prompt[0]["content"])

    def test_fresh_debug_prompt_drops_teacher_and_injects_catalog_once(self) -> None:
        fresh = _fresh_messages([
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "teacher decision"},
            {"role": "user", "content": "teacher observation"},
        ])
        self.assertEqual([item["role"] for item in fresh], ["system", "user"])
        self.assertNotIn("teacher", str(fresh))
        prompted = _augment_messages_for_strict_json(
            fresh,
            local_tools_enabled=True,
            tool_catalog="CATALOG_SENTINEL",
            final_contract="FINAL_SENTINEL",
        )
        joined = "\n".join(item["content"] for item in prompted)
        self.assertEqual(joined.count("CATALOG_SENTINEL"), 1)
        self.assertEqual(joined.count("FINAL_SENTINEL"), 1)


if __name__ == "__main__":
    unittest.main()
