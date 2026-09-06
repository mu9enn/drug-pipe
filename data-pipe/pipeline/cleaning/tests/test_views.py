from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.cleaning.deployment_tools import DeploymentToolSet, LOCAL_TOOLS, load_deployment_tool_set
from pipeline.cleaning.materialize_sft import materialize_sft
from pipeline.cleaning.sft_views import semantic_to_qwen35_sft


def semantic() -> dict:
    return {
        "schema_version": "drug_agent_semantic_trajectory_v1",
        "id": "sample",
        "user_task": "task",
        "events": [
            {"type": "assistant_decision", "source_message_id": "d1", "reasoning": "reason", "tool_calls": [{"name": "tool", "arguments": {}, "source_tool_use_id": "c1"}], "final_answer": None},
            {"type": "tool_observation", "name": "tool", "source_tool_use_id": "c1", "status": "success", "is_error": False, "content": "ok"},
            {"type": "assistant_decision", "source_message_id": "d2", "reasoning": "done", "tool_calls": [], "final_answer": "answer"},
        ],
        "metadata": {"task_type": "kg", "source_session_sha256": "source-sha"},
    }


def deployment_tools(*extra: str) -> DeploymentToolSet:
    names = sorted(LOCAL_TOOLS) + list(extra)
    return DeploymentToolSet(
        tuple({"name": name, "raw_name": name, "description": "", "input_schema": {"type": "object"}} for name in names),
        Path("deployment.json"),
        "sha",
    )


class ViewTest(unittest.TestCase):
    def test_adapter_rejects_deployment_state_in_semantic_metadata(self) -> None:
        source = copy.deepcopy(semantic())
        source["metadata"]["deployment_tool_set_sha256"] = "stale"
        with self.assertRaisesRegex(ValueError, "semantic_contains_deployment"):
            semantic_to_qwen35_sft(
                source,
                deployment_tools=deployment_tools("tool"),
                system_prompt="system",
            )

    def test_structured_sft_preserves_semantic_decisions(self) -> None:
        tools = deployment_tools("tool")
        sft = semantic_to_qwen35_sft(semantic(), deployment_tools=tools, system_prompt="system")
        self.assertEqual([message["role"] for message in sft["messages"]], ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(sft["messages"][2]["tool_calls"][0]["function"]["arguments"], {})
        self.assertNotIn("<thought>", str(sft))

    def test_dynamic_visibility_keeps_local_used_and_deterministic_distractor(self) -> None:
        tools = deployment_tools("tool", "unused_a", "unused_b")
        first = semantic_to_qwen35_sft(
            semantic(), deployment_tools=tools, system_prompt="system",
            tool_visibility="trajectory-plus-distractors",
        )
        second = semantic_to_qwen35_sft(
            semantic(), deployment_tools=tools, system_prompt="system",
            tool_visibility="trajectory-plus-distractors",
        )
        names = {tool["function"]["name"] for tool in first["tools"]}
        self.assertTrue(LOCAL_TOOLS <= names)
        self.assertIn("tool", names)
        self.assertEqual(len(names - LOCAL_TOOLS - {"tool"}), 1)
        self.assertEqual(first["tools"], second["tools"])

    def test_materializer_reads_jsonl_rows_not_reader_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "semantic.jsonl"
            source.write_text(json.dumps(semantic()) + "\n", encoding="utf-8")
            manifest = root / "tools.json"
            manifest.write_text(json.dumps({"tools": [
                {"name": name, "input_schema": {"type": "object"}}
                for name in sorted(LOCAL_TOOLS | {"tool"})
            ]}), encoding="utf-8")
            result = materialize_sft(
                source,
                root / "out",
                deployment_tool_set=manifest,
                system_prompt="system",
            )
            self.assertEqual(result["sft_count"], 1)
            self.assertEqual(
                len(json.loads((root / "out/qwen35_sft.pretty.json").read_text())),
                1,
            )
            self.assertFalse((root / "out/toolrl_candidates.jsonl").exists())
            self.assertTrue((root / "out/materialization_manifest.json").is_file())

    def test_rejects_obsolete_resource_protocol_in_deployment_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tools.json"
            path.write_text(json.dumps({"tools": [{
                "name": "tool",
                "input_schema": {"type": "object", "description": "reuse resource://old/path"},
            }]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "obsolete path-protocol"):
                load_deployment_tool_set(path)
