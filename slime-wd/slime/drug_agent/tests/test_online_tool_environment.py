from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from drug_agent.protocol.react_protocol import parse_runtime_decision, project_final_answer
from drug_agent.tools.artifact_registry import ArtifactRegistry
from drug_agent.tools.local_tools import LocalToolExecutor
from drug_agent.tools.mcp_client import MCPClient
from drug_agent.tools.server_file_materializer import materialize_server_file_result
from drug_agent.tools.tool_executor import MCPToolExecutor
from drug_agent.tools.tool_registry import ToolRegistry


class _CatalogExecutor:
    def __init__(self, count: int = 79, extra_tools: list[dict] | None = None) -> None:
        self.calls = []
        self.count = count
        self.extra_tools = list(extra_tools or [])

    def list_tools(self):
        return [
            {
                "name": f"tool_{index}",
                "description": "live",
                "input_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "integer", "minimum": 1}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            }
            for index in range(self.count)
        ] + self.extra_tools

    def execute(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return {"ok": True, "result": arguments}


class LiveCatalogTest(unittest.TestCase):
    def test_entire_live_catalog_is_authoritative_and_schema_is_complete(self):
        executor = _CatalogExecutor()
        registry = ToolRegistry(executor=executor, include_local_tools=False)
        self.assertEqual(len(registry.list_tools()), 79)
        self.assertEqual(registry.validate_tool_name("tool_78"), (True, None))
        self.assertEqual(registry.validate_tool_name("old_alias"), (False, "tool_not_found_in_registry"))
        self.assertTrue(registry.validate_arguments("tool_0", {"value": 2})[0])
        ok, reason = registry.validate_arguments("tool_0", {"value": 0, "extra": True})
        self.assertFalse(ok)
        self.assertIn("json_schema", reason or "")

    def test_catalog_growth_discovers_and_routes_returned_visualization_tools(self):
        visualization_tools = [
            {
                "name": "visualize_molecule",
                "description": "Render a molecule",
                "input_schema": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                },
            },
            {
                "name": "visualize_protein",
                "description": "Render a protein",
                "input_schema": {
                    "type": "object",
                    "properties": {"pdb_file_path": {"type": "string"}},
                    "required": ["pdb_file_path"],
                },
            },
        ]
        executor = _CatalogExecutor(extra_tools=visualization_tools)
        registry = ToolRegistry(executor=executor, include_local_tools=False)

        self.assertEqual(len(registry.list_tools()), 81)
        calls = [
            ("visualize_molecule", {"input": "CCO"}),
            ("visualize_protein", {"pdb_file_path": "/server/protein.pdb"}),
        ]
        for name, arguments in calls:
            self.assertEqual(registry.validate_tool_name(name), (True, None))
            self.assertEqual(registry.validate_arguments(name, arguments), (True, None))
            self.assertTrue(registry.execute(name, arguments)["ok"])
        self.assertEqual(executor.calls, calls)


class ArtifactRegistryTest(unittest.TestCase):
    def test_server_path_round_trip_and_stable_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ArtifactRegistry(tmp)
            raw = "/data/tool_result/pdbfixer/fixed.pdb"
            first = registry.canonicalize({"output_file": raw})["output_file"]
            second = registry.canonicalize(f"generated {raw}")
            self.assertIn(first, second)
            self.assertEqual(registry.resolve(first), raw)

    def test_local_artifact_maps_to_workspace_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "task"
            registry = ArtifactRegistry(workspace)
            ref = registry.canonicalize({"path": "run_log.md"}, local_result=True)["path"]
            self.assertEqual(ref, "<artifact:local/run_log.md>")
            self.assertEqual(Path(registry.resolve(ref)), workspace / "run_log.md")
            skills = Path(tmp) / "skills"
            skills.mkdir()
            local = LocalToolExecutor(workspace, skills)
            result = local.execute("Write", {"file_path": ref, "content": "ok"})
            self.assertTrue(result["ok"])
            self.assertEqual((workspace / "run_log.md").read_text(), "ok")

    def test_fabricated_local_artifact_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ArtifactRegistry(Path(tmp) / "task")
            malicious = "<artifact:local/../../etc/passwd>"
            self.assertEqual(registry.resolve(malicious), malicious)

    def test_model_final_can_only_reuse_observed_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ArtifactRegistry(tmp)
            known = registry.register("/server/task/known.pdb")
            final = registry.canonicalize(
                {"known": known, "unknown_ref": "<artifact:structure/other.pdb>", "raw": "/server/fake.pdb"},
                register_unknown_paths=False,
            )
            self.assertEqual(final["known"], known)
            self.assertEqual(final["unknown_ref"], "<artifact:unavailable/other.pdb>")
            self.assertEqual(final["raw"], "<artifact:unavailable/fake.pdb>")

    def test_task_registries_do_not_share_raw_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            left = ArtifactRegistry(Path(tmp) / "left")
            right = ArtifactRegistry(Path(tmp) / "right")
            left_ref = left.register("/server/task-left/result.pdb")
            right_ref = right.register("/server/task-right/result.pdb")
            self.assertEqual(left_ref, right_ref)
            self.assertNotEqual(left.resolve(left_ref), right.resolve(right_ref))


class ServerFileMaterializerTest(unittest.TestCase):
    def test_large_base64_is_atomically_materialized_and_removed_from_result(self):
        payload = (b"PDB-DATA\n" * 300_000) + b"END\n"
        encoded = base64.b64encode(payload).decode("ascii")
        with tempfile.TemporaryDirectory() as tmp:
            registry = ArtifactRegistry(Path(tmp) / "task")
            result = materialize_server_file_result(
                {
                    "ok": True,
                    "result": {
                        "status": "success",
                        "file_name": "result.pdb",
                        "base64_string": encoded,
                    },
                    "error": None,
                    "transport_ok": True,
                    "tool_schema_valid": True,
                    "tool_execution_success": True,
                    "tool_semantic_success": True,
                    "semantic_unknown": False,
                    "metadata": {"raw": {"base64_string": encoded}},
                },
                execution_args={"file_path": "/server/result.pdb"},
                artifact_registry=registry,
            )

            expected_sha256 = hashlib.sha256(payload).hexdigest()
            self.assertTrue(result["ok"])
            self.assertEqual(
                result["result"],
                {
                    "status": "success",
                    "artifact": "<artifact:local/result.pdb>",
                    "bytes_written": len(payload),
                    "sha256": expected_sha256,
                },
            )
            self.assertEqual((registry.workspace / "result.pdb").read_bytes(), payload)
            self.assertNotIn("base64", json.dumps(result))
            self.assertFalse(list(registry.workspace.glob("*.tmp")))

    def test_nested_payload_and_untrusted_file_name_stay_inside_workspace(self):
        payload = b"safe"
        with tempfile.TemporaryDirectory() as tmp:
            registry = ArtifactRegistry(Path(tmp) / "task")
            result = materialize_server_file_result(
                {
                    "ok": True,
                    "result": {
                        "content": json.dumps(
                            {
                                "file_name": "../../outside.pdb",
                                "base64_string": base64.b64encode(payload).decode("ascii"),
                            }
                        )
                    },
                    "metadata": {},
                },
                execution_args={"file_path": "/server/fallback.pdb"},
                artifact_registry=registry,
            )
            self.assertTrue(result["ok"])
            self.assertEqual(result["result"]["artifact"], "<artifact:local/outside.pdb>")
            self.assertEqual((registry.workspace / "outside.pdb").read_bytes(), payload)
            self.assertFalse((Path(tmp) / "outside.pdb").exists())

    def test_invalid_base64_returns_compact_error_without_creating_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ArtifactRegistry(Path(tmp) / "task")
            result = materialize_server_file_result(
                {
                    "ok": True,
                    "result": {"file_name": "bad.bin", "base64_string": "not base64!"},
                    "metadata": {"raw": {"large": "discarded"}},
                },
                execution_args={"file_path": "/server/bad.bin"},
                artifact_registry=registry,
            )
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"]["type"], "ServerFileMaterializationError")
            self.assertNotIn("raw", result["metadata"])
            self.assertFalse((registry.workspace / "bad.bin").exists())


class FakeOnlineLoopSmokeTest(unittest.TestCase):
    def test_fake_model_tool_observation_replan_and_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = _CatalogExecutor(count=1)
            registry = ToolRegistry(executor=executor, include_local_tools=False)
            registry.list_tools()
            artifact_registry = ArtifactRegistry(tmp)

            first = parse_runtime_decision(
                '<thought>Call the live tool.</thought>'
                '<tool_call>{"tool_name":"tool_0","arguments":{"value":2}}</tool_call>'
            )
            self.assertTrue(first["ok"])
            call = first["tool_calls"][0]
            self.assertTrue(registry.validate_tool_name(call["tool_name"])[0])
            self.assertTrue(registry.validate_arguments(call["tool_name"], call["arguments"])[0])
            result = registry.execute(call["tool_name"], call["arguments"])
            observation = artifact_registry.canonicalize(
                {"result": result["result"], "output_file": "/server/task/result.sdf"}
            )
            self.assertTrue(observation["output_file"].startswith("<artifact:"))

            terminal = parse_runtime_decision(
                '<thought>The observation is sufficient.</thought>'
                '<final_answer>{"task_type":"pf","selected_smiles":["CCO"],"evidence":[]}</final_answer>'
            )
            self.assertTrue(terminal["ok"])
            self.assertEqual(project_final_answer(terminal["final_answer"]), ["CCO"])


class MCPResultParsingTest(unittest.TestCase):
    def test_structured_content_and_multiple_content_blocks_are_preserved(self):
        class Result:
            structuredContent = {"status": "success", "value": 7}
            content = []

        client = MCPClient(server_url="http://invalid", api_key="not-used")
        self.assertEqual(client.parse_result(Result()), {"status": "success", "value": 7})

        class Text:
            def __init__(self, text):
                self.text = text

        class Multi:
            structuredContent = None
            isError = False
            content = [Text('{"status":"success"}'), Text("details")]

        parsed = client.parse_result(Multi())
        self.assertEqual(parsed["content"][0], {"status": "success"})
        self.assertEqual(parsed["content"][1], {"text": "details"})


class MCPExecutorTaskAffinityTest(unittest.TestCase):
    def test_transport_lifecycle_stays_in_one_asyncio_task(self):
        class FakeClient:
            instances = []

            def __init__(self, **_kwargs):
                self.connected = False
                self.task_ids = []
                self.__class__.instances.append(self)

            def _record(self):
                self.task_ids.append(id(asyncio.current_task()))

            async def connect(self):
                self._record()
                self.connected = True
                return True

            async def list_tools(self):
                self._record()
                return [{"name": "ping", "description": "", "inputSchema": {"type": "object"}}]

            async def call_tool(self, tool_name, arguments):
                self._record()
                return {
                    "parsed": {"status": "success", "tool": tool_name, "arguments": arguments},
                    "raw": {"status": "success"},
                }

            async def disconnect(self):
                self._record()
                self.connected = False

        env = {
            **os.environ,
            "DRUG_AGENT_ALLOW_TOOL_ENV": "1",
            "DRUG_AGENT_TRAINING_OFFLINE": "0",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "drug_agent.tools.tool_executor.MCPClient", FakeClient
        ):
            executor = MCPToolExecutor(connect_on_init=True)
            try:
                self.assertEqual(executor.list_tools()[0]["name"], "ping")
                self.assertTrue(executor.execute("ping", {"value": 1})["ok"])
            finally:
                executor.close()

        self.assertEqual(len(FakeClient.instances), 1)
        self.assertEqual(len(set(FakeClient.instances[0].task_ids)), 1)

    def test_repeated_executor_close_releases_threads_loops_and_file_descriptors(self):
        class FakeClient:
            def __init__(self, **_kwargs):
                self.connected = False

            async def connect(self):
                self.connected = True

            async def disconnect(self):
                self.connected = False

        fd_root = Path("/proc/self/fd")
        before = len(list(fd_root.iterdir())) if fd_root.is_dir() else None
        env = {
            **os.environ,
            "DRUG_AGENT_ALLOW_TOOL_ENV": "1",
            "DRUG_AGENT_TRAINING_OFFLINE": "0",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "drug_agent.tools.tool_executor.MCPClient", FakeClient
        ):
            for _ in range(80):
                executor = MCPToolExecutor(connect_on_init=False)
                owner_thread = executor._thread
                owner_loop = executor._loop
                executor.close()
                executor.close()  # idempotent teardown is required by finally paths
                self.assertFalse(owner_thread.is_alive())
                self.assertTrue(owner_loop.is_closed())

        if before is not None:
            after = len(list(fd_root.iterdir()))
            self.assertLessEqual(after, before + 3, f"file descriptor growth: before={before}, after={after}")

    def test_constructor_connection_failures_do_not_leak_file_descriptors(self):
        class FailingClient:
            def __init__(self, **_kwargs):
                self.connected = False

            async def connect(self):
                raise RuntimeError("fake connection failure")

            async def disconnect(self):
                self.connected = False

        fd_root = Path("/proc/self/fd")
        before = len(list(fd_root.iterdir())) if fd_root.is_dir() else None
        env = {
            **os.environ,
            "DRUG_AGENT_ALLOW_TOOL_ENV": "1",
            "DRUG_AGENT_TRAINING_OFFLINE": "0",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "drug_agent.tools.tool_executor.MCPClient", FailingClient
        ):
            for _ in range(40):
                with self.assertRaisesRegex(RuntimeError, "fake connection failure"):
                    MCPToolExecutor(connect_on_init=True)

        if before is not None:
            after = len(list(fd_root.iterdir()))
            self.assertLessEqual(after, before + 3, f"file descriptor growth: before={before}, after={after}")

    def test_connect_timeout_cancels_owner_operation_before_close(self):
        class SlowClient:
            def __init__(self, **_kwargs):
                self.connected = False
                self.disconnect_task_ids = []

            async def connect(self):
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    self.disconnect_task_ids.append(id(asyncio.current_task()))
                    raise

            async def disconnect(self):
                self.connected = False

        env = {
            **os.environ,
            "DRUG_AGENT_ALLOW_TOOL_ENV": "1",
            "DRUG_AGENT_TRAINING_OFFLINE": "0",
            "MOLCLAW_CONNECT_TIMEOUT_SEC": "0.02",
            "MOLCLAW_CONNECT_MAX_ATTEMPTS": "1",
            "MOLCLAW_TIMEOUT_CLEANUP_GRACE_SEC": "0.2",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "drug_agent.tools.tool_executor.MCPClient", SlowClient
        ):
            executor = MCPToolExecutor(connect_on_init=False)
            owner_thread = executor._thread
            with self.assertRaisesRegex(TimeoutError, "MCP connect timeout"):
                executor.list_tools()
            executor.close()
            self.assertFalse(owner_thread.is_alive())
            self.assertTrue(executor._loop.is_closed())

    def test_transient_connect_failure_is_retried(self):
        class FlakyClient:
            def __init__(self, **_kwargs):
                self.connected = False
                self.attempts = 0

            async def connect(self):
                self.attempts += 1
                if self.attempts < 3:
                    raise RuntimeError("temporary proxy failure")
                self.connected = True
                return True

            async def list_tools(self):
                return {"tools": [{"name": "fix_pdb", "inputSchema": {}}]}

            async def disconnect(self):
                self.connected = False

        env = {
            **os.environ,
            "DRUG_AGENT_ALLOW_TOOL_ENV": "1",
            "DRUG_AGENT_TRAINING_OFFLINE": "0",
            "MOLCLAW_CONNECT_MAX_ATTEMPTS": "3",
            "MOLCLAW_CONNECT_RETRY_BACKOFF_SEC": "0.001",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "drug_agent.tools.tool_executor.MCPClient", FlakyClient
        ):
            executor = MCPToolExecutor(connect_on_init=False)
            try:
                tools = executor.list_tools()
                self.assertEqual([item["name"] for item in tools], ["fix_pdb"])
                self.assertEqual(executor._client.attempts, 3)
            finally:
                executor.close()
