from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from pipeline.claude_agent.run_claude import _extract_result_text_from_stream_jsonl
from pipeline.claude_agent.session_capture import (
    _deepseek_environment,
    extract_assistant_text,
    http_500_retry_delay,
    run_deepseek_harness,
    run_stream_json,
    select_attempt,
    session_format,
    session_has_retryable_http_500,
)


class SessionCaptureTest(unittest.TestCase):
    def test_resolves_deepseek_credentials_from_named_cc_switch_provider(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            database = Path(td) / "cc-switch.db"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "CREATE TABLE providers (id TEXT, app_type TEXT, settings_config TEXT)"
                )
                connection.execute(
                    "INSERT INTO providers VALUES (?, ?, ?)",
                    ("dsv4flash", "claude", json.dumps({"env": {
                        "ANTHROPIC_BASE_URL": "https://example.invalid",
                        "ANTHROPIC_API_KEY": "secret",
                    }})),
                )
            with patch.dict(os.environ, {
                "CC_SWITCH_DB": str(database),
                "DEEPSEEK_BASE_URL": "",
                "DEEPSEEK_API_KEY": "",
            }):
                resolved = _deepseek_environment("dsv4flash")
            self.assertEqual(resolved["DEEPSEEK_BASE_URL"], "https://example.invalid/v1")
            self.assertEqual(resolved["DEEPSEEK_API_KEY"], "secret")

    def _fake_claude(self, root: Path, body: str) -> Path:
        executable = root / "fake-claude"
        executable.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
        executable.chmod(0o755)
        return executable

    def test_detects_only_terminal_api_http_500(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            retryable = root / "retryable.jsonl"
            retryable.write_text(
                json.dumps(
                    {
                        "type": "result",
                        "is_error": True,
                        "result": "API Error: ChatCompletionStreamResponse code': 500",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertTrue(session_has_retryable_http_500(retryable))
            retryable.write_text(
                json.dumps({"type": "result", "is_error": True, "result": "API Error: code 429"})
                + "\n",
                encoding="utf-8",
            )
            self.assertFalse(session_has_retryable_http_500(retryable))
            self.assertEqual([http_500_retry_delay(i) for i in range(1, 7)], [30, 60, 120, 240, 300, 300])

    def test_combined_stream_is_archived_and_selected_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake = self._fake_claude(
                root,
                "import os\n"
                "os.write(1, b'{\"type\":\"system\",\"subtype\":\"init\"}\\n')\n"
                "os.write(2, b'{\"type\":\"result\",\"result\":\"ok\"}\\n')\n",
            )
            command = [str(fake), "--verbose", "--output-format", "stream-json"]
            attempt = run_stream_json(command, cwd=root, archive_root=root)
            canonical = root / "complete_session.jsonl"
            selected = select_attempt(attempt, canonical)

            expected = (
                b'{"type":"system","subtype":"init"}\n'
                b'{"type":"result","result":"ok"}\n'
            )
            self.assertEqual(Path(attempt["session_file"]).read_bytes(), expected)
            self.assertEqual(canonical.read_bytes(), expected)
            self.assertEqual(selected["sha256"], hashlib.sha256(expected).hexdigest())
            self.assertTrue(attempt["raw_session_valid"])
            self.assertEqual(_extract_result_text_from_stream_jsonl(canonical), "")  # Runtime result is not an assistant answer.
            attempt_pretty = json.loads(Path(attempt["pretty_session_file"]).read_text())
            canonical_pretty = json.loads(Path(selected["pretty_session_file"]).read_text())
            self.assertEqual(attempt_pretty, canonical_pretty)
            self.assertEqual([event["type"] for event in canonical_pretty], ["system", "result"])

    def test_pretty_sidecar_preserves_non_json_runtime_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake = self._fake_claude(
                root,
                "import os\n"
                "os.write(1, b'[claude-code:notice] diagnostic\\n')\n"
                "os.write(1, b'{\"type\":\"result\",\"result\":\"ok\"}\\n')\n",
            )
            attempt = run_stream_json(
                [str(fake), "--verbose", "--output-format", "stream-json"],
                cwd=root,
                archive_root=root,
            )
            pretty = json.loads(Path(attempt["pretty_session_file"]).read_text())
            self.assertEqual(pretty[0]["type"], "raw_stream_diagnostic")
            self.assertEqual(pretty[0]["line_number"], 1)
            self.assertEqual(pretty[1]["type"], "result")

    def test_claude_process_uses_bounded_foreground_execution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake = self._fake_claude(
                root,
                "import json, os\n"
                "print(json.dumps({\n"
                " 'type': 'result',\n"
                " 'result': json.dumps({\n"
                "  'concurrency': os.environ.get('CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY'),\n"
                "  'background_disabled': os.environ.get('CLAUDE_CODE_DISABLE_BACKGROUND_TASKS'),\n"
                " })\n"
                "}))\n",
            )
            command = [str(fake), "--verbose", "--output-format", "stream-json"]
            attempt = run_stream_json(command, cwd=root, archive_root=root)
            payload = json.loads(json.loads(Path(attempt["session_file"]).read_text())["result"])
            self.assertEqual(payload["concurrency"], "2")
            self.assertEqual(payload["background_disabled"], "1")

    def test_attempts_never_overwrite_and_empty_output_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake = self._fake_claude(root, "")
            command = [str(fake), "--verbose", "--output-format", "stream-json"]
            first = run_stream_json(command, cwd=root, archive_root=root)
            second = run_stream_json(command, cwd=root, archive_root=root)
            self.assertEqual(first["attempt_index"], 1)
            self.assertEqual(second["attempt_index"], 2)
            self.assertTrue(Path(first["session_file"]).is_file())
            self.assertTrue(Path(second["session_file"]).is_file())
            self.assertFalse(first["raw_session_valid"])
            self.assertEqual(first["byte_count"], 0)

    def test_missing_executable_leaves_empty_raw_file_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            attempt = run_stream_json(
                [str(root / "missing"), "--verbose", "--output-format", "stream-json"],
                cwd=root,
                archive_root=root,
            )
            self.assertEqual(attempt["return_code"], 127)
            self.assertEqual(attempt["failure"], "executable_not_found")
            self.assertTrue(Path(attempt["session_file"]).is_file())
            self.assertEqual(Path(attempt["session_file"]).read_bytes(), b"")

    def test_timeout_preserves_partial_stream_without_runner_marker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake = self._fake_claude(
                root,
                "import os, time\n"
                "os.write(1, b'{\"type\":\"system\",\"subtype\":\"init\"}\\n')\n"
                "time.sleep(2)\n",
            )
            attempt = run_stream_json(
                [str(fake), "--verbose", "--output-format", "stream-json"],
                cwd=root,
                archive_root=root,
                # Leave enough time for the Python shebang interpreter to
                # start on a loaded login host while still timing out during
                # the explicit two-second sleep.
                timeout_sec=0.5,
            )
            raw = Path(attempt["session_file"]).read_text()
            self.assertEqual(attempt["return_code"], 124)
            self.assertTrue(attempt["timed_out"])
            self.assertTrue(attempt["raw_session_valid"])
            self.assertNotIn("[agent-timeout]", raw)

    def test_nonzero_exit_preserves_cli_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake = self._fake_claude(
                root,
                "import os, sys\n"
                "os.write(2, b'{\"type\":\"result\",\"is_error\":true}\\n')\n"
                "sys.exit(9)\n",
            )
            attempt = run_stream_json(
                [str(fake), "--verbose", "--output-format", "stream-json"],
                cwd=root,
                archive_root=root,
            )
            self.assertEqual(attempt["return_code"], 9)
            self.assertTrue(attempt["raw_session_valid"])
            self.assertIn(
                '"is_error":true',
                Path(attempt["session_file"]).read_text(),
            )

    def test_deepseek_harness_captures_canonical_session_without_secret_patch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workdir = root / "workdir"
            workdir.mkdir()
            (workdir / ".git").mkdir()
            node = root / "node"
            node.write_text("#!/bin/sh\nprintf 'v24.19.0\\n'\n", encoding="utf-8")
            node.chmod(0o755)
            fake = root / "fake-dsh"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os\n"
                "from pathlib import Path\n"
                "session=Path(os.environ['DSH_HOME'])/'sessions/project/session-1/session.jsonl'\n"
                "session.parent.mkdir(parents=True)\n"
                "rows=[\n"
                " {'type':'session','version':0},\n"
                " {'type':'assistant/message','data':{'message':{'content':[{'type':'text','text':'done'}]}}},\n"
                " {'type':'turn/end','data':{'reason':{'kind':'completed'}}},\n"
                "]\n"
                "session.write_text(''.join(json.dumps(row)+'\\n' for row in rows))\n"
                "print('done')\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            mcp = root / "mcp.json"
            mcp.write_text(json.dumps({"mcpServers": {"test": {
                "type": "http", "url": "https://example.invalid/mcp",
                "headers": {"Authorization": "secret-mcp-token"},
            }}}), encoding="utf-8")
            with patch.dict(os.environ, {
                "DEEPSEEK_BASE_URL": "https://example.invalid/v1",
                "DEEPSEEK_API_KEY": "secret-model-token",
            }):
                attempt = run_deepseek_harness(
                    "prompt", "system", cwd=workdir, archive_root=workdir,
                    dsh_bin=str(fake), node_bin=str(node), mcp_config_file=mcp,
                )
            session = Path(attempt["session_file"])
            self.assertEqual(attempt["return_code"], 0)
            self.assertEqual(session_format(session), "dsh-session-jsonl-v0")
            self.assertEqual(extract_assistant_text(session, final_only=True), "done")
            self.assertFalse(any(session.parent.glob("*.patch.yml")))
            artifacts = "".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in session.parent.rglob("*") if path.is_file()
            )
            self.assertNotIn("secret-mcp-token", artifacts)
            self.assertNotIn("secret-model-token", artifacts)

    def test_deepseek_harness_rejects_unsupported_node_before_creating_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            node = root / "node"
            node.write_text("#!/bin/sh\nprintf 'v18.20.0\\n'\n", encoding="utf-8")
            node.chmod(0o755)
            fake = self._fake_claude(root, "")
            with patch.dict(os.environ, {
                "DEEPSEEK_BASE_URL": "https://example.invalid/v1",
                "DEEPSEEK_API_KEY": "secret-model-token",
            }):
                with self.assertRaisesRegex(RuntimeError, "requires Node"):
                    run_deepseek_harness(
                        "prompt", "system", cwd=root, archive_root=root,
                        dsh_bin=str(fake), node_bin=str(node),
                    )
            self.assertFalse((root / "attempts").exists())


if __name__ == "__main__":
    unittest.main()
