from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.claude_agent.run_claude import _extract_result_text_from_stream_jsonl
from pipeline.claude_agent.session_capture import (
    http_500_retry_delay,
    run_stream_json,
    select_attempt,
    session_has_retryable_http_500,
)


class SessionCaptureTest(unittest.TestCase):
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
            self.assertEqual(_extract_result_text_from_stream_jsonl(canonical), "ok")

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
            payload = json.loads(_extract_result_text_from_stream_jsonl(Path(attempt["session_file"])))
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
                timeout_sec=0.1,
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


if __name__ == "__main__":
    unittest.main()
