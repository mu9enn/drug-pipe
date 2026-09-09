from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/smoke_deepseek_harness.py"
SPEC = importlib.util.spec_from_file_location("smoke_deepseek_harness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class DeepSeekHarnessSmokeTest(unittest.TestCase):
    def test_summarizes_dsh_tool_round_and_dsml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dsh.jsonl"
            records = [
                {"type": "session", "version": 0, "id": "s", "createdAt": 1, "cwd": "/tmp"},
                {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
                {"type": "assistant/message", "seq": 1, "time": 2, "data": {"turn": 1, "message": {
                    "role": "assistant", "content": [
                        {"type": "text", "text": "<｜DSML｜tool_calls>"},
                        {"type": "tool-call", "id": "c", "name": "read", "arguments": "{}"},
                    ],
                }}},
                {"type": "tool/call", "seq": 2, "time": 3, "data": {"name": "read"}},
                {"type": "tool/result", "seq": 3, "time": 4, "data": {}},
                {"type": "turn/end", "seq": 4, "time": 5, "data": {"reason": {"kind": "completed"}}},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            summary = smoke.summarize_trace(path)
            self.assertEqual(summary["format"], "dsh-session-jsonl-v0")
            self.assertEqual(summary["tool_names"], ["read"])
            self.assertEqual(summary["tool_result_count"], 1)
            self.assertTrue(summary["assistant_dsml_detected"])

    def test_compares_claude_stream_with_non_json_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dsh = root / "dsh.jsonl"
            claude = root / "claude.jsonl"
            dsh.write_text(
                json.dumps({"type": "tool/call", "seq": 1, "data": {"name": "read"}}) + "\n"
                + json.dumps({"type": "tool/result", "seq": 2, "data": {}}) + "\n",
                encoding="utf-8",
            )
            claude.write_text(
                "[claude-code:notice] diagnostic\n"
                + json.dumps({"type": "assistant", "message": {"content": [
                    {"type": "tool_use", "id": "c", "name": "Read", "input": {}},
                ]}}) + "\n"
                + json.dumps({"type": "user", "message": {"content": [
                    {"type": "tool_result", "tool_use_id": "c", "content": "ok"},
                ]}}) + "\n"
                + json.dumps({"type": "result", "subtype": "success", "is_error": False}) + "\n",
                encoding="utf-8",
            )
            report = smoke.compare_traces(dsh, claude)
            self.assertFalse(report["same_format"])
            self.assertEqual(report["claude"]["malformed_line_count"], 1)
            self.assertEqual(report["claude"]["tool_names"], ["Read"])
            self.assertEqual(report["claude"]["tool_result_count"], 1)

    def test_session_discovery_requires_exactly_one_uncompressed_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "found 0"):
                smoke._find_session(home)
            first = home / "sessions/project/session-a/session.jsonl"
            first.parent.mkdir(parents=True)
            first.write_text("{}\n", encoding="utf-8")
            self.assertEqual(smoke._find_session(home), first)
            second = home / "sessions/project/session-b/session.jsonl"
            second.parent.mkdir(parents=True)
            second.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "found 2"):
                smoke._find_session(home)

    def test_generated_patch_keeps_sessions_uncompressed_and_unpacked(self) -> None:
        patch = smoke._dsh_patch("deepseek-v4-flash")
        self.assertIn("compression: none", patch)
        self.assertIn("packChunks: false", patch)
        self.assertIn("provider: deepseek-official", patch)
        self.assertNotIn("DEEPSEEK_API_KEY", patch)

    def test_smoke_pass_requires_false_timeout_and_false_dsml(self) -> None:
        validation = {
            "process_exit_zero": True,
            "timed_out": False,
            "session_found": True,
            "tool_call_recorded": True,
            "tool_result_recorded": True,
            "turn_completed": True,
            "probe_returned": True,
            "assistant_dsml_detected": False,
        }
        self.assertTrue(smoke.smoke_passed(validation))
        validation["assistant_dsml_detected"] = True
        self.assertFalse(smoke.smoke_passed(validation))
        validation["assistant_dsml_detected"] = False
        validation["timed_out"] = True
        self.assertFalse(smoke.smoke_passed(validation))

    def test_run_smoke_captures_fake_dsh_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node = root / "node"
            node.write_text("#!/bin/sh\necho v24.19.0\n", encoding="utf-8")
            node.chmod(0o755)
            dsh = root / "dsh"
            dsh.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os\n"
                "from pathlib import Path\n"
                "marker=Path('dsh_probe.txt').read_text().strip()\n"
                "session=Path(os.environ['DSH_HOME'])/'sessions/project/session-test/session.jsonl'\n"
                "session.parent.mkdir(parents=True)\n"
                "rows=[\n"
                " {'type':'session','version':0,'id':'session-test','createdAt':1,'cwd':str(Path.cwd())},\n"
                " {'type':'turn/start','seq':0,'time':1,'data':{'turn':1}},\n"
                " {'type':'tool/call','seq':1,'time':2,'data':{'name':'read'}},\n"
                " {'type':'tool/result','seq':2,'time':3,'data':{}},\n"
                " {'type':'assistant/message','seq':3,'time':4,'data':{'message':{'content':[{'type':'text','text':marker}]}}},\n"
                " {'type':'turn/end','seq':4,'time':5,'data':{'reason':{'kind':'completed'}}},\n"
                "]\n"
                "session.write_text(''.join(json.dumps(row)+'\\n' for row in rows))\n"
                "print(marker)\n",
                encoding="utf-8",
            )
            dsh.chmod(0o755)
            output = root / "output"
            args = type("Args", (), {
                "output_dir": output,
                "dsh_bin": str(dsh),
                "node_bin": str(node),
                "model": "deepseek-v4-flash",
                "timeout_sec": 10,
            })()
            with patch.dict(os.environ, {
                "DEEPSEEK_BASE_URL": "https://example.invalid/v1",
                "DEEPSEEK_API_KEY": "secret-not-written",
            }):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(smoke.run_smoke(args), 0)
            metadata = json.loads((output / "run_meta.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["passed"])
            self.assertNotIn("secret-not-written", (output / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["session"]["tool_names"], ["read"])


if __name__ == "__main__":
    unittest.main()
