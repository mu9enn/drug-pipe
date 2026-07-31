from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from molclaw_kg.adjudicators.claude_code_runtime import ClaudeCodeRuntime
from molclaw_kg.settings import build_config


class ClaudeCodeRuntimeCaptureTest(unittest.TestCase):
    def _fake_claude(self, root: Path) -> Path:
        executable = root / "fake-claude"
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os\n"
            "from pathlib import Path\n"
            "counter = Path('invocations.txt')\n"
            "n = int(counter.read_text()) + 1 if counter.exists() else 1\n"
            "counter.write_text(str(n))\n"
            "status = 'failed' if n == 1 else 'connected'\n"
            "print(json.dumps({'type':'system','subtype':'init',"
            "'tools':['mcp__molclaw-scp__x'],"
            "'mcp_servers':[{'name':'molclaw-scp','status':status}]}), flush=True)\n"
            "print(json.dumps({'type':'result','result':'{\"ok\":true}'}), flush=True)\n"
            "Path('claude_environment.json').write_text(json.dumps({"
            "'concurrency': os.environ.get('CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY'),"
            "'background_disabled': os.environ.get('CLAUDE_CODE_DISABLE_BACKGROUND_TASKS')}))\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        return executable

    def test_mcp_retry_preserves_both_attempts_and_selects_last(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = build_config(
                root,
                run_id="runtime",
                server_url="http://example.invalid/mcp",
                api_key="test",
            )
            runtime = ClaudeCodeRuntime(config)
            runtime.claude_bin = str(self._fake_claude(root))
            workdir = root / "work"
            run = runtime.run_prompt(
                "prompt",
                run_label="test",
                workdir=workdir,
                builtin_tools="Read",
                allowed_tools="Read",
                disallowed_tools="Bash,WebSearch,Agent",
            )

            self.assertTrue(run.ok)
            self.assertEqual(run.selected_claude_attempt, 2)
            self.assertEqual(len(run.claude_attempts), 2)
            first, second = (Path(path) for path in run.attempt_session_files)
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())
            canonical = Path(run.session_file)
            self.assertEqual(canonical.read_bytes(), second.read_bytes())
            self.assertEqual(
                run.claude_attempts[-1]["sha256"],
                hashlib.sha256(canonical.read_bytes()).hexdigest(),
            )
            self.assertNotIn("[agent-mcp-not-ready]", first.read_text())
            self.assertNotIn("[agent-mcp-not-ready]", canonical.read_text())
            self.assertIn("--tools Read", run.command)
            self.assertIn("--allowedTools Read", run.command)
            self.assertIn("--disallowedTools Bash,WebSearch,Agent", run.command)
            claude_env = json.loads((workdir / "claude_environment.json").read_text())
            self.assertEqual(claude_env["concurrency"], "2")
            self.assertEqual(claude_env["background_disabled"], "1")


if __name__ == "__main__":
    unittest.main()
