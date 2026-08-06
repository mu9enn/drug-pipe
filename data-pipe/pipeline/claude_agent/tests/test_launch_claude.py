from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class LaunchClaudeSmokeTest(unittest.TestCase):
    def test_single_sample_archives_raw_stream_and_selected_copy(self) -> None:
        script = Path(__file__).resolve().parents[1] / "launch_claude.sh"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake = root / "fake-claude"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "cfg = Path(sys.argv[sys.argv.index('--mcp-config') + 1])\n"
                "Path('observed_mcp_config.json').write_text(json.dumps(json.loads(cfg.read_text())))\n"
                "Path('observed_cli_args.json').write_text(json.dumps(sys.argv[1:]))\n"
                "os.write(1, b'{\"type\":\"system\",\"subtype\":\"init\"}\\n')\n"
                "os.write(2, b'{\"type\":\"result\",\"result\":\"ok\"}\\n')\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            workdir = root / "work"
            environment = {
                **os.environ,
                "MOLCLAW_SCP_MCP_URL": "http://example.invalid/mcp",
                "MOLCLAW_SCP_MCP_AUTH": "test",
            }
            process = subprocess.run(
                [
                    "bash",
                    str(script),
                    "--workdir",
                    str(workdir),
                    "--prompt",
                    "test",
                    "--claude-bin",
                    str(fake),
                    "--skip-provider-switch",
                    "--skip-mcp-verify",
                ],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            canonical_skills = (
                Path(__file__).resolve().parents[4]
                / "workdir-skills/molclaw-trajectory-execution"
            )
            self.assertIn(f"skills_root={canonical_skills}", process.stdout)
            observed_config = json.loads((workdir / "observed_mcp_config.json").read_text())
            timeout = observed_config["mcpServers"]["molclaw-scp"]["timeout"]
            self.assertEqual(timeout, 14_400_000)
            self.assertIs(type(timeout), int)
            self.assertTrue((workdir / ".claude/skills/execute-molclaw-trajectory/SKILL.md").is_file())
            self.assertFalse((workdir / "CLAUDE.md").exists())
            self.assertFalse((workdir / "system_prompt.md").exists())
            observed_args = json.loads((workdir / "observed_cli_args.json").read_text())
            self.assertEqual(
                observed_args[observed_args.index("--system-prompt") + 1],
                (canonical_skills / "system_prompt.md").read_text().strip(),
            )
            self.assertEqual(observed_args[observed_args.index("-p") + 1], "test")
            attempt = workdir / "attempts" / "attempt_0001" / "complete_session.jsonl"
            canonical = workdir / "complete_session.jsonl"
            self.assertEqual(attempt.read_bytes(), canonical.read_bytes())
            metadata = json.loads((workdir / "run_meta.json").read_text())
            self.assertEqual(metadata["selected_claude_attempt"], 1)
            self.assertEqual(metadata["mcp_tool_timeout_ms"], 14_400_000)
            self.assertTrue(metadata["raw_session_valid"])
            self.assertEqual(
                metadata["selected_session_sha256"],
                hashlib.sha256(canonical.read_bytes()).hexdigest(),
            )

    def test_rejects_invalid_molclaw_tool_timeout(self) -> None:
        script = Path(__file__).resolve().parents[1] / "launch_claude.sh"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for index, value in enumerate(("999", "14.5", "four-hours"), start=1):
                with self.subTest(value=value):
                    environment = {
                        **os.environ,
                        "MOLCLAW_SCP_MCP_URL": "http://example.invalid/mcp",
                        "MOLCLAW_SCP_MCP_AUTH": "test",
                        "MOLCLAW_MCP_TOOL_TIMEOUT_MS": value,
                    }
                    process = subprocess.run(
                        [
                            "bash",
                            str(script),
                            "--workdir",
                            str(root / f"work_{index}"),
                            "--prompt",
                            "test",
                            "--skip-provider-switch",
                        ],
                        env=environment,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(process.returncode, 0)
                    self.assertIn("must be an integer >= 1000", process.stderr)


if __name__ == "__main__":
    unittest.main()
