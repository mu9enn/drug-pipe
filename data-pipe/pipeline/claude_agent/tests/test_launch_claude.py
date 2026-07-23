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
            skills = root / "skills"
            skills.mkdir()
            fake = root / "fake-claude"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import os\n"
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
                    "--skills-root",
                    str(skills),
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
            attempt = workdir / "attempts" / "attempt_0001" / "complete_session.jsonl"
            canonical = workdir / "complete_session.jsonl"
            self.assertEqual(attempt.read_bytes(), canonical.read_bytes())
            metadata = json.loads((workdir / "run_meta.json").read_text())
            self.assertEqual(metadata["selected_claude_attempt"], 1)
            self.assertTrue(metadata["raw_session_valid"])
            self.assertEqual(
                metadata["selected_session_sha256"],
                hashlib.sha256(canonical.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
