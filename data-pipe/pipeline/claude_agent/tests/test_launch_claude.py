from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class LaunchClaudeSmokeTest(unittest.TestCase):
    def test_legacy_single_sample_entry_is_removed(self) -> None:
        script = Path(__file__).resolve().parents[1] / "launch_claude.sh"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
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
                ],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("Unknown arg: --workdir", process.stderr)
            self.assertFalse(workdir.exists())

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
                            "--run-dataset",
                            "--task",
                            "kg",
                            "--dataset-csv",
                            str(root / f"task_{index}.csv"),
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
