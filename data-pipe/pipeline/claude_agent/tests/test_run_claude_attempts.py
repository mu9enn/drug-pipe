from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline.claude_agent.run_claude import Sample, _run_single_rollout


class RolloutAttemptCaptureTest(unittest.TestCase):
    def test_mcp_ready_retry_keeps_every_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workdir = root / "sample"
            skills = root / "skills"
            skills.mkdir()
            claude_md = root / "CLAUDE.md"
            claude_md.write_text("test", encoding="utf-8")
            mcp_config = root / "mcp.json"
            mcp_config.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "molclaw-scp": {
                                "type": "http",
                                "url": "http://example.invalid/mcp",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            fake = root / "fake-claude"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json\n"
                "from pathlib import Path\n"
                "counter = Path('invocations.txt')\n"
                "n = int(counter.read_text()) + 1 if counter.exists() else 1\n"
                "counter.write_text(str(n))\n"
                "status = 'failed' if n == 1 else 'connected'\n"
                "print(json.dumps({'type':'system','subtype':'init',"
                "'tools':['mcp__molclaw-scp__x'],"
                "'mcp_servers':[{'name':'molclaw-scp','status':status}]}), flush=True)\n"
                "print(json.dumps({'type':'result','result':'<answer>[\"CC\"]</answer>'}), flush=True)\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            sample = Sample(
                row_number=1,
                dataset_index="1",
                question_text="test",
                raw_question_json="",
                candidates=[],
                answer=[],
                n_active=0,
            )
            with patch.dict(
                os.environ,
                {
                    "CLAUDE_MCP_READY_RETRIES": "2",
                    "CLAUDE_MCP_READY_RETRY_WAIT_SEC": "0",
                },
            ):
                result = _run_single_rollout(
                    task="ac",
                    sample=sample,
                    sample_root=workdir,
                    rollout_index=0,
                    num_rollouts=1,
                    prompt="prompt",
                    source_claude_dir=skills,
                    source_claude_md=claude_md,
                    provider="test",
                    claude_bin=str(fake),
                    mcp_config_file=mcp_config,
                    strict_mcp_config=True,
                )

            self.assertEqual(result.return_code, 0)
            metadata = json.loads((workdir / "run_meta.json").read_text())
            self.assertEqual(metadata["selected_claude_attempt"], 2)
            self.assertEqual(len(metadata["claude_attempts"]), 2)
            attempts = [
                Path(item["session_file"]) for item in metadata["claude_attempts"]
            ]
            self.assertTrue(all(path.is_file() for path in attempts))
            canonical = workdir / "complete_session.jsonl"
            self.assertEqual(canonical.read_bytes(), attempts[-1].read_bytes())
            self.assertEqual(
                metadata["selected_session_sha256"],
                hashlib.sha256(canonical.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
