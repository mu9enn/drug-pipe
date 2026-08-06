from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from pipeline.claude_agent.run_claude import (
    Sample,
    _check_session_mcp_ready,
    _load_mcp_tool_timeout_ms,
    _run_single_rollout,
)


class RolloutAttemptCaptureTest(unittest.TestCase):
    def test_reads_numeric_molclaw_server_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = Path(td) / "mcp.json"
            config.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "molclaw-scp": {
                                "type": "http",
                                "url": "http://example.invalid/mcp",
                                "timeout": 14_400_000,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(_load_mcp_tool_timeout_ms(config), 14_400_000)

    def test_max_workers_parallelizes_distinct_task_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            skills = root / "skills"
            skills.mkdir()
            (skills / "CLAUDE.md").write_text("test", encoding="utf-8")
            (skills / ".claude/skills/execute-molclaw-trajectory").mkdir(parents=True)
            (skills / ".claude/skills/execute-molclaw-trajectory/SKILL.md").write_text(
                "---\nname: execute-molclaw-trajectory\ndescription: Test skill.\n---\n",
                encoding="utf-8",
            )
            prompt = root / "prompt.md"
            prompt.write_text("test prompt", encoding="utf-8")
            mcp_config = root / "mcp.json"
            mcp_config.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "molclaw-scp": {
                                "type": "http",
                                "url": "http://example.invalid/mcp",
                                "timeout": 14_400_000,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            dataset = root / "tasks.csv"
            with dataset.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["question_id", "question", "answer", "raw_question_json"],
                )
                writer.writeheader()
                for index in range(1, 4):
                    writer.writerow(
                        {
                            "question_id": index,
                            "question": f"question {index}",
                            "answer": "",
                            "raw_question_json": json.dumps(
                                {"toolchain": {"tools": ["is_valid_smiles"]}}
                            ),
                        }
                    )
            state = root / "state.json"
            state.write_text('{"active":0,"peak":0}', encoding="utf-8")
            fake = root / "fake-claude"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import fcntl, json, os, time\n"
                "from pathlib import Path\n"
                "state = Path(os.environ['FAKE_CLAUDE_STATE'])\n"
                "with state.open('r+') as handle:\n"
                " fcntl.flock(handle, fcntl.LOCK_EX); data=json.load(handle); data['active']+=1; data['peak']=max(data['peak'],data['active']); handle.seek(0); json.dump(data,handle); handle.truncate(); fcntl.flock(handle, fcntl.LOCK_UN)\n"
                "time.sleep(0.08)\n"
                "with state.open('r+') as handle:\n"
                " fcntl.flock(handle, fcntl.LOCK_EX); data=json.load(handle); data['active']-=1; handle.seek(0); json.dump(data,handle); handle.truncate(); fcntl.flock(handle, fcntl.LOCK_UN)\n"
                "print(json.dumps({'type':'system','subtype':'init','tools':['mcp__molclaw-scp__x'],'mcp_servers':[{'name':'molclaw-scp','status':'connected'}]}), flush=True)\n"
                "print(json.dumps({'type':'result','result':'<answer>{\"result\":\"ok\"}</answer>'}), flush=True)\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            script = Path(__file__).resolve().parents[1] / "run_claude.py"
            env = dict(os.environ)
            env["FAKE_CLAUDE_STATE"] = str(state)
            completed = subprocess.run(
                [
                    str(script), "--task", "kg", "--dataset-csv", str(dataset),
                    "--skills-root", str(skills), "--system-prompt-file", str(prompt),
                    "--results-root", str(root / "results"), "--claude-bin", str(fake),
                    "--mcp-config-file", str(mcp_config), "--strict-mcp-config",
                    "--skip-provider-switch", "--max-workers", "2",
                ],
                env=env,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertEqual(json.loads(state.read_text())["peak"], 2)
            run_dir = next((root / "results").glob("molbench_kg_*"))
            config = json.loads((run_dir / "run_config.json").read_text())
            self.assertEqual(config["max_workers"], 2)
            self.assertEqual(config["mcp_tool_timeout_ms"], 14_400_000)
            row_meta_files = sorted(run_dir.glob("row*/run_meta.json"))
            self.assertEqual(len(row_meta_files), 3)
            for row_meta_file in row_meta_files:
                row_meta = json.loads(row_meta_file.read_text())
                self.assertEqual(row_meta["mcp_tool_timeout_ms"], 14_400_000)

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
            counter = root / "invocations.txt"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "counter = Path(os.environ['FAKE_CLAUDE_COUNTER'])\n"
                "n = int(counter.read_text()) + 1 if counter.exists() else 1\n"
                "counter.write_text(str(n))\n"
                "if n == 1: Path('stale_result.md').write_text('attempt one only')\n"
                "if n > 1 and Path('stale_result.md').exists(): sys.exit(9)\n"
                "if n == 2:\n"
                " Path('selected_result.md').write_text('attempt two only')\n"
                " Path('question.json').write_text('corrupted by claude')\n"
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
                    "FAKE_CLAUDE_COUNTER": str(counter),
                },
            ):
                result = _run_single_rollout(
                    task="ac",
                    sample=sample,
                    sample_root=workdir,
                    rollout_index=0,
                    num_rollouts=1,
                    prompt="prompt",
                    system_prompt="Invoke /execute-molclaw-trajectory.",
                    source_claude_dir=skills,
                    provider="test",
                    claude_bin=str(fake),
                    mcp_config_file=mcp_config,
                    strict_mcp_config=True,
                )

            self.assertEqual(result.return_code, 0)
            metadata = json.loads((workdir / "run_meta.json").read_text())
            self.assertEqual(metadata["selected_claude_attempt"], 2)
            self.assertEqual(len(metadata["claude_attempts"]), 2)
            self.assertEqual(counter.read_text(), "2")
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
            attempt_one_workdir = attempts[0].parent / "workdir"
            attempt_two_workdir = attempts[1].parent / "workdir"
            self.assertTrue((attempt_one_workdir / "stale_result.md").is_file())
            self.assertFalse((attempt_two_workdir / "stale_result.md").exists())
            self.assertTrue((attempt_two_workdir / "selected_result.md").is_file())
            self.assertFalse((workdir / "stale_result.md").exists())
            self.assertEqual((workdir / "selected_result.md").read_text(), "attempt two only")
            canonical_question = json.loads((workdir / "question.json").read_text())
            self.assertEqual(canonical_question["question_text"], "test")

    def test_completed_mcp_call_overrides_pending_init_without_retry(self) -> None:
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
            counter = root / "invocations.txt"
            fake = root / "fake-claude"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os\n"
                "from pathlib import Path\n"
                "counter=Path(os.environ['FAKE_CLAUDE_COUNTER'])\n"
                "n=int(counter.read_text())+1 if counter.exists() else 1\n"
                "counter.write_text(str(n))\n"
                "Path('result.md').write_text('fresh result')\n"
                "print(json.dumps({'type':'system','subtype':'init','tools':[],"
                "'mcp_servers':[{'name':'molclaw-scp','status':'pending'}]}), flush=True)\n"
                "print(json.dumps({'type':'assistant','message':{'content':["
                "{'type':'tool_use','id':'call_1','name':'mcp__molclaw-scp__x','input':{}}]}}), flush=True)\n"
                "print(json.dumps({'type':'user','message':{'content':["
                "{'type':'tool_result','tool_use_id':'call_1','content':'{\\\"status\\\":\\\"success\\\"}'}]}}), flush=True)\n"
                "print(json.dumps({'type':'result','result':'<answer>[\\\"CC\\\"]</answer>'}), flush=True)\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            sample = Sample(1, "1", "test", "", [], [], 0)
            with patch.dict(
                os.environ,
                {
                    "CLAUDE_MCP_READY_RETRIES": "2",
                    "CLAUDE_MCP_READY_RETRY_WAIT_SEC": "0",
                    "FAKE_CLAUDE_COUNTER": str(counter),
                },
            ):
                result = _run_single_rollout(
                    task="ac",
                    sample=sample,
                    sample_root=workdir,
                    rollout_index=0,
                    num_rollouts=1,
                    prompt="prompt",
                    system_prompt="Invoke /execute-molclaw-trajectory.",
                    source_claude_dir=skills,
                    provider="test",
                    claude_bin=str(fake),
                    mcp_config_file=mcp_config,
                    strict_mcp_config=True,
                )

            self.assertEqual(result.return_code, 0)
            self.assertEqual(counter.read_text(), "1")
            metadata = json.loads((workdir / "run_meta.json").read_text())
            self.assertEqual(metadata["mcp_ready_reason"], "observed_mcp_tool_result")
            self.assertEqual(metadata["selected_claude_attempt"], 1)
            self.assertEqual((workdir / "result.md").read_text(), "fresh result")

    def test_error_tool_result_does_not_override_pending_init(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            session = Path(td) / "complete_session.jsonl"
            events = [
                {
                    "type": "system",
                    "subtype": "init",
                    "tools": [],
                    "mcp_servers": [{"name": "molclaw-scp", "status": "pending"}],
                },
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call_1",
                                "name": "mcp__molclaw-scp__x",
                            }
                        ]
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call_1",
                                "is_error": True,
                                "content": "MCP server unavailable",
                            }
                        ]
                    },
                },
            ]
            session.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            ready, reason, snapshot = _check_session_mcp_ready(
                session, ["molclaw-scp"]
            )
            self.assertFalse(ready)
            self.assertEqual(reason, "mcp_server_not_connected:molclaw-scp:pending")
            self.assertEqual(snapshot["observed_mcp_tool_results"]["molclaw-scp"], [])

    def test_rerun_uses_new_attempt_and_replaces_promoted_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workdir = root / "sample"
            skills = root / "skills"
            skills.mkdir()
            claude_md = root / "CLAUDE.md"
            claude_md.write_text("test", encoding="utf-8")
            counter = root / "invocations.txt"
            fake = root / "fake-claude"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os\n"
                "from pathlib import Path\n"
                "counter=Path(os.environ['FAKE_CLAUDE_COUNTER'])\n"
                "n=int(counter.read_text())+1 if counter.exists() else 1\n"
                "counter.write_text(str(n))\n"
                "Path('old_only.txt' if n == 1 else 'new_only.txt').write_text(str(n))\n"
                "print(json.dumps({'type':'result','result':'<answer>[\\\"CC\\\"]</answer>'}), flush=True)\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            sample = Sample(1, "1", "test", "", [], [], 0)

            def run_once():
                return _run_single_rollout(
                    task="ac",
                    sample=sample,
                    sample_root=workdir,
                    rollout_index=0,
                    num_rollouts=1,
                    prompt="prompt",
                    system_prompt="Invoke /execute-molclaw-trajectory.",
                    source_claude_dir=skills,
                    provider="test",
                    claude_bin=str(fake),
                    mcp_config_file=None,
                    strict_mcp_config=False,
                )

            with patch.dict(os.environ, {"FAKE_CLAUDE_COUNTER": str(counter)}):
                self.assertEqual(run_once().return_code, 0)
                self.assertTrue((workdir / "old_only.txt").is_file())
                self.assertEqual(run_once().return_code, 0)

            self.assertFalse((workdir / "old_only.txt").exists())
            self.assertEqual((workdir / "new_only.txt").read_text(), "2")
            self.assertTrue(
                (workdir / "attempts" / "attempt_0001" / "complete_session.jsonl").is_file()
            )
            self.assertTrue(
                (workdir / "attempts" / "attempt_0002" / "complete_session.jsonl").is_file()
            )
            manifest = json.loads(
                (workdir / "selected_attempt_artifacts.json").read_text()
            )
            self.assertEqual(manifest["selected_attempt"], 2)
            self.assertEqual(manifest["promoted_entries"], ["new_only.txt"])


if __name__ == "__main__":
    unittest.main()
