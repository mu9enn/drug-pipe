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
from pipeline.output_contracts import normalize_task_prompt

from pipeline.claude_agent.run_claude import (
    Sample,
    _check_session_mcp_ready,
    _load_mcp_tool_timeout_ms,
    _prepare_claude_workdir,
    _run_single_rollout,
)


class QuestionOnlyKGTest(unittest.TestCase):
    def test_question_only_kg_reserves_constrained_tools(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            skills = root / "skills"
            skills.mkdir()
            (skills / "CLAUDE.md").write_text("test", encoding="utf-8")
            (skills / ".claude/skills/L1_tools/test-tool").mkdir(parents=True)
            (skills / ".claude/skills/L1_tools/test-tool/SKILL.md").write_text(
                "---\nname: test-tool\ndescription: Test tool.\n---\n\n# Test tool\n",
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
                            "question": normalize_task_prompt(f"question {index}", "kg"),
                            "answer": "",
                            "raw_question_json": "",
                        }
                    )
            state = root / "state.json"
            state.write_text('{"active":0,"peak":0}', encoding="utf-8")
            fake = root / "fake-claude"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import fcntl, json, os, sys, time\n"
                "from pathlib import Path\n"
                "if any(Path(name).exists() for name in ('prompt.txt','run_meta.json','complete_session.jsonl','attempts')): sys.exit(11)\n"
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
            self.assertEqual(json.loads(state.read_text())["peak"], 1)
            run_dir = next((root / "results").glob("molbench_kg_*"))
            config = json.loads((run_dir / "run_config.json").read_text())
            self.assertEqual(config["max_workers"], 2)
            self.assertEqual(config["mcp_tool_timeout_ms"], 14_400_000)
            self.assertEqual(
                config["source_dataset_sha256"],
                hashlib.sha256(dataset.read_bytes()).hexdigest(),
            )
            row_meta_files = sorted(run_dir.glob("row*/run_meta.json"))
            self.assertEqual(len(row_meta_files), 3)
            for row_meta_file in row_meta_files:
                row_meta = json.loads(row_meta_file.read_text())
                self.assertEqual(row_meta["mcp_tool_timeout_ms"], 14_400_000)
                self.assertEqual(row_meta["source_dataset_sha256"], config["source_dataset_sha256"])
                self.assertEqual(row_meta["system_prompt_sha256"], config["system_prompt_sha256"])
                selected = json.loads((row_meta_file.parent / "selected_attempt_artifacts.json").read_text())
                for key in (
                    "question_sha256", "user_prompt_sha256", "system_prompt_sha256",
                    "selected_session_sha256", "source_dataset_sha256",
                ):
                    self.assertEqual(selected[key], row_meta[key])

