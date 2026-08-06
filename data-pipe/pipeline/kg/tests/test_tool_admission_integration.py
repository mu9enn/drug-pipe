from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[2] / "claude_agent" / "run_claude.py"


class ToolAdmissionIntegrationTest(unittest.TestCase):
    def test_same_compute_tool_is_serial_while_different_tool_overlaps(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            skills = root / "skills"
            skills.mkdir()
            (skills / "system_prompt.md").write_text(
                "Invoke /execute-molclaw-trajectory.", encoding="utf-8"
            )
            (skills / ".claude/skills/execute-molclaw-trajectory").mkdir(parents=True)
            (skills / ".claude/skills/execute-molclaw-trajectory/SKILL.md").write_text(
                "---\nname: execute-molclaw-trajectory\ndescription: Test skill.\n---\n",
                encoding="utf-8",
            )
            event_log = root / "events.jsonl"
            fake_claude = root / "fake-claude"
            fake_claude.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, time\n"
                "from pathlib import Path\n"
                "question = json.loads(Path('question.json').read_text())\n"
                "spec = question['kg_task_spec']\n"
                "task_id = spec['task_id']\n"
                "tool = spec['toolchain']['tools'][0]\n"
                "log = Path(os.environ['ADMISSION_EVENT_LOG'])\n"
                "def emit(event):\n"
                "    with log.open('a', encoding='utf-8') as stream:\n"
                "        stream.write(json.dumps({'event':event,'task_id':task_id,'tool':tool,'time':time.time()}) + '\\n')\n"
                "emit('start')\n"
                "time.sleep(0.25 if tool == 'foldx_tool' else 0.05)\n"
                "emit('end')\n"
                "print(json.dumps({'type':'system','subtype':'init','tools':[],'mcp_servers':[]}))\n"
                "print(json.dumps({'type':'result','result':'done'}))\n",
                encoding="utf-8",
            )
            fake_claude.chmod(0o755)

            dataset = root / "tasks.csv"
            rows = [
                ("foldx-1", "foldx_tool"),
                ("foldx-2", "foldx_tool"),
                ("boltz-1", "pred_binding_affinity_boltz2"),
            ]
            with dataset.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["index", "question_id", "question", "answer", "raw_question_json"],
                )
                writer.writeheader()
                for task_id, tool in rows:
                    spec = {
                        "task_id": task_id,
                        "question": f"Run {task_id}",
                        "toolchain": {"tools": [tool]},
                    }
                    writer.writerow(
                        {
                            "index": task_id,
                            "question_id": task_id,
                            "question": spec["question"],
                            "answer": "[]",
                            "raw_question_json": json.dumps(spec),
                        }
                    )

            process = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--task",
                    "kg",
                    "--dataset-csv",
                    str(dataset),
                    "--skills-root",
                    str(skills),
                    "--results-root",
                    str(root / "results"),
                    "--claude-bin",
                    str(fake_claude),
                    "--max-workers",
                    "2",
                    "--skip-provider-switch",
                ],
                env={**os.environ, "ADMISSION_EVENT_LOG": str(event_log)},
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            events = [
                json.loads(line)
                for line in event_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            intervals: dict[str, dict[str, float]] = {}
            for event in events:
                intervals.setdefault(event["task_id"], {})[event["event"]] = event["time"]

            first_foldx, second_foldx = intervals["foldx-1"], intervals["foldx-2"]
            self.assertGreaterEqual(second_foldx["start"], first_foldx["end"])
            boltz = intervals["boltz-1"]
            self.assertLess(boltz["start"], first_foldx["end"])
            self.assertLess(first_foldx["start"], boltz["end"])

            run_dirs = list((root / "results").glob("molbench_kg_*"))
            self.assertEqual(len(run_dirs), 1)
            summaries = [
                json.loads(line)
                for line in (run_dirs[0] / "run_summary.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            by_id = {row["dataset_index"]: row for row in summaries}
            self.assertEqual(
                by_id["foldx-2"]["tool_admission"]["blocked_by_tools"],
                ["foldx_tool"],
            )


if __name__ == "__main__":
    unittest.main()
