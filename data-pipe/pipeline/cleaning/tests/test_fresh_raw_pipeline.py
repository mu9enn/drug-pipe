from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from pipeline.output_contracts import normalize_task_prompt

from pipeline.cleaning.materialize_sft import materialize_sft
from pipeline.cleaning.python_clean import python_clean
from pipeline.cleaning.skill_native_augmentation import augment_file


class FreshRawPipelineTest(unittest.TestCase):
    def test_collected_contract_is_preserved_through_native_skill_sft(self) -> None:
        drug_pipe = Path(__file__).resolve().parents[4]
        data_pipe = drug_pipe / "data-pipe"
        scene = drug_pipe / "workdir-skills/molclaw-trajectory-execution"
        skills = drug_pipe / "workdir-skills/molclaw-l1-workspace/.agents/skills"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "kg.csv"
            with dataset.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["question_id", "question", "answer", "raw_question_json"],
                )
                writer.writeheader()
                writer.writerow({
                    "question_id": "kg1",
                    "question": normalize_task_prompt("Validate ethanol SMILES with the available tool.", "kg"),
                    "answer": "",
                    "raw_question_json": json.dumps(
                        {"toolchain": {"tools": ["is_valid_smiles"]}}
                    ),
                })

            mcp = root / "mcp.json"
            mcp.write_text(json.dumps({
                "mcpServers": {
                    "molclaw-scp": {
                        "type": "http", "url": "http://example.invalid/mcp", "timeout": 60000,
                    }
                }
            }))
            fake = root / "fake-claude"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "prompt=sys.argv[-1]\n"
                "assert '\"result\"' in prompt and '\"evidence\"' in prompt\n"
                "events=[\n"
                " {'type':'system','subtype':'init','tools':['mcp__molclaw-scp__is_valid_smiles'],'mcp_servers':[{'name':'molclaw-scp','status':'connected'}]},\n"
                " {'type':'assistant','message':{'id':'d1','content':[{'type':'thinking','thinking':'Use the validator.'},{'type':'tool_use','id':'c1','name':'mcp__molclaw-scp__is_valid_smiles','input':{'smiles_list':['CCO']}}]}},\n"
                " {'type':'user','message':{'content':[{'type':'tool_result','tool_use_id':'c1','content':{'valid':True}}]}},\n"
                " {'type':'assistant','message':{'id':'d2','content':[{'type':'thinking','thinking':'Report the grounded result.'},{'type':'text','text':'{\"result\":{\"valid\":true},\"evidence\":[{\"tool\":\"is_valid_smiles\",\"valid\":true}]}' }]}},\n"
                " {'type':'result','subtype':'success','is_error':False,'result':'{\"result\":{\"valid\":true},\"evidence\":[{\"tool\":\"is_valid_smiles\",\"valid\":true}]}'}\n"
                "]\n"
                "for event in events: print(json.dumps(event), flush=True)\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)

            env = dict(os.environ)
            env["PYTHONPATH"] = str(data_pipe)
            completed = subprocess.run(
                [
                    str(data_pipe / "pipeline/claude_agent/run_claude.py"),
                    "--task", "kg", "--dataset-csv", str(dataset),
                    "--skills-root", str(scene), "--results-root", str(root / "raw"),
                    "--claude-bin", str(fake), "--mcp-config-file", str(mcp),
                    "--strict-mcp-config", "--skip-provider-switch", "--max-workers", "1",
                ],
                env=env,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

            semantic_root = root / "semantic"
            manifest = python_clean(root / "raw", semantic_root)
            self.assertEqual(manifest["semantic_valid_count"], 1)
            augmented_root = root / "augmented"
            augment_file(semantic_root / "semantic_trajectories.jsonl", augmented_root, skills)
            sft_root = root / "sft"
            materialize_sft(
                augmented_root / "semantic_trajectories.jsonl",
                sft_root,
                deployment_tool_set=data_pipe / "configs/dsh_molclaw_tool_set.json",
                system_prompt=(data_pipe / "pipeline/cleaning/prompts/qwen35_system.md").read_text(),
                user_prompt_prefix=(drug_pipe / "workdir-skills/molclaw-l1-workspace/prompt_prefix.md").read_text(),
            )

            row = json.loads((sft_root / "qwen35_sft.jsonl").read_text().strip())
            final = json.loads(row["messages"][-1]["content"])
            self.assertEqual(set(final), {"result", "evidence"})
            self.assertNotIn("task_type", final)
            calls = [
                call["function"]["name"]
                for message in row["messages"]
                for call in message.get("tool_calls", [])
            ]
            self.assertEqual(calls[0], "skill")
            self.assertIn("mcp__molclaw-scp__is_valid_smiles", calls)
            task_message = row["messages"][1]["content"]
            self.assertIn('"result"', task_message)
            self.assertIn('"evidence"', task_message)


if __name__ == "__main__":
    unittest.main()
