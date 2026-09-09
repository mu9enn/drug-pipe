from __future__ import annotations

import json
import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from drug_agent.context_summary import ClaudeContextSummarizer, build_source_inventory, validate_context_summary


class ContextSummaryContractTest(unittest.TestCase):
    def setUp(self):
        self.messages = [
            {
                "_source_message_index": 4,
                "role": "assistant",
                "content": '<thought>validate input</thought><tool_call>{"tool_name":"Read","arguments":{"file_path":"artifact.json"}}</tool_call>',
            },
            {
                "_source_message_index": 5,
                "role": "user",
                "content": '<observation tool_name="Read">{"ok":true,"status":"success","artifact":"artifact.json","id":"run-1"}</observation>',
            },
        ]
        self.inventory = build_source_inventory(self.messages)

    def test_grounded_summary_validates(self):
        summary = {
            "schema_version": "react_context_summary_v1",
            "source_context_sha256": "a" * 64,
            "events": [
                {
                    "source_message_indices": [4, 5],
                    "rationale": "Validate the recorded input.",
                    "tool_calls": [{"source_message_index": 4, "tool_name": "Read", "arguments": {"file_path": "artifact.json"}}],
                    "observations": [{
                        "source_message_index": 5,
                        "tool_name": "Read",
                        "status": "success",
                        "artifacts": ["artifact.json"],
                        "paths": [],
                        "ids": ["run-1"],
                        "error": None,
                        "result_summary": "The recorded read succeeded.",
                    }],
                }
            ],
            "unresolved_state": [],
        }
        self.assertEqual(
            validate_context_summary(summary, source_context_sha256="a" * 64, source_inventory=self.inventory),
            [],
        )

    def test_hallucinated_exact_value_is_rejected(self):
        summary = {
            "schema_version": "react_context_summary_v1",
            "source_context_sha256": "a" * 64,
            "events": [{
                "source_message_indices": [5],
                "rationale": "",
                "tool_calls": [],
                "observations": [{
                    "source_message_index": 5,
                    "tool_name": "Read",
                    "status": "success",
                    "artifacts": ["invented.pdb"],
                    "paths": [],
                    "ids": [],
                    "error": None,
                    "result_summary": "",
                }],
            }],
            "unresolved_state": [],
        }
        findings = validate_context_summary(
            summary, source_context_sha256="a" * 64, source_inventory=self.inventory
        )
        self.assertTrue(any(item.startswith("ungrounded_exact_value") for item in findings))

    def test_chunking_keeps_assistant_observation_pair(self):
        chunks = ClaudeContextSummarizer._chunks(self.messages * 3, max_chars=1)
        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 2])

    def test_provider_withholds_target_and_reuses_hash_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "fake-claude"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json\n"
                "from pathlib import Path\n"
                "request=json.loads(Path('context_request.json').read_text())\n"
                "history=Path('omitted_history.json').read_text()\n"
                "assert 'SECRET_CURRENT_GOLD' not in history\n"
                "result={'schema_version':'react_context_summary_v1','source_context_sha256':request['source_context_sha256'],'events':[],'unresolved_state':[]}\n"
                "Path('context_summary.json').write_text(json.dumps(result))\n"
                "print('{\"type\":\"result\",\"result\":\"summary written\"}', flush=True)\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            provider = ClaudeContextSummarizer(
                cache_root=root / "cache", claude_bin=str(fake), timeout_sec=5, max_attempts=3
            )
            first, first_audit = provider.summarize(self.messages)
            second, second_audit = provider.summarize(self.messages)
            self.assertEqual(first, second)
            self.assertFalse(first_audit["calls"][0]["cache_hit"])
            self.assertTrue(second_audit["calls"][0]["cache_hit"])

    def test_deepseek_provider_writes_and_caches_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node = root / "node"
            node.write_text("#!/bin/sh\nprintf 'v24.19.0\\n'\n", encoding="utf-8")
            node.chmod(0o755)
            fake = root / "fake-dsh"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os\n"
                "from pathlib import Path\n"
                "request=json.loads(Path('context_request.json').read_text())\n"
                "result={'schema_version':'react_context_summary_v1','source_context_sha256':request['source_context_sha256'],'events':[],'unresolved_state':[]}\n"
                "Path('context_summary.json').write_text(json.dumps(result))\n"
                "session=Path(os.environ['DSH_HOME'])/'sessions/project/s/session.jsonl'\n"
                "session.parent.mkdir(parents=True)\n"
                "rows=[{'type':'session','version':0},{'type':'assistant/message','data':{'message':{'id':'a','content':[{'type':'text','text':'done'}]}}},{'type':'turn/end','data':{'reason':{'kind':'completed'}}}]\n"
                "session.write_text(''.join(json.dumps(row)+'\\n' for row in rows))\n"
                "print('done')\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            provider = ClaudeContextSummarizer(
                cache_root=root / "cache", harness="deepseek", dsh_bin=str(fake),
                dsh_node_bin=str(node), timeout_sec=5,
            )
            with patch.dict(os.environ, {
                "DEEPSEEK_BASE_URL": "https://example.invalid/v1",
                "DEEPSEEK_API_KEY": "secret",
            }):
                first, audit = provider.summarize(self.messages)
            self.assertEqual(first["schema_version"], "react_context_summary_v1")
            self.assertEqual(audit["harness"], "deepseek")
            self.assertTrue(next((root / "cache").glob("entries/*/attempt-1/.agents/skills"), None))


if __name__ == "__main__":
    unittest.main()
