#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import post_process_sft as sft  # noqa: E402
import scan_molclaw_usage as scan  # noqa: E402


class SanitizerSmokeTest(unittest.TestCase):
    def test_only_local_absolute_roots_are_replaced(self) -> None:
        cases = {
            "/root/example/DrugAgentTools/protein_structures/P35968_2XIR.pdb": "<artifact:protein_structures/P35968_2XIR.pdb>",
            "/home/example/a/b.pdb": "<artifact:protein_structures/b.pdb>",
            "/tmp/x/y.txt": "<artifact:local/y.txt>",
            "/tmp/": "<artifact:local/tmp>",
            "/mnt/data/a.pdb": "<artifact:protein_structures/a.pdb>",
            "/workspace/a/b/c.pdb": "<artifact:protein_structures/c.pdb>",
            "</answer>": "</answer>",
            "</tool_call>": "</tool_call>",
            "</final_answer>": "</final_answer>",
            "skills/tool": "skills/tool",
            "kcal/mol": "kcal/mol",
            "C/C": "C/C",
            "/C": "/C",
            "/c3cc": "/c3cc",
            "http://example.com/a/b": "http://example.com/a/b",
            "https://errors.pydantic.dev/2.0/v/missing_argument": "https://errors.pydantic.dev/2.0/v/missing_argument",
            "[artifact:pdbfixer/a.pdb](artifact:pdbfixer/a.pdb)": "<artifact:pdbfixer/a.pdb>",
            "<artifact:pdbfixer/a.pdb>": "<artifact:pdbfixer/a.pdb>",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(sft._sanitize_text_with_artifacts(raw, {}), expected)


class FpocketSmokeTest(unittest.TestCase):
    def _compress(self, payload: dict) -> dict:
        return sft._compress_fpocket_observation_payload(
            payload,
            sample_id="test",
            actions=[],
            raw_event_index=0,
            tool_use_id="u1",
            path_cache={},
        )

    def test_size_rules_and_artifact(self) -> None:
        no_size = self._compress({"status": "success", "pockets": [{"center": [1, 2, 3], "score": 0.2}]})
        self.assertNotIn("size", no_size["content"]["top_pocket"])
        self.assertEqual(no_size["content"]["artifact"], "<artifact:fpocket/result>")

        real_size = self._compress({"status": "success", "pockets": [{"center": [1, 2, 3], "size": [4, 5, 6]}]})
        self.assertEqual(real_size["content"]["top_pocket"]["size"], [4, 5, 6])

        equal_size = self._compress({"status": "success", "pockets": [{"center": [1, 2, 3], "size": [1, 2, 3]}]})
        self.assertNotIn("size", equal_size["content"]["top_pocket"])

        negative_size = self._compress({"status": "success", "pockets": [{"center": [1, 2, 3], "size": [4, -5, 6]}]})
        self.assertNotIn("size", negative_size["content"]["top_pocket"])

        noisy = self._compress(
            {"status": "success", "pockets": [{"center": [1, 2, 3]}], "files": [f"/tmp/x/{i}.pdb" for i in range(100)]}
        )
        self.assertNotIn("files", noisy["content"])


class ToolAndMetricSmokeTest(unittest.TestCase):
    def test_both_molclaw_namespaces_are_retained(self) -> None:
        self.assertEqual(sft._normalize_tool_name("mcp__molclaw-scp__alpha"), ("alpha", "molclaw-scp"))
        self.assertEqual(sft._normalize_tool_name("mcp__molclaw-vs__beta"), ("beta", "molclaw-vs"))
        self.assertEqual(sft._normalize_tool_name("Read"), (None, None))

    def test_missing_metrics_are_rejected(self) -> None:
        self.assertEqual(scan._missing_metric_reason("vs", scan._metrics_from_record("vs", {})), "missing_vs_top3_hit_num")
        self.assertEqual(scan._missing_metric_reason("ac", scan._metrics_from_record("ac", {})), "missing_ac_is_correct")
        self.assertEqual(scan._missing_metric_reason("pf", scan._metrics_from_record("pf", {})), "missing_pf_precision")
        complete = scan._metrics_from_record(
            "pf",
            {"task_metrics": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "acc": 1.0}},
        )
        self.assertIsNone(scan._missing_metric_reason("pf", complete))


if __name__ == "__main__":
    unittest.main()
