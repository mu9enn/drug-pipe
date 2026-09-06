from __future__ import annotations

import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from drug_agent.toolrl.qwen_native_parser import parse_qwen_native_completion, resolve_parser_config


ROOT = Path(__file__).resolve().parents[1]


class StructuredQwenPipelineTest(unittest.TestCase):
    def test_parser_names_are_required_serving_configuration(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                resolve_parser_config()
            config = resolve_parser_config(reasoning_parser="installed_reasoning", tool_parser="installed_tools")
        self.assertEqual(config.reasoning_parser, "installed_reasoning")
        self.assertEqual(config.tool_parser, "installed_tools")

    def test_launchers_pass_deployment_tools_to_slime_loader(self) -> None:
        sft = (ROOT / "scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh").read_text(encoding="utf-8")
        sft_full = (ROOT / "scripts/run_qwen3_5_4b_drug_sft_full.sh").read_text(encoding="utf-8")
        toolrl = (ROOT / "toolrl/scripts/run_toolrl_grpo.sh").read_text(encoding="utf-8")
        toolrl_full = (ROOT / "toolrl/scripts/run_qwen3_5_4b_toolrl_full.sh").read_text(encoding="utf-8")
        self.assertIn("--input-key messages", sft)
        self.assertIn("--tool-key tools", sft)
        self.assertIn("--loss-mask-type qwen3_5", sft)
        self.assertNotIn("materialize_sft_jsonl", sft)
        self.assertIn("*.jsonl|*.parquet", sft)
        self.assertIn("--input-key prompt", toolrl)
        self.assertIn("--tool-key tools", toolrl)
        self.assertIn("TrajectoryBatchDataSource", toolrl)
        self.assertIn("validate_trajectory_toolrl_batches.py", toolrl)
        self.assertIn("GLOBAL_BATCH_SIZE", toolrl)
        self.assertIn("RBS*n", toolrl)
        self.assertIn("--sglang-reasoning-parser", toolrl)
        self.assertIn("--sglang-tool-call-parser", toolrl)
        self.assertNotIn("toolrl_official_8cee13e", toolrl)
        self.assertIn("qwen35_sft.jsonl", sft_full)
        self.assertIn("qwen35_toolrl.jsonl", toolrl_full)
        self.assertIn("DATASET_SIZE % ROLLOUT_BATCH_SIZE", toolrl_full)

    def test_native_parser_preserves_parallel_calls(self) -> None:
        protocol = ModuleType("sglang.srt.entrypoints.openai.protocol")
        protocol.Function = lambda **kwargs: SimpleNamespace(**kwargs)
        protocol.Tool = lambda **kwargs: SimpleNamespace(**kwargs)

        function_parser = ModuleType("sglang.srt.function_call.function_call_parser")

        class FakeFunctionCallParser:
            def __init__(self, *, tools, tool_call_parser):
                self.tools = tools
                self.tool_call_parser = tool_call_parser

            def has_tool_call(self, body):
                return body == "native calls"

            def parse_non_stream(self, _body):
                return "", [
                    SimpleNamespace(name="tool_a", parameters='{"x":1}'),
                    SimpleNamespace(name="tool_b", parameters='{"y":2}'),
                ]

        function_parser.FunctionCallParser = FakeFunctionCallParser
        reasoning_parser = ModuleType("sglang.srt.parser.reasoning_parser")

        class FakeReasoningParser:
            def __init__(self, *, model_type, stream_reasoning):
                self.model_type = model_type
                self.stream_reasoning = stream_reasoning

            def parse_non_stream(self, _text):
                return "reason", "native calls"

        reasoning_parser.ReasoningParser = FakeReasoningParser
        modules = {
            "sglang": ModuleType("sglang"),
            "sglang.srt": ModuleType("sglang.srt"),
            "sglang.srt.entrypoints": ModuleType("sglang.srt.entrypoints"),
            "sglang.srt.entrypoints.openai": ModuleType("sglang.srt.entrypoints.openai"),
            "sglang.srt.entrypoints.openai.protocol": protocol,
            "sglang.srt.function_call": ModuleType("sglang.srt.function_call"),
            "sglang.srt.function_call.function_call_parser": function_parser,
            "sglang.srt.parser": ModuleType("sglang.srt.parser"),
            "sglang.srt.parser.reasoning_parser": reasoning_parser,
        }
        tools = [
            {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
            for name in ("tool_a", "tool_b")
        ]
        with patch.dict(sys.modules, modules):
            parsed = parse_qwen_native_completion(
                "rendered native output",
                tools_schema=tools,
                reasoning_parser="installed_reasoning",
                tool_parser="installed_tools",
            )
        self.assertTrue(parsed["ok"])
        self.assertEqual([call["name"] for call in parsed["tool_calls"]], ["tool_a", "tool_b"])
        self.assertEqual(parsed["tool_calls"][1]["arguments"], {"y": 2})


if __name__ == "__main__":
    unittest.main()
