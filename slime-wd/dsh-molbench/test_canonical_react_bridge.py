from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import canonical_react_bridge as bridge


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs.get("enable_thinking") is False
        return "CHAT:" + json.dumps(messages, ensure_ascii=False) + "\nASSISTANT:"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text.encode("utf-8"))}


def tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "mcp__molclaw-scp__calculate_mol_basic_info",
                "description": "calculate",
                "parameters": {
                    "type": "object",
                    "required": ["smiles_list"],
                    "properties": {"smiles_list": {"type": "array"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "read file",
                "parameters": {"type": "object"},
            },
        },
        {"type": "function", "function": {"name": "todo_write", "parameters": {}}},
    ]


def request(messages=None, stream=False):
    marker = bridge.TASK_MARKER + json.dumps({
        "task_id": "molbench_ms1_001",
        "task_type": "pf",
        "question": "Which molecules pass?",
    })
    return {
        "model": "qwen3.5-9b-canonical-bridge",
        "messages": messages or [{"role": "user", "content": marker}],
        "tools": tools(),
        "stream": stream,
    }


class FakeBridge(bridge.CanonicalBridge):
    def __init__(self, outputs):
        self._outputs = iter(outputs)
        tmp = Path(tempfile.mkdtemp()) / "bridge.jsonl"
        super().__init__(bridge.BridgeConfig(
            upstream_url="http://unused/generate",
            model_id="qwen3.5-9b-canonical-bridge",
            max_new_tokens=16384,
            max_prompt_tokens=100000,
            max_steps=128,
            repeat_limit=3,
            log_path=tmp,
        ), FakeTokenizer())

    def call_upstream(self, input_ids):
        return next(self._outputs)


class BridgeTests(unittest.TestCase):
    def test_name_and_catalog_mapping(self):
        specs, name_map = bridge.tool_catalog(tools())
        self.assertEqual([item["name"] for item in specs], ["calculate_mol_basic_info", "Read"])
        self.assertEqual(
            bridge.canonical_to_dsh_name("calculate_mol_basic_info", name_map),
            "mcp__molclaw-scp__calculate_mol_basic_info",
        )
        self.assertEqual(bridge.canonical_to_dsh_name("Read", name_map), "read")

    def test_prompt_replaces_dsh_protocol(self):
        task = bridge.extract_task(request()["messages"])
        specs, _ = bridge.tool_catalog(tools())
        text = bridge.build_prompt_text(FakeTokenizer(), task, specs, "")
        self.assertIn("canonical ReAct XML", text)
        self.assertIn("calculate_mol_basic_info", text)
        self.assertNotIn("mcp__molclaw-scp__", text)
        self.assertIn("Which molecules pass?", text)

    def test_multi_call_becomes_structured_dsh_calls(self):
        raw = (
            "<thought>calculate first</thought>"
            "<tool_call>{\"tool_name\":\"calculate_mol_basic_info\","
            "\"arguments\":{\"smiles_list\":[\"CCO\"]}}"
            "{\"tool_name\":\"Read\","
            "\"arguments\":{\"file_path\":\"result.md\"}}</tool_call>"
        )
        result = FakeBridge([(raw, "stop")]).complete(request())
        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(len(choice["message"]["tool_calls"]), 2)
        self.assertEqual(
            choice["message"]["tool_calls"][0]["function"]["name"],
            "mcp__molclaw-scp__calculate_mol_basic_info",
        )
        self.assertEqual(choice["message"]["tool_calls"][1]["function"]["name"], "read")
        self.assertEqual(choice["message"]["content"], "<thought>calculate first</thought>")

    def test_tool_result_reconstructs_observation(self):
        marker = request()["messages"][0]
        messages = [
            marker,
            {
                "role": "assistant",
                "content": "<thought>calculate</thought>",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "mcp__molclaw-scp__calculate_mol_basic_info",
                        "arguments": "{\"smiles_list\":[\"CCO\"]}",
                    },
                }],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "{\"mw\":46}"},
        ]
        trace = bridge.reconstruct_trace(messages)
        self.assertIn("<tool_call>", trace)
        self.assertIn('<observation tool_name="calculate_mol_basic_info">', trace)
        self.assertIn('"mw":46', trace)

    def test_malformed_output_finishes_without_empty_tool_call(self):
        result = FakeBridge([("garbage </parameter>", "stop")]).complete(request())
        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(choice["message"]["content"], "garbage </parameter>")
        self.assertNotIn("tool_calls", choice["message"])

    def test_wrong_final_type_gets_internal_feedback(self):
        wrong = '<thought>x</thought><final_answer>{"task_type":"ac","answer_smiles":"CCO","evidence":[]}</final_answer>'
        right = '<thought>fixed</thought><final_answer>{"task_type":"pf","selected_smiles":["CCO"],"evidence":[]}</final_answer>'
        result = FakeBridge([(wrong, "stop"), (right, "stop")]).complete(request())
        content = result["choices"][0]["message"]["content"]
        self.assertIn("FinalTaskTypeMismatch", content)
        self.assertTrue(content.endswith(right))

    def test_length_is_terminal(self):
        result = FakeBridge([("<thought>unfinished", "length")]).complete(request())
        self.assertEqual(result["choices"][0]["finish_reason"], "length")

    def test_repetition_guard_stops_third_identical_decision(self):
        raw = '<thought>x</thought><tool_call>{"tool_name":"Read","arguments":{}}</tool_call>'
        instance = FakeBridge([(raw, "stop"), (raw, "stop"), (raw, "stop")])
        marker = request()["messages"][0]
        first_messages = [marker]
        second_messages = first_messages + [
            {
                "role": "assistant",
                "content": "<thought>x</thought>",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "first"},
        ]
        third_messages = second_messages + [
            {
                "role": "assistant",
                "content": "<thought>x</thought>",
                "tool_calls": [{
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call-2", "content": "second"},
        ]
        self.assertEqual(
            instance.complete(request(first_messages))["choices"][0]["finish_reason"],
            "tool_calls",
        )
        self.assertEqual(
            instance.complete(request(second_messages))["choices"][0]["finish_reason"],
            "tool_calls",
        )
        third = instance.complete(request(third_messages))["choices"][0]
        self.assertEqual(third["finish_reason"], "stop")
        self.assertNotIn("tool_calls", third["message"])

    def test_fresh_conversation_resets_repetition_state_for_same_task_id(self):
        raw = '<thought>x</thought><tool_call>{"tool_name":"Read","arguments":{}}</tool_call>'
        instance = FakeBridge([(raw, "stop"), (raw, "stop")])
        first = instance.complete(request())["choices"][0]
        fresh = instance.complete(request())["choices"][0]
        self.assertEqual(first["finish_reason"], "tool_calls")
        self.assertEqual(fresh["finish_reason"], "tool_calls")

    def test_native_thinking_leak_is_not_silently_recovered(self):
        raw = (
            "native reasoning</think>"
            '<thought>x</thought><tool_call>{"tool_name":"Read","arguments":{}}</tool_call>'
        )
        choice = FakeBridge([(raw, "stop")]).complete(request())["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertNotIn("tool_calls", choice["message"])
        self.assertEqual(choice["message"]["content"], raw)

    def test_interactive_user_question_tool_is_forbidden_for_benchmark(self):
        body = request()
        body["tools"] = body["tools"] + [{
            "type": "function",
            "function": {
                "name": "ask_user_question",
                "description": "pause for human input",
                "parameters": {"type": "object"},
            },
        }]
        with self.assertRaisesRegex(RuntimeError, "forbidden interactive tools"):
            FakeBridge([]).complete(body)


if __name__ == "__main__":
    unittest.main()
