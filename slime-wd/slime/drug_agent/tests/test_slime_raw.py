from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from drug_agent.protocol.slime_raw import PROFILE_NAME, parse_decision, render_prompt, task_messages
from drug_agent.tools.slime_raw_local_tools import SlimeRawLocalToolExecutor, workspace_only_tool_specs


def test_profile_has_stable_short_name():
    assert PROFILE_NAME == "slime-raw"


def test_launcher_selects_the_dedicated_runtime_and_manifest_profile():
    drug_agent_root = Path(__file__).resolve().parents[1]
    wrapper = (drug_agent_root / "scripts/run_slime_raw_eval.sh").read_text(encoding="utf-8")
    launcher = (drug_agent_root / "scripts/run_molbench_eval.sh").read_text(encoding="utf-8")

    assert "DRUG_AGENT_EVAL_PROFILE=slime-raw" in wrapper
    assert "drug_agent.rollout.generate_with_slime_raw.generate" in launcher
    assert '--eval-profile "$DRUG_AGENT_EVAL_PROFILE"' in launcher


def test_prompt_contains_only_neutral_protocol_catalog_and_contract():
    messages = task_messages(
        [
            {"role": "system", "content": "model-specific old system prompt"},
            {"role": "user", "content": "benchmark question"},
            {"role": "assistant", "content": "teacher history"},
        ],
        tool_catalog="Available tools: test_tool",
        final_contract="Required terminal payload: test_schema",
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[1]["content"] == "benchmark question"
    system = messages[0]["content"]
    assert "Available tools: test_tool" in system
    assert "Required terminal payload: test_schema" in system
    assert "model-specific old system prompt" not in system
    assert "teacher history" not in system
    for forbidden in ("/no_think", "Qwen", "<|im_end|>", "SKILL.md", "skills/L1_tools"):
        assert forbidden not in system


def test_prompt_uses_native_chat_template_without_model_specific_switches():
    class Tokenizer:
        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return "rendered"

    state = type("State", (), {"tokenizer": Tokenizer()})()
    assert render_prompt(
        state,
        "question",
        tool_catalog="tools",
        final_contract="final",
    ) == "rendered"
    assert state.tokenizer.kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
    }


def test_local_tool_catalog_and_executor_do_not_expose_skills():
    specs = workspace_only_tool_specs(
        [
            {
                "name": "Read",
                "description": "Read workspace or skill catalog.",
                "executor": "local_sandbox",
                "input_schema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string", "description": "skills/L1_tools/..."}},
                },
            }
        ]
    )
    assert "skill" not in json.dumps(specs).lower()

    with tempfile.TemporaryDirectory() as tmp:
        executor = SlimeRawLocalToolExecutor(Path(tmp) / "workspace", Path(tmp) / "unused")
        result = executor.execute("Read", {"file_path": "skills/L1_tools/example/SKILL.md"})
    assert result["ok"] is False
    assert "only task-workspace" in result["error"]["message"]


def test_strict_parser_accepts_only_exact_canonical_decisions():
    parsed = parse_decision(
        '<thought>inspect</thought><tool_call>{"tool_name":"test_tool","arguments":{"x":1}}</tool_call>'
    )
    assert parsed["ok"] is True
    assert parsed["decision_type"] == "tool_call"
    assert parsed["tool_calls"][0]["arguments"] == {"x": 1}

    final = parse_decision(
        '<final_answer>{"task_type":"e2e","result":"done","evidence":[]}</final_answer>'
    )
    assert final["ok"] is True
    assert final["decision_type"] == "final_answer"


@pytest.mark.parametrize(
    "output",
    [
        '<think></think><tool_call>{"tool_name":"test_tool","arguments":{}}</tool_call>',
        '<tool_call>{"tool_name":"test_tool","arguments":{}}</tool_call><|im_end|>',
        '<tool_call>```json\n{"tool_name":"test_tool","arguments":{}}\n```</tool_call>',
        '</tool_call>',
        '<tool_call>{"tool_name":"a","arguments":{}}, {"tool_name":"b","arguments":{}}</tool_call>',
        '<tool_call>{"tool_name":"test_tool","arguments":{}}</tool_call>trailing text',
    ],
)
def test_strict_parser_rejects_compatibility_variants(output):
    assert parse_decision(output)["ok"] is False
