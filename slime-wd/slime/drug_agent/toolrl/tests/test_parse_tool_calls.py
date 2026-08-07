from __future__ import annotations

from drug_agent.toolrl.parse_tool_calls import parse_tool_calls


def test_parse_tool_calls_single_and_multi_call():
    text = (
        "<thought>first</thought>"
        '<tool_call>{"tool_name":"mcp__molclaw-scp__fix_pdb","arguments":{"input_path":"/tmp/a.pdb","remove_water":true}}</tool_call>'
        '<tool_call>{"tool_name":"is_valid_smiles","arguments":{"smiles_list":["CCO"]}}</tool_call>'
    )
    parsed = parse_tool_calls(text, keep_non_molclaw=True)
    assert parsed["ok"] is True
    assert parsed["tool_call_count"] == 2
    assert parsed["molclaw_tool_call_count"] == 2
    assert parsed["tool_calls"][0]["tool_name"] == "fix_pdb"
    assert parsed["tool_calls"][0]["tool_name_raw"] == "mcp__molclaw-scp__fix_pdb"
    assert parsed["tool_calls"][1]["arguments"]["smiles_list"] == ["CCO"]


def test_parse_tool_calls_filters_non_molclaw():
    text = (
        "<thought>first</thought>"
        '<tool_call>{"tool_name":"Write","arguments":{"path":"x.txt"}}</tool_call>'
        '<tool_call>{"tool_name":"fix_pdb","arguments":{"input_path":"/tmp/a.pdb"}}</tool_call>'
    )
    parsed = parse_tool_calls(text)
    assert parsed["ok"] is True
    assert parsed["tool_call_count"] == 1
    assert parsed["molclaw_tool_call_count"] == 1
    assert parsed["non_molclaw_tool_call_count"] == 1
    assert parsed["local_tool_call_count"] == 1
    assert parsed["unsupported_tool_call_count"] == 0
    assert parsed["tool_calls"][0]["tool_name"] == "fix_pdb"


def test_parse_tool_calls_preserves_runtime_local_tool_spelling():
    text = '<thought>read</thought><tool_call>{"tool_name":"Read","arguments":{"file_path":"skills/L1_tools/x/SKILL.md"}}</tool_call>'
    parsed = parse_tool_calls(text, keep_non_molclaw=True)
    assert parsed["ok"] is True
    assert parsed["tool_calls"][0]["tool_name"] == "Read"
    assert parsed["local_tool_call_count"] == 1
    assert parsed["supported_tool_call_count"] == 1
    assert parsed["unsupported_tool_call_count"] == 0


def test_parse_tool_calls_local_classification_is_case_insensitive():
    text = '<thought>read</thought><tool_call>{"tool_name":"read","arguments":{"file_path":"run_log.md"}}</tool_call>'
    parsed = parse_tool_calls(text, keep_non_molclaw=True)
    assert parsed["molclaw_tool_call_count"] == 0
    assert parsed["local_tool_call_count"] == 1
    assert parsed["supported_tool_call_count"] == 1
    assert parsed["tool_calls"][0]["tool_name"] == "Read"


def test_parse_tool_calls_rejects_malformed_json():
    text = '<tool_call>{"tool_name":"fix_pdb","arguments":{}}</tool_call><tool_call>{"tool_name":}</tool_call>'
    parsed = parse_tool_calls(text)
    assert parsed["ok"] is False
    assert parsed["error_type"] in {"ReactJSONDecodeError", "ReactFormatError"}
