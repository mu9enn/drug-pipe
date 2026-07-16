from __future__ import annotations

import json
from pathlib import Path

from drug_agent.toolrl.convert_react_to_toolrl_steps import convert_react_to_toolrl_steps


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_converter_builds_step_level_samples(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    record = {
        "schema_version": "drug_agent_sft_react_json_v1",
        "id": "sample-1",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user prompt"},
            {
                "role": "assistant",
                "content": '<thought>t1</thought><tool_call>{"tool_name":"fix_pdb","arguments":{"input_path":"/tmp/a.pdb","remove_water":true}}</tool_call>',
            },
            {
                "role": "user",
                "content": '<observation tool_name="fix_pdb">{"ok":true,"status":"success","content":{"output_file":"x.pdb"}}</observation>',
            },
            {
                "role": "assistant",
                "content": '<thought>t2</thought><tool_call>{"tool_name":"is_valid_smiles","arguments":{"smiles_list":["CCO","CCN"]}}</tool_call>',
            },
            {"role": "user", "content": '<observation tool_name="is_valid_smiles">{"ok":true}</observation>'},
            {"role": "assistant", "content": '<final_answer>{"answer":{"summary":"done","evidence":[],"result":{}}}</final_answer>'},
        ],
    }
    _write_json(input_dir / "sample.json", record)

    output_path = tmp_path / "toolrl.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    report = convert_react_to_toolrl_steps(input_dir, output_path, skipped_report_path=skipped_path)
    assert report["ok"] is True
    assert report["kept_rows"] == 2
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert rows[0]["metadata"]["assistant_index"] == 2
    assert rows[0]["prompt"][-1]["role"] == "user"
    assert "final_answer" not in rows[0]["prompt"][-1]["content"]
    assert rows[1]["metadata"]["assistant_index"] == 4
    assert rows[1]["prompt"][-1]["role"] == "user"
    assert rows[1]["target_tool_calls"][0]["tool_name"] == "is_valid_smiles"


def test_converter_skips_final_answer_only_and_malformed(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    record = {
        "schema_version": "drug_agent_sft_react_json_v1",
        "id": "sample-2",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": '<final_answer>{"answer":{"summary":"done","evidence":[],"result":{}}}</final_answer>'},
            {"role": "assistant", "content": '<tool_call>{"tool_name":"fix_pdb","arguments":{"input_path":"/tmp/a.pdb"}}</tool_call>'},
        ],
    }
    _write_json(input_dir / "sample.json", record)

    output_path = tmp_path / "toolrl.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    report = convert_react_to_toolrl_steps(input_dir, output_path, skipped_report_path=skipped_path)
    assert report["ok"] is True
    assert report["kept_rows"] == 1
    skipped = [json.loads(line) for line in skipped_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(item["skip_reason"] == "no_molclaw_tool_calls" for item in skipped)
