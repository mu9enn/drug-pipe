from drug_agent.data.tool_call_token_stats import (
    _tool_calls_from_assistant,
    summarize_tokens,
)


def test_extracts_each_tool_call_json_without_xml_tags() -> None:
    content = (
        '<thought>Check and render.</thought>'
        '<tool_call>{"arguments":{"smiles":"CCO"},"tool_name":"is_valid_smiles"}</tool_call>'
        '<tool_call>{"arguments":{"smiles":"CCO"},"tool_name":"visualize_molecule"}</tool_call>'
    )
    rows = _tool_calls_from_assistant(
        content,
        dataset="sft",
        record_id="sample",
        decision_index=2,
    )

    assert [row.tool_name for row in rows] == ["is_valid_smiles", "visualize_molecule"]
    assert all("<tool_call>" not in row.json_text for row in rows)
    assert rows[0].json_text == '{"arguments":{"smiles":"CCO"},"tool_name":"is_valid_smiles"}'


def test_token_summary_uses_nearest_rank_percentiles() -> None:
    summary = summarize_tokens([1, 2, 3, 4, 100])

    assert summary == {
        "count": 5,
        "total_tokens": 110,
        "mean": 22.0,
        "min": 1,
        "p50": 3,
        "p90": 100,
        "p95": 100,
        "p99": 100,
        "max": 100,
        "gt_4096": 0,
        "gt_8192": 0,
        "gt_16384": 0,
    }
