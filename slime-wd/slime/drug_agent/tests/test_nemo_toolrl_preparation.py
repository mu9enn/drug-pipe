import json

from drug_agent.scripts.prepare_nemo_toolrl import encoding_text, prompt_contains_catalog
from drug_agent.toolrl.v8_dataset import stable_json
from drug_agent.scripts.materialize_toolrl_v8_length_tiers import materialize
from drug_agent.toolrl.selector_window import window_document


def test_nemo_comparison_separates_explicit_modes_and_plaintext_failures():
    from drug_agent.scripts.run_nemo_toolrl import comparison_scope
    doc=window_fixture()
    doc["history_before_current_decision"][-1]["content"]="The operation timed out."
    doc["current_teacher_response"]["tool_calls"]=[{"function":{"name":"compute","arguments":{}}}]
    row={"scope":json.dumps({"prior_outcome":"unknown"}),"text":json.dumps(doc)}
    real=comparison_scope(row)
    assert json.loads(real)["prior_outcome"]=="error"
    doc["current_teacher_response"]["tool_calls"][0]["function"]["arguments"]["dry_run"]=True
    row["text"]=json.dumps(doc)
    assert comparison_scope(row)!=real
    doc["history_before_current_decision"][-1]["content"]="Documentation: The operation timed out. is an example."
    row["text"]=json.dumps(doc)
    assert json.loads(comparison_scope(row))["prior_outcome"]=="unknown"


def window_fixture():
    history = [{"role":"system","content":"Do not overwrite files"}, {"role":"user","content":"repair protein"}]
    for i in range(4):
        history.extend([{"role":"assistant","reasoning_content":"reason"*20,
                         "tool_calls":[{"id":str(i),"function":{"name":"read","arguments":{"path":str(i)}}}]},
                        {"role":"tool","name":"read","tool_call_id":str(i),"content":"observation"*30}])
    return {"history_before_current_decision":history,"current_call_tool_definitions":[{"name":"write"}],
            "current_teacher_response":{"role":"assistant","reasoning_content":"current","content":"complete target"}}


def test_recent_window_keeps_task_current_and_whole_contiguous_interactions():
    doc=window_fixture()
    original=stable_json(doc)
    text,n,audit=window_document(doc,len,limit=1600)
    result=json.loads(text)
    assert n<=1600 and audit["lossy"] and not audit["protected_reason"]
    history=result["history_before_current_decision"]
    assert history[:2]==doc["history_before_current_decision"][:2]
    assert history[-2:]==doc["history_before_current_decision"][-2:]
    assert result["current_teacher_response"]==doc["current_teacher_response"]
    assert stable_json(doc)==original
    assert len(history[2:])%2==0


def test_large_current_or_latest_interaction_is_protected_not_sliced():
    doc=window_fixture()
    text,n,audit=window_document(doc,len,limit=400)
    assert audit["protected_reason"]=="latest_complete_interaction_does_not_fit"
    doc["current_teacher_response"]["content"]="x"*2000
    text,n,audit=window_document(doc,len,limit=1600)
    assert audit["protected_reason"]=="fixed_task_current_definitions_over_limit"
    assert json.loads(text)["current_teacher_response"]["content"]=="x"*2000


def test_only_explicit_historical_payloads_are_limited():
    doc=window_fixture()
    doc["history_before_current_decision"][-1]["content"]=json.dumps({"base64":"A"*10000,"error":"bad format","path":"/server/file","code":"B"*9000})
    doc["current_teacher_response"]["content"]="C"*9000
    text,n,audit=window_document(doc,len,limit=50000)
    result=json.loads(text)
    payload=json.loads(result["history_before_current_decision"][-1]["content"])
    assert "omitted" in payload["base64"]
    assert payload["error"]=="bad format" and payload["code"]=="B"*9000
    assert result["current_teacher_response"]["content"]=="C"*9000
    assert audit["payload_edits"][0]["reason"]=="explicit_base64"


def test_short_history_and_loaded_skill_are_preserved():
    doc=window_fixture()
    doc["history_before_current_decision"][-1]["name"]="skill"
    doc["history_before_current_decision"][-1]["content"]=json.dumps({"file_content":"skill instructions"*800})
    text,n,audit=window_document(doc,len,limit=50000)
    assert json.loads(text)==doc
    assert not audit["lossy"]


def test_later_user_constraint_survives_window_and_incomplete_pair_is_rejected():
    import pytest
    doc=window_fixture()
    doc["history_before_current_decision"].insert(4,{"role":"user","content":"New constraint: do not overwrite"})
    text,n,audit=window_document(doc,len,limit=1600)
    assert "New constraint: do not overwrite" in text
    doc["history_before_current_decision"][-1]["tool_call_id"]="incorrect"
    with pytest.raises(ValueError,match="Incomplete"):
        window_document(doc,len,limit=1600)


def test_full_document_preserves_history_parameters_and_target_boundaries():
    large = "before" + "x" * 100000 + "middle parameter" + "y" * 100000 + "after"
    calls = [{"function": {"name": name, "arguments": {"content": large}}} for name in ("write", "read")]
    row = {"id": "not-content", "prompt": [{"role": "user", "content": large}],
           "tools": [{"type": "function", "function": {"name": name, "description": large}} for name in ("write", "read", "unused")],
           "label": {"target_assistant": {"role": "assistant", "reasoning_content": large, "tool_calls": calls}},
           "metadata": {"selection_description": "do not embed this", "future": "excluded"}}
    doc = json.loads(encoding_text(row))
    assert doc["history_before_current_decision"] == row["prompt"]
    assert doc["current_call_tool_definitions"] == row["tools"][:2]
    assert doc["current_teacher_response"] == row["label"]["target_assistant"]
    assert set(doc) == {"history_before_current_decision", "current_call_tool_definitions", "current_teacher_response"}


def test_catalog_check_compares_definitions_not_names():
    tools = [{"type": "function", "function": {"name": "read", "description": "read a file"}}]
    serialized = stable_json(tools)
    row = {"prompt": [{"role": "system", "content": "Use read when necessary."}]}
    assert not prompt_contains_catalog(row, serialized)
    row["prompt"][0]["content"] = "<tools>\n" + json.dumps(tools[0], indent=2) + "\n</tools>"
    assert prompt_contains_catalog(row, serialized)
    different = [{"type": "function", "function": {"name": "read", "description": "different semantics"}}]
    assert not prompt_contains_catalog(row, stable_json(different))


def test_system_constraints_and_loaded_skills_are_not_factored_out():
    history = [{"role": "system", "content": "Task-specific constraint: do not modify the receptor."},
               {"role": "tool", "content": "<skill_content>Complete instructions</skill_content>"}]
    row = {"prompt": history, "tools": [],
           "label": {"target_assistant": {"role": "assistant", "content": "final"}}}
    doc = json.loads(encoding_text(row))
    assert doc["history_before_current_decision"] == history
    assert doc["current_call_tool_definitions"] == []


def test_length_partition_can_omit_pretty_without_changing_rows(tmp_path):
    rows = [{"id": str(i), "prompt": [{"role": "user", "content": "original"}], "tools": [], "label": {"x": i}} for i in range(3)]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    details = tmp_path / "details.jsonl"
    fields = ["reasoning_tokens", "structured_action_or_final_tokens", "native_action_or_final_tokens", "prompt_plus_reference_tokens"]
    details.write_text("".join(json.dumps({"decision_id": str(i), "prompt_tokens": 10,
                                          "full_reference_tokens": n, **dict.fromkeys(fields, 1)}) + "\n"
                               for i, n in enumerate([100, 40000, 70000])))
    root = tmp_path / "output"
    result = materialize(source, details, root, emit_pretty=False)
    for i, tier in enumerate(("standard_32k", "long_64k_candidate", "oversize_review")):
        assert json.loads((root / f"{tier}.jsonl").read_text()) == rows[i]
        assert result["outputs"][tier]["records"] == 1
        assert "pretty_json" not in result["outputs"][tier]
    assert not list(root.glob("*.pretty.json"))
