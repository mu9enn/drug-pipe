import json


def test_threshold_calibration_matches_official_rule_and_preserves_ties():
    from drug_agent.scripts.sweep_nemo_thresholds import calibrate
    scores = [0.50 + i/200 for i in range(80)]
    result = calibrate(scores,100)
    assert 20 <= result["selected"]["retained"] <= 30
    for trial in result["boundary_trials"]:
        assert trial["retained"] == 100-sum(s >= 1-trial["eps"] for s in scores)
    assert result == calibrate(scores,100)
    # With 40 protected/tiny records the target cannot be reached.
    assert calibrate(scores[:60],100)["selected"] is None
    # Equal scores cannot be split to manufacture a target count.
    assert calibrate([0.9]*90,100)["selected"] is None

from drug_agent.scripts.prepare_nemo_toolrl import encoding_text, prompt_contains_catalog
from drug_agent.toolrl.v8_dataset import stable_json
from drug_agent.scripts.materialize_toolrl_v8_length_tiers import materialize
from drug_agent.toolrl.selector_window import window_document


def test_nemo_comparison_separates_explicit_modes_and_plaintext_failures():
    from drug_agent.toolrl.nemo_scope import comparison_scope
    doc=window_fixture()
    doc["history_before_current_decision"][-1]["content"]="The operation timed out."
    doc["current_teacher_response"]["tool_calls"]=[{"function":{"name":"compute","arguments":{}}}]
    row={"prompt":doc["history_before_current_decision"], "label":{"target_assistant":doc["current_teacher_response"]}, "metadata":{"task_type":"AC"}}
    real=comparison_scope(row)
    assert json.loads(real)["prior_outcome"]=="error"
    doc["current_teacher_response"]["tool_calls"][0]["function"]["arguments"]["dry_run"]=True
    assert comparison_scope(row)!=real
    doc["history_before_current_decision"][-1]["content"]="Documentation: The operation timed out. is an example."
    assert json.loads(comparison_scope(row))["prior_outcome"]=="unknown"


def test_native_scope_cross_task_and_position_specific_skills():
    from copy import deepcopy
    from drug_agent.toolrl.nemo_scope import comparison_scope
    row = {"prompt":[], "label":{"target_assistant":{"tool_calls":[
        {"function":{"name":"skill", "arguments":{"name":"molclaw-a"}}},
        {"function":{"name":"skill", "arguments":{"name":"molclaw-b"}}}]}},
        "metadata":{"task_type":"AC", "comparison_scope":"invalid old metadata"}}
    other = deepcopy(row)
    other["metadata"]["task_type"] = "KG"
    assert comparison_scope(row) == comparison_scope(other)
    other["label"]["target_assistant"]["tool_calls"].reverse()
    assert comparison_scope(row) != comparison_scope(other)
    row["label"]["target_assistant"] = {"content":"final"}
    other["label"]["target_assistant"] = {"content":"final"}
    assert comparison_scope(row) != comparison_scope(other)


def test_prepare_is_independent_and_rejects_mixed_catalogs(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import pytest
    import drug_agent.scripts.prepare_nemo_toolrl as prep
    monkeypatch.setattr(prep, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(prep, "_init_tokenizer", lambda path: None)
    def encode(line):
        row = json.loads(line)
        return row, encoding_text(row), 10, {"protected_reason":None, "lossy":False}, 10
    monkeypatch.setattr(prep, "_encode_row", encode)
    source = tmp_path / "input.jsonl"
    rows = [{"id":str(i), "prompt":[{"role":"user","content":"task"}],
             "tools":[{"function":{"name":"read","description":"same"}}],
             "label":{"target_assistant":{"role":"assistant","content":"done"}},
             "metadata":{"trajectory_index":0,"decision_ordinal":i,"trajectory_decision_count":2,"task_type":"AC"}}
            for i in range(2)]
    def run(destination):
        source.write_text("".join(json.dumps(r)+"\n" for r in rows))
        monkeypatch.setattr("sys.argv", ["prepare", "--input", str(source), "--model", str(tmp_path), "--output-root", str(destination)])
        prep.main()
    run(tmp_path / "valid")
    manifest = json.loads((tmp_path / "valid/preparation_manifest.json").read_text())
    assert manifest["records"] == 2 and len(manifest["public_catalog_versions"]) == 1
    rows[1]["tools"][0]["function"]["description"] = "changed definition; same name"
    with pytest.raises(ValueError, match="Mixed full tool catalog"):
        run(tmp_path / "mixed")
    assert not (tmp_path / "mixed/preparation_manifest.json").exists()
    rows[1]["tools"][0]["function"]["description"] = "same"
    rows[0]["prompt"].insert(0, {"role":"system", "content":"Shared instructions"})
    rows[1]["prompt"].insert(0, {"role":"system", "content":"Shared instructions"})
    run(tmp_path / "shared_system")
    manifest = json.loads((tmp_path / "shared_system/preparation_manifest.json").read_text())
    assert len(manifest["public_system_versions"]) == 1
    rows[1]["prompt"][0]["content"] = "Different task-specific constraint"
    with pytest.raises(ValueError, match="Mixed system message versions"):
        run(tmp_path / "mixed_system")
    assert not (tmp_path / "mixed_system/preparation_manifest.json").exists()


def test_runner_reads_prepared_scope_without_reclassifying(tmp_path):
    from drug_agent.scripts.run_nemo_toolrl import load_rows
    from drug_agent.toolrl.nemo_scope import VERSION
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "preparation_manifest.json").write_text(json.dumps({"comparison_boundary_version":VERSION}))
    row = {"id":"a", "scope":"already-prepared", "text":"not reparsed by runner", "tokens":1}
    (prepared / "eligible_text.jsonl").write_text(json.dumps(row)+"\n")
    assert load_rows(tmp_path, False) == [row]


def test_embedding_cache_ignores_scope_but_rejects_changed_text(tmp_path, monkeypatch):
    import hashlib
    import numpy as np
    import pytest
    from drug_agent.scripts.run_nemo_toolrl import read_embedding, embedding_binding
    monkeypatch.setattr("importlib.metadata.version", lambda name:"test-version")
    assert "preparation_sha256" not in embedding_binding()
    row = {"id":"one", "text":"same input", "scope":"old scope"}
    path = tmp_path / "vector.npz"
    vector = np.ones(4096, dtype=np.float32) / 64
    np.savez(path, vector=vector, id=row["id"], text_sha256=hashlib.sha256(row["text"].encode()).hexdigest())
    row["scope"] = "new scope; no re-encoding"
    np.testing.assert_array_equal(read_embedding(path,row), vector)
    row["text"] = "different input"
    with pytest.raises(AssertionError, match="Cached text changed"):
        read_embedding(path,row)


def test_full_limit_embedding_requests_are_isolated_without_truncation():
    from types import SimpleNamespace
    from drug_agent.scripts.run_nemo_toolrl import embed_checked
    import pytest
    seen = []
    expected = {"short":[1], "full":[2]*32768}
    def embed(texts, **kwargs):
        seen.append(texts)
        return [SimpleNamespace(prompt_token_ids=expected[t]) for t in texts]
    outputs = embed_checked(embed, ["short","full"], [expected["short"],expected["full"]])
    assert seen == [["short"],["full"]]
    assert len(outputs[1].prompt_token_ids) == 32768
    seen.clear()
    embed_checked(embed, ["short","short"], [[1],[1]])
    assert seen == [["short","short"]]
    with pytest.raises(AssertionError):
        embed_checked(embed, ["full"], [[2]*32769])


def test_export_uses_independent_bound_lengths_and_original_rows(tmp_path, monkeypatch):
    import hashlib
    from drug_agent.scripts.materialize_nemo_toolrl import main
    from drug_agent.scripts.audit_toolrl_v8_lengths import _tokenizer_identity
    from drug_agent.toolrl.v8_dataset import sha256_file
    import pytest
    def put(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    rows = [{"id":str(i), "prompt":[{"role":"user","content":"task"}], "tools":[],
             "label":{"target_assistant":{"role":"assistant","content":"answer"},"target_tool_calls":[]},
             "metadata":{"trajectory_index":0,"decision_ordinal":i,"source_id":"trajectory", "task_type":"ac","decision_type":"final_answer"}}
            for i in range(3)]
    source = tmp_path / "mother.jsonl"
    source.write_text("".join(json.dumps(r)+"\n" for r in rows))
    root = tmp_path / "new-independent-run"
    put(root / "prepared/preparation_manifest.json", {"input":str(source),"input_sha256":sha256_file(source), "records":3,"counts":{"eligible":3}})
    (root / "prepared/eligible_text.jsonl").write_text("comparison copy\n")
    (root / "prepared/encoding_audit.jsonl").write_text("".join(json.dumps({"id":r["id"],"disposition":"eligible", "scope":json.dumps({"prior_outcome":"unknown"}), "training_content_sha256":hashlib.sha256(stable_json({k:r[k] for k in ("prompt","tools","label")}).encode()).hexdigest()})+"\n" for r in rows))
    put(root / "encoding_complete.json", {"records":3})
    put(root / "official_dedup/report.json", {"records":3,"prepared_text_sha256":sha256_file(root / "prepared/eligible_text.jsonl"),"threshold_removed_counts":{"0.0001":1}})
    put(root / "official_dedup/removed_eps_0.0001.json", ["1"])
    model = tmp_path / "tokenizer"
    put(model / "tokenizer_config.json", {"chat_template":"fixture"})
    lengths = tmp_path / "independent-lengths.jsonl"
    fields = ["reasoning_tokens","structured_action_or_final_tokens","native_action_or_final_tokens","prompt_plus_reference_tokens"]
    lengths.write_text("".join(json.dumps({"decision_id":r["id"], "prompt_tokens":10, "full_reference_tokens":20, **dict.fromkeys(fields,1)})+"\n" for r in rows))
    report_path = tmp_path / "independent-length-report.json"
    report = {"input":{"sha256":sha256_file(source),"records":3},"details_sha256":sha256_file(lengths), "tokenizer":str(model),"tokenizer_identity":_tokenizer_identity(str(model)),"context_limit":262144,"native_template":{"add_generation_prompt":True,"enable_thinking":True,"tools_included":True}}
    put(report_path, report)
    monkeypatch.setattr("sys.argv", ["export", "--run-root",str(root), "--eps","0.0001", "--length-details",str(lengths), "--length-report",str(report_path)])
    main()
    delivered = list(map(json.loads, (root / "delivery_eps_0.0001/standard_32k.jsonl").open()))
    assert delivered == [rows[0],rows[2]]
    report["input"]["sha256"] = "different source"
    put(report_path, report)
    with pytest.raises(AssertionError, match="different source"):
        main()


def window_fixture():
    history = [{"role":"system","content":"Do not overwrite files"}, {"role":"user","content":"repair protein"}]
    for i in range(4):
        history.extend([{"role":"assistant","reasoning_content":"reason"*20,
                         "tool_calls":[{"id":str(i),"function":{"name":"read","arguments":{"path":str(i)}}}]},
                        {"role":"tool","name":"read","tool_call_id":str(i),"content":"observation"*30}])
    return {"history_before_current_decision":history,"current_call_tool_definitions":[{"name":"write"}],
            "current_teacher_response":{"role":"assistant","reasoning_content":"current","content":"complete target"}}


def test_recent_window_keeps_task_current_and_whole_contiguous_assistants():
    doc=window_fixture()
    original=stable_json(doc)
    text,n,audit=window_document(doc,len,limit=1000)
    result=json.loads(text)
    assert n<=1000 and audit["lossy"] and not audit["protected_reason"]
    history=result["history_before_current_decision"]
    assert history[0]==doc["history_before_current_decision"][1]
    assert history[-1]==doc["history_before_current_decision"][-2]
    assert result["current_teacher_response"]==doc["current_teacher_response"]
    assert stable_json(doc)==original
    assert all(m["role"] == "assistant" for m in history[1:])


def test_large_current_or_latest_interaction_is_protected_not_sliced():
    doc=window_fixture()
    text,n,audit=window_document(doc,len,limit=400)
    assert audit["protected_reason"]=="latest_complete_assistant_does_not_fit"
    doc["current_teacher_response"]["content"]="x"*2000
    text,n,audit=window_document(doc,len,limit=1600)
    assert audit["protected_reason"]=="fixed_task_current_definitions_over_limit"
    assert json.loads(text)["current_teacher_response"]["content"]=="x"*2000


def test_all_tool_results_are_excluded_but_current_parameters_are_intact():
    doc=window_fixture()
    doc["history_before_current_decision"][-1]["content"]=json.dumps({"base64":"A"*10000,"error":"bad format","path":"/server/file","code":"B"*9000})
    doc["current_teacher_response"]["content"]="C"*9000
    text,n,audit=window_document(doc,len,limit=50000)
    result=json.loads(text)
    assert all(m["role"] not in {"tool","system"} for m in result["history_before_current_decision"])
    assert result["current_teacher_response"]["content"]=="C"*9000
    assert len(audit["excluded_tool_result_indices"]) == 4


def test_even_loaded_skill_results_are_excluded():
    doc=window_fixture()
    doc["history_before_current_decision"][-1]["name"]="skill"
    doc["history_before_current_decision"][-1]["content"]=json.dumps({"file_content":"skill instructions"*800})
    text,n,audit=window_document(doc,len,limit=50000)
    assert "skill instructions" not in text
    assert json.loads(text)["history_before_current_decision"] == [m for m in doc["history_before_current_decision"] if m["role"] not in {"system","tool"}]
    assert audit["lossy"]


def test_later_user_constraint_survives_and_results_never_enter_window():
    doc=window_fixture()
    doc["history_before_current_decision"].insert(4,{"role":"user","content":"New constraint: do not overwrite"})
    text,n,audit=window_document(doc,len,limit=1600)
    assert "New constraint: do not overwrite" in text
    doc["history_before_current_decision"][-1]["tool_call_id"]="incorrect"
    changed,_,_=window_document(doc,len,limit=1600)
    assert changed == text


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
