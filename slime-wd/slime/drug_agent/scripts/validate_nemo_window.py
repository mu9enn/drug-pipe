"""Verify every prepared comparison copy against its immutable source row."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from drug_agent.toolrl.v8_dataset import sha256_file, stable_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,required=True)
    args=parser.parse_args()
    root=args.run_root
    prep=json.loads((root/"prepared/preparation_manifest.json").read_text())
    audit={r["id"]:r for r in map(json.loads,(root/"prepared/encoding_audit.jsonl").open())}
    copies=iter(map(json.loads,(root/"prepared/eligible_text.jsonl").open()))
    counts=Counter()
    source=Path(prep["input"])
    assert sha256_file(source)==prep["input_sha256"]
    strip=lambda m:{k:v for k,v in m.items() if k!="step_loss_mask"}
    for row in map(json.loads,source.open()):
        entry=audit.pop(row["id"])
        content=stable_json({k:row[k] for k in ("prompt","tools","label")})
        assert hashlib.sha256(content.encode()).hexdigest()==entry["training_content_sha256"]
        counts[entry["disposition"]]+=1
        if entry["disposition"]!="eligible":
            continue
        copy=next(copies)
        assert copy["id"]==row["id"] and copy["tokens"]<=32768
        assert hashlib.sha256(copy["text"].encode()).hexdigest()==entry["text_sha256"]
        doc=json.loads(copy["text"])
        assert doc["current_teacher_response"]==strip(row["label"]["target_assistant"])
        tools={t["function"]["name"]:t for t in row["tools"]}
        expected=[tools[c["function"]["name"]] for c in row["label"]["target_assistant"].get("tool_calls",[])]
        assert doc["current_call_tool_definitions"]==expected
        omitted=set(entry["window"]["omitted_message_indices"])
        assert all(row["prompt"][i]["role"] not in {"system","user","developer"} for i in omitted)
        excluded_system={i for i,m in enumerate(row["prompt"]) if m["role"] == "system"}
        excluded_tool={i for i,m in enumerate(row["prompt"]) if m["role"] == "tool"}
        assert excluded_system == set(entry["window"]["excluded_system_message_indices"])
        assert excluded_tool == set(entry["window"]["excluded_tool_result_indices"])
        system=stable_json([strip(row["prompt"][i]) for i in sorted(excluded_system)])
        version=hashlib.sha256(system.encode()).hexdigest()
        assert version in prep["public_system_versions"]
        assert json.loads((root/"prepared/system_messages"/f"{version}.json").read_text()) == json.loads(system)
        indices=[i for i in range(len(row["prompt"])) if i not in omitted | excluded_system | excluded_tool]
        history=doc["history_before_current_decision"]
        assert len(history)==len(indices)
        assert all(m["role"] not in {"system","tool"} for m in history)
        assert history==[strip(row["prompt"][i]) for i in indices]
        counts["windowed_eligible"]+=bool(omitted)
    assert not audit and next(copies,None) is None
    report={"counts":dict(counts),"source_sha256":prep["input_sha256"],
            "current_target_and_definitions_exact":True,"all_user_developer_constraints_exact":True,
            "common_system_saved_separately":True,"no_tool_results_in_encoding":True,
            "retained_history_messages_exact_in_original_order":True,
            "training_source_unchanged":True,
            "embedding_not_yet_asserted_by_this_check":True}
    (root/"window_integrity_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
