"""Recover original decisions by official NeMo IDs; never rebuild training text."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from drug_agent.toolrl.v8_dataset import sha256_file, stable_json
from drug_agent.scripts.materialize_toolrl_v8_length_tiers import materialize
from drug_agent.scripts.audit_toolrl_v8_lengths import _tokenizer_identity


def coverage(row, counters, scope):
    meta = row["metadata"]
    counters["task"][meta["task_type"]] += 1
    counters["trajectory"][meta["source_id"]] += 1
    counters["decision_type"][meta["decision_type"]] += 1
    counters["prior_outcome"][scope["prior_outcome"]] += 1
    for call in row["label"].get("target_tool_calls", []):
        counters["tool_calls"][call["name"]] += 1
        if call["name"] == "skill":
            counters["skill_calls"][call["arguments"]["name"]] += 1
    if meta["decision_type"] == "final_answer":
        counters["final_task"][meta["task_type"]] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--qwen-source", type=Path, help="Optional additional exact mother-history verification")
    parser.add_argument("--length-details", type=Path, required=True)
    parser.add_argument("--length-report", type=Path, required=True)
    args = parser.parse_args()
    args.eps = str(args.eps)
    root = args.run_root.resolve()
    prep = json.loads((root / "prepared/preparation_manifest.json").read_text())
    official = json.loads((root / "official_dedup/report.json").read_text())
    encoded = json.loads((root / "encoding_complete.json").read_text())
    assert encoded["records"] == official["records"] == prep["counts"]["eligible"]
    source = Path(prep["input"])
    assert sha256_file(source) == prep["input_sha256"]
    assert official["prepared_text_sha256"] == sha256_file(root / "prepared/eligible_text.jsonl")
    lengths = json.loads(args.length_report.read_text())
    assert lengths["input"]["sha256"] == prep["input_sha256"], "Length audit belongs to a different source"
    assert lengths["details_sha256"] == sha256_file(args.length_details), "Length details changed"
    assert lengths["input"]["records"] == prep["records"]
    assert lengths["tokenizer_identity"].get("files"), "Length audit must pin local tokenizer assets"
    assert _tokenizer_identity(lengths["tokenizer"]) == lengths["tokenizer_identity"], "Training tokenizer/template changed"
    assert lengths["native_template"] == {"add_generation_prompt":True,"enable_thinking":True,"tools_included":True}
    assert lengths["context_limit"] == 262144
    qwen = {}
    if args.qwen_source:
        for row in map(json.loads, args.qwen_source.open()):
            assert row["id"] not in qwen
            messages = [{k:v for k,v in m.items() if k != "step_loss_mask"} for m in row["messages"]]
            qwen[row["id"]] = (messages, [i for i,m in enumerate(messages) if m["role"] == "assistant"], row["tools"])
    audit = {r["id"]:r for r in map(json.loads, (root / "prepared/encoding_audit.jsonl").open())}
    removed_list = json.loads((root / f"official_dedup/removed_eps_{args.eps}.json").read_text())
    removed = set(removed_list)
    assert len(removed) == len(removed_list) == official["threshold_removed_counts"][args.eps]
    assert removed <= {i for i,r in audit.items() if r["disposition"] == "eligible"}
    destination = root / f"delivery_eps_{args.eps}"
    destination.mkdir(exist_ok=True)
    selected_path = destination / "retained_decisions.jsonl"
    counters = {phase:{name:Counter() for name in ("task","trajectory","decision_type","prior_outcome","tool_calls","skill_calls","final_task")} for phase in ("before","after")}
    selected, selected_hashes, reasons = set(), {}, Counter()
    previous_order = (-1, -1)
    with source.open() as stream, selected_path.open("w") as output, (destination / "selection_audit.jsonl").open("w") as selection, (destination / "pre_budget_ids.jsonl").open("w") as ids:
        for line in stream:
            row = json.loads(line)
            ident = row["id"]
            if args.qwen_source:
                messages, positions, tools = qwen[row["metadata"]["source_id"]]
                position = positions[row["metadata"]["decision_ordinal"]]
                assert row["prompt"] == messages[:position], "History changed or contains current/future content"
                assert row["label"]["target_assistant"] == messages[position]
                assert row["tools"] == tools
            entry = audit.pop(ident)
            content = stable_json({k:row[k] for k in ("prompt","tools","label")})
            assert hashlib.sha256(content.encode()).hexdigest() == entry["training_content_sha256"]
            order = (row["metadata"]["trajectory_index"],row["metadata"]["decision_ordinal"])
            assert order > previous_order
            previous_order = order
            scope = json.loads(entry["scope"])
            coverage(row,counters["before"],scope)
            reason = "recent_context_similarity_not_selected" if ident in removed else entry["disposition"]
            if ident not in removed:
                output.write(line)
                ids.write(json.dumps({"id":ident})+"\n")
                selected.add(ident)
                selected_hashes[ident] = hashlib.sha256(stable_json(row).encode()).hexdigest()
                coverage(row,counters["after"],scope)
            reasons[reason] += 1
            selection.write(json.dumps({"id":ident,"selected":ident not in removed,"reason":reason})+"\n")
    assert not audit and len(selected)+len(removed) == prep["records"]
    tiers = materialize(selected_path,args.length_details,destination,emit_pretty=False)
    observed = set()
    for item in tiers["outputs"].values():
        order = (-1,-1)
        for row in map(json.loads,Path(item["jsonl"]).open()):
            assert row["id"] not in observed and row["id"] in selected
            assert hashlib.sha256(stable_json(row).encode()).hexdigest() == selected_hashes[row["id"]]
            observed.add(row["id"])
            key = (row["metadata"]["trajectory_index"],row["metadata"]["decision_ordinal"])
            assert key > order
            order = key
    assert observed == selected
    report = {"input_records":prep["records"],"encoded_records":encoded["records"],
              "protected_records":prep["counts"].get("protected_window_exception",0),
              "semantic_not_selected":len(removed),"budget_not_selected":0,
              "retained_records":len(selected),"eps":args.eps,"reasons":dict(reasons),
              "coverage":counters,"length_tiers":tiers,
              "budget_note":"No budget sampling; this is NeMo selection on lossy recent-context copies, not proof of exact duplicate records.",
              "length_reuse_basis":{"report":str(args.length_report),"report_sha256":sha256_file(args.length_report),
                                    "details_sha256":lengths["details_sha256"],"tokenizer_identity":lengths["tokenizer_identity"]},
              "source_sha256":prep["input_sha256"],"selected_sha256":sha256_file(selected_path),
              "checks":{"original_lines_recovered":True,"training_content_unchanged":True,"ids_unique":True,"ordered_exact_tier_partition":True,"exact_original_history_prefix_no_future":bool(args.qwen_source)},
              "not_verified":["actual RL generation","backward-pass memory"] +
                             ([] if args.qwen_source else ["additional mother-trajectory prefix verification"])}
    (destination / "selection_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("coverage","length_tiers")},indent=2))


if __name__ == "__main__":
    main()
