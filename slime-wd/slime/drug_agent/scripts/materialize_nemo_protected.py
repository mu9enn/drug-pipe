"""Deliver the explicitly protected corpus when no full document fits NeMo's encoder.

This is an empty-eligible-input outcome, not a substitute deduplication algorithm.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from drug_agent.scripts.materialize_toolrl_v8_length_tiers import materialize
from drug_agent.scripts.validate_trajectory_toolrl_batches import validate_file
from drug_agent.toolrl.v8_dataset import sha256_file, stable_json, _strip_training_fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    prep = json.loads((root / "prepared/preparation_manifest.json").read_text())
    if prep["counts"].get("eligible", 0):
        raise ValueError("Eligible records exist: actual official NeMo processing is required.")
    source = Path(prep["input"])
    assert sha256_file(source) == prep["input_sha256"]
    old_validation = json.loads((root.parent / "final_validation.json").read_text())
    assert old_validation["all_decisions"]["sha256"] == prep["input_sha256"]
    old_materialization = json.loads((source.parent / "materialization_manifest.json").read_text())
    assert sha256_file(args.sft) == old_materialization["qwen_adapter_source"]["sha256"]
    qwen = {r["id"]: r for r in map(json.loads, args.sft.open())}
    audit = {r["id"]:r for r in map(json.loads, (root / "prepared/encoding_audit.jsonl").open())}
    coverage = {k: Counter() for k in ("tools", "skills", "tasks", "final_tasks", "trajectories", "prior_outcomes")}
    checked = 0
    lengths = []
    with source.open() as inp, (root / "pre_budget_ids.jsonl").open("w") as ids:
        for line in inp:
            row = json.loads(line)
            ident = row["id"]
            entry = audit.pop(ident)
            assert entry["disposition"] == "protected_over_embedding_limit"
            assert hashlib.sha256(stable_json({k:row[k] for k in ("prompt","tools","label")}).encode()).hexdigest() == entry["training_content_sha256"]
            meta = row["metadata"]
            original = qwen[meta["source_id"]]
            assistants = [(i,m) for i,m in enumerate(original["messages"]) if m["role"] == "assistant"]
            index, target = assistants[meta["decision_ordinal"]]
            assert row["prompt"] == [_strip_training_fields(m) for m in original["messages"][:index]], ident
            assert row["label"]["target_assistant"] == _strip_training_fields(target), ident
            assert row["tools"] == original["tools"], ident
            assert ident == f'{meta["source_id"]}::decision_{meta["decision_ordinal"]:04d}'
            for call in row["label"].get("target_tool_calls", []):
                coverage["tools"][call["name"]] += 1
                if call["name"] == "skill":
                    coverage["skills"][call["arguments"]["name"]] += 1
            coverage["tasks"][meta["task_type"]] += 1
            coverage["trajectories"][meta["source_id"]] += 1
            coverage["prior_outcomes"][json.loads(entry["scope"])["prior_outcome"]] += 1
            if meta["decision_type"] == "final_answer":
                coverage["final_tasks"][meta["task_type"]] += 1
            ids.write(json.dumps({"id": ident, "reason": entry["disposition"]}) + "\n")
            lengths.append(entry["tokens"])
            checked += 1
    assert not audit and checked == prep["records"]
    selected = root / "retained_decisions.jsonl"
    if not selected.exists():
        selected.symlink_to(source)
    assert sha256_file(selected) == prep["input_sha256"]
    tier = materialize(selected, root.parent / "02_length_audit_all/length_details.jsonl", root / "length_tiers", emit_pretty=False)
    seen = set()
    source_digests = {r["id"]:r["training_content_sha256"] for r in map(json.loads, (root / "prepared/encoding_audit.jsonl").open())}
    for output in tier["outputs"].values():
        for row in map(json.loads, Path(output["jsonl"]).open()):
            assert row["id"] not in seen
            seen.add(row["id"])
            assert hashlib.sha256(stable_json({k:row[k] for k in ("prompt","tools","label")}).encode()).hexdigest() == source_digests[row["id"]]
    assert seen == set(source_digests)
    order = validate_file(root / "length_tiers/standard_32k.jsonl", 8)
    report = {
        "status": "protected_data_delivered; official_nemo_not_run_no_eligible_documents",
        "input_records": checked, "encoded_records": 0, "protected_overlength_records": checked,
        "semantic_removed": 0, "budget_removed": 0, "budget_requested": 2527,
        "budget_conflict": "All full documents exceed 32K; protected records alone exceed requested budget.",
        "input_sha256": prep["input_sha256"], "embedding_tokens_min": min(lengths), "embedding_tokens_max": max(lengths),
        "coverage": {k:dict(v) for k,v in coverage.items()}, "coverage_counts": {k:len(v) for k,v in coverage.items()},
        "validation": {"source_sft_prefix_and_target_checks": checked, "future_message_leakage": 0,
                       "training_content_mismatches": 0, "tier_missing_or_duplicate_ids": 0,
                       "order": order, "gpu_allocated": False, "generation_or_backward_run": False},
        "length_reuse": {"prior_validation": str(root.parent / "final_validation.json"),
                         "basis": "Source SHA matches previously validated all_decisions; only selection eligibility is added in separate files. Original Qwen3.5-9B template and budgets are retained.",
                         "previous_metadata_only_rebuild": old_validation["length_details"]["reuse_note"]},
        "length_tiers": tier["outputs"],
    }
    (root / "delivery_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "coverage"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
