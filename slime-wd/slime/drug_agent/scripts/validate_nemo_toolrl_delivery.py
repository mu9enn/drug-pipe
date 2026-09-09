"""CPU-only real Slime Dataset and SGLang parser checks; never starts RL."""
import argparse
import json
import tempfile
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer
from tokenizers import Tokenizer

from slime.utils.data import Dataset
from drug_agent.scripts.validate_qwen_native_toolrl_roundtrip import _completion
from drug_agent.toolrl.qwen_native_parser import parse_qwen_native_completion
from drug_agent.toolrl.trajectory_data_source import TrajectoryBatchDataSource
from drug_agent.toolrl.v8_dataset import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--length-details", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reasoning-parser", required=True)
    parser.add_argument("--tool-parser", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rust = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    rust.no_truncation()
    details = {r["decision_id"]:r for r in map(json.loads, args.length_details.open())}
    checked = 0
    parse_checks = 0
    length_checks = 0
    failures = []
    seen_patterns = set()
    sampled_ids = []
    with tempfile.TemporaryDirectory(prefix="nemo-reader-", dir=args.report.parent) as tmp, args.input.open() as source:
        batch_path = Path(tmp) / "batch.jsonl"
        while lines := list(islice(source, 64)):
            # A bounded copy used by the actual loader; automatically removed.
            batch_path.write_text("".join(lines))
            rows = [json.loads(line) for line in lines]
            dataset = Dataset(str(batch_path), tokenizer, None, None,
                              prompt_key="prompt", label_key="label", tool_key="tools",
                              apply_chat_template=True, apply_chat_template_kwargs={"enable_thinking":True})
            assert len(dataset.samples) == len(rows)
            for row, sample in zip(rows, dataset.samples, strict=True):
                assert sample.label == row["label"]
                assert sample.metadata["tools"] == row["tools"]
                detail = details[row["id"]]
                assert detail["prompt_tokens"] <= 229376 and detail["full_reference_tokens"] <= 32768
                target = row["label"]["target_assistant"]
                full = tokenizer.apply_chat_template(row["prompt"]+[target], tools=row["tools"], tokenize=False, add_generation_prompt=False, enable_thinking=True)
                assert full.startswith(sample.prompt), row["id"]
                checked += 1
                pattern = (tuple(row["metadata"]["tool_names"]), row["metadata"]["task_type"],
                           tuple(c["arguments"].get("name", "") for c in row["label"]["target_tool_calls"] if c["name"] == "skill"))
                if pattern not in seen_patterns:
                    seen_patterns.add(pattern)
                    sampled_ids.append(row["id"])
                    actual_len = len(rust.encode(sample.prompt, add_special_tokens=False).ids)
                    assert actual_len == detail["prompt_tokens"], (row["id"], actual_len, detail["prompt_tokens"])
                    length_checks += 1
                    parsed = parse_qwen_native_completion(_completion(tokenizer, row), tools_schema=row["tools"],
                                                         reasoning_parser=args.reasoning_parser, tool_parser=args.tool_parser)
                    actual = [{"name":c["name"], "arguments":c["arguments"]} for c in parsed["tool_calls"]]
                    valid = parsed["ok"] and actual == row["label"]["target_tool_calls"]
                    if row["label"]["decision_type"] == "final_answer":
                        valid = valid and parsed["final_answer"] == row["label"]["target_final_answer"]
                    if not valid:
                        failures.append({"id":row["id"], "error":parsed.get("error_message"), "tool_calls_match":actual == row["label"]["target_tool_calls"]})
                    parse_checks += 1
            reader = TrajectoryBatchDataSource.__new__(TrajectoryBatchDataSource)
            reader.args = SimpleNamespace(n_samples_per_prompt=4, rollout_batch_size=8)
            reader.samples = dataset.samples
            reader.cursor = reader.epoch_id = reader.sample_group_index = reader.sample_index = 0
            groups = reader.get_samples(len(rows))
            assert len({g[0].group_index for g in groups}) == len(rows)
            for row, group in zip(rows, groups, strict=True):
                assert len(group) == 4 and all(s.label == row["label"] for s in group)
                assert len({s.group_index for s in group}) == 1
            if checked % 512 == 0:
                print(checked, "loaded/rendered/grouped; parser failures:", len(failures), flush=True)
    report = {"records_loaded_rendered_grouped":checked, "parser_roundtrip_checked":parse_checks,
              "retokenized_input_checks":length_checks, "checked_ids":sampled_ids, "parser_failures":failures,
              "input_sha256":sha256_file(args.input), "reasoning_parser":args.reasoning_parser,"tool_parser":args.tool_parser,
              "versions":{name:__import__(name).__version__ for name in ("transformers","tokenizers","sglang","ray","jinja2")},
              "tokenizer_assets":{p.name:sha256_file(p) for p in args.model.iterdir() if p.name in ("tokenizer.json","tokenizer_config.json","chat_template.jinja")},
              "runtime_generation_backward_memory_gate": "not_run", "status":"pass" if not failures else "parser_review_required"}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "checked_ids"}, indent=2))


if __name__ == "__main__":
    main()
