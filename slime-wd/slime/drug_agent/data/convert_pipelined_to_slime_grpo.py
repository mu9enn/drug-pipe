from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.constants import (
    DEFAULT_MAX_STEPS,
    GRPO_OUT_ROOT,
    PIPELINED_DATA,
    RAW_TASK_TYPES,
    SFT_OUTPUTS_ANSWER_HIT,
)
from drug_agent.data.common import (
    basename_to_task_type,
    discover_raw_files,
    index_rl_rows,
    load_sft_rows_by_id,
    load_usage_summary_by_basename,
    parse_raw_trajectory_file,
)
from drug_agent.protocol.prompts import build_grpo_prompt_messages
from drug_agent.utils import normalize_tool_name, write_json, write_jsonl


def load_allowlist(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        values = obj.get("allowed_tools")
        if isinstance(values, list):
            return {normalize_tool_name(x) for x in values if isinstance(x, str) and x.strip()}
    if isinstance(obj, list):
        return {normalize_tool_name(x) for x in obj if isinstance(x, str) and x.strip()}
    return set()


def normalize_allowed_tools(raw_tools: list[Any], allowlist: set[str], allow_all: bool) -> list[str]:
    out: list[str] = []
    for item in raw_tools:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s:
            continue
        if not s.startswith("mcp__"):
            continue
        bare = normalize_tool_name(s)
        if not bare:
            continue
        if (not allow_all) and allowlist and bare not in allowlist:
            continue
        if bare not in out:
            out.append(bare)
    return out


def derive_instruction(env_task: dict[str, Any], raw_info: dict[str, Any] | None) -> str:
    base = env_task.get("instruction") if isinstance(env_task.get("instruction"), str) else ""
    question_text = ""
    if raw_info is not None and isinstance(raw_info.get("question_text"), str):
        question_text = raw_info["question_text"].strip()

    if question_text:
        return question_text
    if base and base.strip() and base.strip().lower() != "task execution session":
        return base.strip()
    if base:
        return base.strip()
    return ""


def derive_inputs(env_task: dict[str, Any], raw_info: dict[str, Any] | None) -> dict[str, Any]:
    inputs = env_task.get("inputs") if isinstance(env_task.get("inputs"), dict) else {}
    merged = dict(inputs)

    if raw_info and isinstance(raw_info.get("question_payload"), dict):
        q = raw_info["question_payload"]
        if q.get("question_text") and "question_text" not in merged:
            merged["question_text"] = q.get("question_text")
        if q.get("answer") is not None and "ground_truth_answer" not in merged:
            merged["ground_truth_answer"] = q.get("answer")
        if q.get("candidates") is not None and "candidates" not in merged:
            merged["candidates"] = q.get("candidates")
        if q.get("task") and "task" not in merged:
            merged["task"] = q.get("task")

    return merged


def build_label(
    *,
    sample_id: str,
    task_id: str,
    task_type: str,
    raw_info: dict[str, Any] | None,
    usage_info: dict[str, Any] | None,
    sft_row: dict[str, Any] | None,
) -> dict[str, Any]:
    ground_truth = None
    if raw_info is not None:
        if raw_info.get("question_answer") is not None:
            ground_truth = raw_info.get("question_answer")

    expected = {}
    if usage_info:
        for key in ["answer_hit_pass", "vs_top3_hit_num", "vs_top10_hit_num", "ac_is_correct", "pf_f1", "pf_is_correct"]:
            if usage_info.get(key) is not None:
                expected[key] = usage_info.get(key)

    metadata = {
        "sample_id": sample_id,
        "task_id": task_id,
        "task_type": task_type,
        "raw_source_path": raw_info.get("source_path") if raw_info else None,
        "usage_summary": usage_info,
        "sft_metadata": (sft_row or {}).get("metadata") if isinstance((sft_row or {}).get("metadata"), dict) else {},
        "raw_final_result": (raw_info or {}).get("final_result"),
    }

    return {
        "task_id": task_id,
        "task_type": task_type,
        "ground_truth": ground_truth,
        "expected": expected,
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert pipelined_data to slime GRPO prompt-data jsonl (hybrid join)")
    parser.add_argument("--input-root", type=str, default=str(PIPELINED_DATA))
    parser.add_argument("--output-root", type=str, default=str(GRPO_OUT_ROOT))
    parser.add_argument("--max-samples-per-task-type", type=int, default=None)
    parser.add_argument("--allowlist", type=str, default=str(Path(__file__).resolve().parents[1] / "tools/allowlist_v0.json"))
    parser.add_argument("--allow-all", action="store_true")
    parser.add_argument("--rl-prompts-path", type=str, default=str(SFT_OUTPUTS_ANSWER_HIT / "mcp_rl_prompts_all.jsonl"))
    parser.add_argument("--sft-path", type=str, default=str(SFT_OUTPUTS_ANSWER_HIT / "mcp_sft_all.jsonl"))
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rl_prompts_path = Path(args.rl_prompts_path)
    if not rl_prompts_path.exists():
        rl_prompts_path = input_root / "sft_outputs_answer_hit/mcp_rl_prompts_all.jsonl"
    sft_path = Path(args.sft_path)
    if not sft_path.exists():
        sft_path = input_root / "sft_outputs_answer_hit/mcp_sft_all.jsonl"

    allowlist = load_allowlist(Path(args.allowlist) if args.allowlist else None)
    usage_by_basename = load_usage_summary_by_basename(input_root / "molclaw_usage_summary.csv")
    sft_by_id = load_sft_rows_by_id(sft_path)
    rl_rows = index_rl_rows(rl_prompts_path)

    raw_files_by_type = discover_raw_files(input_root)
    raw_path_by_basename: dict[str, Path] = {}
    for _, paths in raw_files_by_type.items():
        for p in paths:
            raw_path_by_basename[p.name] = p

    raw_cache: dict[str, dict[str, Any]] = {}

    out_rows: dict[str, list[dict[str, Any]]] = {k: [] for k in RAW_TASK_TYPES}
    skipped: list[dict[str, Any]] = []
    counters = Counter()

    for row_idx, rl in enumerate(rl_rows):
        counters["input_rows"] += 1
        sample_id = rl.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            skipped.append({"source": "rl_prompts", "task_id": None, "skip_reason": "missing_sample_id", "details": {"row_idx": row_idx}})
            counters["skip_missing_sample_id"] += 1
            continue

        sft_row = sft_by_id.get(sample_id)
        if sft_row is None:
            skipped.append({"source": "rl_prompts", "task_id": sample_id, "skip_reason": "missing_sft_row_for_id", "details": {}})
            counters["skip_missing_sft_row"] += 1
            continue

        metadata = sft_row.get("metadata") if isinstance(sft_row.get("metadata"), dict) else {}
        trajectory_path = metadata.get("trajectory_path")
        if not isinstance(trajectory_path, str):
            skipped.append({"source": "sft_outputs", "task_id": sample_id, "skip_reason": "missing_trajectory_path", "details": {}})
            counters["skip_missing_trajectory_path"] += 1
            continue

        basename = Path(trajectory_path).name
        raw_path = raw_path_by_basename.get(basename)
        if raw_path is None:
            skipped.append({"source": "raw", "task_id": sample_id, "skip_reason": "raw_trajectory_not_found", "details": {"basename": basename}})
            counters["skip_raw_not_found"] += 1
            continue

        if basename not in raw_cache:
            raw_cache[basename] = parse_raw_trajectory_file(raw_path)
        raw_info = raw_cache[basename]

        usage = usage_by_basename.get(basename)

        env_kwargs = rl.get("env_kwargs") if isinstance(rl.get("env_kwargs"), dict) else {}
        env_task = env_kwargs.get("task") if isinstance(env_kwargs.get("task"), dict) else {}

        task_type = env_task.get("task_type")
        if not isinstance(task_type, str) or task_type not in RAW_TASK_TYPES:
            task_type = raw_info.get("raw_task_type")
        if task_type not in RAW_TASK_TYPES:
            skipped.append({"source": "hybrid", "task_id": sample_id, "skip_reason": "unknown_task_type", "details": {"task_type": task_type}})
            counters["skip_unknown_task_type"] += 1
            continue

        task_id = env_task.get("task_id") if isinstance(env_task.get("task_id"), str) and env_task.get("task_id") else sample_id
        instruction = derive_instruction(env_task, raw_info)
        if not instruction:
            skipped.append({"source": "hybrid", "task_id": task_id, "skip_reason": "missing_instruction", "details": {"basename": basename}})
            counters["skip_missing_instruction"] += 1
            continue

        inputs = derive_inputs(env_task, raw_info)

        allowed_tools_raw = env_task.get("allowed_tools") if isinstance(env_task.get("allowed_tools"), list) else []
        allowed_tools = normalize_allowed_tools(allowed_tools_raw, allowlist=allowlist, allow_all=args.allow_all)
        if not allowed_tools:
            skipped.append(
                {
                    "source": "hybrid",
                    "task_id": task_id,
                    "skip_reason": "allowed_tools_empty_after_filter",
                    "details": {
                        "basename": basename,
                        "allow_all": args.allow_all,
                        "allowlist_size": len(allowlist),
                        "raw_allowed_tools": allowed_tools_raw,
                    },
                }
            )
            counters["skip_allowed_tools_empty"] += 1
            continue

        max_steps = env_task.get("max_steps")
        if not isinstance(max_steps, int) or max_steps <= 0:
            max_steps = DEFAULT_MAX_STEPS

        prompt_messages = build_grpo_prompt_messages(
            task_id=task_id,
            task_type=task_type,
            instruction=instruction,
            inputs=inputs,
            allowed_tools=allowed_tools,
            max_steps=max_steps,
        )

        label = build_label(
            sample_id=sample_id,
            task_id=task_id,
            task_type=task_type,
            raw_info=raw_info,
            usage_info=usage,
            sft_row=sft_row,
        )

        data_source = f"drug_agent_{task_type}"
        env_payload = {
            "task_id": task_id,
            "task_type": task_type,
            "instruction": instruction,
            "inputs": inputs,
            "allowed_tools": allowed_tools,
            "max_steps": max_steps,
            "data_source": data_source,
        }

        sample = {
            "prompt": prompt_messages,
            "label": label,
            "data_source": data_source,
            "env_kwargs": env_payload,
            "extra_info": {
                "index": len(out_rows[task_type]),
                "source_path": str(raw_path),
                "rl_prompt_id": sample_id,
                "trajectory_basename": basename,
                "original_metadata": metadata,
                "usage_summary": usage,
            },
            "metadata": {
                "task_id": task_id,
                "task_type": task_type,
                "data_source": data_source,
                "env_kwargs": env_payload,
                "extra_info": {
                    "source_path": str(raw_path),
                    "rl_prompt_id": sample_id,
                    "trajectory_basename": basename,
                    "usage_summary": usage,
                },
                "drug_agent_quality": {
                    "answer_hit_pass": usage.get("answer_hit_pass") if usage else None,
                    "ac_is_correct": usage.get("ac_is_correct") if usage else None,
                    "pf_f1": usage.get("pf_f1") if usage else None,
                    "vs_top10_hit_num": usage.get("vs_top10_hit_num") if usage else None,
                },
            },
        }

        out_rows[task_type].append(sample)
        counters[f"kept_{task_type}"] += 1
        counters["kept_total"] += 1

    # optional size cap
    if args.max_samples_per_task_type is not None and args.max_samples_per_task_type >= 0:
        for task_type in RAW_TASK_TYPES:
            out_rows[task_type] = out_rows[task_type][: args.max_samples_per_task_type]

    mixed = []
    for task_type in RAW_TASK_TYPES:
        mixed.extend(out_rows[task_type])

    for task_type in RAW_TASK_TYPES:
        write_jsonl(output_root / f"{task_type}.jsonl", out_rows[task_type])
    write_jsonl(output_root / "mixed.jsonl", mixed)
    write_jsonl(output_root / "skipped_report.jsonl", skipped)

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "rl_prompts_path": str(rl_prompts_path),
        "sft_path": str(sft_path),
        "allowlist": str(args.allowlist),
        "allow_all": bool(args.allow_all),
        "max_samples_per_task_type": args.max_samples_per_task_type,
        "counts": {
            **dict(counters),
            "out_ac": len(out_rows["ac"]),
            "out_pf": len(out_rows["pf"]),
            "out_vs": len(out_rows["vs"]),
            "out_mixed": len(mixed),
            "skipped": len(skipped),
        },
    }
    write_json(output_root / "manifest.json", manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
