from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from drug_agent.protocol.react_protocol import parse_react_sequence
from drug_agent.tools.local_tools import LOCAL_TOOL_NAMES


@dataclass(frozen=True)
class ToolCallOccurrence:
    dataset: str
    record_id: str
    decision_index: int
    call_index: int
    tool_name: str
    json_text: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: record must be a JSON object")
            yield line_no, payload


def _tool_calls_from_assistant(
    content: str,
    *,
    dataset: str,
    record_id: str,
    decision_index: int,
) -> list[ToolCallOccurrence]:
    parsed = parse_react_sequence(content, role="assistant")
    if not parsed.get("ok"):
        raise ValueError(
            f"{dataset}:{record_id}:{decision_index}: invalid assistant ReAct: "
            f"{parsed.get('error_type')}: {parsed.get('error_message')}"
        )

    rows: list[ToolCallOccurrence] = []
    call_index = 0
    for block in parsed.get("blocks") or []:
        if not isinstance(block, dict) or block.get("kind") != "tool_call":
            continue
        payload = block.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"{dataset}:{record_id}:{decision_index}: tool call payload is not an object")
        tool_name = payload.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError(f"{dataset}:{record_id}:{decision_index}: tool call has no tool_name")
        json_text = str(block.get("body") or "").strip()
        if not json_text:
            raise ValueError(f"{dataset}:{record_id}:{decision_index}: empty tool-call JSON")
        # The protocol parser already validates the object. Reload here so this
        # report fails loudly if its exact tokenized text is ever not JSON.
        if not isinstance(json.loads(json_text), dict):
            raise ValueError(f"{dataset}:{record_id}:{decision_index}: tool-call JSON is not an object")
        rows.append(
            ToolCallOccurrence(
                dataset=dataset,
                record_id=record_id,
                decision_index=decision_index,
                call_index=call_index,
                tool_name=tool_name,
                json_text=json_text,
            )
        )
        call_index += 1
    return rows


def collect_sft(path: Path) -> tuple[int, int, list[ToolCallOccurrence]]:
    record_count = 0
    target_count = 0
    rows: list[ToolCallOccurrence] = []
    for line_no, record in _read_jsonl(path):
        record_count += 1
        record_id = str(record.get("id") or f"line_{line_no}")
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"{path}:{line_no}: messages must be a list")
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(f"{path}:{line_no}: assistant content must be a string")
            calls = _tool_calls_from_assistant(
                content,
                dataset="sft",
                record_id=record_id,
                decision_index=message_index,
            )
            if calls:
                target_count += 1
                rows.extend(calls)
    return record_count, target_count, rows


def collect_toolrl(path: Path) -> tuple[int, int, list[ToolCallOccurrence]]:
    record_count = 0
    target_count = 0
    rows: list[ToolCallOccurrence] = []
    for line_no, record in _read_jsonl(path):
        record_count += 1
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        record_id = str(metadata.get("source_id") or f"line_{line_no}")
        decision_index = int(metadata.get("assistant_index") or 0)
        target = record.get("target_assistant")
        content = target.get("content") if isinstance(target, dict) else None
        if not isinstance(content, str):
            raise ValueError(f"{path}:{line_no}: target_assistant.content must be a string")
        calls = _tool_calls_from_assistant(
            content,
            dataset="toolrl",
            record_id=record_id,
            decision_index=decision_index,
        )
        if calls:
            target_count += 1
            rows.extend(calls)
    return record_count, target_count, rows


def collect_gad(path: Path) -> tuple[int, int, list[ToolCallOccurrence]]:
    record_count = 0
    target_count = 0
    rows: list[ToolCallOccurrence] = []
    for line_no, record in _read_jsonl(path):
        record_count += 1
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        record_id = str(metadata.get("source_id") or metadata.get("sample_id") or f"line_{line_no}")
        decision_index = int(metadata.get("assistant_index") or 0)
        content = record.get("teacher_response")
        if not isinstance(content, str):
            raise ValueError(f"{path}:{line_no}: teacher_response must be a string")
        calls = _tool_calls_from_assistant(
            content,
            dataset="gad",
            record_id=record_id,
            decision_index=decision_index,
        )
        if calls:
            target_count += 1
            rows.extend(calls)
    return record_count, target_count, rows


def _nearest_rank(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_tokens(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {
            "count": 0,
            "total_tokens": 0,
            "mean": 0.0,
            "min": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "gt_4096": 0,
            "gt_8192": 0,
            "gt_16384": 0,
        }
    return {
        "count": len(values),
        "total_tokens": sum(values),
        "mean": round(sum(values) / len(values), 2),
        "min": min(values),
        "p50": _nearest_rank(values, 0.50),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
        "max": max(values),
        "gt_4096": sum(value > 4096 for value in values),
        "gt_8192": sum(value > 8192 for value in values),
        "gt_16384": sum(value > 16384 for value in values),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def generate_report(
    *,
    sft_path: Path,
    toolrl_path: Path,
    gad_path: Path,
    tokenizer_json: Path,
    output_dir: Path,
    tokenizer_name: str,
    count_tokens: Callable[[str], int],
) -> dict[str, Any]:
    sources = {
        "sft": sft_path.resolve(),
        "toolrl": toolrl_path.resolve(),
        "gad": gad_path.resolve(),
    }
    collectors = {
        "sft": collect_sft,
        "toolrl": collect_toolrl,
        "gad": collect_gad,
    }
    records_by_dataset: dict[str, int] = {}
    targets_by_dataset: dict[str, int] = {}
    occurrences: list[ToolCallOccurrence] = []
    for dataset, path in sources.items():
        record_count, target_count, rows = collectors[dataset](path)
        records_by_dataset[dataset] = record_count
        targets_by_dataset[dataset] = target_count
        occurrences.extend(rows)

    occurrence_rows: list[dict[str, Any]] = []
    dataset_tokens: dict[str, list[int]] = defaultdict(list)
    tool_tokens: dict[str, list[int]] = defaultdict(list)
    tool_dataset_counts: dict[str, Counter[str]] = defaultdict(Counter)
    unique_json: dict[str, set[str]] = defaultdict(set)
    for item in occurrences:
        token_count = int(count_tokens(item.json_text))
        json_hash = hashlib.sha256(item.json_text.encode("utf-8")).hexdigest()
        kind = "local" if item.tool_name in LOCAL_TOOL_NAMES else "molclaw"
        occurrence_rows.append(
            {
                "dataset": item.dataset,
                "record_id": item.record_id,
                "decision_index": item.decision_index,
                "call_index": item.call_index,
                "tool_kind": kind,
                "tool_name": item.tool_name,
                "json_chars": len(item.json_text),
                "json_bytes": len(item.json_text.encode("utf-8")),
                "json_tokens": token_count,
                "json_sha256": json_hash,
            }
        )
        dataset_tokens[item.dataset].append(token_count)
        tool_tokens[item.tool_name].append(token_count)
        tool_dataset_counts[item.tool_name][item.dataset] += 1
        unique_json[item.dataset].add(json_hash)

    output_dir.mkdir(parents=True, exist_ok=True)
    occurrence_fields = [
        "dataset",
        "record_id",
        "decision_index",
        "call_index",
        "tool_kind",
        "tool_name",
        "json_chars",
        "json_bytes",
        "json_tokens",
        "json_sha256",
    ]
    _write_csv(output_dir / "tool_call_occurrences.csv", occurrence_rows, occurrence_fields)

    dataset_rows: list[dict[str, Any]] = []
    for dataset in ("sft", "toolrl", "gad"):
        row = {
            "dataset": dataset,
            "records": records_by_dataset[dataset],
            "tool_call_targets": targets_by_dataset[dataset],
            "unique_tool_call_json": len(unique_json[dataset]),
            **summarize_tokens(dataset_tokens[dataset]),
        }
        dataset_rows.append(row)
    combined_values = [int(row["json_tokens"]) for row in occurrence_rows]
    dataset_rows.append(
        {
            "dataset": "all_representations",
            "records": sum(records_by_dataset.values()),
            "tool_call_targets": sum(targets_by_dataset.values()),
            "unique_tool_call_json": len({row["json_sha256"] for row in occurrence_rows}),
            **summarize_tokens(combined_values),
        }
    )
    dataset_fields = [
        "dataset",
        "records",
        "tool_call_targets",
        "unique_tool_call_json",
        "count",
        "total_tokens",
        "mean",
        "min",
        "p50",
        "p90",
        "p95",
        "p99",
        "max",
        "gt_4096",
        "gt_8192",
        "gt_16384",
    ]
    _write_csv(output_dir / "dataset_summary.csv", dataset_rows, dataset_fields)

    tool_rows: list[dict[str, Any]] = []
    for tool_name in sorted(tool_tokens):
        counts = tool_dataset_counts[tool_name]
        tool_rows.append(
            {
                "tool_kind": "local" if tool_name in LOCAL_TOOL_NAMES else "molclaw",
                "tool_name": tool_name,
                "sft_count": counts["sft"],
                "toolrl_count": counts["toolrl"],
                "gad_count": counts["gad"],
                **summarize_tokens(tool_tokens[tool_name]),
            }
        )
    tool_fields = [
        "tool_kind",
        "tool_name",
        "sft_count",
        "toolrl_count",
        "gad_count",
        "count",
        "total_tokens",
        "mean",
        "min",
        "p50",
        "p90",
        "p95",
        "p99",
        "max",
        "gt_4096",
        "gt_8192",
        "gt_16384",
    ]
    _write_csv(output_dir / "tool_summary.csv", tool_rows, tool_fields)

    longest = sorted(occurrence_rows, key=lambda row: (-int(row["json_tokens"]), str(row["dataset"])))[:20]
    manifest = {
        "schema_version": "drug_agent_tool_call_token_stats_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "sft": "all assistant tool-call targets",
            "toolrl": "target_assistant only; prompt history excluded",
            "gad": "teacher_response only; prompt/state history excluded",
            "tokenized_text": "JSON inside each <tool_call> block; XML tags excluded",
            "percentiles": "nearest-rank",
            "all_representations_warning": "ToolRL and GAD are derived from SFT; combined counts are serialized training occurrences, not unique scientific actions.",
        },
        "tokenizer": {
            "name": tokenizer_name,
            "tokenizer_json": str(tokenizer_json.resolve()),
            "sha256": _sha256(tokenizer_json),
            "add_special_tokens": False,
        },
        "sources": {
            dataset: {
                "path": str(path),
                "sha256": _sha256(path),
                "records": records_by_dataset[dataset],
            }
            for dataset, path in sources.items()
        },
        "dataset_summary": dataset_rows,
        "longest_occurrences": longest,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    dataset_md = _markdown_table(
        ["数据表示", "records", "tool-call targets", "JSON calls", "unique JSON", "total tokens", "mean", "min", "P50", "P90", "P95", "P99", "max", ">4096", ">8192", ">16384"],
        [
            [
                row["dataset"],
                row["records"],
                row["tool_call_targets"],
                row["count"],
                row["unique_tool_call_json"],
                row["total_tokens"],
                row["mean"],
                row["min"],
                row["p50"],
                row["p90"],
                row["p95"],
                row["p99"],
                row["max"],
                row["gt_4096"],
                row["gt_8192"],
                row["gt_16384"],
            ]
            for row in dataset_rows
        ],
    )
    tool_md = _markdown_table(
        ["类型", "工具", "SFT", "ToolRL", "GAD", "合计", "mean", "min", "P50", "P90", "P95", "P99", "max", ">4096", ">8192", ">16384"],
        [
            [
                row["tool_kind"],
                row["tool_name"],
                row["sft_count"],
                row["toolrl_count"],
                row["gad_count"],
                row["count"],
                row["mean"],
                row["min"],
                row["p50"],
                row["p90"],
                row["p95"],
                row["p99"],
                row["max"],
                row["gt_4096"],
                row["gt_8192"],
                row["gt_16384"],
            ]
            for row in tool_rows
        ],
    )
    report = f"""# Tool-call JSON token 统计

本表由 `{Path(__file__).name}` 生成。统计对象是当前正式 `live_tool_catalog_v2` 的训练目标：SFT assistant、ToolRL `target_assistant`、GAD `teacher_response`。ToolRL/GAD 的 prompt history 不计入，以免同一历史调用被重复放大。

tokenizer：`{tokenizer_name}`；统计文本仅为 `<tool_call>` 标签内部的 JSON，不包含 XML 标签；`add_special_tokens=false`。分位数采用 nearest-rank。

注意：ToolRL 和 GAD 都从 canonical SFT 派生；`all_representations` 表示实际序列化到三类训练文件中的出现次数，不能解释为互不重复的科学动作数。需要观察原始 canonical 分布时，以 `sft` 行为准。

## 按训练数据表示

{dataset_md}

## 按工具（跨三种训练表示）

{tool_md}

## 配套文件

- `dataset_summary.csv`：按 SFT/ToolRL/GAD 汇总。
- `tool_summary.csv`：按工具名汇总。
- `tool_call_occurrences.csv`：每个实际训练 target 中每个 tool-call JSON 的 token 数与 SHA256，不复制敏感或超长参数正文。
- `manifest.json`：输入路径/hash、tokenizer hash、统计口径和最长 20 个调用定位。
"""
    (output_dir / "TOOL_CALL_TOKEN_STATS.md").write_text(report, encoding="utf-8")
    return manifest


def _default_paths() -> dict[str, Path]:
    wd = Path(__file__).resolve().parents[3]
    root = wd / "outputs" / "slime_drug_agent_data" / "live_tool_catalog_v2"
    return {
        "sft": root / "react_trajectories.jsonl",
        "toolrl": root / "toolrl" / "toolrl_steps.jsonl",
        "gad": root / "gad" / "gad_steps.jsonl",
        "tokenizer_json": wd / "data" / "Qwen3.5-9B" / "tokenizer.json",
        "output_dir": root / "tool_call_token_stats",
    }


def main() -> int:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description="Count Qwen tokens in training-target tool-call JSON payloads.")
    parser.add_argument("--sft", type=Path, default=defaults["sft"])
    parser.add_argument("--toolrl", type=Path, default=defaults["toolrl"])
    parser.add_argument("--gad", type=Path, default=defaults["gad"])
    parser.add_argument("--tokenizer-json", type=Path, default=defaults["tokenizer_json"])
    parser.add_argument("--tokenizer-name", default="Qwen3.5-9B")
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    args = parser.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit(
            "The exact tokenizer is required. Run this command in the Slime training environment "
            "where the `tokenizers` package is installed."
        ) from exc

    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    manifest = generate_report(
        sft_path=args.sft,
        toolrl_path=args.toolrl,
        gad_path=args.gad,
        tokenizer_json=args.tokenizer_json,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        count_tokens=lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids),
    )
    combined = manifest["dataset_summary"][-1]
    print(
        f"wrote {args.output_dir}: calls={combined['count']} "
        f"tokens={combined['total_tokens']} max={combined['max']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
