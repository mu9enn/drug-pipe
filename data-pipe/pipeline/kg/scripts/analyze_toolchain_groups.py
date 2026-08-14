#!/usr/bin/env python3
"""Compare tool-chain consistency and diversity within and across task groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_PATH = Path(__file__).resolve()
BASE_ANALYZER_PATH = SCRIPT_PATH.with_name("analyze_transferred_toolchains.py")
BASE_SPEC = importlib.util.spec_from_file_location("drug_pipe_base_toolchain_analysis", BASE_ANALYZER_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError(f"cannot import shared tool-chain analysis functions from {BASE_ANALYZER_PATH}")
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)

TOOLKG_ORDER = tuple("ABCDEFGH")
MOLBENCH_ORDER = ("MS1_PF", "MS2_AC", "MS3_VS")
GROUP_ORDER = TOOLKG_ORDER + MOLBENCH_ORDER
LOCAL_TOOLS = set(BASE.LOCAL_TOOL_NAMES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return BASE._load_jsonl(path)


def _validate_mapping(mapping: dict[str, Any]) -> tuple[dict[int, str], dict[str, str], list[tuple[str, str]]]:
    if mapping.get("schema_version") != "drug_pipe_toolchain_group_mapping_v1":
        raise ValueError("unsupported tool-chain group mapping schema")
    patterns = mapping.get("toolkg_primary_patterns")
    molbench = mapping.get("molbench_ms_groups")
    mapped = mapping.get("mapped_comparisons")
    if not isinstance(patterns, dict) or tuple(patterns) != TOOLKG_ORDER:
        raise ValueError("Tool-KG mapping must define ordered groups A-H")
    index_to_group: dict[int, str] = {}
    for group, specification in patterns.items():
        indices = specification.get("indices") if isinstance(specification, dict) else None
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"Tool-KG group {group} must contain non-empty indices")
        for value in indices:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"Tool-KG group {group} contains a non-integer index")
            if value in index_to_group:
                raise ValueError(f"Tool-KG index {value} appears in multiple groups")
            index_to_group[value] = group
    if set(index_to_group) != set(range(1, 241)):
        missing = sorted(set(range(1, 241)) - set(index_to_group))
        extra = sorted(set(index_to_group) - set(range(1, 241)))
        raise ValueError(f"Tool-KG mapping must cover exactly 1-240; missing={missing}, extra={extra}")
    if not isinstance(molbench, dict) or tuple(molbench) != MOLBENCH_ORDER:
        raise ValueError("MolBench mapping must define ordered groups MS1_PF, MS2_AC, MS3_VS")
    prefix_to_group: dict[str, str] = {}
    for group, specification in molbench.items():
        prefix = specification.get("trajectory_id_prefix") if isinstance(specification, dict) else None
        if not isinstance(prefix, str) or not prefix:
            raise ValueError(f"MolBench group {group} has no trajectory_id_prefix")
        if prefix in prefix_to_group:
            raise ValueError(f"duplicate MolBench trajectory prefix: {prefix}")
        prefix_to_group[prefix] = group
    comparisons: list[tuple[str, str]] = []
    if not isinstance(mapped, list):
        raise ValueError("mapped_comparisons must be an array")
    for item in mapped:
        left = item.get("toolkg_group") if isinstance(item, dict) else None
        right = item.get("molbench_group") if isinstance(item, dict) else None
        if left not in TOOLKG_ORDER or right not in MOLBENCH_ORDER:
            raise ValueError(f"invalid mapped comparison: {item!r}")
        comparisons.append((left, right))
    return index_to_group, prefix_to_group, comparisons


def _record(
    *,
    record_id: str,
    dataset: str,
    group: str,
    source_index: int | None,
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    if trajectory.get("schema_version") != BASE.TRAJECTORY_SCHEMA:
        raise ValueError(f"{record_id}: unsupported trajectory schema {trajectory.get('schema_version')!r}")
    _, actual, local_hist = BASE._extract_actual_calls(
        trajectory, record_id=record_id, local_tools=LOCAL_TOOLS
    )
    canonical = BASE._run_length_collapse(actual)
    ordered_unique = BASE._ordered_unique(actual)
    return {
        "record_id": record_id,
        "dataset": dataset,
        "group": group,
        "source_index": source_index,
        "actual_chain": actual,
        "canonical_chain": canonical,
        "ordered_unique_chain": ordered_unique,
        "tool_set_signature": sorted(set(actual)),
        "actual_call_count": len(actual),
        "unique_tool_count": len(set(actual)),
        "ignored_local_call_count": sum(local_hist.values()),
        "ignored_local_tool_hist": dict(sorted(local_hist.items())),
    }


def _load_records(
    *,
    toolkg_root: Path,
    molbench_trajectories: Path,
    index_to_group: dict[int, str],
    prefix_to_group: dict[str, str],
) -> list[dict[str, Any]]:
    toolkg_path = toolkg_root / "data/react_trajectories.jsonl"
    metadata_path = toolkg_root / "metadata/toolkg/toolkg_metadata_240.jsonl"
    trajectories = BASE._unique_index(_load_jsonl(toolkg_path), "id", label="Tool-KG trajectory")
    metadata = BASE._unique_index(
        _load_jsonl(metadata_path), "training_record_id", label="Tool-KG metadata"
    )
    if set(trajectories) != set(metadata) or len(trajectories) != 240:
        raise ValueError("Tool-KG trajectories and metadata must form a 240-record ID bijection")
    records: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for record_id, trajectory in trajectories.items():
        index = metadata[record_id].get("training_record_index")
        if not isinstance(index, int) or isinstance(index, bool) or index not in index_to_group:
            raise ValueError(f"{record_id}: invalid training_record_index {index!r}")
        if index in seen_indices:
            raise ValueError(f"duplicate Tool-KG training_record_index: {index}")
        seen_indices.add(index)
        records.append(
            _record(
                record_id=record_id,
                dataset="toolkg",
                group=index_to_group[index],
                source_index=index,
                trajectory=trajectory,
            )
        )
    if seen_indices != set(range(1, 241)):
        raise ValueError("Tool-KG training_record_index must cover exactly 1-240")

    seen_ids: set[str] = set()
    for trajectory in _load_jsonl(molbench_trajectories):
        record_id = trajectory.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("MolBench trajectory has no non-empty id")
        matched = [(prefix, group) for prefix, group in prefix_to_group.items() if record_id.startswith(prefix)]
        if not matched:
            continue
        if len(matched) != 1:
            raise ValueError(f"{record_id}: matches multiple MolBench prefixes")
        if record_id in seen_ids:
            raise ValueError(f"duplicate selected MolBench trajectory id: {record_id}")
        seen_ids.add(record_id)
        records.append(
            _record(
                record_id=record_id,
                dataset="molbench",
                group=matched[0][1],
                source_index=None,
                trajectory=trajectory,
            )
        )
    counts = Counter(record["group"] for record in records)
    if any(counts[group] == 0 for group in GROUP_ORDER):
        raise ValueError(f"one or more analysis groups are empty: {dict(counts)}")
    return sorted(records, key=lambda item: (GROUP_ORDER.index(item["group"]), item["source_index"] or 0, item["record_id"]))


def _pair_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    jaccard_distance = BASE._jaccard_distance(
        set(left["tool_set_signature"]), set(right["tool_set_signature"])
    )
    ordered_edit_distance = 1.0 - BASE._normalized_edit_similarity(
        left["ordered_unique_chain"], right["ordered_unique_chain"]
    )
    canonical_edit_distance = 1.0 - BASE._normalized_edit_similarity(
        left["canonical_chain"], right["canonical_chain"]
    )
    return {
        "tool_set_jaccard_similarity": 1.0 - jaccard_distance,
        "tool_set_jaccard_distance": jaccard_distance,
        "ordered_unique_edit_similarity": 1.0 - ordered_edit_distance,
        "ordered_unique_edit_distance": ordered_edit_distance,
        "canonical_edit_similarity": 1.0 - canonical_edit_distance,
        "canonical_edit_distance": canonical_edit_distance,
    }


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else None


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    return BASE._distribution(list(values))


def _js_similarity(left: Counter[str], right: Counter[str]) -> float:
    tools = set(left) | set(right)
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not tools:
        return 1.0
    if not left_total or not right_total:
        return 0.0
    divergence = 0.0
    for tool in tools:
        p = left[tool] / left_total
        q = right[tool] / right_total
        midpoint = (p + q) / 2
        if p:
            divergence += 0.5 * p * math.log2(p / midpoint)
        if q:
            divergence += 0.5 * q * math.log2(q / midpoint)
    return 1.0 - divergence


def _signature_summary(records: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    counter = Counter(tuple(record[key]) for record in records)
    diversity = BASE._diversity(counter)
    return {
        **diversity,
        "unique_signature_count": len(counter),
        "uniqueness_ratio": len(counter) / len(records) if records else 0.0,
        "singleton_signature_count": sum(count == 1 for count in counter.values()),
        "top_signature_share": max(counter.values(), default=0) / len(records) if records else 0.0,
    }


def _within_group(
    group: str, records: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(records, 2):
        pair_rows.append(
            {
                "group": group,
                "record_id_a": left["record_id"],
                "record_id_b": right["record_id"],
                **_pair_metrics(left, right),
            }
        )
    calls = Counter(tool for record in records for tool in record["actual_chain"])
    summary = {
        "group": group,
        "dataset": records[0]["dataset"],
        "sample_count": len(records),
        "pair_count": len(pair_rows),
        "actual_call_count": sum(record["actual_call_count"] for record in records),
        "tool_vocabulary_size": len(calls),
        "chain_length": _distribution(record["actual_call_count"] for record in records),
        "unique_tool_count": _distribution(record["unique_tool_count"] for record in records),
        "tool_call_diversity": BASE._diversity(calls),
        "raw_signature": _signature_summary(records, "actual_chain"),
        "canonical_signature": _signature_summary(records, "canonical_chain"),
        "ordered_unique_signature": _signature_summary(records, "ordered_unique_chain"),
        "tool_set_signature": _signature_summary(records, "tool_set_signature"),
        "mean_pairwise_tool_set_similarity": _mean(
            row["tool_set_jaccard_similarity"] for row in pair_rows
        ),
        "mean_pairwise_tool_set_distance": _mean(
            row["tool_set_jaccard_distance"] for row in pair_rows
        ),
        "mean_pairwise_ordered_similarity": _mean(
            row["ordered_unique_edit_similarity"] for row in pair_rows
        ),
        "mean_pairwise_ordered_distance": _mean(
            row["ordered_unique_edit_distance"] for row in pair_rows
        ),
        "mean_pairwise_canonical_similarity": _mean(
            row["canonical_edit_similarity"] for row in pair_rows
        ),
    }
    return summary, pair_rows


def _compare_groups(
    group_a: str,
    records_a: Sequence[dict[str, Any]],
    group_b: str,
    records_b: Sequence[dict[str, Any]],
    *,
    include_pairs: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    same_group = group_a == group_b
    pairs = itertools.combinations(records_a, 2) if same_group else itertools.product(records_a, records_b)
    pair_rows: list[dict[str, Any]] = []
    metric_values: dict[str, list[float]] = {
        "tool_set_jaccard_similarity": [],
        "tool_set_jaccard_distance": [],
        "ordered_unique_edit_similarity": [],
        "ordered_unique_edit_distance": [],
        "canonical_edit_similarity": [],
        "canonical_edit_distance": [],
    }
    nearest_a: dict[str, float] = {record["record_id"]: 0.0 for record in records_a}
    nearest_b: dict[str, float] = {record["record_id"]: 0.0 for record in records_b}
    exact_tool_set_a: set[str] = set()
    exact_tool_set_b: set[str] = set()
    for left, right in pairs:
        metrics = _pair_metrics(left, right)
        for name, value in metrics.items():
            metric_values[name].append(value)
        similarity = metrics["tool_set_jaccard_similarity"]
        nearest_a[left["record_id"]] = max(nearest_a[left["record_id"]], similarity)
        nearest_b[right["record_id"]] = max(nearest_b[right["record_id"]], similarity)
        if left["tool_set_signature"] == right["tool_set_signature"]:
            exact_tool_set_a.add(left["record_id"])
            exact_tool_set_b.add(right["record_id"])
        if include_pairs:
            pair_rows.append(
                {
                    "group_a": group_a,
                    "group_b": group_b,
                    "record_id_a": left["record_id"],
                    "record_id_b": right["record_id"],
                    **metrics,
                }
            )
    calls_a = Counter(tool for record in records_a for tool in record["actual_chain"])
    calls_b = Counter(tool for record in records_b for tool in record["actual_chain"])
    vocabulary_a = set(calls_a)
    vocabulary_b = set(calls_b)
    union = vocabulary_a | vocabulary_b
    intersection = vocabulary_a & vocabulary_b
    summary = {
        "group_a": group_a,
        "group_b": group_b,
        "dataset_a": records_a[0]["dataset"],
        "dataset_b": records_b[0]["dataset"],
        "sample_count_a": len(records_a),
        "sample_count_b": len(records_b),
        "pair_count": len(metric_values["tool_set_jaccard_similarity"]),
        "mean_cross_tool_set_similarity": _mean(metric_values["tool_set_jaccard_similarity"]),
        "mean_cross_tool_set_distance": _mean(metric_values["tool_set_jaccard_distance"]),
        "mean_cross_ordered_similarity": _mean(metric_values["ordered_unique_edit_similarity"]),
        "mean_cross_ordered_distance": _mean(metric_values["ordered_unique_edit_distance"]),
        "mean_cross_canonical_similarity": _mean(metric_values["canonical_edit_similarity"]),
        "mean_cross_canonical_distance": _mean(metric_values["canonical_edit_distance"]),
        "mean_nearest_neighbor_similarity_a_to_b": _mean(nearest_a.values()),
        "mean_nearest_neighbor_similarity_b_to_a": _mean(nearest_b.values()),
        "symmetric_nearest_neighbor_similarity": _mean(
            [*_non_null([_mean(nearest_a.values())]), *_non_null([_mean(nearest_b.values())])]
        ),
        "vocabulary_size_a": len(vocabulary_a),
        "vocabulary_size_b": len(vocabulary_b),
        "vocabulary_intersection_size": len(intersection),
        "vocabulary_jaccard_similarity": len(intersection) / len(union) if union else 1.0,
        "vocabulary_a_covered_by_b": len(intersection) / len(vocabulary_a) if vocabulary_a else 1.0,
        "vocabulary_b_covered_by_a": len(intersection) / len(vocabulary_b) if vocabulary_b else 1.0,
        "tool_call_js_similarity": _js_similarity(calls_a, calls_b),
        "exact_tool_set_match_rate_a": len(exact_tool_set_a) / len(records_a) if records_a else 0.0,
        "exact_tool_set_match_rate_b": len(exact_tool_set_b) / len(records_b) if records_b else 0.0,
    }
    return summary, pair_rows


def _non_null(values: Iterable[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def _flatten_within(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "group": summary["group"],
        "dataset": summary["dataset"],
        "sample_count": summary["sample_count"],
        "pair_count": summary["pair_count"],
        "actual_call_count": summary["actual_call_count"],
        "tool_vocabulary_size": summary["tool_vocabulary_size"],
        "chain_length_mean": summary["chain_length"]["mean"],
        "chain_length_p50": summary["chain_length"]["p50"],
        "chain_length_p95": summary["chain_length"]["p95"],
        "tool_call_shannon_entropy": summary["tool_call_diversity"]["shannon_entropy"],
        "tool_call_effective_shannon": summary["tool_call_diversity"]["effective_shannon"],
        "raw_unique_signature_count": summary["raw_signature"]["unique_signature_count"],
        "raw_uniqueness_ratio": summary["raw_signature"]["uniqueness_ratio"],
        "tool_set_unique_signature_count": summary["tool_set_signature"]["unique_signature_count"],
        "tool_set_uniqueness_ratio": summary["tool_set_signature"]["uniqueness_ratio"],
        "mean_pairwise_tool_set_similarity": summary["mean_pairwise_tool_set_similarity"],
        "mean_pairwise_tool_set_distance": summary["mean_pairwise_tool_set_distance"],
        "mean_pairwise_ordered_similarity": summary["mean_pairwise_ordered_similarity"],
        "mean_pairwise_ordered_distance": summary["mean_pairwise_ordered_distance"],
        "mean_pairwise_canonical_similarity": summary["mean_pairwise_canonical_similarity"],
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    status = run("status", "--short").splitlines()
    return {
        "repo_root": str(repo_root),
        "commit": run("rev-parse", "HEAD") or None,
        "dirty": bool(status),
        "status_entry_count": len(status),
    }


def _plots(
    output_dir: Path,
    records_by_group: dict[str, list[dict[str, Any]]],
    within: dict[str, dict[str, Any]],
    all_comparisons: list[dict[str, Any]],
    mapped: list[dict[str, Any]],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required unless --skip-plots is used") from exc
    figures = output_dir / "figures"
    figures.mkdir()
    paths: list[str] = []

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].bar(GROUP_ORDER, [within[group]["mean_pairwise_tool_set_similarity"] for group in GROUP_ORDER])
    axes[0].set_ylabel("Mean similarity")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Within-group tool-set consistency")
    axes[1].bar(GROUP_ORDER, [within[group]["mean_pairwise_tool_set_distance"] for group in GROUP_ORDER])
    axes[1].set_ylabel("Mean distance")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Within-group tool-set diversity")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = figures / "within_group_consistency_diversity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(output_dir)))

    lookup = {
        tuple(sorted((row["group_a"], row["group_b"]), key=GROUP_ORDER.index)): row
        for row in all_comparisons
    }
    matrix: list[list[float]] = []
    for left in GROUP_ORDER:
        values: list[float] = []
        for right in GROUP_ORDER:
            key = tuple(sorted((left, right), key=GROUP_ORDER.index))
            values.append(float(lookup[key]["mean_cross_tool_set_similarity"]))
        matrix.append(values)
    fig, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(GROUP_ORDER)), GROUP_ORDER, rotation=35, ha="right")
    axis.set_yticks(range(len(GROUP_ORDER)), GROUP_ORDER)
    axis.set_title("Mean pairwise tool-set similarity")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    path = figures / "all_group_similarity_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(output_dir)))

    labels = [f"{row['group_a']}↔{row['group_b']}" for row in mapped]
    metric_names = (
        "mean_cross_tool_set_similarity",
        "symmetric_nearest_neighbor_similarity",
        "vocabulary_jaccard_similarity",
        "tool_call_js_similarity",
    )
    offsets = (-0.27, -0.09, 0.09, 0.27)
    fig, axis = plt.subplots(figsize=(12, 6))
    x_values = list(range(len(labels)))
    for metric, offset in zip(metric_names, offsets):
        axis.bar([value + offset for value in x_values], [row[metric] for row in mapped], width=0.18, label=metric)
    axis.set_xticks(x_values, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Similarity")
    axis.set_title("Mapped Tool-KG vs MolBench comparisons")
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = figures / "mapped_comparison_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(output_dir)))

    all_calls = Counter(tool for records in records_by_group.values() for record in records for tool in record["actual_chain"])
    top_tools = [tool for tool, _ in all_calls.most_common(30)]
    heatmap: list[list[float]] = []
    for group in GROUP_ORDER:
        calls = Counter(tool for record in records_by_group[group] for tool in record["actual_chain"])
        total = sum(calls.values())
        heatmap.append([calls[tool] / total if total else 0.0 for tool in top_tools])
    fig, axis = plt.subplots(figsize=(18, 7))
    image = axis.imshow(heatmap, aspect="auto", cmap="magma")
    axis.set_xticks(range(len(top_tools)), top_tools, rotation=65, ha="right", fontsize=7)
    axis.set_yticks(range(len(GROUP_ORDER)), GROUP_ORDER)
    axis.set_title("Per-group call-share profiles for corpus top-30 tools")
    fig.colorbar(image, ax=axis, label="Call share")
    fig.tight_layout()
    path = figures / "group_tool_profile_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(output_dir)))
    return paths


def _report(
    *,
    generated_at: str,
    within: dict[str, dict[str, Any]],
    mapped: list[dict[str, Any]],
    all_comparisons: list[dict[str, Any]],
    mapping: dict[str, Any],
    figures: Sequence[str],
) -> str:
    pattern_names = mapping["toolkg_primary_patterns"]
    molbench_names = mapping["molbench_ms_groups"]
    toolkg_between = [
        row
        for row in all_comparisons
        if row["group_a"] in TOOLKG_ORDER
        and row["group_b"] in TOOLKG_ORDER
        and row["group_a"] != row["group_b"]
    ]
    molbench_between = [
        row
        for row in all_comparisons
        if row["group_a"] in MOLBENCH_ORDER
        and row["group_b"] in MOLBENCH_ORDER
        and row["group_a"] != row["group_b"]
    ]
    closest_toolkg = max(toolkg_between, key=lambda row: row["mean_cross_tool_set_similarity"])
    closest_molbench = max(molbench_between, key=lambda row: row["mean_cross_tool_set_similarity"])
    most_consistent_toolkg = max(
        TOOLKG_ORDER, key=lambda group: within[group]["mean_pairwise_tool_set_similarity"]
    )
    most_diverse_toolkg = max(
        TOOLKG_ORDER, key=lambda group: within[group]["mean_pairwise_tool_set_distance"]
    )
    lines = [
        "# Tool-KG 主分类与 MolBench MS1/2/3 Tool-Chain 对比分析",
        "",
        f"- 生成时间（UTC）：`{generated_at}`",
        "- Tool-chain 仅使用 `<tool_call>`；忽略 `Read/Write/Edit/Bash/Grep/Glob`。",
        "- 每个 benchmark 问题当前只有一条轨迹，因此“MS 内部一致性”指同一 MS 类型不同问题之间的轨迹相似度。",
        "",
        "> 组内一致性使用平均 pairwise similarity；diversity 使用与其互补的 pairwise distance。"
        "跨数据集没有逐题配对，因此报告全 cross-pair 与双向 nearest-neighbor similarity，而不使用 expected recall/precision。",
        "",
        "## 组内一致性与 Diversity",
        "",
        "| Group | Description | N | Tool-set similarity | Jaccard distance | Ordered similarity | Tool-set uniqueness |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in GROUP_ORDER:
        specification = pattern_names[group] if group in TOOLKG_ORDER else molbench_names[group]
        item = within[group]
        lines.append(
            f"| `{group}` | {specification['name']} | {item['sample_count']} | "
            f"{item['mean_pairwise_tool_set_similarity']:.4f} | {item['mean_pairwise_tool_set_distance']:.4f} | "
            f"{item['mean_pairwise_ordered_similarity']:.4f} | {item['tool_set_signature']['uniqueness_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Tool-KG 与对应 MolBench 类型",
            "",
            "| Mapping | Cross similarity | Symmetric nearest neighbor | Vocab Jaccard | Call-distribution JS similarity |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in mapped:
        lines.append(
            f"| `{item['group_a']}↔{item['group_b']}` | {item['mean_cross_tool_set_similarity']:.4f} | "
            f"{item['symmetric_nearest_neighbor_similarity']:.4f} | {item['vocabulary_jaccard_similarity']:.4f} | "
            f"{item['tool_call_js_similarity']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 主要观察",
            "",
            f"- Tool-KG A–H 中，`{most_consistent_toolkg}` 的组内 tool-set similarity 最高"
            f"（{within[most_consistent_toolkg]['mean_pairwise_tool_set_similarity']:.4f}）；"
            f"`{most_diverse_toolkg}` 的组内 Jaccard distance 最高"
            f"（{within[most_diverse_toolkg]['mean_pairwise_tool_set_distance']:.4f}）。",
            f"- A–H 不同类别间最接近的是 `{closest_toolkg['group_a']}↔{closest_toolkg['group_b']}`，"
            f"平均 tool-set similarity={closest_toolkg['mean_cross_tool_set_similarity']:.4f}；"
            "完整类别间关系见热图和 `all_group_pair_comparisons.csv`。",
            f"- MolBench 内部 `MS1_PF` 最一致（{within['MS1_PF']['mean_pairwise_tool_set_similarity']:.4f}），"
            f"`MS2_AC` 最多样（distance={within['MS2_AC']['mean_pairwise_tool_set_distance']:.4f}）；"
            f"不同 MS 类型间最接近的是 `{closest_molbench['group_a']}↔{closest_molbench['group_b']}`"
            f"（similarity={closest_molbench['mean_cross_tool_set_similarity']:.4f}）。",
            f"- `F↔MS1_PF` 的全 cross similarity 仅 {mapped[0]['mean_cross_tool_set_similarity']:.4f}，"
            f"但 MS1 的工具词表被 F 覆盖 {mapped[0]['vocabulary_b_covered_by_a']:.4f}；"
            "这表示主目标相似，但 Tool-KG F 的执行方式更宽。",
            f"- `B↔MS2_AC` 的调用分布最接近（JS similarity={mapped[1]['tool_call_js_similarity']:.4f}）；"
            f"`B↔MS3_VS` 虽覆盖 MS3 全部工具词表（{mapped[2]['vocabulary_b_covered_by_a']:.4f}），"
            f"调用分布相似度仅 {mapped[2]['tool_call_js_similarity']:.4f}，主要反映 VS 中 docking 重复调用占主导。",
            "",
            "### 指标解释",
            "",
            "- Cross similarity：两组所有跨组轨迹对的平均 tool-set Jaccard similarity；越高表示整体执行工具更接近。",
            "- Symmetric nearest neighbor：每条轨迹在另一组找最相似轨迹，再对两个方向平均；衡量是否存在对应执行模式。",
            "- Vocab Jaccard：两组整体工具词表的交并比；只说明覆盖范围，不说明使用频率。",
            "- JS similarity：`1 - Jensen-Shannon divergence(base=2)`；越高表示工具调用频率分布越接近。",
            "- Ordered similarity：首次出现有序链的 normalized edit similarity；同时考虑工具选择和首次出现顺序。",
            "",
            "## 边界",
            "",
            "- `B` 同时映射 MS2 和 MS3，因为二者都属于已知候选比较/筛选，但执行工具可能明显不同。",
            "- A、D、E、H 没有 standalone MolBench-MS 对照；它们仍包含在 A–H 内部和组间矩阵中。",
            "- 指标描述轨迹结构，不直接判断科学正确性或额外工具是否必要。",
        ]
    )
    if figures:
        lines.extend(["", "## Figures", ""])
        for figure in figures:
            lines.extend([f"![{Path(figure).stem}]({figure})", ""])
    return "\n".join(lines) + "\n"


def analyze(
    *,
    toolkg_root: Path,
    molbench_trajectories: Path,
    mapping_path: Path,
    output_dir: Path,
    skip_plots: bool,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    mapping = _load_json(mapping_path)
    index_to_group, prefix_to_group, mapped_specs = _validate_mapping(mapping)
    input_paths = [
        toolkg_root / "data/react_trajectories.jsonl",
        toolkg_root / "metadata/toolkg/toolkg_metadata_240.jsonl",
        molbench_trajectories,
        mapping_path,
        BASE_ANALYZER_PATH,
    ]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    input_hashes = {str(path): _sha256(path) for path in input_paths}
    records = _load_records(
        toolkg_root=toolkg_root,
        molbench_trajectories=molbench_trajectories,
        index_to_group=index_to_group,
        prefix_to_group=prefix_to_group,
    )
    records_by_group = {
        group: [record for record in records if record["group"] == group] for group in GROUP_ORDER
    }
    within: dict[str, dict[str, Any]] = {}
    within_pairs: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        within[group], rows = _within_group(group, records_by_group[group])
        within_pairs.extend(rows)

    all_comparisons: list[dict[str, Any]] = []
    for left_index, group_a in enumerate(GROUP_ORDER):
        for group_b in GROUP_ORDER[left_index:]:
            comparison, _ = _compare_groups(
                group_a,
                records_by_group[group_a],
                group_b,
                records_by_group[group_b],
                include_pairs=False,
            )
            all_comparisons.append(comparison)
    mapped: list[dict[str, Any]] = []
    mapped_pairs: list[dict[str, Any]] = []
    for group_a, group_b in mapped_specs:
        comparison, rows = _compare_groups(
            group_a,
            records_by_group[group_a],
            group_b,
            records_by_group[group_b],
            include_pairs=True,
        )
        mapped.append(comparison)
        mapped_pairs.extend(rows)

    tool_rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        group_records = records_by_group[group]
        calls = Counter(tool for record in group_records for tool in record["actual_chain"])
        documents = Counter(tool for record in group_records for tool in set(record["actual_chain"]))
        for tool, count in sorted(calls.items(), key=lambda item: (-item[1], item[0])):
            tool_rows.append(
                {
                    "group": group,
                    "dataset": group_records[0]["dataset"],
                    "tool_name": tool,
                    "call_count": count,
                    "call_share": count / sum(calls.values()),
                    "sample_count": documents[tool],
                    "sample_coverage": documents[tool] / len(group_records),
                }
            )

    generated_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema_version": "drug_pipe_toolchain_group_analysis_v1",
        "generated_at_utc": generated_at,
        "record_count": len(records),
        "dataset_counts": dict(Counter(record["dataset"] for record in records)),
        "group_counts": {group: len(records_by_group[group]) for group in GROUP_ORDER},
        "within_group": within,
        "mapped_comparisons": mapped,
        "all_group_pair_comparisons": all_comparisons,
        "method": {
            "ignored_local_tools": list(BASE.LOCAL_TOOL_NAMES),
            "tool_name_matching": "exact_after_mcp_prefix_removal",
            "within_consistency": "mean_pairwise_similarity",
            "within_diversity": "mean_pairwise_distance",
            "cross_consistency": "all-cross-pair and bidirectional-nearest-neighbor similarity",
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp_", dir=output_dir.parent))
    try:
        with (staging / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_csv(staging / "within_group_summary.csv", [_flatten_within(within[group]) for group in GROUP_ORDER])
        _write_csv(staging / "within_group_pairwise.csv", within_pairs)
        _write_csv(staging / "all_group_pair_comparisons.csv", all_comparisons)
        _write_csv(staging / "mapped_comparisons.csv", mapped)
        _write_csv(staging / "mapped_pairwise.csv", mapped_pairs)
        _write_csv(staging / "tool_usage_by_group.csv", tool_rows)
        figures = [] if skip_plots else _plots(staging, records_by_group, within, all_comparisons, mapped)
        (staging / "report.md").write_text(
            _report(
                generated_at=generated_at,
                within=within,
                mapped=mapped,
                all_comparisons=all_comparisons,
                mapping=mapping,
                figures=figures,
            ),
            encoding="utf-8",
        )
        output_hashes = {
            str(path.relative_to(staging)): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        repo_root = SCRIPT_PATH.parents[4]
        manifest = {
            "schema_version": "drug_pipe_toolchain_group_analysis_manifest_v1",
            "generated_at_utc": generated_at,
            "parameters": {
                "toolkg_root": str(toolkg_root),
                "molbench_trajectories": str(molbench_trajectories),
                "mapping": str(mapping_path),
                "output_dir": str(output_dir),
                "skip_plots": skip_plots,
            },
            "input_sha256": input_hashes,
            "analyzer": {
                "path": str(SCRIPT_PATH),
                "sha256": _sha256(SCRIPT_PATH),
                "shared_analyzer_path": str(BASE_ANALYZER_PATH),
                "shared_analyzer_sha256": _sha256(BASE_ANALYZER_PATH),
                "git": _git_provenance(repo_root),
            },
            "record_count": len(records),
            "output_sha256": output_hashes,
            "notes": ["manifest.json is excluded from output_sha256", "source inputs are read-only"],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if {str(path): _sha256(path) for path in input_paths} != input_hashes:
            raise RuntimeError("an input changed while analysis was running")
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output_dir": str(output_dir),
        "record_count": len(records),
        "group_counts": summary["group_counts"],
        "within_pair_count": len(within_pairs),
        "mapped_pair_count": len(mapped_pairs),
        "plots_generated": not skip_plots,
    }


def main() -> int:
    repo_root = SCRIPT_PATH.parents[4]
    default_toolkg = Path(
        "/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug_wd/drug_pipe_240_transfer_20260811"
    )
    default_molbench = (
        repo_root
        / "slime-wd/outputs/slime_drug_agent_data/live_tool_catalog_v4/react_trajectories.jsonl"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkg-root", type=Path, default=default_toolkg)
    parser.add_argument("--molbench-trajectories", type=Path, default=default_molbench)
    parser.add_argument(
        "--mapping", type=Path, default=SCRIPT_PATH.parents[1] / "toolchain_group_mapping_v1.json"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    arguments = parser.parse_args()
    toolkg_root = arguments.toolkg_root.expanduser().resolve()
    output_dir = (
        arguments.output_dir.expanduser().resolve()
        if arguments.output_dir
        else toolkg_root / "documentation/toolchain_group_comparison"
    )
    result = analyze(
        toolkg_root=toolkg_root,
        molbench_trajectories=arguments.molbench_trajectories.expanduser().resolve(),
        mapping_path=arguments.mapping.expanduser().resolve(),
        output_dir=output_dir,
        skip_plots=arguments.skip_plots,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
