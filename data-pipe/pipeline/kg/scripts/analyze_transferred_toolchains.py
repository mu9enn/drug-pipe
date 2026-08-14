#!/usr/bin/env python3
"""Analyze expected and executed tool chains in a transferred Drug-Pipe corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


TOOL_CALL_RE = re.compile(r"<tool_call>([\s\S]*?)</tool_call>")
LOCAL_TOOL_NAMES = ("Read", "Write", "Edit", "Bash", "Grep", "Glob")
BATCH_ORDER = ("old_100", "historical_40", "new_100")
ALIGNMENT_CLASSES = (
    "exact",
    "expected_subsequence_with_extras",
    "full_coverage_order_deviation",
    "partial_overlap",
    "no_overlap",
)
SIGNATURE_VIEWS = ("raw", "canonical", "ordered_unique", "tool_set")
TRAJECTORY_SCHEMA = "drug_agent_sft_react_json_v1"
METADATA_SCHEMA = "drug_pipe_toolkg_metadata_link_v1"
METADATA_MANIFEST_SCHEMA = "drug_pipe_toolkg_metadata_manifest_v1"
EXPECTED_TRAJECTORY_SCHEMA = "trajectory_v2_graph"


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL record is not an object")
            records.append(value)
    return records


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON value is not an object")
    return value


def _validate_input_schemas(
    trajectories: Sequence[dict[str, Any]],
    metadata: Sequence[dict[str, Any]],
    metadata_manifest: dict[str, Any],
) -> None:
    for position, record in enumerate(trajectories, 1):
        if record.get("schema_version") != TRAJECTORY_SCHEMA:
            raise ValueError(
                f"trajectory record {position}: expected schema_version {TRAJECTORY_SCHEMA!r}, "
                f"found {record.get('schema_version')!r}"
            )
    for position, record in enumerate(metadata, 1):
        if record.get("schema_version") != METADATA_SCHEMA:
            raise ValueError(
                f"metadata record {position}: expected schema_version {METADATA_SCHEMA!r}, "
                f"found {record.get('schema_version')!r}"
            )
    if metadata_manifest.get("schema_version") != METADATA_MANIFEST_SCHEMA:
        raise ValueError(
            f"metadata manifest: expected schema_version {METADATA_MANIFEST_SCHEMA!r}, "
            f"found {metadata_manifest.get('schema_version')!r}"
        )
    if metadata_manifest.get("record_count") != len(metadata):
        raise ValueError(
            "metadata manifest record_count differs from metadata JSONL count: "
            f"{metadata_manifest.get('record_count')!r} != {len(metadata)}"
        )


def _normalize_tool_name(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: tool name must be a non-empty string")
    name = value.strip()
    if name.startswith("mcp__") and "__" in name:
        name = name.rsplit("__", 1)[-1]
    if not name:
        raise ValueError(f"{context}: tool name is empty after MCP-prefix normalization")
    return name


def _unique_index(records: Iterable[dict[str, Any]], key: str, *, label: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records, 1):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} record {position}: missing non-empty {key}")
        value = value.strip()
        if value in index:
            raise ValueError(f"{label}: duplicate {key}: {value}")
        index[value] = record
    return index


def _extract_actual_calls(
    trajectory: dict[str, Any],
    *,
    record_id: str,
    local_tools: set[str],
) -> tuple[list[str], list[str], Counter[str]]:
    messages = trajectory.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{record_id}: messages must be an array")
    all_calls: list[str] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{record_id}: message {message_index} is not an object")
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"{record_id}: assistant message {message_index} content is not a string")
        for call_index, match in enumerate(TOOL_CALL_RE.finditer(content)):
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{record_id}: assistant message {message_index} tool call {call_index} "
                    f"contains invalid JSON: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{record_id}: assistant message {message_index} tool call {call_index} is not an object"
                )
            all_calls.append(
                _normalize_tool_name(
                    payload.get("tool_name"),
                    context=f"{record_id}: assistant message {message_index} tool call {call_index}",
                )
            )
    local_hist = Counter(name for name in all_calls if name in local_tools)
    domain_calls = [name for name in all_calls if name not in local_tools]
    return all_calls, domain_calls, local_hist


def _extract_expected(
    metadata: dict[str, Any],
    *,
    record_id: str,
) -> tuple[list[str], list[dict[str, str]], str, str | None]:
    source_task = metadata.get("source_task")
    if not isinstance(source_task, dict):
        raise ValueError(f"{record_id}: source_task must be an object")
    toolchain = source_task.get("toolchain")
    if not isinstance(toolchain, dict):
        raise ValueError(f"{record_id}: source_task.toolchain must be an object")
    raw_tools = toolchain.get("tools")
    if not isinstance(raw_tools, list) or not raw_tools:
        raise ValueError(f"{record_id}: source_task.toolchain.tools must be a non-empty array")
    expected = [
        _normalize_tool_name(name, context=f"{record_id}: expected tool {index}")
        for index, name in enumerate(raw_tools)
    ]

    trajectory = source_task.get("expected_trajectory")
    if not isinstance(trajectory, dict) or trajectory.get("schema_version") != EXPECTED_TRAJECTORY_SCHEMA:
        found = trajectory.get("schema_version") if isinstance(trajectory, dict) else None
        raise ValueError(
            f"{record_id}: expected_trajectory.schema_version must be "
            f"{EXPECTED_TRAJECTORY_SCHEMA!r}, found {found!r}"
        )
    plan = trajectory.get("execution_plan") if isinstance(trajectory, dict) else None
    raw_order = plan.get("tool_order") if isinstance(plan, dict) else None
    if not isinstance(raw_order, list):
        raise ValueError(f"{record_id}: expected_trajectory.execution_plan.tool_order is missing")
    plan_order = [
        _normalize_tool_name(name, context=f"{record_id}: execution-plan tool {index}")
        for index, name in enumerate(raw_order)
    ]
    if plan_order != expected:
        raise ValueError(f"{record_id}: toolchain.tools differs from expected execution-plan tool_order")

    raw_edges = toolchain.get("edges")
    if not isinstance(raw_edges, list):
        raise ValueError(f"{record_id}: source_task.toolchain.edges must be an array")
    edges: list[dict[str, str]] = []
    for index, edge in enumerate(raw_edges):
        if not isinstance(edge, dict):
            raise ValueError(f"{record_id}: expected edge {index} is not an object")
        source = _normalize_tool_name(edge.get("source_tool"), context=f"{record_id}: edge {index} source")
        target = _normalize_tool_name(edge.get("target_tool"), context=f"{record_id}: edge {index} target")
        edge_type = edge.get("edge_type")
        if not isinstance(edge_type, str) or not edge_type.strip():
            raise ValueError(f"{record_id}: expected edge {index} has no edge_type")
        edges.append({"source_tool": source, "target_tool": target, "edge_type": edge_type.strip()})
    expected_pairs = list(zip(expected, expected[1:]))
    actual_pairs = [(edge["source_tool"], edge["target_tool"]) for edge in edges]
    if actual_pairs != expected_pairs:
        raise ValueError(f"{record_id}: toolchain.edges does not describe the linear toolchain.tools sequence")
    if toolchain.get("hops") != len(expected) - 1:
        raise ValueError(f"{record_id}: toolchain.hops differs from len(toolchain.tools) - 1")

    batch = metadata.get("source_batch")
    if not isinstance(batch, str) or not batch.strip():
        raise ValueError(f"{record_id}: source_batch must be a non-empty string")
    task_id = source_task.get("task_id")
    if task_id is not None and not isinstance(task_id, str):
        raise ValueError(f"{record_id}: source_task.task_id must be a string or null")
    return expected, edges, batch.strip(), task_id


def _run_length_collapse(sequence: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in sequence:
        if not result or value != result[-1]:
            result.append(value)
    return result


def _ordered_unique(sequence: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in sequence:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_value in left:
        current = [0]
        for index, right_value in enumerate(right, 1):
            if left_value == right_value:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _levenshtein_distance(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for right_index, right_value in enumerate(right, 1):
        current = [right_index]
        for left_index, left_value in enumerate(left, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[left_index] + 1,
                    previous[left_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _normalized_edit_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 1.0
    return 1.0 - (_levenshtein_distance(left, right) / denominator)


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return 1.0 - (len(left & right) / len(union))


def _alignment_class(expected: Sequence[str], actual: Sequence[str], lcs_length: int) -> str:
    expected_set = set(expected)
    actual_set = set(actual)
    if list(expected) == list(actual):
        return "exact"
    if lcs_length == len(expected):
        return "expected_subsequence_with_extras"
    if expected_set <= actual_set:
        return "full_coverage_order_deviation"
    if expected_set & actual_set:
        return "partial_overlap"
    return "no_overlap"


def _edge_preservation(
    actual: Sequence[str],
    canonical: Sequence[str],
    edges: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, tool in enumerate(actual):
        positions[tool].append(index)
    canonical_pairs = set(zip(canonical, canonical[1:]))
    results: list[dict[str, Any]] = []
    for edge in edges:
        source = edge["source_tool"]
        target = edge["target_tool"]
        source_positions = positions.get(source, [])
        target_positions = positions.get(target, [])
        precedence = bool(
            source_positions and target_positions and min(source_positions) < max(target_positions)
        )
        results.append(
            {
                **edge,
                "precedence_preserved": precedence,
                "canonical_adjacency_preserved": (source, target) in canonical_pairs,
            }
        )
    return results


def _analyze_record(
    trajectory: dict[str, Any],
    metadata: dict[str, Any],
    *,
    record_id: str,
    local_tools: set[str],
    catalog_tools: set[str],
) -> dict[str, Any]:
    all_calls, actual, local_hist = _extract_actual_calls(
        trajectory, record_id=record_id, local_tools=local_tools
    )
    expected, edges, batch, task_id = _extract_expected(metadata, record_id=record_id)
    canonical = _run_length_collapse(actual)
    ordered_unique = _ordered_unique(actual)
    tool_set = sorted(set(actual))
    expected_set = set(expected)
    actual_set = set(actual)
    overlap = expected_set & actual_set
    recall = len(overlap) / len(expected_set) if expected_set else 1.0
    precision = len(overlap) / len(actual_set) if actual_set else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    union = expected_set | actual_set
    jaccard = len(overlap) / len(union) if union else 1.0
    lcs_length = _lcs_length(expected, actual)
    edge_results = _edge_preservation(actual, canonical, edges)
    edge_count = len(edge_results)
    precedence_count = sum(bool(edge["precedence_preserved"]) for edge in edge_results)
    adjacency_count = sum(bool(edge["canonical_adjacency_preserved"]) for edge in edge_results)
    actual_counts = Counter(actual)
    expected_counts = Counter(expected)
    repeated_expected_calls = sum(
        max(0, actual_counts[name] - expected_counts[name]) for name in expected_set
    )
    question_match = metadata.get("question_match")

    return {
        "training_record_id": record_id,
        "training_record_index": metadata.get("training_record_index"),
        "source_batch": batch,
        "source_task_id": task_id,
        "question_exact_match": (
            bool(question_match.get("exact")) if isinstance(question_match, dict) else None
        ),
        "all_actual_calls": all_calls,
        "ignored_local_call_count": sum(local_hist.values()),
        "ignored_local_tool_hist": dict(sorted(local_hist.items())),
        "expected_chain": expected,
        "actual_chain": actual,
        "canonical_chain": canonical,
        "ordered_unique_chain": ordered_unique,
        "tool_set_signature": tool_set,
        "expected_tool_count": len(expected),
        "actual_call_count": len(actual),
        "canonical_call_count": len(canonical),
        "actual_unique_tool_count": len(actual_set),
        "missing_expected_tools": sorted(expected_set - actual_set),
        "unexpected_actual_tools": sorted(actual_set - expected_set),
        "unexpected_actual_call_count": sum(name not in expected_set for name in actual),
        "repeated_expected_call_count": repeated_expected_calls,
        "unknown_catalog_tools": sorted(actual_set - catalog_tools),
        "expected_recall": recall,
        "actual_precision": precision,
        "f1": f1,
        "jaccard_similarity": jaccard,
        "lcs_length": lcs_length,
        "ordered_expected_coverage": lcs_length / len(expected) if expected else 1.0,
        "canonical_edit_similarity": _normalized_edit_similarity(expected, canonical),
        "exact_chain_match": expected == actual,
        "exact_canonical_chain_match": expected == canonical,
        "alignment_class": _alignment_class(expected, actual, lcs_length),
        "expected_edge_count": edge_count,
        "edge_precedence_recall": precedence_count / edge_count if edge_count else 1.0,
        "edge_canonical_adjacency_recall": adjacency_count / edge_count if edge_count else 1.0,
        "expected_edges": edge_results,
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(numbers),
        "min": min(numbers),
        "p25": _percentile(numbers, 0.25),
        "p50": _percentile(numbers, 0.50),
        "p75": _percentile(numbers, 0.75),
        "p90": _percentile(numbers, 0.90),
        "p95": _percentile(numbers, 0.95),
        "max": max(numbers),
        "mean": statistics.fmean(numbers),
    }


def _diversity(counter: Counter[Any]) -> dict[str, Any]:
    total = sum(counter.values())
    categories = len(counter)
    if total == 0:
        return {
            "total": 0,
            "category_count": 0,
            "shannon_entropy": 0.0,
            "normalized_shannon_entropy": 0.0,
            "simpson_concentration": 0.0,
            "simpson_diversity": 0.0,
            "effective_shannon": 0.0,
            "effective_simpson": 0.0,
        }
    probabilities = [count / total for count in counter.values()]
    shannon = -sum(probability * math.log(probability) for probability in probabilities)
    concentration = sum(probability * probability for probability in probabilities)
    return {
        "total": total,
        "category_count": categories,
        "shannon_entropy": shannon,
        "normalized_shannon_entropy": shannon / math.log(categories) if categories > 1 else 0.0,
        "simpson_concentration": concentration,
        "simpson_diversity": 1.0 - concentration,
        "effective_shannon": math.exp(shannon),
        "effective_simpson": 1.0 / concentration if concentration else 0.0,
    }


def _signature(record: dict[str, Any], view: str, *, expected: bool = False) -> tuple[str, ...]:
    if expected:
        sequence = record["expected_chain"]
        if view == "raw" or view == "canonical":
            return tuple(sequence)
        if view == "ordered_unique":
            return tuple(_ordered_unique(sequence))
        if view == "tool_set":
            return tuple(sorted(set(sequence)))
    else:
        keys = {
            "raw": "actual_chain",
            "canonical": "canonical_chain",
            "ordered_unique": "ordered_unique_chain",
            "tool_set": "tool_set_signature",
        }
        return tuple(record[keys[view]])
    raise ValueError(f"unknown signature view: {view}")


def _signature_summary(counter: Counter[tuple[str, ...]], sample_count: int) -> dict[str, Any]:
    metrics = _diversity(counter)
    singleton_count = sum(count == 1 for count in counter.values())
    top_count = max(counter.values(), default=0)
    return {
        **metrics,
        "unique_signature_count": len(counter),
        "uniqueness_ratio": len(counter) / sample_count if sample_count else 0.0,
        "singleton_signature_count": singleton_count,
        "singleton_signature_rate": singleton_count / len(counter) if counter else 0.0,
        "top_signature_count": top_count,
        "top_signature_share": top_count / sample_count if sample_count else 0.0,
    }


def _transition_counter(records: Sequence[dict[str, Any]], key: str) -> tuple[Counter[tuple[str, str]], Counter[tuple[str, str]]]:
    calls: Counter[tuple[str, str]] = Counter()
    documents: Counter[tuple[str, str]] = Counter()
    for record in records:
        pairs = list(zip(record[key], record[key][1:]))
        calls.update(pairs)
        documents.update(set(pairs))
    return calls, documents


def _expected_transition_data(
    records: Sequence[dict[str, Any]],
) -> tuple[Counter[tuple[str, str]], Counter[tuple[str, str]], dict[tuple[str, str], Counter[str]]]:
    calls: Counter[tuple[str, str]] = Counter()
    documents: Counter[tuple[str, str]] = Counter()
    relation_types: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for record in records:
        pairs: list[tuple[str, str]] = []
        for edge in record["expected_edges"]:
            pair = (edge["source_tool"], edge["target_tool"])
            pairs.append(pair)
            relation_types[pair][edge["edge_type"]] += 1
        calls.update(pairs)
        documents.update(set(pairs))
    return calls, documents, relation_types


def _transition_summary(counter: Counter[tuple[str, str]]) -> dict[str, Any]:
    total = sum(counter.values())
    self_loops = sum(count for (source, target), count in counter.items() if source == target)
    return {
        **_diversity(counter),
        "self_loop_count": self_loops,
        "self_loop_rate": self_loops / total if total else 0.0,
    }


def _load_catalog(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: tool catalog must be an object")
    limits = value.get("limits")
    if not isinstance(limits, dict):
        raise ValueError(f"{path}: tool catalog limits must be an object")
    names: list[str] = []
    for values in limits.values():
        if not isinstance(values, list):
            raise ValueError(f"{path}: each tool catalog limit group must be an array")
        names.extend(_normalize_tool_name(name, context=f"{path}: catalog") for name in values)
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: duplicate tool names in catalog")
    return set(names)


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    status = run("status", "--short").splitlines()
    return {
        "repo_root": str(repo_root),
        "commit": run("rev-parse", "HEAD") or None,
        "dirty": bool(status),
        "status_entry_count": len(status),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _per_trajectory_csv_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    json_fields = {
        "expected_chain",
        "actual_chain",
        "canonical_chain",
        "ordered_unique_chain",
        "tool_set_signature",
        "missing_expected_tools",
        "unexpected_actual_tools",
        "unknown_catalog_tools",
        "ignored_local_tool_hist",
    }
    fields = (
        "training_record_id",
        "training_record_index",
        "source_batch",
        "source_task_id",
        "question_exact_match",
        "expected_tool_count",
        "actual_call_count",
        "canonical_call_count",
        "actual_unique_tool_count",
        "ignored_local_call_count",
        "unexpected_actual_call_count",
        "repeated_expected_call_count",
        "expected_recall",
        "actual_precision",
        "f1",
        "jaccard_similarity",
        "ordered_expected_coverage",
        "canonical_edit_similarity",
        "exact_chain_match",
        "exact_canonical_chain_match",
        "alignment_class",
        "expected_edge_count",
        "edge_precedence_recall",
        "edge_canonical_adjacency_recall",
        *sorted(json_fields),
    )
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {field: record.get(field) for field in fields}
        for field in json_fields:
            row[field] = _json_compact(row[field])
        rows.append(row)
    return rows


def _tool_usage_rows(
    records: Sequence[dict[str, Any]], catalog_tools: set[str]
) -> list[dict[str, Any]]:
    actual_calls = Counter(tool for record in records for tool in record["actual_chain"])
    actual_documents = Counter(
        tool for record in records for tool in set(record["actual_chain"])
    )
    expected_calls = Counter(tool for record in records for tool in record["expected_chain"])
    expected_documents = Counter(
        tool for record in records for tool in set(record["expected_chain"])
    )
    batch_calls: dict[str, Counter[str]] = {}
    batch_documents: dict[str, Counter[str]] = {}
    for batch in BATCH_ORDER:
        selected = [record for record in records if record["source_batch"] == batch]
        batch_calls[batch] = Counter(tool for record in selected for tool in record["actual_chain"])
        batch_documents[batch] = Counter(
            tool for record in selected for tool in set(record["actual_chain"])
        )
    tools = sorted(set(actual_calls) | set(expected_calls))
    total_actual_calls = sum(actual_calls.values())
    rows: list[dict[str, Any]] = []
    for tool in tools:
        row: dict[str, Any] = {
            "tool_name": tool,
            "catalog_known": tool in catalog_tools,
            "actual_call_count": actual_calls[tool],
            "actual_call_share": actual_calls[tool] / total_actual_calls if total_actual_calls else 0.0,
            "actual_sample_count": actual_documents[tool],
            "actual_sample_coverage": actual_documents[tool] / len(records) if records else 0.0,
            "expected_occurrence_count": expected_calls[tool],
            "expected_sample_count": expected_documents[tool],
        }
        for batch in BATCH_ORDER:
            row[f"{batch}_actual_call_count"] = batch_calls[batch][tool]
            row[f"{batch}_actual_sample_count"] = batch_documents[batch][tool]
        rows.append(row)
    return sorted(rows, key=lambda row: (-int(row["actual_call_count"]), row["tool_name"]))


def _signature_rows_and_summary(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"actual": {}, "expected_reference": {}}
    for source, expected in (("actual", False), ("expected_reference", True)):
        for view in SIGNATURE_VIEWS:
            counter = Counter(_signature(record, view, expected=expected) for record in records)
            summary[source][view] = _signature_summary(counter, len(records))
            for rank, (signature, count) in enumerate(
                sorted(counter.items(), key=lambda item: (-item[1], item[0])), 1
            ):
                rows.append(
                    {
                        "source": source,
                        "signature_view": view,
                        "rank": rank,
                        "count": count,
                        "share": count / len(records) if records else 0.0,
                        "signature_json": _json_compact(list(signature)),
                    }
                )
    return rows, summary


def _transition_rows_and_summary(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_calls, raw_docs = _transition_counter(records, "actual_chain")
    canonical_calls, canonical_docs = _transition_counter(records, "canonical_chain")
    expected_calls, expected_docs, relation_types = _expected_transition_data(records)
    views = {
        "actual_raw": (raw_calls, raw_docs),
        "actual_canonical": (canonical_calls, canonical_docs),
        "expected_reference": (expected_calls, expected_docs),
    }
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    expected_pairs = set(expected_calls)
    for view, (calls, documents) in views.items():
        summary[view] = _transition_summary(calls)
        total = sum(calls.values())
        for (source, target), count in sorted(calls.items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "transition_view": view,
                    "source_tool": source,
                    "target_tool": target,
                    "transition_count": count,
                    "transition_share": count / total if total else 0.0,
                    "sample_count": documents[(source, target)],
                    "self_loop": source == target,
                    "is_expected_edge_pair": (source, target) in expected_pairs,
                    "expected_edge_occurrence_count": expected_calls[(source, target)],
                    "expected_edge_types_json": _json_compact(
                        sorted(relation_types.get((source, target), {}).items())
                    ),
                }
            )
    return rows, summary


def _pairwise_rows_and_summary(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped_jaccard: dict[str, list[float]] = defaultdict(list)
    grouped_edit: dict[str, list[float]] = defaultdict(list)
    overall_jaccard: list[float] = []
    overall_edit: list[float] = []
    batch_rank = {batch: index for index, batch in enumerate(BATCH_ORDER)}
    for left, right in itertools.combinations(records, 2):
        left_set = set(left["tool_set_signature"])
        right_set = set(right["tool_set_signature"])
        jaccard_distance = _jaccard_distance(left_set, right_set)
        ordered_edit_distance = 1.0 - _normalized_edit_similarity(
            left["ordered_unique_chain"], right["ordered_unique_chain"]
        )
        batches = sorted(
            (left["source_batch"], right["source_batch"]),
            key=lambda batch: (batch_rank.get(batch, len(batch_rank)), batch),
        )
        batch_pair = f"{batches[0]}__{batches[1]}"
        grouped_jaccard[batch_pair].append(jaccard_distance)
        grouped_edit[batch_pair].append(ordered_edit_distance)
        overall_jaccard.append(jaccard_distance)
        overall_edit.append(ordered_edit_distance)
        rows.append(
            {
                "training_record_id_a": left["training_record_id"],
                "training_record_id_b": right["training_record_id"],
                "source_batch_a": left["source_batch"],
                "source_batch_b": right["source_batch"],
                "batch_pair": batch_pair,
                "tool_set_jaccard_distance": jaccard_distance,
                "ordered_unique_edit_distance": ordered_edit_distance,
            }
        )
    summary = {
        "pair_count": len(rows),
        "overall": {
            "tool_set_jaccard_distance": _distribution(overall_jaccard),
            "ordered_unique_edit_distance": _distribution(overall_edit),
        },
        "by_batch_pair": {
            batch_pair: {
                "pair_count": len(grouped_jaccard[batch_pair]),
                "tool_set_jaccard_distance": _distribution(grouped_jaccard[batch_pair]),
                "ordered_unique_edit_distance": _distribution(grouped_edit[batch_pair]),
            }
            for batch_pair in sorted(grouped_jaccard)
        },
    }
    return rows, summary


def _edge_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    overall = Counter()
    for record in records:
        for edge in record["expected_edges"]:
            for counter in (overall, by_type[edge["edge_type"]]):
                counter["expected_count"] += 1
                counter["precedence_preserved_count"] += int(edge["precedence_preserved"])
                counter["canonical_adjacency_preserved_count"] += int(
                    edge["canonical_adjacency_preserved"]
                )

    def serialize(counter: Counter[str]) -> dict[str, Any]:
        expected = counter["expected_count"]
        return {
            **dict(counter),
            "precedence_recall": counter["precedence_preserved_count"] / expected if expected else 1.0,
            "canonical_adjacency_recall": (
                counter["canonical_adjacency_preserved_count"] / expected if expected else 1.0
            ),
        }

    return {
        "overall": serialize(overall),
        "by_edge_type": {edge_type: serialize(counter) for edge_type, counter in sorted(by_type.items())},
    }


def _consistency_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "expected_recall",
        "actual_precision",
        "f1",
        "jaccard_similarity",
        "ordered_expected_coverage",
        "canonical_edit_similarity",
        "edge_precedence_recall",
        "edge_canonical_adjacency_recall",
    )
    alignment_counts = Counter(record["alignment_class"] for record in records)
    overlap_sum = sum(
        len(set(record["expected_chain"]) & set(record["actual_chain"])) for record in records
    )
    expected_sum = sum(len(set(record["expected_chain"])) for record in records)
    actual_sum = sum(len(set(record["actual_chain"])) for record in records)
    return {
        "alignment_class_counts": {name: alignment_counts[name] for name in ALIGNMENT_CLASSES},
        "exact_chain_match_count": sum(record["exact_chain_match"] for record in records),
        "exact_canonical_chain_match_count": sum(
            record["exact_canonical_chain_match"] for record in records
        ),
        "metric_distributions": {
            name: _distribution(record[name] for record in records) for name in metric_names
        },
        "micro_expected_recall": overlap_sum / expected_sum if expected_sum else 1.0,
        "micro_actual_precision": overlap_sum / actual_sum if actual_sum else 0.0,
        "edge_preservation": _edge_summary(records),
    }


def _batch_summary_rows(
    records: Sequence[dict[str, Any]], pairwise_summary: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    groups = (("overall", list(records)),) + tuple(
        (batch, [record for record in records if record["source_batch"] == batch])
        for batch in BATCH_ORDER
    )
    for label, selected in groups:
        actual_calls = Counter(tool for record in selected for tool in record["actual_chain"])
        alignment = Counter(record["alignment_class"] for record in selected)
        signature_counts = {
            view: len(Counter(_signature(record, view) for record in selected))
            for view in SIGNATURE_VIEWS
        }
        if label == "overall":
            pair_stats = pairwise_summary["overall"]
        else:
            pair_stats = pairwise_summary["by_batch_pair"].get(f"{label}__{label}", {})
        row = {
            "source_batch": label,
            "sample_count": len(selected),
            "actual_call_count": sum(record["actual_call_count"] for record in selected),
            "actual_tool_vocabulary_size": len(actual_calls),
            "actual_call_shannon_entropy": _diversity(actual_calls)["shannon_entropy"],
            "actual_call_effective_shannon": _diversity(actual_calls)["effective_shannon"],
            "actual_chain_length_mean": _distribution(
                record["actual_call_count"] for record in selected
            )["mean"],
            "actual_chain_length_p50": _distribution(
                record["actual_call_count"] for record in selected
            )["p50"],
            "actual_chain_length_p95": _distribution(
                record["actual_call_count"] for record in selected
            )["p95"],
            "expected_recall_mean": _distribution(record["expected_recall"] for record in selected)[
                "mean"
            ],
            "actual_precision_mean": _distribution(record["actual_precision"] for record in selected)[
                "mean"
            ],
            "ordered_expected_coverage_mean": _distribution(
                record["ordered_expected_coverage"] for record in selected
            )["mean"],
            "canonical_edit_similarity_mean": _distribution(
                record["canonical_edit_similarity"] for record in selected
            )["mean"],
            "raw_unique_signature_count": signature_counts["raw"],
            "canonical_unique_signature_count": signature_counts["canonical"],
            "ordered_unique_signature_count": signature_counts["ordered_unique"],
            "tool_set_unique_signature_count": signature_counts["tool_set"],
            "within_batch_jaccard_distance_mean": (
                pair_stats.get("tool_set_jaccard_distance", {}).get("mean")
                if isinstance(pair_stats, dict)
                else None
            ),
            "within_batch_ordered_edit_distance_mean": (
                pair_stats.get("ordered_unique_edit_distance", {}).get("mean")
                if isinstance(pair_stats, dict)
                else None
            ),
            "alignment_class_counts_json": _json_compact(
                {name: alignment[name] for name in ALIGNMENT_CLASSES}
            ),
        }
        rows.append(row)
        detail[label] = {
            **row,
            "actual_chain_length_distribution": _distribution(
                record["actual_call_count"] for record in selected
            ),
            "actual_unique_tool_count_distribution": _distribution(
                record["actual_unique_tool_count"] for record in selected
            ),
            "actual_call_diversity": _diversity(actual_calls),
        }
    return rows, detail


def _plot_outputs(
    output_dir: Path,
    records: Sequence[dict[str, Any]],
    tool_rows: Sequence[dict[str, Any]],
    signature_summary: dict[str, Any],
    pairwise_summary: dict[str, Any],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required for plots; install requirements or use --skip-plots") from exc

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=False)
    written: list[str] = []

    alignment = Counter(record["alignment_class"] for record in records)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar(range(len(ALIGNMENT_CLASSES)), [alignment[name] for name in ALIGNMENT_CLASSES])
    axes[0].set_xticks(range(len(ALIGNMENT_CLASSES)))
    axes[0].set_xticklabels(
        ["exact", "subsequence+", "full/order", "partial", "none"], rotation=20, ha="right"
    )
    axes[0].set_ylabel("Trajectories")
    axes[0].set_title("Alignment classes")
    metric_names = (
        "expected_recall",
        "actual_precision",
        "ordered_expected_coverage",
        "canonical_edit_similarity",
        "edge_precedence_recall",
        "edge_canonical_adjacency_recall",
    )
    boxplot_labels = ["recall", "precision", "ordered", "edit", "edge order", "edge adjacent"]
    version_match = re.match(r"^(\d+)\.(\d+)", matplotlib.__version__)
    use_tick_labels = bool(
        version_match and tuple(int(value) for value in version_match.groups()) >= (3, 9)
    )
    label_argument = {"tick_labels" if use_tick_labels else "labels": boxplot_labels}
    axes[1].boxplot(
        [[record[name] for record in records] for name in metric_names],
        showmeans=True,
        **label_argument,
    )
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Consistency metrics")
    fig.tight_layout()
    path = figures / "consistency_overview.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path.relative_to(output_dir)))

    top = list(tool_rows[:30])
    labels = [row["tool_name"] for row in reversed(top)]
    fig, axes = plt.subplots(1, 2, figsize=(17, 10))
    axes[0].barh(labels, [row["actual_call_count"] for row in reversed(top)])
    axes[0].set_xlabel("Calls")
    axes[0].set_title("Top-30 tools by call count")
    axes[1].barh(labels, [row["actual_sample_coverage"] for row in reversed(top)])
    axes[1].set_xlabel("Sample coverage")
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Top-30 tool sample coverage")
    fig.tight_layout()
    path = figures / "tool_usage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path.relative_to(output_dir)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    lengths = [record["actual_call_count"] for record in records]
    axes[0].hist(lengths, bins=min(40, max(5, len(set(lengths)))))
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Domain tool calls (log scale)")
    axes[0].set_ylabel("Trajectories")
    axes[0].set_title("Actual chain length")
    unique_counts = [
        signature_summary["actual"][view]["unique_signature_count"] for view in SIGNATURE_VIEWS
    ]
    axes[1].bar(SIGNATURE_VIEWS, unique_counts)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylabel("Unique signatures")
    axes[1].set_title("Signature diversity")
    fig.tight_layout()
    path = figures / "chain_length_and_signatures.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path.relative_to(output_dir)))

    def matrix(metric: str) -> list[list[float]]:
        values: list[list[float]] = []
        order = {batch: index for index, batch in enumerate(BATCH_ORDER)}
        for left in BATCH_ORDER:
            row: list[float] = []
            for right in BATCH_ORDER:
                pair = sorted((left, right), key=lambda batch: order[batch])
                key = f"{pair[0]}__{pair[1]}"
                value = pairwise_summary["by_batch_pair"].get(key, {}).get(metric, {}).get("mean")
                row.append(float("nan") if value is None else float(value))
            values.append(row)
        return values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, values, title in (
        (axes[0], matrix("tool_set_jaccard_distance"), "Mean tool-set Jaccard distance"),
        (axes[1], matrix("ordered_unique_edit_distance"), "Mean ordered-unique edit distance"),
    ):
        image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis")
        axis.set_xticks(range(len(BATCH_ORDER)), BATCH_ORDER, rotation=25, ha="right")
        axis.set_yticks(range(len(BATCH_ORDER)), BATCH_ORDER)
        axis.set_title(title)
        for row_index, row in enumerate(values):
            for column_index, value in enumerate(row):
                axis.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", color="white")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = figures / "batch_pairwise_distance_heatmaps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path.relative_to(output_dir)))
    return written


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _render_report(
    *,
    generated_at: str,
    transfer_root: Path,
    records: Sequence[dict[str, Any]],
    consistency: dict[str, Any],
    signature_summary: dict[str, Any],
    transition_summary: dict[str, Any],
    pairwise_summary: dict[str, Any],
    batch_details: dict[str, Any],
    tool_rows: Sequence[dict[str, Any]],
    local_hist: Counter[str],
    catalog_tools: set[str],
    plot_paths: Sequence[str],
) -> str:
    unknown = sorted(
        {tool for record in records for tool in record["actual_chain"] if tool not in catalog_tools}
    )
    metric_distributions = consistency["metric_distributions"]
    lines = [
        "# Drug-Pipe 240 Tool-Chain 一致性与 Diversity 分析",
        "",
        f"- 生成时间（UTC）：`{generated_at}`",
        f"- 数据目录：`{transfer_root}`",
        f"- 轨迹数：{len(records)}",
        f"- 实际非本地工具词表：{len({tool for record in records for tool in record['actual_chain']})}",
        f"- 成对比较数：{pairwise_summary['pair_count']}",
        "",
        "> 这里的一致性表示实际调用链与隐藏 Tool-KG 参考链的结构接近程度；"
        "预期链未暴露给 agent，也不是强制执行规范，因此这些指标不直接代表科学正确性。",
        "",
        "## 输入与解析完整性",
        "",
        "- 轨迹与 metadata 按 `training_record_id` 完整一一连接。",
        "- 实际链只来自 assistant 消息中的 `<tool_call>`，不从 observation 反推。",
        f"- 忽略的本地调用总数：{sum(local_hist.values())}；分布：`{_json_compact(dict(sorted(local_hist.items())))}`。",
        f"- 非 catalog 工具名：`{_json_compact(unknown)}`；保留原拼写，不做别名映射。",
        "",
        "## 一致性",
        "",
        "| Alignment class | Count | Share |",
        "|---|---:|---:|",
    ]
    counts = consistency["alignment_class_counts"]
    for name in ALIGNMENT_CLASSES:
        lines.append(f"| `{name}` | {counts[name]} | {counts[name] / len(records):.4f} |")
    lines.extend(
        [
            "",
            "| Metric | Mean | P50 | P90 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in (
        "expected_recall",
        "actual_precision",
        "f1",
        "jaccard_similarity",
        "ordered_expected_coverage",
        "canonical_edit_similarity",
        "edge_precedence_recall",
        "edge_canonical_adjacency_recall",
    ):
        dist = metric_distributions[name]
        lines.append(
            f"| `{name}` | {_fmt(dist['mean'])} | {_fmt(dist['p50'])} | {_fmt(dist['p90'])} |"
        )
    edge_overall = consistency["edge_preservation"]["overall"]
    lines.extend(
        [
            "",
            f"- Micro expected recall：{consistency['micro_expected_recall']:.4f}",
            f"- Micro actual precision：{consistency['micro_actual_precision']:.4f}",
            f"- Expected-edge precedence recall：{edge_overall['precedence_recall']:.4f}",
            f"- Expected-edge canonical adjacency recall：{edge_overall['canonical_adjacency_recall']:.4f}",
            "",
            "## Actual Tool-Chain Diversity",
            "",
            "| Signature view | Unique | Uniqueness ratio | Singleton rate | Effective Shannon |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for view in SIGNATURE_VIEWS:
        data = signature_summary["actual"][view]
        lines.append(
            f"| `{view}` | {data['unique_signature_count']} | {data['uniqueness_ratio']:.4f} | "
            f"{data['singleton_signature_rate']:.4f} | {data['effective_shannon']:.2f} |"
        )
    raw_transition = transition_summary["actual_raw"]
    canonical_transition = transition_summary["actual_canonical"]
    lines.extend(
        [
            "",
            f"- Raw directed transitions：{raw_transition['category_count']} 类，"
            f"self-loop rate={raw_transition['self_loop_rate']:.4f}，"
            f"effective Shannon={raw_transition['effective_shannon']:.2f}。",
            f"- Canonical directed transitions：{canonical_transition['category_count']} 类，"
            f"effective Shannon={canonical_transition['effective_shannon']:.2f}。",
            f"- 全体 pairwise tool-set Jaccard distance mean="
            f"{pairwise_summary['overall']['tool_set_jaccard_distance']['mean']:.4f}。",
            f"- 全体 pairwise ordered-unique edit distance mean="
            f"{pairwise_summary['overall']['ordered_unique_edit_distance']['mean']:.4f}。",
            "",
            "### Batch 对比",
            "",
            "| Batch | Samples | Calls | Vocabulary | Recall mean | Precision mean | Within Jaccard |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for batch in BATCH_ORDER:
        data = batch_details[batch]
        lines.append(
            f"| `{batch}` | {data['sample_count']} | {data['actual_call_count']} | "
            f"{data['actual_tool_vocabulary_size']} | {_fmt(data['expected_recall_mean'])} | "
            f"{_fmt(data['actual_precision_mean'])} | {_fmt(data['within_batch_jaccard_distance_mean'])} |"
        )
    lines.extend(
        [
            "",
            "### 高频实际工具",
            "",
            "| Tool | Calls | Sample coverage | In catalog |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in tool_rows[:20]:
        lines.append(
            f"| `{row['tool_name']}` | {row['actual_call_count']} | "
            f"{row['actual_sample_coverage']:.4f} | {row['catalog_known']} |"
        )
    if plot_paths:
        lines.extend(["", "## Figures", ""])
        for path in plot_paths:
            lines.append(f"![{Path(path).stem}]({path})")
            lines.append("")
    lines.extend(
        [
            "## 方法说明",
            "",
            "- `canonical` 只折叠相邻重复调用；非相邻重试和循环仍保留。",
            "- `ordered_unique` 保留每个工具第一次出现的顺序；pairwise edit distance 使用该视图。",
            "- Edge precedence 表示某次 source 调用早于某次 target 调用；adjacency 要求二者在 canonical 链中相邻。",
            "- Shannon 使用自然对数；effective Shannon 为 `exp(H)`，effective Simpson 为 `1/sum(p^2)`。",
            "- 完整逐条指标、signature、transition 和 pairwise 数据见同目录结构化文件。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    *,
    transfer_root: Path,
    output_dir: Path,
    tool_catalog: Path,
    expected_count: int,
    skip_plots: bool,
) -> dict[str, Any]:
    trajectories_path = transfer_root / "data/react_trajectories.jsonl"
    metadata_path = transfer_root / "metadata/toolkg/toolkg_metadata_240.jsonl"
    metadata_manifest_path = transfer_root / "metadata/toolkg/toolkg_metadata_manifest.json"
    input_paths = (trajectories_path, metadata_path, metadata_manifest_path, tool_catalog)
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    input_hashes_before = {str(path): _sha256(path) for path in input_paths}
    trajectory_records = _load_jsonl(trajectories_path)
    metadata_records = _load_jsonl(metadata_path)
    metadata_manifest = _load_json_object(metadata_manifest_path)
    _validate_input_schemas(trajectory_records, metadata_records, metadata_manifest)
    if expected_count > 0:
        if len(trajectory_records) != expected_count:
            raise ValueError(
                f"expected {expected_count} trajectories, found {len(trajectory_records)}"
            )
        if len(metadata_records) != expected_count:
            raise ValueError(f"expected {expected_count} metadata records, found {len(metadata_records)}")
    trajectories = _unique_index(trajectory_records, "id", label="trajectory")
    metadata = _unique_index(metadata_records, "training_record_id", label="metadata")
    missing_metadata = sorted(set(trajectories) - set(metadata))
    missing_trajectories = sorted(set(metadata) - set(trajectories))
    if missing_metadata or missing_trajectories:
        raise ValueError(
            "trajectory/metadata ID mismatch: "
            f"missing_metadata={missing_metadata}, missing_trajectories={missing_trajectories}"
        )
    catalog_tools = _load_catalog(tool_catalog)
    local_tools = set(LOCAL_TOOL_NAMES)
    ordered_ids = sorted(
        trajectories,
        key=lambda record_id: (
            metadata[record_id].get("training_record_index")
            if isinstance(metadata[record_id].get("training_record_index"), int)
            else math.inf,
            record_id,
        ),
    )
    records = [
        _analyze_record(
            trajectories[record_id],
            metadata[record_id],
            record_id=record_id,
            local_tools=local_tools,
            catalog_tools=catalog_tools,
        )
        for record_id in ordered_ids
    ]
    unknown_batches = sorted({record["source_batch"] for record in records} - set(BATCH_ORDER))
    if unknown_batches:
        raise ValueError(f"unexpected source_batch values: {unknown_batches}")

    generated_at = datetime.now(timezone.utc).isoformat()
    consistency = _consistency_summary(records)
    actual_call_counter = Counter(tool for record in records for tool in record["actual_chain"])
    local_hist = Counter(
        tool for record in records for tool, count in record["ignored_local_tool_hist"].items() for _ in range(count)
    )
    tool_rows = _tool_usage_rows(records, catalog_tools)
    signature_rows, signature_summary = _signature_rows_and_summary(records)
    transition_rows, transition_summary = _transition_rows_and_summary(records)
    pairwise_rows, pairwise_summary = _pairwise_rows_and_summary(records)
    expected_pairs = len(records) * (len(records) - 1) // 2
    if len(pairwise_rows) != expected_pairs:
        raise AssertionError(f"expected {expected_pairs} pairwise rows, found {len(pairwise_rows)}")
    batch_rows, batch_details = _batch_summary_rows(records, pairwise_summary)
    chain_length_summary = {
        "actual": _distribution(record["actual_call_count"] for record in records),
        "canonical": _distribution(record["canonical_call_count"] for record in records),
        "ordered_unique": _distribution(record["actual_unique_tool_count"] for record in records),
        "expected": _distribution(record["expected_tool_count"] for record in records),
    }
    unknown_catalog_tools = sorted(set(actual_call_counter) - catalog_tools)
    summary = {
        "schema_version": "drug_pipe_toolchain_analysis_v1",
        "generated_at_utc": generated_at,
        "record_count": len(records),
        "source_batches": dict(Counter(record["source_batch"] for record in records)),
        "input_integrity": {
            "trajectory_metadata_id_bijection": True,
            "tool_call_parse_error_count": 0,
            "expected_chain_invariant_error_count": 0,
            "catalog_tool_count": len(catalog_tools),
            "actual_domain_tool_vocabulary_size": len(actual_call_counter),
            "unknown_catalog_tools": unknown_catalog_tools,
            "ignored_local_tools": list(LOCAL_TOOL_NAMES),
            "ignored_local_tool_call_count": sum(local_hist.values()),
            "ignored_local_tool_hist": dict(sorted(local_hist.items())),
        },
        "consistency": consistency,
        "chain_length_distributions": chain_length_summary,
        "tool_call_diversity": _diversity(actual_call_counter),
        "chain_signature_diversity": signature_summary,
        "transition_diversity": transition_summary,
        "pairwise_diversity": pairwise_summary,
        "batch_analysis": batch_details,
    }

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp_", dir=output_dir.parent))
    try:
        with (staging / "per_trajectory.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        per_csv_rows = _per_trajectory_csv_rows(records)
        _write_csv(staging / "per_trajectory.csv", per_csv_rows, list(per_csv_rows[0]) if per_csv_rows else [])
        _write_csv(staging / "tool_usage.csv", tool_rows, list(tool_rows[0]) if tool_rows else [])
        _write_csv(
            staging / "transition_usage.csv",
            transition_rows,
            list(transition_rows[0]) if transition_rows else [],
        )
        _write_csv(
            staging / "chain_signatures.csv",
            signature_rows,
            list(signature_rows[0]) if signature_rows else [],
        )
        _write_csv(
            staging / "pairwise_distances.csv",
            pairwise_rows,
            list(pairwise_rows[0]) if pairwise_rows else [],
        )
        _write_csv(staging / "batch_summary.csv", batch_rows, list(batch_rows[0]) if batch_rows else [])
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        plot_paths = [] if skip_plots else _plot_outputs(
            staging, records, tool_rows, signature_summary, pairwise_summary
        )
        report = _render_report(
            generated_at=generated_at,
            transfer_root=transfer_root,
            records=records,
            consistency=consistency,
            signature_summary=signature_summary,
            transition_summary=transition_summary,
            pairwise_summary=pairwise_summary,
            batch_details=batch_details,
            tool_rows=tool_rows,
            local_hist=local_hist,
            catalog_tools=catalog_tools,
            plot_paths=plot_paths,
        )
        (staging / "report.md").write_text(report, encoding="utf-8")

        repo_root = Path(__file__).resolve().parents[4]
        output_hashes = {
            str(path.relative_to(staging)): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": "drug_pipe_toolchain_analysis_manifest_v1",
            "generated_at_utc": generated_at,
            "parameters": {
                "transfer_root": str(transfer_root),
                "output_dir": str(output_dir),
                "tool_catalog": str(tool_catalog),
                "expected_count": expected_count,
                "skip_plots": skip_plots,
                "ignored_local_tools": list(LOCAL_TOOL_NAMES),
            },
            "input_sha256": input_hashes_before,
            "analyzer": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
                "git": _git_provenance(repo_root),
            },
            "record_count": len(records),
            "output_sha256": output_hashes,
            "notes": [
                "manifest.json is excluded from output_sha256 to avoid self-referential hashing",
                "the transfer package SHA256SUMS file is intentionally not modified",
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        input_hashes_after = {str(path): _sha256(path) for path in input_paths}
        if input_hashes_after != input_hashes_before:
            raise RuntimeError("one or more input files changed while the analysis was running")
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "transfer_root": str(transfer_root),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "pairwise_row_count": len(pairwise_rows),
        "actual_domain_tool_vocabulary_size": len(actual_call_counter),
        "unknown_catalog_tools": unknown_catalog_tools,
        "plots_generated": not skip_plots,
    }


def main() -> int:
    script_path = Path(__file__).resolve()
    default_catalog = script_path.parents[1] / "molclaw_tool_concurrency_v1.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <transfer-root>/documentation/toolchain_analysis",
    )
    parser.add_argument("--tool-catalog", type=Path, default=default_catalog)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=0,
        help="Fail unless both input JSONL files contain this many records; 0 disables the check.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    transfer_root = args.transfer_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else transfer_root / "documentation/toolchain_analysis"
    )
    result = analyze(
        transfer_root=transfer_root,
        output_dir=output_dir,
        tool_catalog=args.tool_catalog.expanduser().resolve(),
        expected_count=args.expected_count,
        skip_plots=args.skip_plots,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
