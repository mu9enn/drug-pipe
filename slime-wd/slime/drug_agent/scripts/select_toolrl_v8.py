#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from drug_agent.toolrl.v8_dataset import (
    DESCRIPTION_VERSION,
    EMBEDDING_INSTRUCTION,
    cluster_and_select,
    embedding_text,
    sha256_file,
)


def _load_projection(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        rows = []
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            skill_names = [
                str(call.get("arguments", {}).get("name") or "")
                for call in row.get("label", {}).get("target_tool_calls", [])
                if call.get("name") == "skill"
            ]
            rows.append(
                {
                    "id": row["id"],
                    "metadata": {
                        "source_id": metadata.get("source_id"),
                        "task_type": metadata.get("task_type"),
                        "decision_type": metadata.get("decision_type"),
                        "tool_names": metadata.get("tool_names", []),
                        "comparison_scope": metadata.get("comparison_scope"),
                        "selection_description": metadata.get("selection_description"),
                        "skill_names": skill_names,
                    },
                }
            )
        return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_pretty_array(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            output.write("[\n")
            for index, row in enumerate(rows):
                if index:
                    output.write(",\n")
                rendered = json.dumps(row, ensure_ascii=False, indent=2)
                output.write("\n".join("  " + line for line in rendered.splitlines()))
            output.write("\n]\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _embedding_model_identity(model: str) -> dict[str, Any]:
    path = Path(model)
    if not path.is_dir():
        return {"huggingface_id": model}
    files = {}
    for name in ("config.json", "modules.json", "config_sentence_transformers.json", "model.safetensors"):
        candidate = path / name
        if candidate.is_file():
            files[name] = {"bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)}
    return {"path": str(path.resolve()), "files": files}


def encode(input_path: Path, output_root: Path, model: str, batch_size: int) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    rows = _load_projection(input_path)
    texts = [embedding_text(row) for row in rows]
    model_impl = SentenceTransformer(model, trust_remote_code=True)
    vectors = model_impl.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    output_root.mkdir(parents=True, exist_ok=True)
    vector_path = output_root / "decision_embeddings.npy"
    np.save(vector_path, vectors, allow_pickle=False)
    ids_hash = hashlib.sha256("\n".join(str(row["id"]) for row in rows).encode()).hexdigest()
    manifest = {
        "schema_version": "toolrl_v8_embedding_cache_v1",
        "model": model,
        "model_identity": _embedding_model_identity(model),
        "sentence_transformers_version": __import__("sentence_transformers").__version__,
        "description_version": DESCRIPTION_VERSION,
        "embedding_instruction": EMBEDDING_INSTRUCTION,
        "input": {"path": str(input_path.resolve()), "sha256": sha256_file(input_path), "records": len(rows), "ids_sha256": ids_hash},
        "vectors": {"path": vector_path.name, "shape": list(vectors.shape), "dtype": str(vectors.dtype), "sha256": sha256_file(vector_path)},
    }
    (output_root / "embedding_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _load_bound_embeddings(
    input_path: Path, embedding_root: Path, rows: list[dict[str, Any]]
) -> tuple[np.ndarray, dict[str, Any]]:
    embedding_manifest = json.loads((embedding_root / "embedding_manifest.json").read_text(encoding="utf-8"))
    source = embedding_manifest.get("input") or {}
    if source.get("sha256") != sha256_file(input_path) or int(source.get("records", -1)) != len(rows):
        raise ValueError("embedding cache is not bound to this decision input")
    vector_path = Path(embedding_manifest["vectors"]["path"])
    if vector_path.is_absolute():
        raise ValueError("embedding manifest vector path must be relative to its cache root")
    vector_path = embedding_root / vector_path
    if sha256_file(vector_path) != embedding_manifest["vectors"]["sha256"]:
        raise ValueError("embedding vector cache hash mismatch")
    vectors = np.load(vector_path, allow_pickle=False)
    if list(vectors.shape) != embedding_manifest["vectors"]["shape"]:
        raise ValueError("embedding vector cache shape mismatch")
    return vectors, embedding_manifest


def calibrate(
    input_path: Path,
    embedding_root: Path,
    output_path: Path,
    thresholds: list[float],
) -> dict[str, Any]:
    rows = _load_projection(input_path)
    vectors, embedding_manifest = _load_bound_embeddings(input_path, embedding_root, rows)
    sweeps: list[dict[str, Any]] = []
    for threshold in thresholds:
        _, summary, assignments = cluster_and_select(
            rows, vectors, distance_threshold=threshold, total_budget=None
        )
        members: dict[int, list[dict[str, Any]]] = defaultdict(list)
        row_by_id = {str(row["id"]): row for row in rows}
        for assignment in assignments:
            members[int(assignment["cluster_id"])].append(assignment)
        inspections = []
        for cluster in summary["largest_clusters"][:20]:
            cluster_members = members[int(cluster["cluster_id"])]
            example_rows = []
            for assignment in sorted(
                cluster_members,
                key=lambda item: (float(item["representative_similarity"]), str(item["decision_id"])),
            )[:12]:
                row = row_by_id[str(assignment["decision_id"])]
                description = str(row["metadata"].get("selection_description") or "")
                example_rows.append({
                    "decision_id": assignment["decision_id"],
                    "representative_id": assignment["representative_id"],
                    "representative_similarity": assignment["representative_similarity"],
                    "description_excerpt": description[:1200],
                })
            scope = json.loads(row_by_id[str(cluster_members[0]["decision_id"])]["metadata"]["comparison_scope"])
            inspections.append({**cluster, "comparison_scope": scope, "least_similar_examples": example_rows})
        sweeps.append({"summary": summary, "largest_cluster_inspections": inspections})
    report = {
        "schema_version": "toolrl_v8_threshold_calibration_v1",
        "input": {"path": str(input_path.resolve()), "sha256": sha256_file(input_path)},
        "embedding_cache": embedding_manifest,
        "threshold_sweeps": sweeps,
        "note": "Thresholds are candidates for human inspection, not universal defaults.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def select(
    input_path: Path,
    embedding_root: Path,
    output_root: Path,
    threshold: float,
    budget: int | None,
    min_final_records: int = 0,
) -> dict[str, Any]:
    rows = _load_projection(input_path)
    vectors, embedding_manifest = _load_bound_embeddings(input_path, embedding_root, rows)
    selected_indices, summary, assignments = cluster_and_select(
        rows,
        vectors,
        distance_threshold=threshold,
        total_budget=budget,
        min_final_records=min_final_records,
    )
    selected_set = set(selected_indices)
    assignment_by_id = {item["decision_id"]: item for item in assignments}

    output_root.mkdir(parents=True, exist_ok=True)
    selected_path = output_root / "selected_decisions.jsonl"
    pretty_path = output_root / "selected_decisions.pretty.json"
    audit_path = output_root / "selection_audit.jsonl"
    selected_tmp = selected_path.with_name(f".{selected_path.name}.tmp.{os.getpid()}")
    pretty_tmp = pretty_path.with_name(f".{pretty_path.name}.tmp.{os.getpid()}")
    selected_projection: list[dict[str, Any]] = []
    try:
        with input_path.open(encoding="utf-8") as source, selected_tmp.open("w", encoding="utf-8") as jsonl_output, pretty_tmp.open("w", encoding="utf-8") as pretty_output:
            pretty_output.write("[\n")
            pretty_index = 0
            for index, line in enumerate(line for line in source if line.strip()):
                if index not in selected_set:
                    continue
                row = json.loads(line)
                row["metadata"]["selector"] = assignment_by_id[row["id"]]
                jsonl_output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                if pretty_index:
                    pretty_output.write(",\n")
                rendered = json.dumps(row, ensure_ascii=False, indent=2)
                pretty_output.write("\n".join("  " + value for value in rendered.splitlines()))
                pretty_index += 1
                selected_projection.append(rows[index])
            pretty_output.write("\n]\n")
        os.replace(selected_tmp, selected_path)
        os.replace(pretty_tmp, pretty_path)
    finally:
        selected_tmp.unlink(missing_ok=True)
        pretty_tmp.unlink(missing_ok=True)
    _write_jsonl(audit_path, assignments)
    _write_pretty_array(output_root / "selection_audit.pretty.json", assignments)

    def breakdown(subset: list[dict[str, Any]]) -> dict[str, Any]:
        task = Counter(str(row["metadata"].get("task_type") or "unknown") for row in subset)
        tools = Counter(name for row in subset for name in row["metadata"].get("tool_names", []))
        skills = Counter(
            skill_name
            for row in subset
            for skill_name in row["metadata"].get("skill_names", [])
        )
        contexts = Counter(
            json.loads(row["metadata"]["comparison_scope"]).get("prior_outcome", "unknown")
            for row in subset
        )
        return {
            "trajectories": len({str(row["metadata"].get("source_id") or "") for row in subset}),
            "task_type": dict(task),
            "tools": dict(tools),
            "skills": dict(skills),
            "prior_outcome": dict(contexts),
        }

    selected_sources = {
        str(row["metadata"].get("source_id") or "") for row in selected_projection
    }
    unrepresented = []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["metadata"].get("source_id") or "")].append(row)
    for source_id, source_rows in sorted(by_source.items()):
        if source_id in selected_sources:
            continue
        unrepresented.append({
            "source_id": source_id,
            "task_type": source_rows[0]["metadata"].get("task_type"),
            "decision_count": len(source_rows),
            "representatives": [
                {
                    "decision_id": row["id"],
                    "representative_id": assignment_by_id[row["id"]]["representative_id"],
                    "representative_similarity": assignment_by_id[row["id"]]["representative_similarity"],
                }
                for row in source_rows
            ],
        })

    report = {
        "schema_version": "toolrl_v8_homogeneous_selector_v1",
        "selector": "Qwen3-Embedding-0.6B + scoped complete-linkage + center/farthest representatives",
        "input": {"path": str(input_path.resolve()), "sha256": sha256_file(input_path)},
        "embedding_cache": embedding_manifest,
        "selection": summary,
        "counts_before": breakdown(rows),
        "counts_after": breakdown(selected_projection),
        "unrepresented_trajectories": unrepresented,
        "outputs": {
            "selected_jsonl": str(selected_path.resolve()),
            "selected_sha256": sha256_file(selected_path),
            "selected_pretty_json": str(pretty_path.resolve()),
            "audit_jsonl": str(audit_path.resolve()),
        },
        "ordering": "canonical trajectory_index then original decision_ordinal",
        "unselected_label": "representative_downsampling_not_selected",
    }
    (output_root / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode or select v8 ToolRL decisions.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    encode_parser = subparsers.add_parser("encode")
    encode_parser.add_argument("--input", required=True, type=Path)
    encode_parser.add_argument("--output-root", required=True, type=Path)
    encode_parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    encode_parser.add_argument("--batch-size", type=int, default=32)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--input", required=True, type=Path)
    select_parser.add_argument("--embedding-root", required=True, type=Path)
    select_parser.add_argument("--output-root", required=True, type=Path)
    select_parser.add_argument("--distance-threshold", required=True, type=float)
    select_parser.add_argument("--budget", type=int)
    select_parser.add_argument("--min-final-records", type=int, default=0)
    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--input", required=True, type=Path)
    calibrate_parser.add_argument("--embedding-root", required=True, type=Path)
    calibrate_parser.add_argument("--output", required=True, type=Path)
    calibrate_parser.add_argument(
        "--distance-thresholds", type=float, nargs="+", required=True
    )
    args = parser.parse_args()
    if args.command == "encode":
        result = encode(args.input.resolve(), args.output_root.resolve(), args.model, args.batch_size)
    elif args.command == "select":
        result = select(
            args.input.resolve(),
            args.embedding_root.resolve(),
            args.output_root.resolve(),
            args.distance_threshold,
            args.budget,
            args.min_final_records,
        )
    else:
        result = calibrate(
            args.input.resolve(),
            args.embedding_root.resolve(),
            args.output.resolve(),
            args.distance_thresholds,
        )
    if args.command == "calibrate":
        print(json.dumps({
            "schema_version": result["schema_version"],
            "output": str(args.output.resolve()),
            "threshold_summaries": [sweep["summary"] for sweep in result["threshold_sweeps"]],
        }, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
