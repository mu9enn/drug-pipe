#!/usr/bin/env python3
"""Aggregate the eight aligned-v4 original checkpoint MolBench runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


MODELS = ("9b", "27b", "35b-a3b", "122b-a10b")
VARIANTS = ("flat", "hier")
METRICS = (
    ("strict_format_rate",),
    ("metrics", "rdkit_bench_all", "acc"),
    ("metrics", "rdkit_bench_all", "f1"),
    ("metrics", "acnet_curated_all", "acc"),
    ("metrics", "acnet_curated_all", "valid_rate"),
)


def nested(value: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return float(current) if isinstance(current, (int, float)) else None


def run_name(model: str, variant: str, stamp: str) -> str:
    workspace = "l1-flat" if variant == "flat" else "legacy-hierarchy"
    return f"aligned-v4-{model}-original-{workspace}-full-{stamp}"


def load_run(root: Path, model: str, variant: str, stamp: str) -> dict[str, Any]:
    directory = root / run_name(model, variant, stamp)
    summary_path = directory / "evaluation_summary.json"
    if not summary_path.is_file():
        return {"run_dir": str(directory), "status": "incomplete", "summary": None}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    complete = (
        summary.get("sample_count") == 87
        and summary.get("completed_count", 0) + summary.get("failed_or_missing_count", 0) == 87
        and summary.get("infra_failure_count") == 0
        and summary.get("publishable") is True
        and manifest.get("evaluation_contract", {}).get("contract_version") == "molbench_answer_contract_v4"
        and len(manifest.get("sample_ids", [])) == 87
    )
    return {
        "run_dir": str(directory),
        "status": "publishable" if complete else "incomplete",
        "summary": summary,
    }


def fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--models", default="9b", help="Comma-separated checkpoint tags")
    args = parser.parse_args()
    models = args.models.split(",")
    if not models or any(model not in MODELS for model in models):
        parser.error("unknown model selection")

    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for model in models:
        runs[model] = {
            variant: load_run(args.outputs_root, model, variant, args.stamp)
            for variant in VARIANTS
        }

    differences: dict[str, Any] = {}
    for model, variants in runs.items():
        flat = variants["flat"]["summary"] or {}
        hier = variants["hier"]["summary"] or {}
        if any(variants[variant]["status"] != "publishable" for variant in VARIANTS):
            differences[model] = None
            continue
        model_differences: dict[str, float | None] = {}
        for path in METRICS:
            flat_value = nested(flat, path)
            hier_value = nested(hier, path)
            model_differences[".".join(path)] = (
                None if flat_value is None or hier_value is None
                else hier_value - flat_value
            )
        differences[model] = model_differences

    payload = {
        "schema_version": "dsh_molbench_pretrained_matrix_summary_v1",
        "stamp": args.stamp,
        "all_publishable": all(
            run["status"] == "publishable"
            for variants in runs.values() for run in variants.values()
        ),
        "runs": runs,
        "hierarchy_minus_flat": differences,
    }
    atomic_write(args.output_json, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# Aligned-v4 original checkpoint MolBench matrix",
        "",
        "| Model | Variant | Status | Completed | Model failures | Infra retries | Strict format | MS1 acc | MS1 F1 | MS2 acc | MS2 valid |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        for variant in VARIANTS:
            run = runs[model][variant]
            summary = run["summary"] or {}
            lines.append(
                f"| {model} | {variant} | {run['status']} | "
                f"{summary.get('completed_count', 0)}/87 | "
                f"{summary.get('model_failure_count', '—')} | "
                f"{summary.get('infra_retry_count', '—')} | "
                f"{fmt(nested(summary, ('strict_format_rate',)))} | "
                f"{fmt(nested(summary, ('metrics', 'rdkit_bench_all', 'acc')))} | "
                f"{fmt(nested(summary, ('metrics', 'rdkit_bench_all', 'f1')))} | "
                f"{fmt(nested(summary, ('metrics', 'acnet_curated_all', 'acc')))} | "
                f"{fmt(nested(summary, ('metrics', 'acnet_curated_all', 'valid_rate')))} |"
            )
    lines.extend(["", "## Hierarchy − flat", ""])
    for model in models:
        delta = differences[model]
        if delta is None:
            lines.append(f"- {model}: incomplete; no official comparison published.")
        else:
            values = ", ".join(
                f"{key}={'—' if value is None else f'{value:+.4f}'}"
                for key, value in delta.items()
            )
            lines.append(f"- {model}: {values}")
    atomic_write(args.output_markdown, "\n".join(lines) + "\n")

    print(json.dumps({
        "all_publishable": payload["all_publishable"],
        "output_json": str(args.output_json),
        "output_markdown": str(args.output_markdown),
    }))


if __name__ == "__main__":
    main()
