from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from drug_agent.constants import DEFAULT_SYSTEM_PROMPT
from drug_agent.utils import ensure_dir, write_json, write_jsonl


EXPECTED_COUNTS = {"molbench_ms1": 50, "molbench_ms2": 33, "molbench_ms3": 25, "molbench_mo": 78}
SELECTABLE_SUITES = set(EXPECTED_COUNTS) | {"molbench_mo_edit", "molbench_mo_opt"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(_normalized_prompt(text).encode("utf-8")).hexdigest()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sample(
    *,
    task_id: str,
    task_type: str,
    prompt: str,
    label: dict[str, Any],
    suite: str,
    subtask: str,
    source_path: Path,
    max_steps: int,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "prompt": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "label": label,
        "metadata": {
            "env_kwargs": {
                "task_id": task_id,
                "task_type": task_type,
                "data_source": suite,
                "max_steps": max_steps,
            },
            "benchmark": {
                "suite": suite,
                "subtask": subtask,
                "source_path": str(source_path),
            },
        },
    }


def _suite_group(sample: dict[str, Any]) -> str:
    suite = str(sample["metadata"]["benchmark"]["suite"])
    return "molbench_mo" if suite.startswith("molbench_mo_") else suite


def _select_samples(
    samples: list[dict[str, Any]],
    *,
    selected_suites: list[str] | None,
    limit_per_suite: int,
) -> list[dict[str, Any]]:
    suites = list(dict.fromkeys(selected_suites or []))
    unknown = sorted(set(suites) - SELECTABLE_SUITES)
    if unknown:
        raise ValueError(f"Unknown MolBench suite selection: {unknown}")
    if limit_per_suite < 0:
        raise ValueError("limit_per_suite must be non-negative (0 means unlimited)")
    if not suites and limit_per_suite == 0:
        return list(samples)

    active_suites = suites or list(EXPECTED_COUNTS)
    kept: list[dict[str, Any]] = []
    counts: dict[str, int] = {suite: 0 for suite in active_suites}
    for sample in samples:
        raw_suite = str(sample["metadata"]["benchmark"]["suite"])
        grouped_suite = _suite_group(sample)
        suite = next(
            (candidate for candidate in active_suites if candidate in {raw_suite, grouped_suite}),
            None,
        )
        if suite is None:
            continue
        if limit_per_suite and counts[suite] >= limit_per_suite:
            continue
        kept.append(sample)
        counts[suite] += 1
    return kept


def _extract_pair(question: str) -> tuple[str, str]:
    match_a = re.search(r"Molecule A:\s*([^\n]+)", question)
    match_b = re.search(r"Molecule B:\s*([^\n]+)", question)
    return (
        match_a.group(1).strip() if match_a else "",
        match_b.group(1).strip() if match_b else "",
    )


def _extract_source_smiles(query: str, marker: str) -> str:
    match = re.search(rf"{re.escape(marker)}\s*([^\n]+)", query, re.IGNORECASE)
    if not match:
        return ""
    value = match.group(1).strip().strip('"')
    if marker.lower().startswith("input"):
        value = re.split(r",\s*Functional Group", value, maxsplit=1, flags=re.IGNORECASE)[0]
    return value.strip().rstrip(".")


def build_molbench_dataset(
    molbench_root: str | Path,
    output_dir: str | Path,
    *,
    selected_suites: list[str] | None = None,
    limit_per_suite: int = 0,
    max_steps: int = 0,
) -> dict[str, Any]:
    root = Path(molbench_root).expanduser().resolve()
    output = ensure_dir(output_dir)
    data_root = root / "data"
    required = {
        "ms1": data_root / "molbench-ms-1/molbench-ms-1.csv",
        "ms2": data_root / "molbench-ms-2/molbench-ms-2.csv",
        "ms3": data_root / "molbench-ms-3/molbench-ms-3.csv",
        "mo_manifest": data_root / "molbench-mo/manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MolBench source files missing: {missing}")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative (0 means unlimited)")

    exclusion_path = Path(__file__).with_name("molbench_exclusions.json")
    exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
    excluded_hashes = set(exclusion["molbench_ms2_prompt_sha256"])
    samples: list[dict[str, Any]] = []
    overlap_audit: list[dict[str, Any]] = []

    for index, row in enumerate(_csv_rows(required["ms1"]), start=1):
        prompt = row.get("prompt") or row.get("\ufeffprompt") or ""
        samples.append(
            _sample(
                task_id=f"molbench_ms1_{index:03d}", task_type="pf", prompt=prompt,
                label={"answer": row.get("answer", "")}, suite="molbench_ms1", subtask="filter",
                source_path=required["ms1"], max_steps=max_steps,
            )
        )

    for index, row in enumerate(_csv_rows(required["ms2"]), start=1):
        question = row.get("question") or row.get("\ufeffquestion") or ""
        digest = _prompt_hash(question)
        if digest in excluded_hashes:
            overlap_audit.append(
                {"source_row": index, "target": row.get("target"), "prompt_sha256": digest, "reason": "training_overlap"}
            )
            continue
        s1, s2 = _extract_pair(question)
        samples.append(
            _sample(
                task_id=f"molbench_ms2_{index:03d}", task_type="ac", prompt=question,
                label={"answer": row.get("answer", ""), "target": row.get("target", ""), "s1": s1, "s2": s2},
                suite="molbench_ms2", subtask="qa", source_path=required["ms2"], max_steps=max_steps,
            )
        )

    for index, row in enumerate(_csv_rows(required["ms3"]), start=1):
        question_text = row.get("questions", "")
        try:
            question_payload = json.loads(question_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"MS-3 row {index} has invalid questions JSON: {exc}") from exc
        candidates = question_payload.get("candidates") if isinstance(question_payload, dict) else None
        if not isinstance(candidates, list) or len(candidates) != 60:
            raise ValueError(f"MS-3 row {index} must contain exactly 60 candidates")
        answer = json.loads(row.get("answer") or "[]")
        samples.append(
            _sample(
                task_id=f"molbench_ms3_{index:03d}", task_type="vs", prompt=question_text,
                label={"answer": answer, "answer_score": json.loads(row.get("answer_score") or "[]"), "candidates": candidates, "index": row.get("index")},
                suite="molbench_ms3", subtask="ranking", source_path=required["ms3"], max_steps=max_steps,
            )
        )

    mo_files = sorted((data_root / "molbench-mo").glob("molbench-mo-*/*.json"))
    for path in mo_files:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"MO source must contain a list: {path}")
        for index, row in enumerate(rows, start=1):
            query = str(row.get("query") or "")
            meta = row.get("meta") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            subtask = str(row.get("subtask") or path.stem)
            if row.get("task") == "mol_edit":
                task_type = "mol_edit"
                source_smiles = _extract_source_smiles(query, "Input Molecule:")
                label = {"source_smiles": source_smiles, "subtask": subtask, **meta}
                suite = "molbench_mo_edit"
            else:
                task_type = "mol_opt_physchem"
                source_smiles = _extract_source_smiles(query, "Source Molecule:")
                label = {"source_smiles": source_smiles, "subtask": subtask, **meta}
                suite = "molbench_mo_opt"
            samples.append(
                _sample(
                    task_id=f"{suite}_{subtask}_{index:03d}", task_type=task_type, prompt=query,
                    label=label, suite=suite, subtask=subtask, source_path=path, max_steps=max_steps,
                )
            )

    source_counts = {
        "molbench_ms1": sum(s["metadata"]["benchmark"]["suite"] == "molbench_ms1" for s in samples),
        "molbench_ms2": sum(s["metadata"]["benchmark"]["suite"] == "molbench_ms2" for s in samples),
        "molbench_ms3": sum(s["metadata"]["benchmark"]["suite"] == "molbench_ms3" for s in samples),
        "molbench_mo": sum(s["metadata"]["benchmark"]["suite"].startswith("molbench_mo_") for s in samples),
    }
    if source_counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected source benchmark counts: {source_counts}, expected {EXPECTED_COUNTS}")
    if len(overlap_audit) != 4:
        raise ValueError(f"Expected four MS-2 overlap exclusions, found {len(overlap_audit)}")

    samples = _select_samples(
        samples,
        selected_suites=selected_suites,
        limit_per_suite=limit_per_suite,
    )
    counts = {
        suite: sum(_suite_group(sample) == suite for sample in samples)
        for suite in EXPECTED_COUNTS
    }
    if not samples:
        raise ValueError("MolBench selection produced no samples")

    dataset_path = output / "molbench_eval.jsonl"
    write_jsonl(dataset_path, samples)
    write_jsonl(output / "overlap_audit.jsonl", overlap_audit)
    source_files = sorted({Path(s["metadata"]["benchmark"]["source_path"]) for s in samples} | set(required.values()))
    raw_mo_manifest = json.loads(required["mo_manifest"].read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "drug_agent_molbench_eval_manifest_v1",
        "molbench_root": str(root),
        "dataset_path": str(dataset_path),
        "counts": counts,
        "source_counts": source_counts,
        "total": len(samples),
        "selection": {
            "suites": list(dict.fromkeys(selected_suites or [])),
            "limit_per_suite": limit_per_suite,
            "max_steps": max_steps,
        },
        "excluded_training_overlap": len(overlap_audit),
        "source_files": [{"path": str(path), "sha256": _sha256_file(path)} for path in source_files],
        "source_mo_manifest_total": raw_mo_manifest.get("total_samples"),
        "available_mo_total": source_counts["molbench_mo"],
        "selected_mo_total": counts["molbench_mo"],
        "missing_mo_target_optimization": int(raw_mo_manifest.get("total_samples", 0)) - source_counts["molbench_mo"],
    }
    write_json(output / "benchmark_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fresh held-out Drug-Agent MolBench evaluation JSONL")
    parser.add_argument("--molbench-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suite", action="append", default=[])
    parser.add_argument("--limit-per-suite", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(build_molbench_dataset(
        args.molbench_root,
        args.output_dir,
        selected_suites=args.suite,
        limit_per_suite=args.limit_per_suite,
        max_steps=args.max_steps,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
