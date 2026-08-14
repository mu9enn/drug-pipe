#!/usr/bin/env python3
"""Build deterministic candidate sets for ToolRL prompt-length gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _centered_window(items: list[tuple[int, int, str]], position: int, count: int) -> list[tuple[int, int, str]]:
    width = min(count, len(items))
    start = max(0, min(position - width // 2, len(items) - width))
    return items[start : start + width]


def build_length_probes(input_path: Path, output_dir: Path, *, candidates_per_tier: int = 4) -> dict[str, Any]:
    if candidates_per_tier < 1:
        raise ValueError("candidates_per_tier must be positive")
    # Weighted copies intentionally collapse by decision key: a length gate
    # verifies capacity, not sampling weight.
    by_key: dict[tuple[str, int, str], tuple[int, int, str]] = {}
    with input_path.open(encoding="utf-8") as source:
        for source_line, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            key = (
                str(metadata.get("source_id") or metadata.get("task_id") or ""),
                int(metadata.get("assistant_index", -1)),
                str(metadata.get("decision_type") or ""),
            )
            tokens = int(metadata.get("prompt_tokens_final", -1))
            if tokens < 0:
                raise ValueError(f"missing prompt_tokens_final at {input_path}:{source_line}")
            by_key.setdefault(key, (tokens, source_line, line))
    ranked = sorted(by_key.values(), key=lambda item: (item[0], item[1]))
    if not ranked:
        raise ValueError(f"no records in {input_path}")

    last = len(ranked) - 1
    positions = {
        "shortest": 0,
        "p50": round(last * 0.50),
        "p95": round(last * 0.95),
        "near_limit": last,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    probes = {}
    for name, position in positions.items():
        candidates = _centered_window(ranked, position, candidates_per_tier)
        path = output_dir / f"toolrl_{name}.jsonl"
        tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        tmp.write_text(
            "".join(line if line.endswith("\n") else line + "\n" for _, _, line in candidates),
            encoding="utf-8",
        )
        os.replace(tmp, path)
        probes[name] = {
            "path": str(path.resolve()),
            "candidate_count": len(candidates),
            "min_prompt_tokens": min(item[0] for item in candidates),
            "max_prompt_tokens": max(item[0] for item in candidates),
            "candidates": [
                {"source_line": source_line, "prompt_tokens": tokens}
                for tokens, source_line, _ in candidates
            ],
            "sha256": _sha256(path),
        }
    manifest = {
        "schema_version": "toolrl_length_probe_candidates_v2",
        "source": {"path": str(input_path.resolve()), "sha256": _sha256(input_path)},
        "unique_decisions": len(ranked),
        "candidates_per_tier": candidates_per_tier,
        "probes": probes,
    }
    manifest_path = output_dir / "manifest.json"
    tmp = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidates-per-tier", default=4, type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            build_length_probes(
                args.input.resolve(),
                args.output_dir.resolve(),
                candidates_per_tier=args.candidates_per_tier,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
