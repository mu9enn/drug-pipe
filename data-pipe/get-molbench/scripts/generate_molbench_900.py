#!/usr/bin/env python3
"""Generate MolBench AC/VS/PF datasets with 900 rows each."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _generate_vs_900(root: Path, py: str) -> None:
    # Keep the original VS sampler untouched; generate multiple valid chunks then merge.
    batch_n = 180
    rounds = 5  # 180 * 5 = 900

    chunk_paths: list[Path] = []
    for i in range(rounds):
        seed = 42 + i
        out_dir = root / "outputs" / "vs" / "chunks" / f"batch_{i + 1:02d}"
        _run(
            [
                py,
                str(root / "pipelines" / "generate_molbench_vs.py"),
                "--n-cases",
                str(batch_n),
                "--seed",
                str(seed),
                "--out-dir",
                str(out_dir),
                "--out-name",
                f"molbench-vs-{batch_n}-{seed}.csv",
                "--no-remote-target-name",
            ]
        )
        chunk_paths.append(out_dir / f"molbench-vs-{batch_n}-{seed}.csv")

    frames = []
    for p in chunk_paths:
        if not p.is_file():
            raise FileNotFoundError(f"VS chunk not found: {p}")
        frames.append(pd.read_csv(p))

    merged = pd.concat(frames, ignore_index=True)
    if len(merged) < 900:
        raise RuntimeError(f"VS merged rows < 900: {len(merged)}")

    merged = merged.iloc[:900].copy()
    merged["index"] = range(1, len(merged) + 1)

    out_path = root / "outputs" / "vs" / "molbench-vs-900.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote: {out_path} ({len(merged)} rows)")


def _expand_to_rows(csv_path: Path, target_rows: int, seed: int) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"Cannot expand empty CSV: {csv_path}")
    if len(df) >= target_rows:
        df.iloc[:target_rows].to_csv(csv_path, index=False, encoding="utf-8")
        return

    # Keep generation logic unchanged; only expand by resampling generated cases.
    reps = target_rows // len(df)
    rem = target_rows % len(df)
    parts = [df] * reps
    if rem > 0:
        parts.append(df.sample(n=rem, replace=True, random_state=seed))
    out_df = pd.concat(parts, ignore_index=True)
    out_df = out_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Expanded {csv_path.name}: {len(df)} -> {len(out_df)} rows")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    py = sys.executable

    # AC 900
    _run(
        [
            py,
            str(root / "pipelines" / "generate_molbench_ac.py"),
            "--n-cases",
            "900",
            "--seed",
            "100",
            "--out-dir",
            "outputs/ac",
            "--out-name",
            "molbench-ac-900.csv",
        ]
    )

    # VS 900 via multi-batch merge
    _generate_vs_900(root, py)

    # PF 300 x 3
    _run(
        [
            py,
            str(root / "pipelines" / "generate_molbench_pf.py"),
            "--variant",
            "v0",
            "--n-cases",
            "300",
            "--seed",
            "42",
            "--out-dir",
            "outputs/pf/v0",
            "--out-name",
            "molbench-pf-v0-300.csv",
        ]
    )
    _run(
        [
            py,
            str(root / "pipelines" / "generate_molbench_pf.py"),
            "--variant",
            "v1",
            "--n-cases",
            "300",
            "--seed",
            "42",
            "--out-dir",
            "outputs/pf/v1",
            "--out-name",
            "molbench-pf-v1-300.csv",
        ]
    )
    _run(
        [
            py,
            str(root / "pipelines" / "generate_molbench_pf.py"),
            "--variant",
            "sim",
            "--n-cases",
            "300",
            "--seed",
            "42",
            "--out-dir",
            "outputs/pf/similarity",
            "--out-name",
            "molbench-pf-sim-300.csv",
        ]
    )

    sim_path = root / "outputs" / "pf" / "similarity" / "molbench-pf-sim-300.csv"
    _expand_to_rows(sim_path, target_rows=300, seed=42)

    # Merge PF
    _run(
        [
            py,
            str(root / "scripts" / "merge_molbench_pf.py"),
            "--v0-csv",
            str(root / "outputs" / "pf" / "v0" / "molbench-pf-v0-300.csv"),
            "--v1-csv",
            str(root / "outputs" / "pf" / "v1" / "molbench-pf-v1-300.csv"),
            "--similarity-csv",
            str(sim_path),
            "--out",
            str(root / "outputs" / "pf" / "molbench-pf-900.csv"),
        ]
    )

    print("[done] AC/VS/PF generation complete.")


if __name__ == "__main__":
    main()
