#!/usr/bin/env python3
"""Pipeline entry for MolBench-PF (molecular property filtering / similarity)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


SCRIPT_BY_VARIANT = {
    "v0": "make_rdkit_benchmark_v0.py",
    "v1": "make_rdkit_benchmark_v1.py",
    "similarity": "molecular_similiar.py",
}

VARIANT_ALIAS = {
    "v0": "v0",
    "v1": "v1",
    "sim": "similarity",
    "similarity": "similarity",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MolBench-PF dataset.")
    parser.add_argument("--variant", choices=sorted(VARIANT_ALIAS), required=True)
    parser.add_argument("--n-cases", type=int, required=True, help="Number of cases")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument(
        "--out-name",
        default="",
        help="Optional output filename. Default: molbench-pf-<variant>-<n_cases>-<seed>.csv",
    )
    parser.add_argument(
        "--input",
        default="data/CARA/Task/VS_All.tsv",
        help="Input CARA TSV for RDKit tasks",
    )
    parser.add_argument(
        "--retry-seeds",
        type=int,
        default=0,
        help="Retry with seed+offset when output rows < n-cases (useful for similarity variant).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    canonical_variant = VARIANT_ALIAS[args.variant]
    rdkit_script = project_root / "scripts" / SCRIPT_BY_VARIANT[canonical_variant]

    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variant_tag = "sim" if canonical_variant == "similarity" else canonical_variant
    out_name = args.out_name.strip() or f"molbench-pf-{variant_tag}-{args.n_cases}-{args.seed}.csv"
    out_path = out_dir / out_name

    input_path = str((project_root / args.input).resolve())

    def _run_once(seed: int, out_csv: Path) -> int:
        cmd = [
            sys.executable,
            str(rdkit_script),
            "--input",
            input_path,
            "--out",
            str(out_csv),
            "--n_cases",
            str(args.n_cases),
            "--seed",
            str(seed),
        ]
        subprocess.run(cmd, check=True)
        try:
            df = pd.read_csv(out_csv)
        except Exception:
            return 0
        return len(df)

    # Most variants are deterministic enough; similarity may need retries.
    max_tries = max(1, args.retry_seeds + 1)
    best_rows = -1
    best_seed = args.seed
    last_tmp: Path | None = None

    for i in range(max_tries):
        seed_i = args.seed + i
        with tempfile.NamedTemporaryFile(prefix="pf_gen_", suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        rows = _run_once(seed_i, tmp_path)
        if rows > best_rows:
            best_rows = rows
            best_seed = seed_i
            if last_tmp is not None and last_tmp.exists():
                last_tmp.unlink(missing_ok=True)
            last_tmp = tmp_path
        else:
            tmp_path.unlink(missing_ok=True)
        if rows >= args.n_cases:
            break

    if last_tmp is None or not last_tmp.exists():
        raise RuntimeError("PF generation failed: no output CSV was produced.")

    last_tmp.replace(out_path)

    if best_rows < args.n_cases:
        raise RuntimeError(
            f"PF generation produced only {best_rows}/{args.n_cases} rows "
            f"(best seed={best_seed}). Increase --retry-seeds or lower --n-cases."
        )

    print(f"Wrote: {out_path} (seed={best_seed}, rows={best_rows})")


if __name__ == "__main__":
    main()
