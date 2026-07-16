#!/usr/bin/env python3
"""Pipeline entry for MolBench-VS (virtual screening)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MolBench-VS dataset.")
    parser.add_argument("--n-cases", type=int, required=True, help="Number of cases")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument(
        "--out-name",
        default="",
        help="Optional output filename. Default: molbench-vs-<n_cases>-<seed>.csv",
    )
    parser.add_argument(
        "--input",
        default="data/CARA/Task/VS_All.tsv",
        help="Input CARA VS TSV",
    )
    parser.add_argument(
        "--value-types",
        default="IC50,Kd,Ki",
        help="Comma-separated Value Type list",
    )
    parser.add_argument("--n-candidates", type=int, default=60)
    parser.add_argument("--min-active", type=int, default=6)
    parser.add_argument("--max-active", type=int, default=10)
    parser.add_argument("--threshold-pchembl", type=float, default=6.0)
    parser.add_argument("--max-tries-per-assay", type=int, default=200)
    parser.add_argument(
        "--no-remote-target-name",
        action="store_true",
        help="Disable ChEMBL API target name lookup",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    vs_script = project_root / "scripts" / "generate_molbench_vs.py"

    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name.strip() or f"molbench-vs-{args.n_cases}-{args.seed}.csv"

    cmd = [
        sys.executable,
        str(vs_script),
        "--input",
        str((project_root / args.input).resolve()),
        "--out-dir",
        str(out_dir),
        "--sizes",
        str(args.n_cases),
        "--seed",
        str(args.seed),
        "--value-types",
        args.value_types,
        "--n-candidates",
        str(args.n_candidates),
        "--min-active",
        str(args.min_active),
        "--max-active",
        str(args.max_active),
        "--threshold-pchembl",
        str(args.threshold_pchembl),
        "--max-tries-per-assay",
        str(args.max_tries_per_assay),
    ]
    if args.no_remote_target_name:
        cmd.append("--no-remote-target-name")

    subprocess.run(cmd, check=True)

    src = out_dir / f"MolBench-vs-{args.n_cases}.csv"
    dst = out_dir / out_name
    if src.exists():
        if dst.exists():
            dst.unlink()
        src.rename(dst)
    else:
        raise FileNotFoundError(f"VS generator output not found: {src}")
    print(f"Wrote: {dst}")


if __name__ == "__main__":
    main()
