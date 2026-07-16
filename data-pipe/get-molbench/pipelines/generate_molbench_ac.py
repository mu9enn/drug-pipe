#!/usr/bin/env python3
"""Pipeline entry for MolBench-AC (binding affinity comparison)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MolBench-AC dataset.")
    parser.add_argument("--n-cases", type=int, required=True, help="Number of cases")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument(
        "--out-name",
        default="",
        help="Optional output filename. Default: molbench-ac-<n_cases>-<seed>.csv",
    )
    parser.add_argument(
        "--input-csv",
        default="data/ACNet/mmp_ac_s_distinct.csv",
        help="Input ACNet CSV",
    )
    parser.add_argument(
        "--target-dict-xlsx",
        default="data/ACNet/target_dictionary.xlsx",
        help="Target dictionary XLSX",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    ac_script = project_root / "scripts" / "generate_dataset_ACNet_v0.2.py"

    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name.strip() or f"molbench-ac-{args.n_cases}-{args.seed}.csv"
    out_path = out_dir / out_name

    cmd = [
        sys.executable,
        str(ac_script),
        "--input-csv",
        str((project_root / args.input_csv).resolve()),
        "--target-dict-xlsx",
        str((project_root / args.target_dict_xlsx).resolve()),
        "--n-cases",
        str(args.n_cases),
        "--seed",
        str(args.seed),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
