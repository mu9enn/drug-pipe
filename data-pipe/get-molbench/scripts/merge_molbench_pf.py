#!/usr/bin/env python3
"""Merge three PF variants (v0/v1/similarity) into one MolBench-PF CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


def _load_with_tag(path: Path, tag: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"PF source file not found: {path}")
    try:
        df = pd.read_csv(path)
    except EmptyDataError as e:
        raise ValueError(
            f"PF source file has no rows/columns: {path}. "
            "Please regenerate this variant with a different seed or higher retry budget."
        ) from e
    if df.empty:
        raise ValueError(f"PF source file is empty: {path}")
    df = df.copy()
    df["source_variant"] = tag
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge PF v0/v1/similarity into one CSV.")
    parser.add_argument("--v0-csv", required=True, help="PF v0 CSV path")
    parser.add_argument("--v1-csv", required=True, help="PF v1 CSV path")
    parser.add_argument(
        "--similarity-csv",
        default="",
        help="Optional PF similarity CSV path",
    )
    parser.add_argument("--out", required=True, help="Merged output CSV path")
    args = parser.parse_args()

    v0_df = _load_with_tag(Path(args.v0_csv), "v0")
    v1_df = _load_with_tag(Path(args.v1_csv), "v1")
    frames = [v0_df, v1_df]
    if args.similarity_csv.strip():
        sim_df = _load_with_tag(Path(args.similarity_csv), "similarity")
        frames.append(sim_df)

    merged = pd.concat(frames, ignore_index=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Merged rows: {len(merged)}")
    print("Variant counts:")
    print(merged["source_variant"].value_counts().to_string())
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
