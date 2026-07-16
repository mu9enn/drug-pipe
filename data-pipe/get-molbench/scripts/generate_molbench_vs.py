#!/usr/bin/env python3
"""Generate MolBench-vs style CSV datasets in batch.

Output schema matches the current bench expectation:
- index
- questions
- task_type
- answer
- answer_score
- n_active
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd


DEFAULT_VALUE_TYPES = ["IC50", "Kd", "Ki"]


@dataclass
class BuildConfig:
    n_candidates: int = 60
    min_active: int = 6
    max_active: int = 10
    threshold_pchembl: float = 6.0
    max_tries_per_assay: int = 200


def parse_sizes(raw: str) -> List[int]:
    sizes: List[int] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        value = int(x)
        if value <= 0:
            raise ValueError(f"Invalid size: {value}")
        sizes.append(value)
    if not sizes:
        raise ValueError("--sizes is empty")
    return sizes


def resolve_target_name(target_id: str, timeout_sec: int = 10) -> Optional[str]:
    url = f"https://www.ebi.ac.uk/chembl/api/data/target/{target_id}.json"
    try:
        with urlopen(url, timeout=timeout_sec) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read().decode("utf-8"))
            value = payload.get("pref_name")
            if isinstance(value, str) and value.strip():
                return value.strip()
            return None
    except (URLError, TimeoutError, ValueError):
        return None


def build_question(target_id: str, target_name: str, candidates: Sequence[str], n_candidates: int) -> str:
    prompt = {
        "task": "You are a computational medicinal chemist tasked with docking score ranking for a set of candidate molecules against a specific drug discovery target.",
        "objective": f"Rank all {n_candidates} candidate compounds by their predicted docking scores against the target, from the most favorable (strongest binding) to the least favorable.",
        "output_format": f"Return a JSON array of exactly {n_candidates} SMILES strings, ordered from the most favorable docking score to the least.",
        "target_chembl_id": target_id,
        "target_name": target_name,
        "note": f"The candidate list contains {n_candidates} molecules. Your goal is to provide a complete ranking of all molecules based on docking performance, reflecting their structural complementarity to the target.",
        "candidates": list(candidates),
    }
    return json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))


def sample_case(
    assay_df: pd.DataFrame,
    cfg: BuildConfig,
    rng: random.Random,
    target_name_cache: Dict[str, str],
    use_remote_target_name: bool,
) -> Optional[Dict[str, str]]:
    pool = assay_df.drop_duplicates(subset=["Smiles"]).copy()
    if len(pool) < cfg.n_candidates:
        return None

    target_id = str(pool.iloc[0]["Target ChEMBL ID"])

    actives = pool[pool["pChEMBL Value"] >= cfg.threshold_pchembl]
    inactives = pool[pool["pChEMBL Value"] < cfg.threshold_pchembl]

    if len(actives) < cfg.min_active:
        return None

    max_k = min(cfg.max_active, len(actives), cfg.n_candidates)
    feasible_k = [
        k for k in range(cfg.min_active, max_k + 1) if len(inactives) >= (cfg.n_candidates - k)
    ]
    if not feasible_k:
        return None

    for _ in range(cfg.max_tries_per_assay):
        n_active = rng.choice(feasible_k)

        sampled_active = actives.sample(n=n_active, random_state=rng.randint(0, 10**9))
        sampled_inactive = inactives.sample(
            n=cfg.n_candidates - n_active,
            random_state=rng.randint(0, 10**9),
        )
        sampled = (
            pd.concat([sampled_active, sampled_inactive], ignore_index=True)
            .sample(frac=1.0, random_state=rng.randint(0, 10**9))
            .reset_index(drop=True)
        )

        actives_sorted = sampled[sampled["pChEMBL Value"] >= cfg.threshold_pchembl].sort_values(
            by="pChEMBL Value", ascending=False
        )
        if not (cfg.min_active <= len(actives_sorted) <= cfg.max_active):
            continue

        if target_id not in target_name_cache:
            target_name = target_id
            if use_remote_target_name:
                resolved = resolve_target_name(target_id)
                if resolved:
                    target_name = resolved
            target_name_cache[target_id] = target_name

        question = build_question(
            target_id=target_id,
            target_name=target_name_cache[target_id],
            candidates=sampled["Smiles"].astype(str).tolist(),
            n_candidates=cfg.n_candidates,
        )

        answer_smiles = actives_sorted["Smiles"].astype(str).tolist()
        answer_scores = [round(float(x), 4) for x in actives_sorted["pChEMBL Value"].tolist()]

        return {
            "questions": question,
            "task_type": "zero-shot",
            "answer": json.dumps(answer_smiles, ensure_ascii=False, separators=(",", ":")),
            "answer_score": json.dumps(answer_scores, ensure_ascii=False, separators=(",", ":")),
            "n_active": str(len(answer_smiles)),
        }

    return None


def build_dataset(
    input_tsv: Path,
    n_cases: int,
    seed: int,
    cfg: BuildConfig,
    value_types: Sequence[str],
    use_remote_target_name: bool,
) -> pd.DataFrame:
    rng = random.Random(seed)

    df = pd.read_csv(input_tsv, sep="\t", dtype=str)
    required = [
        "Assay ChEMBL ID",
        "Target Cluster 0.3",
        "Target ChEMBL ID",
        "Smiles",
        "Value Type",
        "pChEMBL Value",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["Value Type"].isin(value_types)].copy()
    df["pChEMBL Value"] = pd.to_numeric(df["pChEMBL Value"], errors="coerce")
    df["Smiles"] = df["Smiles"].astype(str)
    df = df.dropna(subset=["Smiles", "pChEMBL Value"])

    df = df.drop_duplicates(subset=["Assay ChEMBL ID", "Smiles"])

    eligible_assays: List[str] = []
    for assay_id, g in df.groupby("Assay ChEMBL ID"):
        if len(g) < cfg.n_candidates:
            continue
        n_active = int((g["pChEMBL Value"] >= cfg.threshold_pchembl).sum())
        n_inactive = int((g["pChEMBL Value"] < cfg.threshold_pchembl).sum())
        if n_active < cfg.min_active:
            continue
        if n_inactive < (cfg.n_candidates - cfg.max_active):
            continue
        eligible_assays.append(assay_id)

    if not eligible_assays:
        raise RuntimeError("No assay passes the sampling constraints.")

    eligible_df = df[df["Assay ChEMBL ID"].isin(eligible_assays)].copy()

    one_per_cluster: List[str] = []
    for _, g in eligible_df.groupby("Target Cluster 0.3"):
        assays = g["Assay ChEMBL ID"].drop_duplicates().tolist()
        one_per_cluster.append(rng.choice(assays))

    rng.shuffle(one_per_cluster)

    selected_assays = one_per_cluster[: min(len(one_per_cluster), n_cases)]
    if len(selected_assays) < n_cases:
        remaining = [a for a in eligible_assays if a not in set(selected_assays)]
        rng.shuffle(remaining)
        need = n_cases - len(selected_assays)
        selected_assays.extend(remaining[:need])

    rows: List[Dict[str, str]] = []
    cache: Dict[str, str] = {}

    for assay_id in selected_assays:
        assay_df = eligible_df[eligible_df["Assay ChEMBL ID"] == assay_id]
        row = sample_case(
            assay_df=assay_df,
            cfg=cfg,
            rng=rng,
            target_name_cache=cache,
            use_remote_target_name=use_remote_target_name,
        )
        if row is not None:
            rows.append(row)

    # Backfill from all eligible assays if some selected assays fail in sampling.
    if len(rows) < n_cases:
        fallback_assays = [a for a in eligible_assays if a not in set(selected_assays)]
        rng.shuffle(fallback_assays)
        for assay_id in fallback_assays:
            assay_df = eligible_df[eligible_df["Assay ChEMBL ID"] == assay_id]
            row = sample_case(
                assay_df=assay_df,
                cfg=cfg,
                rng=rng,
                target_name_cache=cache,
                use_remote_target_name=use_remote_target_name,
            )
            if row is not None:
                rows.append(row)
            if len(rows) >= n_cases:
                break

    if len(rows) < n_cases:
        raise RuntimeError(
            f"Only generated {len(rows)} rows. Try smaller --sizes or looser constraints."
        )

    out_df = pd.DataFrame(rows[:n_cases])
    out_df.insert(0, "index", range(1, len(out_df) + 1))
    return out_df


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(description="Batch-generate MolBench-vs style CSV files.")
    parser.add_argument(
        "--input",
        default=str(project_root / "data" / "CARA" / "Task" / "VS_All.tsv"),
        help="Path to CARA VS_All.tsv",
    )
    parser.add_argument(
        "--out-dir",
        default=str(project_root / "outputs" / "MolBench-vs"),
        help="Output directory for MolBench-vs-<N>.csv files",
    )
    parser.add_argument(
        "--sizes",
        default="25",
        help="Comma-separated case counts, e.g. 25 or 25,50,100",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--value-types",
        default=",".join(DEFAULT_VALUE_TYPES),
        help="Comma-separated endpoint types from Value Type column",
    )
    parser.add_argument("--n-candidates", type=int, default=60)
    parser.add_argument("--min-active", type=int, default=6)
    parser.add_argument("--max-active", type=int, default=10)
    parser.add_argument("--threshold-pchembl", type=float, default=6.0)
    parser.add_argument("--max-tries-per-assay", type=int, default=200)
    parser.add_argument(
        "--no-remote-target-name",
        action="store_true",
        help="Do not query ChEMBL API for target names; use target ID as fallback name",
    )
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes)
    value_types = [x.strip() for x in args.value_types.split(",") if x.strip()]

    cfg = BuildConfig(
        n_candidates=args.n_candidates,
        min_active=args.min_active,
        max_active=args.max_active,
        threshold_pchembl=args.threshold_pchembl,
        max_tries_per_assay=args.max_tries_per_assay,
    )

    input_tsv = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, n_cases in enumerate(sizes):
        run_seed = args.seed + i
        out_df = build_dataset(
            input_tsv=input_tsv,
            n_cases=n_cases,
            seed=run_seed,
            cfg=cfg,
            value_types=value_types,
            use_remote_target_name=(not args.no_remote_target_name),
        )
        out_path = out_dir / f"MolBench-vs-{n_cases}.csv"
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
