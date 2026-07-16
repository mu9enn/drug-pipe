# make_benchmark.py
# Usage:
#   python make_benchmark.py --input LO_All.tsv --out benchmark.csv --n_cases 15 --seed 42
# 限制条件改成其他的

import argparse
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors


# ---------- Property oracle (RDKit) ----------
def _safe_mol(smiles: str) -> Optional[Chem.Mol]:
    try:
        m = Chem.MolFromSmiles(smiles)
        return m
    except Exception:
        return None


def calc_props(smiles: str) -> Optional[Dict[str, float]]:
    """Return RDKit properties. None if invalid SMILES."""
    m = _safe_mol(smiles)
    if m is None:
        return None

    return {
        # Lipinski properties
        "MolWt": float(Descriptors.MolWt(m)),
        "MolLogP": float(Crippen.MolLogP(m)),
        "HBD": float(rdMolDescriptors.CalcNumHBD(m)),
        "HBA": float(rdMolDescriptors.CalcNumHBA(m)),

        # polarity
        "TPSA": float(rdMolDescriptors.CalcTPSA(m)),

        # flexibility
        "RotB": float(Descriptors.NumRotatableBonds(m)),

        # ring / topology
        "RingCount": float(rdMolDescriptors.CalcNumRings(m)),
        "AromaticRings": float(rdMolDescriptors.CalcNumAromaticRings(m)),

        # complexity / 3D character
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(m)),

        # atom composition
        "HeavyAtoms": float(rdMolDescriptors.CalcNumHeavyAtoms(m)),
        "HeteroAtoms": float(rdMolDescriptors.CalcNumHeteroatoms(m)),
    }


def lipinski_ro5_ok(p: Dict[str, float]) -> bool:
    """Strict Lipinski Rule of Five"""
    return (
        p["MolWt"] <= 500.0
        and p["HBD"] <= 5.0
        and p["HBA"] <= 10.0
        and p["MolLogP"] <= 5.0
    )


# ---------- Constraint system ----------
@dataclass
class ExtraConstraint:
    name: str
    op: str
    threshold: float


def apply_extra(p: Dict[str, float], c: ExtraConstraint) -> bool:
    v = p[c.name]
    if c.op == ">=":
        return v >= c.threshold
    if c.op == "<=":
        return v <= c.threshold
    raise ValueError(f"Unknown op: {c.op}")


def quantile_threshold(values: List[float], q: float) -> float:
    """Compute quantile threshold and round for stability"""
    t = float(np.quantile(np.array(values, dtype=float), q))
    return float(np.round(t, 2))


def build_prompt(smiles10: List[str], constraints: List[ExtraConstraint]) -> str:
    lines = []
    lines.append("You are a cheminformatics assistant.")
    lines.append("")
    lines.append("Task:")
    lines.append("From the following SMILES list, output ALL molecules that satisfy ALL constraints below.")
    lines.append("You must ONLY choose from the given SMILES and output them EXACTLY as provided (original strings).")
    lines.append("If none satisfy, output an empty line.")
    lines.append("")
    lines.append("SMILES:")
    lines.extend(smiles10)
    lines.append("")
    lines.append("Constraints:")
    lines.append("1) Lipinski Rule of Five MUST ALL be satisfied:")
    lines.append("   - MolWt <= 500")
    lines.append("   - NumHDonors <= 5")
    lines.append("   - NumHAcceptors <= 10")
    lines.append("   - MolLogP <= 5")
    lines.append("2) Additional constraints:")
    for c in constraints:
        lines.append(f"   - {c.name} {c.op} {c.threshold}")
    lines.append("")
    lines.append("Output format:")
    lines.append("Print each satisfying SMILES on its own line, and nothing else.")
    return "\n".join(lines)


def oracle_select(
    smiles10: List[str],
    props_map: Dict[str, Dict[str, float]],
    constraints: List[ExtraConstraint],
) -> List[str]:

    selected = []

    for s in smiles10:

        p = props_map[s]

        if not lipinski_ro5_ok(p):
            continue

        ok = True

        for c in constraints:
            if not apply_extra(p, c):
                ok = False
                break

        if ok:
            selected.append(s)

    return selected


# ---------- Case generation with k in [1,5] ----------

EXTRA_POOL = [

    ("RingCount", "<=", [0.3, 0.4, 0.5, 0.6]),
    ("RingCount", ">=", [0.4, 0.5, 0.6, 0.7]),

    ("AromaticRings", "<=", [0.3, 0.4, 0.5, 0.6]),
    ("AromaticRings", ">=", [0.4, 0.5, 0.6, 0.7]),

    ("TPSA", "<=", [0.3, 0.4, 0.5, 0.6]),
    ("TPSA", ">=", [0.4, 0.5, 0.6, 0.7]),

    ("FractionCSP3", "<=", [0.3, 0.4, 0.5, 0.6]),
    ("FractionCSP3", ">=", [0.4, 0.5, 0.6, 0.7]),

    ("HeavyAtoms", "<=", [0.3, 0.4, 0.5, 0.6]),
    ("HeavyAtoms", ">=", [0.4, 0.5, 0.6, 0.7]),

    ("HeteroAtoms", "<=", [0.3, 0.4, 0.5, 0.6]),
    ("HeteroAtoms", ">=", [0.4, 0.5, 0.6, 0.7]),
]


@dataclass
class CaseMeta:
    assay_id: str
    seed: int
    smiles: List[str]
    constraints: List[ExtraConstraint]
    selected_count: int
    attempts: int


def try_make_case(
    assay_id: str,
    smiles_pool: List[str],
    rng: random.Random,
    max_attempts: int = 60,
    k_min: int = 1,
    k_max: int = 5,
) -> Optional[Tuple[str, str, CaseMeta]]:

    for attempt in range(1, max_attempts + 1):

        smiles10 = rng.sample(smiles_pool, 10)

        props_map = {}

        valid = True

        for s in smiles10:

            p = calc_props(s)

            if p is None:
                valid = False
                break

            props_map[s] = p

        if not valid:
            continue

        n_extra = 2 if rng.random() < 0.8 else 3

        constraints: List[ExtraConstraint] = []

        used = set()

        for _ in range(n_extra):

            for _inner in range(20):

                name, op, qs = rng.choice(EXTRA_POOL)

                key = (name, op)

                if key in used:
                    continue

                used.add(key)

                values = [props_map[s][name] for s in smiles10]

                q = rng.choice(qs)

                thr = quantile_threshold(values, q)

                constraints.append(
                    ExtraConstraint(name=name, op=op, threshold=thr)
                )

                break

        selected = oracle_select(smiles10, props_map, constraints)

        k = len(selected)

        if k_min <= k <= k_max:

            prompt = build_prompt(smiles10, constraints)

            answer = "\n".join(selected)

            meta = CaseMeta(
                assay_id=assay_id,
                seed=rng.randint(0, 2**31 - 1),
                smiles=smiles10,
                constraints=constraints,
                selected_count=k,
                attempts=attempt,
            )

            return prompt, answer, meta

    return None


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="benchmark.csv")

    ap.add_argument("--n_cases", type=int, default=15)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--assay_col", default="Assay ChEMBL ID")

    ap.add_argument("--smiles_col", default="Smiles")

    ap.add_argument("--max_assay_tries", type=int, default=2000)

    args = ap.parse_args()

    rng = random.Random(args.seed)

    df = pd.read_csv(args.input, sep="\t", dtype=str)

    df = df[[args.assay_col, args.smiles_col]].dropna()

    df[args.smiles_col] = df[args.smiles_col].astype(str)

    g = df.groupby(args.assay_col)[args.smiles_col].apply(
        lambda s: list(dict.fromkeys(s.tolist()))
    )

    eligible = [
        (assay, smi_list)
        for assay, smi_list in g.items()
        if len(smi_list) >= 10
    ]

    if len(eligible) == 0:
        raise RuntimeError("No eligible assays with >=10 unique SMILES found.")

    rng.shuffle(eligible)

    rows = []

    used_assays = set()

    assay_try = 0

    while len(rows) < args.n_cases and assay_try < args.max_assay_tries:

        assay_try += 1

        assay_id, smiles_pool = eligible[assay_try % len(eligible)]

        if assay_id in used_assays and len(eligible) >= args.n_cases:
            continue

        case = try_make_case(assay_id, smiles_pool, rng)

        if case is None:
            continue

        prompt, answer, meta = case

        used_assays.add(assay_id)

        rows.append({
            "prompt": prompt,
            "answer": answer,
            "meta": json.dumps({
                **asdict(meta),
                "constraints": [asdict(c) for c in meta.constraints],
            }, ensure_ascii=False),
        })

    if len(rows) < args.n_cases:
        raise RuntimeError(
            f"Only generated {len(rows)} cases; try increasing --max_assay_tries."
        )

    out_df = pd.DataFrame(rows)

    out_df.to_csv(args.out, index=False, encoding="utf-8")

    print(f"Wrote {len(rows)} cases to {args.out}")


if __name__ == "__main__":
    main()
