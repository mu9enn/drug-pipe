# make_similarity_benchmark.py
import argparse
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

# -------------------------------
# Basic utilities
# -------------------------------

def safe_mol(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None

def canonical_smiles(smiles: str):
    m = safe_mol(smiles)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True)

# -------------------------------
# Morgan fingerprint similarity
# -------------------------------

def morgan_fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )

def calc_similarity(smiles_a, smiles_b):
    ma = safe_mol(smiles_a)
    mb = safe_mol(smiles_b)
    if ma is None or mb is None:
        return None
    fa = morgan_fp(ma)
    fb = morgan_fp(mb)
    return float(TanimotoSimilarity(fa, fb))

# -------------------------------
# Difficulty (Updated logic)
# -------------------------------

def difficulty_from_gap(gap: float):
    # 根据 Top1 和 Top2 的差距定义难度
    if gap > 0.15:
        return "easy"
    if gap > 0.08:
        return "medium"
    return "hard"

# -------------------------------
# Prompt builders (Updated to single output)
# -------------------------------

def build_prompt_tanimoto(smiles10):
    lines = []
    lines.append("You are a cheminformatics assistant.")
    lines.append("")
    lines.append("Task:")
    lines.append("Find the molecule MOST similar to the FIRST molecule.")
    lines.append("Similarity is defined as Morgan fingerprint (radius=2, 2048 bits) Tanimoto similarity.")
    lines.append("")
    lines.append("SMILES:")
    lines.extend(smiles10)
    lines.append("")
    lines.append("Output format:")
    lines.append("Print ONLY the selected SMILES and nothing else.")
    return "\n".join(lines)

def build_prompt_morgan(smiles10):
    lines = []
    lines.append("You are a cheminformatics assistant.")
    lines.append("")
    lines.append("Task:")
    lines.append("Find the molecule sharing the MOST structural fragments with the FIRST molecule.")
    lines.append("Fragments are defined by Morgan fingerprint (radius=2).")
    lines.append("")
    lines.append("SMILES:")
    lines.extend(smiles10)
    lines.append("")
    lines.append("Output format:")
    lines.append("Print ONLY the selected SMILES and nothing else.")
    return "\n".join(lines)

# -------------------------------
# Case generation (Updated logic)
# -------------------------------

@dataclass
class CaseMeta:
    assay_id: str
    similarities: List[float]
    selected_count: int
    gap: float

def try_make_case(assay_id, smiles_pool, rng):
    if len(smiles_pool) < 10:
        return None

    smiles10 = rng.sample(smiles_pool, 10)
    query = smiles10[0]
    sims = []

    for s in smiles10[1:]:
        sim = calc_similarity(query, s)
        if sim is None:
            return None
        # 过滤掉极端情况
        if sim < 0.20 or sim > 0.98:
            return None
        sims.append((s, sim))

    # 按相似度从高到低排序
    sims.sort(key=lambda x: x[1], reverse=True)

    if len(sims) < 2:
        return None

    # 核心修改：检查第一名和第二名的差距
    top1_sim = sims[0][1]
    top2_sim = sims[1][1]
    gap = top1_sim - top2_sim

    if gap <= 0.02:
        return None

    # 固定只选最相似的一个
    k = 1
    selected = [sims[0][0]]

    difficulty = difficulty_from_gap(gap)
    task_type = rng.choice(["morgan", "tanimoto"])

    if task_type == "morgan":
        prompt = build_prompt_morgan(smiles10)
    else:
        prompt = build_prompt_tanimoto(smiles10)

    answer = selected[0]

    meta = CaseMeta(
        assay_id=assay_id,
        similarities=[round(x[1], 3) for x in sims],
        selected_count=k,
        gap=round(gap, 3)
    )

    return prompt, answer, task_type, difficulty, meta

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="benchmark.csv")
    ap.add_argument("--n_cases", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--assay_col", default="Assay ChEMBL ID")
    ap.add_argument("--smiles_col", default="Smiles")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    
    # 假设输入是 tsv
    df = pd.read_csv(args.input, sep="\t", dtype=str)
    df = df[[args.assay_col, args.smiles_col]].dropna()

    # 标准化 SMILES
    df[args.smiles_col] = df[args.smiles_col].apply(canonical_smiles)
    df = df.dropna()

    g = df.groupby(args.assay_col)[args.smiles_col].apply(
        lambda s: list(set(s.tolist()))
    )

    eligible = [
        (assay, smi)
        for assay, smi in g.items()
        if len(smi) >= 10
    ]

    rows = []
    attempts = 0
    max_attempts = args.n_cases * 100 # 防止死循环

    while len(rows) < args.n_cases and attempts < max_attempts:
        attempts += 1
        assay_id, pool = rng.choice(eligible)
        case = try_make_case(assay_id, pool, rng)

        if case is None:
            continue

        prompt, answer, task_type, difficulty, meta = case

        rows.append({
            "prompt": prompt,
            "answer": answer,
            "task_type": task_type,
            "difficulty": difficulty,
            "meta": json.dumps(asdict(meta), ensure_ascii=False)
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False, encoding="utf-8")

    print(f"Generated {len(rows)} cases after {attempts} attempts. Wrote to {args.out}")

if __name__ == "__main__":
    main()