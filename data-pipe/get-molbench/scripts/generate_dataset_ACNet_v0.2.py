import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_qa(r):
    target = r["target_name"]
    s1, k1 = r["c1"], float(r["Ki_1"])
    s2, k2 = r["c2"], float(r["Ki_2"])

    # 随机决定问“更高Ki”还是“更低Ki”
    ask_lower = np.random.rand() < 0.5

    if ask_lower:
        # 更低Ki = 更强结合（通常）
        q = (
            f"You are a computational medicinal chemist. For the target {target}, "
            f"you are given two small molecules:\n\n"
            f"Molecule A: {s1}\n"
            f"Molecule B: {s2}\n\n"
            f"""Predict which molecule is more likely to have higher binding affinity (lower Ki) against the target.
                Use available evidence and tool outputs to make a justified decision whenever possible.
                Only output the corresponding SMILES. """
        )
        ans = s1 if k1 < k2 else s2
    else:
        q = (
            f"You are a computational medicinal chemist. For the target {target}, "
            f"you are given two small molecules:\n\n"
            f"Molecule A: {s1}\n"
            f"Molecule B: {s2}\n\n"
            f"""Predict which molecule is more likely to have lower binding affinity (higher Ki) against the target.
Use available evidence and tool outputs to make a justified decision whenever possible.
Only output the corresponding SMILES. """
        )
        ans = s1 if k1 > k2 else s2

    return pd.Series(
        {
            "question": q,
            "answer": ans,
            "target": target,
            "s1": s1,
            "k1": k1,
            "s2": s2,
            "k2": k2,
        }
    )


def build_dataset(csv_path: Path, dict_xlsx_path: Path, n_cases: int, seed: int) -> pd.DataFrame:
    np.random.seed(seed)

    # ====== 读入数据 ======
    df = pd.read_csv(csv_path)
    # 期望列：c1 Ki_1 c2 Ki_2 tid
    needed = {"c1", "Ki_1", "c2", "Ki_2", "tid"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV缺少列: {missing}，当前列为: {list(df.columns)}")

    # tid可能读成字符串，这里尽量转成整数
    df["tid_raw"] = df["tid"]
    df["tid"] = pd.to_numeric(df["tid"], errors="coerce")

    # ====== 读入字典并做tid->target_name映射（第一个sheet）=====
    dict_df = pd.read_excel(dict_xlsx_path, sheet_name=0)
    # 期望列：tid target_name num_compounds（至少前两列）
    if "tid" not in dict_df.columns or "target_name" not in dict_df.columns:
        raise ValueError(f"字典表缺少 tid/target_name 列，当前列为: {list(dict_df.columns)}")

    dict_df["tid"] = pd.to_numeric(dict_df["tid"], errors="coerce")
    tid2target = dict_df.set_index("tid")["target_name"].to_dict()

    df["target_name"] = df["tid"].map(tid2target)

    # 如果有tid在字典里找不到，做个兜底标记（也可以选择直接丢弃）
    df["target_name"] = df["target_name"].fillna("UNKNOWN_TARGET")

    # ====== tid排序，等间隔分层抽样 ======
    df_sorted = df.sort_values(["tid", "Ki_1", "Ki_2"], kind="mergesort").reset_index(drop=True)

    unique_tids = df_sorted["tid"].dropna().unique()
    unique_tids = np.array(sorted(unique_tids.tolist(), key=lambda x: (str(type(x)), x)))

    rows = []
    if len(unique_tids) >= n_cases:
        # 从按tid排序后的unique tid中“等间隔”取N个tid
        idxs = np.linspace(0, len(unique_tids) - 1, n_cases)
        idxs = np.round(idxs).astype(int)
        chosen_tids = unique_tids[idxs]

        # 每个tid随机取1条记录
        for tid in chosen_tids:
            g = df_sorted[df_sorted["tid"] == tid]
            rows.append(g.sample(n=1, random_state=np.random.randint(0, 10**9)).iloc[0])
    else:
        # tid种类不足N：先每个tid取1条，再从所有tid里补齐到N（允许同一tid多条）
        for tid in unique_tids:
            g = df_sorted[df_sorted["tid"] == tid]
            rows.append(g.sample(n=1, random_state=np.random.randint(0, 10**9)).iloc[0])

        remain = n_cases - len(rows)
        if remain > 0:
            idxs = np.linspace(0, len(unique_tids) - 1, remain)
            idxs = np.round(idxs).astype(int)
            extra_tids = unique_tids[idxs]
            for tid in extra_tids:
                g = df_sorted[df_sorted["tid"] == tid]
                rows.append(g.sample(n=1, random_state=np.random.randint(0, 10**9)).iloc[0])

    sampled = pd.DataFrame(rows).reset_index(drop=True)
    return sampled.apply(make_qa, axis=1)


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Generate ACNet-style binding affinity comparison cases.")
    parser.add_argument(
        "--input-csv",
        default=str(script_dir / "mmp_ac_s_distinct.csv"),
        help="Input CSV path (expects c1/Ki_1/c2/Ki_2/tid columns)",
    )
    parser.add_argument(
        "--target-dict-xlsx",
        default=str(script_dir / "target_dictionary.xlsx"),
        help="Target dictionary XLSX path",
    )
    parser.add_argument("--n-cases", type=int, default=40, help="Number of QA cases to generate")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    out_df = build_dataset(
        csv_path=Path(args.input_csv),
        dict_xlsx_path=Path(args.target_dict_xlsx),
        n_cases=args.n_cases,
        seed=args.seed,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"Done. 写出: {args.out}")
    print(out_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
