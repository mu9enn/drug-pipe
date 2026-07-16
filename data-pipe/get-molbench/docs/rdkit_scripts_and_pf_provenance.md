# RDKit 三脚本对比、用法与 `Molecular Property Filtering` 来源说明

## 1. 三个脚本的定位

### `make_rdkit_benchmark_v0.py`
- 任务类型：**分子性质过滤（Property Filtering）**
- 每题输入：10 个 SMILES
- 规则：
  - 固定 Lipinski Rule of Five
  - 额外约束从较小属性池采样（`MolLogP/TPSA/RotB/MolWt`）
- 输出列：`prompt`, `answer`, `meta`
- 典型用途：较基础、可解释的规则筛选任务

### `make_rdkit_benchmark_v1.py`
- 任务类型：**分子性质过滤（Property Filtering）**
- 每题输入：10 个 SMILES
- 规则：
  - 固定 Lipinski Rule of Five
  - 扩展属性池（例如 `RingCount`, `AromaticRings`, `FractionCSP3`, `HeavyAtoms`, `HeteroAtoms`）
- 输出列：`prompt`, `answer`, `meta`
- 典型用途：属性维度更丰富、题目多样性更高

### `molecular_similiar.py`
- 任务类型：**结构相似性检索（Similarity）**
- 每题输入：10 个 SMILES（第一个作为 query）
- 规则：Morgan fingerprint + Tanimoto，相似性差距（top1-top2 gap）控制难度
- 输出列：`prompt`, `answer`, `task_type`, `difficulty`, `meta`
- 典型用途：分子结构近邻识别/相似性推断任务

---

## 2. 统一入口用法（推荐）

通过 `pipelines/generate_molbench_pf.py` 选择变体：

```bash
# v0: 基础属性过滤
python pipelines/generate_molbench_pf.py \
  --variant v0 --n-cases 50 --seed 42 --out-dir outputs/pf_v0

# v1: 扩展属性过滤
python pipelines/generate_molbench_pf.py \
  --variant v1 --n-cases 50 --seed 42 --out-dir outputs/pf_v1

# similarity: 相似性任务
python pipelines/generate_molbench_pf.py \
  --variant similarity --n-cases 50 --seed 42 --out-dir outputs/pf_sim
```

输出统一命名为：`molbench-pf-<N>.csv`。

---

## 3. `MolBench-MS - Molecular Property Filtering.csv` 的来源拆解

目标文件：`examples/MolBench-MS - Molecular Property Filtering.csv`

### 结论
该文件是**混合数据集**，不是单一“属性过滤”来源：
- 含有 Property Filtering 题型（`From the following SMILES list, output ALL molecules ...`）
- 也含有 Similarity 题型（`Find the molecule MOST similar...` / `Morgan fingerprint` / `Tanimoto`）

### 观测特征
- 总行数：50
- 其中 similarity 题型：15
- 其余为 property filtering 题型：35

### 属性过滤部分的约束特征
在 35 条 property filtering 中可以看到：
- 基础约束（Lipinski）
- 扩展约束（如 `AromaticRings`, `FractionCSP3`, `HeavyAtoms`, `HeteroAtoms`, `RingCount`）

这说明其“属性过滤”部分至少包含了 `v1` 风格的扩展属性约束；并非只来自 `v0`。

---

## 4. 如何复现同类型混合 PF 数据（建议流程）

下面流程复现的是“同类型、同范式”混合集（不追求逐字节一致）：

```bash
# 1) 生成 35 条属性过滤（推荐 v1）
python pipelines/generate_molbench_pf.py \
  --variant v1 --n-cases 35 --seed 42 --out-dir outputs/pf_mix

# 2) 生成 15 条相似性题
python pipelines/generate_molbench_pf.py \
  --variant similarity --n-cases 15 --seed 42 --out-dir outputs/pf_mix_sim

# 3) 合并为一个 mixed 文件（保留 question/answer）
python - <<'PY'
import pandas as pd
prop = pd.read_csv('outputs/pf_mix/molbench-pf-35.csv')
sim = pd.read_csv('outputs/pf_mix_sim/molbench-pf-15.csv')
prop_df = prop.rename(columns={'prompt': 'question'})[['question', 'answer']]
sim_df = sim.rename(columns={'prompt': 'question'})[['question', 'answer']]
out = pd.concat([prop_df, sim_df], ignore_index=True)
out.to_csv('outputs/pf_mix/molbench-pf-50-mixed.csv', index=False)
print('Wrote outputs/pf_mix/molbench-pf-50-mixed.csv', len(out))
PY
```

如果你希望保留 `meta/difficulty/task_type` 等列，可以在合并脚本中单独处理列对齐。

---

## 5. 选择建议
- 想做“规则筛选”为主：优先 `v1`，`v0` 作为基础对照。
- 想做“结构相似性推断”：用 `similarity`。
- 想贴近 `examples/Molecular Property Filtering.csv` 的混合风格：`v1 + similarity` 混合构建。
