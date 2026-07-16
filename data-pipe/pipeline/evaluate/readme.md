# MolBench-VS Evaluation

本目录负责对 `results_dir/preds/molbench_vs/molbench_vs.json` 进行评估，并回写逐样本指标与审计信息。

## 1. 入口

```bash
python evaluate/run_eval_bench.py /path/to/results_dir
```

输出：
- `results_dir/bench_scores.json`
- 同时更新 `preds/molbench_vs/molbench_vs.json` 中每条 entry 的 `metrics` 与 `eval_audit`

## 2. 指标定义

核心指标：
- `top3_avg_hit_num`
- `top10_avg_hit_num`
- `n_samples`

逐条目指标：
- `metrics.top3_hit_num`
- `metrics.top10_hit_num`

## 3. canonical 评估策略

- 若环境可导入 RDKit：
  - 对 `answer`、`json_results.ranking`、`candidates` 进行 canonical SMILES 归一化后再计算 hit 指标。
- 若 RDKit 不可用：
  - 自动退化为字符串匹配，并在 `bench_scores.json.audit` 记录 `rdkit_error`。

## 4. 审计输出（bench_scores.json.audit）

- `rdkit_available`
- `rdkit_error`
- `quality_issue_hist`
  - `length_mismatch`
  - `outside_candidate_set`
  - `duplicate_predictions`
  - `empty_candidate_set`
- `invalid_smiles_hist`
  - `candidate`
  - `answer`
  - `ranking`

## 5. 输入格式要求

每个 prediction entry 至少应包含：
- `answer`: ground-truth 列表
- `json_results.ranking`: 预测排序列表
- `candidates`: 候选集合（用于一致性审计，强烈建议）

