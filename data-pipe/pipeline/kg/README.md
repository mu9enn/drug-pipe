# KG Module

`pipeline/kg` 将 `tool-kg` 的 sampled questions 转为可执行任务，并接入统一执行与审计链路。
当前主协议为 `kg_task_spec_v0.2`，原生消费 `expected_trajectory.schema_version=trajectory_v2_graph`。

## 1) 检查输入样本

```bash
python pipeline/kg/scripts/inspect_kg_samples.py \
  --kg-run-dir /path/to/tool-kg/runs/<run_id>
```

## 2) 构建 KG 任务集

```bash
python pipeline/kg/scripts/build_kg_task_dataset.py \
  --kg-run-dir /path/to/tool-kg/runs/<run_id> \
  --output-dir pipeline/kg/data/<run_id> \
  --max-samples 10
```

说明：
- 输入优先读取 `sample_results/sample_success_v2.jsonl`（若不存在则回退 `sample_success.jsonl`）。
- 轨迹校验按 v2 graph 协议执行。

## 3) 执行 KG 任务

```bash
bash pipeline/kg/run_kg_pipeline.sh \
  --kg-task-file pipeline/kg/data/<run_id>/kg_sampled_tasks.jsonl \
  --n-cases 3
```

## 4) 审计 KG rollout

```bash
python pipeline/kg/scripts/scan_kg_rollouts.py \
  --results-root results/kg_sampled \
  --output-dir results/kg_sampled_audit
```

KG 执行结果写入 `results/kg_sampled/`，反馈文件默认写回 `pipeline/kg/data/<kg_run_id>/kg_execution_feedback.jsonl`。
