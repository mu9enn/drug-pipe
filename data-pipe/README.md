# data-pipe

统一分子任务执行与后处理工程（`vs/ac/pf/e2e/kg`），当前采用硬切后的 `pipeline/` 架构。

## 目录结构


## 一键入口

执行（只跑并落盘 raw）：

```bash
bash pipeline/claude_agent/run_execute.sh --run-dataset --task vs --dataset-csv molbench/molbench-vs-30.csv
```

评测（仅 `vs/ac/pf`）：

```bash
bash pipeline/evaluate/run_evaluate.sh results/<run_dir> vs
```

后处理（全量，从 raw 会话重建）：

```bash
bash scripts/run_postprocess.sh --results-root results
```

`run_postprocess.sh` 固定流程：

1. `trajectory_exporter.py`
2. `scan_molclaw_usage.py`
3. `post_process_sft.py`

输出位于：


## 规则约定


## 常用工作流

生成并下发 AC/VS/PF：

```bash
bash scripts/run_molbench_workflow.sh --seed 42 --n-cases 30
```

构建与运行 KG：

```bash
python pipeline/kg/scripts/build_kg_task_dataset.py \
  --kg-run-dir /path/to/tool-kg/runs/<run_id> \
  --output-dir pipeline/kg/data/<run_id>

bash pipeline/kg/run_kg_pipeline.sh \
  --kg-task-file pipeline/kg/data/<run_id>/kg_sampled_tasks.jsonl \
  --n-cases 1
```

构建与运行 E2E：

```bash
bash pipeline/e2e/run_e2e_pipeline.sh --questions E2E-Q01,E2E-Q02
```
