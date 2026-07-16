# Claude Agent (Execution Only)

`pipeline/claude_agent` 只负责执行任务并落盘 raw 产物，不做后处理。

## 入口

```bash
bash pipeline/claude_agent/run_execute.sh --run-dataset --task vs --dataset-csv molbench/molbench-vs-30.csv
```

或直接：

```bash
bash pipeline/claude_agent/launch_claude.sh --run-dataset --task ac --dataset-csv molbench/molbench-ac-30.csv
```

## 产物

每个 run 在 `results/molbench_<task>_<provider>_run_<timestamp>/` 下生成：

- `run_config.json`
- `run_summary.jsonl`
- `completion_report.json`
- `row*/question.json`
- `row*/prompt.txt`
- `row*/complete_session.jsonl`
- `row*/parsed_answer.json`
- `row*/run_meta.json`
- `preds/molbench_<task>/...`

说明：轨迹导出与 SFT/RL 转换已迁移到 `pipeline/postprocess`。
