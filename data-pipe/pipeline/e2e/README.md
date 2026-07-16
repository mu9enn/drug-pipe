# E2E Module

`pipeline/e2e` 负责把 `molbench/MolBench-E2E/questions/*.md` 组织成执行 CSV，并调用统一执行入口 `pipeline/claude_agent`。

## 运行

```bash
bash pipeline/e2e/run_e2e_pipeline.sh
```

或指定题目：

```bash
bash pipeline/e2e/run_e2e_pipeline.sh --questions E2E-Q03,E2E-Q05
```

## 产物

- `pipeline/e2e/runs/<timestamp>/e2e_dataset.csv`
- `pipeline/e2e/runs/<timestamp>/dataset_manifest.json`
- `pipeline/e2e/runs/<timestamp>/pipeline.log`
- `pipeline/e2e/runs/<timestamp>/manifest.json`

真实执行结果统一写入根目录：`results/`。
