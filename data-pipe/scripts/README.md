# Scripts Workflow

`scripts/` 放的是常用的一键入口，按顺序可以走完样本生成、raw 处理、以及 LLM clean 三步。

## 1. 生成样本、启动 Claude、逐样本评测

```bash
bash scripts/run_molbench_workflow.sh --seed 602 --n-cases 120
```

这一步会做这些事：

- 生成 AC / VS / PF 的 MolBench 样本
- 合并 PF v0 / v1
- 通过 tmux 把 `claude -p` 任务发到三个窗口里执行
- 产出 raw `complete_session.jsonl`
- 对 `vs/ac/pf` 做逐样本评测，写出 `bench_scores.json`

## 2. 后处理 raw 会话

```bash
bash scripts/run_postprocess.sh --results-root results --output-root results/postprocess_candidates
```

全量重跑应使用新的 `--output-root`。Stage 2 按操作规范保留旧候选，不会自动清空目录；反复写入旧目录会产生 `__dup2`、`__dup3`。

这一步会做这些事：

- 从 raw `complete_session.jsonl` 重新导出 `trajectories/*`
- 扫描并筛出 molclaw usage 候选
- 生成干净的 `mcp_sft_all/` 和 `mcp_rl_prompts_all.jsonl`
- 做 accepted 判别和格式清洗
- 生成非阻塞的 `pre_llm_semantic_report.{json,md}`，只标记需由 LLM 修复的 ranking/status 问题

如果你想只导出更严格的训练子集，可以在第二步加：

```bash
--answer-hit-only
```

## 3. LLM Clean 轨迹

```bash
bash scripts/run_llm_clean.sh results/postprocess_candidates/sft_outputs/mcp_sft_all
```

这一步会做这些事：

- 遍历输入目录顶层的 `*.json`
- 为每条轨迹创建独立的 `cc-workdir/<source_stem>/`
- 复制源 JSON 到 workdir 后调用本地 Claude Code
- 生成 `<source_stem>-cleaned.json`
- 把所有 cleaned 文件收集到 `<input_dir>/cleaned/`
- 自动执行脚本2：删除 observation debug metadata、保守清理相对路径
- 把通过最终 gate 的样本写入 `<input_dir>/cleaned_final/`
- 把未解决的 VS ranking / observation status 冲突隔离到 `<input_dir>/cleaned_final_reports/quarantine/`
- 对 `cleaned_final/` 执行 post-LLM validator，生成 `cleaned_final_validation.{json,md}`
- 把 validator 发现的其他 P0 invalid 样本移动到 `<input_dir>/cleaned_final_reports/quarantine_validator/`

这一步不接 MCP，不复制 `.claude`，也不写 wrapper / manifest / `run_meta.json` / `complete_session.jsonl`。

`run_llm_clean.sh` 也可以直接对其他 JSON 轨迹目录使用，只要目录里是顶层 `*.json` 文件即可。

如需跳过 LLM、只对已有 `cleaned/` 重跑脚本2：

```bash
bash scripts/run_llm_clean.sh \
  results/postprocess_candidates/sft_outputs/mcp_sft_all \
  --skip-llm
```

`cleaned/` 始终保留 LLM 原始输出；`cleaned_final/` 才是训练候选。脚本2和 validator 都不会自动重排 VS，也不会自动改写 observation status。
