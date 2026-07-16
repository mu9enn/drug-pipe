# Postprocess Module

`pipeline/postprocess` 负责把 raw `complete_session.jsonl` 转成可训练、可审计的数据产物。
它是一个纯后处理模块，不负责在线推理，也不负责训练。

## 职责边界

固定链路为：

1. `trajectory_exporter.py`
   从 raw `complete_session.jsonl` 重建 `trajectories/*`。
2. `scan_molclaw_usage.py`
   汇总 accepted 候选会话，并产出统一筛选表。
3. `post_process_sft.py`
   把候选会话转成 ReAct-style SFT / RL 数据。

这里不引入 reward，也不改 PPO / GRPO / slime 训练代码。

## 一键运行

```bash
bash scripts/run_postprocess.sh --results-root results
```

常用参数：

```bash
bash scripts/run_postprocess.sh \
  --results-root results \
  --answer-hit-only \
  --split-multi-tool-calls
```

## ReAct SFT 协议

`post_process_sft.py` 会把 accepted 会话清洗成统一的 ReAct-style 消息流：

- `assistant` 里的思考文本 -> `<thought>...</thought>`
- `mcp__molclaw-scp__*` / `mcp__molclaw-vs__*` 工具调用 -> `<tool_call>...</tool_call>`
- 对应 `tool_result` -> `<observation tool_name="...">...</observation>`
- 最终回答 -> `<final_answer>{...task-aware final_answer...}</final_answer>`
- `/root`、`/home`、`/tmp`、`/mnt`、`/workspace` 下的本地绝对路径会被替换为纯文本 `<artifact:...>` 占位符
- `fpocket_toolkit` 的 observation 会被额外压缩成只保留 top pocket 的轻量结构

默认行为：

- 保留原始多轮 ReAct 语义
- 默认合并连续 assistant raw event
- 默认 observation 使用 `role=user`
- 只有显式启用 `--split-multi-tool-calls` 时才拆分多 tool call

当前 SFT 样本主体只保留：

- `schema_version`
- `id`
- `messages`

其余审计信息会落到 `cleaning_report` / `cleaning_report_index.jsonl` / `schema_validation_report.json`。

`final_answer` 按 `task_type` 细分为：

- `ac`: `answer_smiles` / `short_reason` / `evidence`
- `vs`: `ranked_smiles` / `selected_smiles` / `short_reason` / `evidence`
- `pf`: `selected_smiles` / `labels`(optional) / `short_reason` / `evidence`
- `kg` / `e2e`: 保留最小任务结果结构

## 清洗规则

- `complete_session.jsonl` 最后一行若为 `[runner-error]...`，该样本不能进入 accepted。
- 同时保留 `mcp__molclaw-scp__*` 和 `mcp__molclaw-vs__*` 工具调用
- 非 MCP 工具调用及其结果会被删除
- VS/AC/PF accepted trajectory 缺少必需 `task_metrics` 时，Stage 2 不复制候选，并写入 `stage2_rejected_candidates.jsonl`
- observation 会被结构化为：
  - `ok`
  - `tool_name`
  - `status`
  - `content`
  - `metadata`
- 只剥离 final answer 或 observation 外层的 triple-backtick 包裹，不删除内部内容
- 会生成每样本 `cleaning_report`

## 输出

默认输出到：

- `results/postprocess_candidates/molclaw_usage_summary.csv`
- `results/postprocess_candidates/stage2_rejected_candidates.jsonl`（VS/AC/PF 缺少必需指标的候选）
- `results/postprocess_candidates/sft_outputs/mcp_sft_all/`
- `results/postprocess_candidates/sft_outputs/mcp_sft_all.jsonl`（兼容副本，内容同目录内每样本 JSON）
- `results/postprocess_candidates/sft_outputs/mcp_rl_prompts_all.jsonl`
- `results/postprocess_candidates/sft_outputs/rejected_samples.jsonl`
- `results/postprocess_candidates/sft_outputs/cleaning_reports/`

验证报告：

- `results/postprocess_candidates/sft_outputs/schema_validation_report.json`
- `results/postprocess_candidates/sft_outputs/schema_validation_report.md`

## 兼容性说明

- `--answer-hit-only` 只影响 `vs/ac/pf` 的候选筛选，不改变 ReAct 清洗本身
- `kg/e2e` 不做 answer-hit 过滤
- RL 输出 schema 保持不变
- 如果下游还读旧 JSON action schema，只更新消费侧，不回退后处理格式
- `mcp_sft_all/` 是主阅读入口；每条样本单独一个 pretty JSON 文件

## LLM Clean 与 Final Hard-Clean

用户入口会依次执行 LLM semantic repair、脚本2 final hard-clean 和最终 validator：

```bash
bash scripts/run_llm_clean.sh results/.../mcp_sft_all
```

`cleaned/` 保留 LLM 原始输出，`cleaned_final/` 是删除 observation debug metadata 并通过 gate 的训练候选。仍有 VS ranking 或 observation status 冲突的样本进入 `cleaned_final_reports/quarantine/`；最终 validator 发现的其他 P0 invalid 样本进入 `cleaned_final_reports/quarantine_validator/`。脚本不会自动重排 VS final answer，也不会自动修改 observation error status。

`scripts/run_postprocess.sh` 最后还会生成 pre-LLM semantic report。该报告只使用
`needs_llm_semantic_repair/repair_reasons` 做提示，不拒绝或重写样本；post-LLM validator
则使用 `*_after_llm_clean` 错误作为最终 invalid gate。
