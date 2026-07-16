# Mol-Pipeline Handoff (2026-05-26 11:55:27 CST)

## 0) 项目一句话定位
`data-pipe` 是一个把 **MolBench 数据生成（get-molbench）** 与 **Agent 执行/轨迹导出/评估（ms_pipeline）** 打通的统一工程，并额外提供了 `e2e_pipeline` 用于跑 MolBench-E2E 问题集。

---

## 1) 已确认事实（代码与目录现状）

## 1.1 目录结构（当前真实状态）
- 根目录：`<data-pipe-root>`
- 子项目：
  - `get-molbench/`
  - `ms_pipeline/`（已取代旧命名 `vs_pipeline`）
  - `e2e_pipeline/`（新增）
  - `scripts/`
- 根目录存在：`.env`, `.env.template`, `.gitignore`, `README.md`

## 1.2 关键脚本（当前存在）
- 一键 AC/VS/PF 工作流：`scripts/run_molbench_workflow.sh`
- E2E 数据集构建：`e2e_pipeline/scripts/build_e2e_dataset.py`
- E2E 运行入口：`e2e_pipeline/run_e2e_pipeline.sh`
- 任务执行主入口：`ms_pipeline/claude_agent/run_claude.py`
- 编排入口：`ms_pipeline/claude_agent/test_flow_claude.sh`
- 路由与 MCP 配置：`ms_pipeline/claude_agent/launch_claude.sh`
- 轨迹导出：`ms_pipeline/claude_agent/trajectory_exporter.py`
- reward 历史回填：`ms_pipeline/claude_agent/backfill_no_reward_from_molclaw.py`
- molclaw 轨迹筛选：`ms_pipeline/scan_molclaw_usage.py`
- SFT 后处理：`scripts/post_process_sft.py`

## 1.3 当前 Git 工作树状态（重要）
当前仓库是 **dirty state**（未全部提交），包含多处 `M` 与 `??`（例如 `e2e_pipeline/`、`post_process_sft.py`、`scan_molclaw_usage.py` 等）。
这意味着：接手时应先决定是否分批 commit，再继续新改动，避免上下文污染。

---

## 2) 核心目标与动机（最终版）

本项目后续推进的主目标已经收敛为三条：
1. **稳定跑通多任务基线**：`vs/ac/pf/e2e` 在统一链路内可运行、可导出、可审计。
2. **构建可训练数据资产**：从 `complete_session.jsonl` 中筛出有效 `molclaw` 样本，转换为 SLIME 可用 SFT 数据。
3. **去除不稳健 reward 逻辑**：导出轨迹不再写 reward 字段，降低噪声与错误归因风险。

---

## 3) 决策演化与关键 Pivot（只保留对后续有价值的）

## 3.1 目录合并与命名
- 早期：`get-molbench` 与 `vs_pipeline` 分离。
- 最终采用：统一到 `data-pipe`，并将 `vs_pipeline` 更名为 `ms_pipeline`。
- 原因：减少路径耦合、便于统一自动化、便于后续 GitHub 管理。

## 3.2 数据集文件命名策略
- 最终采用：所有生成文件默认带 seed（AC/VS/PF 都统一）。
- 原因：同规模不同随机种子批次需可追溯，避免覆盖与歧义。

## 3.3 PF `sim` 变体处理
- 被弱化/默认跳过：自动工作流里默认只跑 `v0 + v1` 并 merge，`sim` 保留接口兼容但不进入常用自动流程。
- 原因：`sim` 变体稳定性较差、常出现样本不足。

## 3.4 轨迹 reward 逻辑
- 最终采用：**删除导出中的 reward 字段（非置零）**。
- 原因：现有 reward 口径不 solid，保留会误导后续 RL/SFT 数据质量判断。

## 3.5 E2E 任务语义
- 最终采用：`e2e` 作为新任务类型并入 `ms_pipeline`，但 **默认跳过评估**，且轨迹 **统一 accepted**。
- 原因：E2E 目标是“全链路执行与轨迹沉淀”，不是当前阶段的自动打分 benchmark。

## 3.6 E2E 超时策略
- 最终采用：仅 `e2e` 取消超时（`timeout=None`）；`vs/ac/pf` 仍保留超时机制。
- 触发原因：出现 `task timed out after 1200 seconds`，而 E2E 问题天然长程。

## 3.7 SFT schema 的 tool role
- 最终采用：`post_process_sft.py` 默认 `--tool-role-mode user_observation`，而非 `role=tool`。
- 原因：兼容更多 tokenizer/chat template；并通过 `step_loss_mask=0` 避免 observation 参与 loss。

## 3.8 `scan_molclaw_usage` 评估输出形态
- 早期想法：每个样本单独评估 JSON。
- 最终采用：只输出总 CSV（含单样本指标列），不额外写每样本 JSON。
- 原因：工程更简洁，足够支撑筛选与审计。

---

## 4) 当前采用的技术路线（可执行口径）

## 4.1 统一 AC/VS/PF 一键工作流
入口：`scripts/run_molbench_workflow.sh`

能力：
- 参数仅 `--seed`、`--n-cases`
- 自动生成：
  - AC: `N` 条，seed=`S`
  - VS: `N` 条，seed=`S`
  - PF: `v0=floor(N/2), seed=S` + `v1=N-floor(N/2), seed=S+1`，再 merge 成 `N`
- 输出目录固定：
  - `get-molbench/outputs/auto/ac`
  - `get-molbench/outputs/auto/vs`
  - `get-molbench/outputs/auto/pf`
- 下发 tmux：
  - `vs_pipe-2:0` 运行 `task=vs`
  - `ac_pipe-4:0` 运行 `task=ac`
  - `pf_pipe-5:0` 运行 `task=pf`
- 通过 `ENV_BOOTSTRAP` 注入 `.env` 到 `tmux send-keys` 命令，避免 tmux 会话缺环境变量。

## 4.2 ms_pipeline 任务路由
入口：`ms_pipeline/claude_agent/test_flow_claude.sh`

当前任务支持：`vs|ac|pf|e2e`
- `vs`: `skills/` + `system_prompt_result.md` + `molclaw-vs`
- `ac/pf/e2e`: `skills_full/` + `system_prompt_FULL.md` + `molclaw-scp`

## 4.3 run_claude 执行与产物
入口：`ms_pipeline/claude_agent/run_claude.py`

核心行为：
- 每个样本落盘：`question.json`, `prompt.txt`, `complete_session.jsonl`, `parsed_answer.json`, `run_meta.json`
- 支持 rollout 与并发 rollout
- 自动输出 `preds/molbench_<task>/...`
- 自动做完成性检查 `completion_report.json`
- 默认调用 `trajectory_exporter.py` 导出轨迹

E2E 特殊行为：
- `timeout_sec = None`（仅 e2e）
- 最终答案保留原始文本路径，不做 SMILES/JSON 严格约束

## 4.4 trajectory_exporter 导出语义
入口：`ms_pipeline/claude_agent/trajectory_exporter.py`

导出文件：
- `trajectories/trajectory_level.jsonl`
- `trajectories/step_level.jsonl`
- `trajectories/accepted.jsonl`
- `trajectories/rejected.jsonl`
- `trajectories/dataset_summary.json`

关键规则：
- reward 字段已移除（step 和 trajectory 顶层均不写 reward/reward_outcome）
- `vs/ac/pf` 继续走任务规则判 accepted/rejected
- `e2e` 固定 accepted，`task_metrics={}`，不走 reject 门

## 4.5 历史 run 的无 reward 回填
入口：`ms_pipeline/claude_agent/backfill_no_reward_from_molclaw.py`

逻辑：
1. 扫描 `results/**/complete_session.jsonl`
2. 命中包含 `"name":"mcp__molclaw` 的样本
3. 聚合到 run 级
4. 对命中 run 重新调用 exporter 覆盖导出
5. 输出 report（含 reward 字段清理验证）

## 4.6 molclaw 使用样本抽取与命中过滤
入口：`ms_pipeline/scan_molclaw_usage.py`

新增参数：
- `--use-accepted-only`：只扫描已被 trajectories/accepted.jsonl 引用的样本
- `--answer-hit-only`：
  - VS: `top3_hit_num >= 1`
  - AC: `is_correct == True`
  - PF: `is_correct == True`（`acc==1`）
- `--preprocess`：对复制后的 jsonl 做去模型/去旧路径清洗

指标来源（先后顺序）：
1. 优先从 `run_dir/trajectories/trajectory_level.jsonl` 的 `task_metrics` 直接读
2. 匹配失败时，回退到 `parsed_answer.json + question.json` 重算

输出：
- 按任务复制 `jsonl` 到 `output_root/{vs,ac,pf}`
- 单个总表 CSV（包含 `metric_source` 和各任务指标列）

## 4.7 SLIME SFT 后处理
入口：`scripts/post_process_sft.py`

当前行为：
- **已支持同时处理 `vs/ac/pf`**（不再需要 `--task` 参数）
- 输入：`scan_molclaw_usage` 导出的 `{vs,ac,pf}/*.jsonl`
- 输出：
  - `mcp_sft_train.jsonl`
  - `mcp_sft_valid.jsonl`
  - `mcp_rl_prompts_train.jsonl`
  - `mcp_rl_prompts_valid.jsonl`
  - `rejected_samples.jsonl`
  - `dataset_manifest.json`
  - `schema_validation_report.md`
- 默认 tool observation 兼容模式：`--tool-role-mode user_observation`

---

## 5) 数据格式与指标口径（接手必看）

## 5.1 `trajectory_level.jsonl` 的单样本指标字段
来源文件：`ms_pipeline/results/<run>/trajectories/trajectory_level.jsonl`

字段位置：`task_metrics`
- VS: `top3_hit_num`, `top10_hit_num`
- AC: `acc`, `is_correct`
- PF: `precision`, `recall`, `f1`, `acc`

`scan_molclaw_usage.py` 的“从 trajectory 直接取 metric”即读取该 `task_metrics`。

## 5.2 `scan_molclaw_usage.py` 的 answer-hit 判定
- VS：`vs_top3_hit_num >= 1`
- AC：`ac_is_correct == True`
- PF：`pf_is_correct == True`（由 `acc==1` 推导）

## 5.3 SFT 消息 schema（当前推荐）
每条样本结构：
- `messages`
- `tools`
- `metadata`

训练相关：
- `assistant` turn `step_loss_mask=1`
- `system/user/observation` turn `step_loss_mask=0`
- observation 默认 `role=user` + `<observation ...>` 包裹（更兼容）

---

## 6) 环境与配置共识

## 6.1 `.env` 的作用
`.env` 用于统一提供 MCP endpoint/token 与可选默认 provider。
主要变量模板在 `.env.template`，包括：
- `MOLCLAW_VS_MCP_*`
- `MOLCLAW_SCP_MCP_*`
- `CC_SWITCH_PROVIDER`
- `PYTHON_BIN`

## 6.2 tmux 是否自动继承 env
- 普通 tmux session 不保证读取当前 shell 的新 env。
- 当前一键脚本通过在 `send-keys` 命令前追加 `set -a; source .env; set +a` 方式显式注入，已覆盖该问题。

---

## 7) 已放弃/不再采用的路线（保留原因）

1. **在自动化里强跑 PF sim**：放弃默认启用，因稳定性差、经常找不到足够样本。
2. **E2E 任务沿用固定超时**：放弃，改为 e2e 不限时。
3. **SFT 强依赖 `role=tool`**：暂不作为默认，优先兼容模式 `user_observation`。
4. **scan 输出每样本独立评估 JSON**：放弃，统一写入总 CSV 指标列。
5. **把数据复制到 `ms_pipeline/molbench` 才跑**：放弃，改为直接传绝对路径，降低重复拷贝。

---

## 8) 当前不确定性 / 待验证项

1. **全量 E2E 8 题在“无超时”下的长时稳定性**：代码已改，但需要一次完整实跑确认资源占用与失败恢复策略。
2. **`ms_pipeline/readme.md` 文档与实际行为不完全一致**：该文档仍含旧路径/旧描述，不能作为唯一事实来源。
3. **`run_claude.py` 仍通过 `-p <prompt>` 传 prompt**：在本项目规模下尚未暴露 `ARG_MAX` 问题；若 prompt 继续增大，需改 stdin 传入（该风险在 `tool-kg` 已出现过）。
4. **`ms_pipeline/results/` 已在 `.gitignore` 中整体忽略**：如果后续希望版本化少量结果样例，需要额外白名单策略。

---

## 9) 风险与注意事项

1. **Secrets 风险**：仓库存在 `.env` 与 `ms_pipeline/.mcp.json` 本地文件；提交前必须检查是否被忽略、是否意外入库。
2. **工作树脏状态风险**：当前已有多文件修改，建议先整理 commit 边界再继续开发。
3. **轨迹筛选偏差风险**：`--answer-hit-only` 会显著收缩样本，适合高质 SFT，不适合做行为覆盖统计。
4. **任务混合数据风险**：`post_process_sft.py` 已支持多任务混合，但训练时需确认下游 prompt 模板能处理 task-type 差异。
5. **E2E 无评估分数风险**：当前不会产出 `bench_scores.json`（设计如此），若需要量化指标需单独定义 evaluator。

---

## 10) 接手者下一步执行清单（按优先级）

## P0（立即执行）
1. **整理并提交当前改动**
   - 先做一次 scoped commit（建议按模块拆分：`e2e_pipeline` / `scan+postprocess` / `reward-removal`）。
2. **全量跑一次 `scan_molclaw_usage.py`（你已清理有效 runs）**
   - 推荐命令：
   ```bash
   python ms_pipeline/scan_molclaw_usage.py \
     --results-root ms_pipeline/results \
     --output-root ms_pipeline/results/used_molclaw_accepted_hit_0526 \
     --use-accepted-only \
     --answer-hit-only \
     --preprocess
   ```
3. **将抽取结果转成 SFT 数据**
   - 推荐命令：
   ```bash
   python scripts/post_process_sft.py \
     --input-root ms_pipeline/results/used_molclaw_accepted_hit_0526 \
     --output-dir ms_pipeline/results/used_molclaw_accepted_hit_0526/sft_outputs
   ```

## P1（本周内）
1. **跑全量 E2E 8题回归**
   - 命令：
   ```bash
   bash e2e_pipeline/run_e2e_pipeline.sh
   ```
   - 检查：`completion_report.json`、`trajectories/accepted.jsonl` 行数、是否无 `rejected`。
2. **修正文档一致性**
   - 重点更新：`ms_pipeline/readme.md`（当前与真实实现存在偏差）。
3. **增加最小自动化回归脚本**
   - 检查项：reward 字段不存在、e2e 无评估、scan 的 answer-hit 生效、post_process 输出完整。

## P2（后续增强）
1. **SFT 数据质量增强**：加入更细粒度 rejection reason 统计与可视化。
2. **训练链路对接**：将 `mcp_sft_{train,valid}.jsonl` 接入真实 SLIME 训练 config 并记录可复现实验配置。
3. **可恢复运行能力**：长任务增加 resume/checkpoint（尤其 e2e）。

---

## 11) 任务分工建议（给不同 agent）

- **Coding Agent**
  - 完成回归脚本、修 README 一致性、补充 `post_process_sft.py` 单元测试（schema 校验、mask 校验、tool-call 配对校验）。
- **Research Agent**
  - 定义多任务混合 SFT 的采样配比（vs/ac/pf 比例、answer-hit 阈值敏感性）并设计 ablation。
- **Ops/Infra Agent**
  - 固化 `.env` 管理与密钥轮换策略；配置 tmux session 自检脚本；给 e2e 长跑增加 watchdog 报警。
- **Writing Agent**
  - 统一主 README、`ms_pipeline/readme.md`、`e2e_pipeline/README.md` 的术语、命令与路径，确保无旧路径残留。

---

## 12) 快速事实索引（接手时常查）

- 项目根：`<data-pipe-root>`
- 一键 AC/VS/PF：`bash scripts/run_molbench_workflow.sh --seed <S> --n-cases <N>`
- E2E 8题全跑：`bash e2e_pipeline/run_e2e_pipeline.sh`
- molclaw 扫描：`python ms_pipeline/scan_molclaw_usage.py ...`
- SFT 后处理：`python scripts/post_process_sft.py --input-root <dir>`
- 单样本指标来源（优先）：`ms_pipeline/results/<run>/trajectories/trajectory_level.jsonl::task_metrics`

