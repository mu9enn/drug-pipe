# data-pipe 深度交接文档（截至 2026-06-03）

这份文档的目标不是“给一个概览”，而是让接手者在**不看聊天记录**的前提下，也能把当前 `data-pipe` 的全流程、文件边界、任务分工、产物结构、以及 2026-05-27 之后的关键演进完整接上。

当前版本的核心结论可以先记成一句话：

> `data-pipe` 已经从“执行 + 评测 + 后处理耦合在一起的脚本工程”，演进成了一个以 `complete_session.jsonl` 为唯一 raw 真源、以 `claude_agent / evaluate / postprocess` 三段式为骨架、并且可继续扩展到 `e2e` 与 `kg` 的分层数据管线。

---

## 1. 项目定位与当前边界

### 1.1 项目现在到底做什么

`data-pipe` 是一个面向分子类任务的统一工程，当前覆盖：

- `vs`
- `ac`
- `pf`
- `e2e`
- `kg`

它的职责不是“直接训练模型”，而是把**任务数据 → Claude Code 执行 → 原始会话落盘 → 任务评测 → 轨迹重建 → SFT/RL 可训练数据导出**这条链完整打通。

### 1.2 当前仓库的结构性约束

根目录下几个最重要的目录已经固定下来：

- `get-molbench/`：MolBench 数据生成
- `pipeline/claude_agent/`：只做执行，不做后处理
- `pipeline/evaluate/`：只做 `vs/ac/pf` 评测
- `pipeline/postprocess/`：只做轨迹重建、usage 筛选、SFT/RL 转换
- `pipeline/e2e/`：E2E 数据构建与运行
- `pipeline/kg/`：KG-sampled 数据构建、运行与审计
- `results/`：统一运行产物根目录
- `molbench/`：统一本地数据集目录
- `skills/skills_vs`、`skills/skills_full`：任务技能与系统提示集合
- `scripts/`：一键 shell 入口和少量说明文档

### 1.3 当前总原则

现在的三个硬约束是：

1. **raw 会话是唯一真源**
   - 后处理必须从 `complete_session.jsonl` 重建
   - 不应该把 reward、审计统计、路径噪声写回训练正文

2. **执行、评测、后处理彻底解耦**
   - 执行层只负责跑完并落盘
   - 评测层只负责给 `vs/ac/pf` 打分
   - 后处理层只负责重建轨迹和转训练格式

3. **训练样本要尽量小而干净**
   - `SFT/RL` 导出不再保留一堆执行审计字段
   - `final_answer` 必须按任务类型定 schema
   - 本地绝对路径、工程 chatter、超大 observation 都要清洗

---

## 2. 2026-05-27 之后的关键演进

这一部分是“为什么今天的 `data-pipe` 和 5/27 之前已经不是同一个系统”的核心。

### 2.1 5/27 的大 pivot：`ms_pipeline` 后处理架构重构

这次重构把过去混在一起的逻辑拆成了三层：

1. **执行层**：`pipeline/claude_agent`
2. **评测层**：`pipeline/evaluate`
3. **后处理层**：`pipeline/postprocess`

这一步的实际效果不是“只是挪文件”，而是把整个系统的真源从“脚本内部自动串联”变成了“raw 会话 + 独立评测 + 独立后处理”。

它带来的最重要变化有四个：

- `run_claude` 不再自动做后处理
- `trajectory_exporter` 不再依赖执行阶段的副作用
- `scan_molclaw_usage` 成为独立候选筛选器
- `post_process_sft` 成为真正的训练数据生成器

### 2.2 后续演进的方向不是“继续加复杂度”，而是“加分层”

5/27 之后的变化可以概括成下面几类：

- **去 reward 化**
  - 轨迹导出不再写 reward / reward_outcome
  - 后处理也不把 reward 当成训练目标

- **任务扩展**
  - 从只围绕 `vs/ac/pf`，扩展到 `e2e` 和 `kg`

- **MCP / provider 更稳**
  - 不再硬编码 qwen
  - provider 由 `cc-switch` 外部控制
  - 每次运行前都检查 MCP 是否真的连接上

- **SFT 样本更干净**
  - 从通用 JSON action 演进为 ReAct-style 文本协议
  - 之后又收紧成 task-aware final answer
  - 再进一步瘦身，只保留训练真正需要的字段

- **向下游训练框架对齐**
  - 逐步形成可导出的 bundle
  - 为后续 `verl_wd` / slime SFT 做字段契约准备

---

## 3. 整体流程图：当前三段式主链路

可以把当前 `data-pipe` 的主链路理解成下面这个顺序：

```text
get-molbench / molbench 题目构建
        ↓
scripts/run_molbench_workflow.sh
        ↓
pipeline/claude_agent
  - 生成 prompt
  - 调 claude -p
  - 落盘 complete_session.jsonl
  - 对 vs/ac/pf 写评测入口预测文件
        ↓
pipeline/evaluate   (仅 vs/ac/pf)
  - 写 bench_scores.json
  - 回写逐样本指标
        ↓
scripts/run_postprocess.sh
  1) trajectory_exporter.py
  2) scan_molclaw_usage.py
  3) post_process_sft.py
        ↓
results/postprocess_candidates/
  - trajectories/*
  - molclaw_usage_summary.csv
  - sft_outputs/mcp_sft_all/
  - sft_outputs/mcp_rl_prompts_all.jsonl
  - cleaning reports / manifests / validation reports
```

这个顺序是当前推荐的“完整主流程”。

如果只想跑 E2E 或 KG，则会走对应支线，但最后仍然会落回到 `pipeline/claude_agent` 和 `pipeline/postprocess` 这两套通用设施上。

---

## 4. 第一步：样本生成 + Claude 执行 + 原始会话落盘

这一层的真正目标是把问题跑完，并把原始会话完整留住，而不是直接做训练数据。

### 4.1 主入口：`scripts/run_molbench_workflow.sh`

这是常规 MolBench 主工作流的一键入口。

典型用法：

```bash
bash scripts/run_molbench_workflow.sh --seed 42 --n-cases 30
```

它做的事情是：

1. 生成 AC / VS / PF 的任务 CSV
2. 合并 PF 的 `v0` / `v1`
3. 通过 tmux 把三类任务下发给 Claude Code
4. 产出 raw `complete_session.jsonl`
5. 对 `vs/ac/pf` 做逐样本评测，写 `bench_scores.json`

### 4.2 输入数据来自哪里

数据生成发生在 `get-molbench/`：

- `generate_molbench_ac.py`
- `generate_molbench_vs.py`
- `generate_molbench_pf.py`
- `merge_molbench_pf.py`

默认输出位置：

- `get-molbench/outputs/auto/ac`
- `get-molbench/outputs/auto/vs`
- `get-molbench/outputs/auto/pf`

命名规则现在已经统一成“全部带 seed”：

- `molbench-ac-<n>-<seed>.csv`
- `molbench-vs-<n>-<seed>.csv`
- `molbench-pf-v0-...`
- `molbench-pf-v1-...`
- `molbench-pf-<n>-<seed>.csv`

### 4.3 任务路由和技能选择

当前执行层的任务路由是明确分开的：

- `vs`
  - 用 `skills/skills_vs`
  - 用 `system_prompt_result.md`
  - 走 `molclaw-vs`

- `ac / pf / e2e / kg`
  - 用 `skills/skills_full`
  - 用 `system_prompt_FULL.md`
  - 默认走 `molclaw-scp`

这个路由已经不再是“写死某个模型/某个 provider 的脚本”了。
当前做法是：

- 任务路由由脚本决定
- 模型 provider 由你运行前的 `cc-switch` 决定
- MCP server 由任务类型决定

### 4.4 raw 运行结果的目录结构

每次运行会落到：

```text
results/molbench_<task>_<provider>_run_<timestamp>/
```

典型内容包括：

- `run_config.json`
- `run_summary.jsonl`
- `completion_report.json`
- `row*/question.json`
- `row*/prompt.txt`
- `row*/complete_session.jsonl`
- `row*/parsed_answer.json`
- `row*/run_meta.json`
- `preds/molbench_<task>/...`

如果是多 rollout，会有：

- `row*/rollout0001/`
- `row*/rollout0002/`
- ...

`trajectory_exporter` 已经兼容这两种结构：

- 单 rollout：`row_dir/parsed_answer.json`
- 多 rollout：`row_dir/rolloutXXXX/parsed_answer.json`

### 4.5 `complete_session.jsonl` 的地位

`complete_session.jsonl` 是原始会话流，是后处理的唯一真源。

它记录的是：

- system/init
- assistant
- user
- tool_use / tool_result
- 最终回答

后处理阶段不应该再依赖“人工总结过的别的文本”，只应从这份 raw 会话流重建轨迹。

### 4.6 这一层当前最重要的稳定性改进

5/27 后最明显的变化是：

- 不再在仓库内部硬编码 provider
- 不再默认把 qwen 写死进流程
- 所有任务的 Claude 调用均不设置执行超时，会等待任务自然结束或由用户显式中断
- Claude 运行前会检查 MCP init 是否真的 ready
- workdir / prompt / session 这些产物更完整

这使得 `run_claude` 不只是“能调用一次模型”，而是一个可审计的执行器。

---

## 5. 第二步：评测层 `pipeline/evaluate`

这一层只负责 `vs/ac/pf` 的评测。

### 5.1 入口

```bash
bash pipeline/evaluate/run_evaluate.sh results/molbench_vs_qwen-397b_run_20260529_130417
```

或者直接：

```bash
python pipeline/evaluate/run_eval_bench.py /path/to/results_dir
```

### 5.2 为什么评测层独立

在 5/27 之前，执行、评测、轨迹导出容易绑死在一起，问题是：

- 一旦执行流程出错，很难判断是模型输出坏了，还是评测逻辑坏了
- 同一份 raw 会话很难独立重算
- 评测结果、轨迹结果、训练结果混在一起，不利于审计

现在拆开以后：

- 执行层只管把任务跑完
- 评测层只管算指标并回写逐样本结果
- 后处理层可以直接消费评测后的结果

### 5.3 `vs/ac/pf` 的评测口径

#### VS

核心指标：

- `top3_avg_hit_num`
- `top10_avg_hit_num`

逐样本信息写回：

- `metrics.top3_hit_num`
- `metrics.top10_hit_num`

审计输出还会记录：

- 候选集外预测
- 长度不匹配
- 重复预测
- 空候选集
- invalid SMILES

#### AC

核心指标：

- `accuracy`

逐样本信息写回：

- `metrics.is_correct`

#### PF

核心指标：

- `exact_set_match_rate`
- `avg_f1`
- `single_answer_accuracy`

逐样本信息写回：

- `metrics.precision`
- `metrics.recall`
- `metrics.f1`
- `metrics.acc`

### 5.4 canonical / RDKit 的处理方式

如果环境能导入 RDKit：

- 对答案、排序结果、候选集做 canonical SMILES 归一化后再评测

如果不能导入 RDKit：

- 退化为字符串匹配
- 同时在审计里记录 `rdkit_error`

### 5.5 评测层的产物

评测后会得到：

- `results_dir/bench_scores.json`
- 同时更新 `preds/molbench_<task>/molbench_<task>.json`

这意味着 `bench_scores.json` 并不是单纯“一个分数文件”，它也是逐样本指标和审计的聚合入口。

---

## 6. 第三步：后处理层 `pipeline/postprocess`

这一层是当前变化最多、也最值得认真理解的一层。

它的设计目标不是“再跑一遍模型”，而是把 raw 会话转成：

- 可审计轨迹
- 可筛选候选
- 可训练的 ReAct SFT
- 可交付给下游训练框架的 bundle

### 6.1 后处理总入口：`scripts/run_postprocess.sh`

注意：这个脚本现在在 `scripts/` 下，而不是旧的 `pipeline/postprocess/` 路径里。

典型命令：

```bash
bash scripts/run_postprocess.sh --results-root results --output-root results/postprocess_candidates
```

可选参数：

- `--answer-hit-only`
- `--split-multi-tool-calls`
- `--skip-export`
- `--skip-scan`
- `--skip-sft`

### 6.2 三段式后处理固定顺序

`run_postprocess.sh` 的内部顺序是固定的：

1. `trajectory_exporter.py`
2. `scan_molclaw_usage.py`
3. `post_process_sft.py`

它们各自只负责一件事。

---

### 6.3 第 1 段：`trajectory_exporter.py`

#### 6.3.1 它做什么

它负责把 raw `complete_session.jsonl` 重建成轨迹数据。

输出：

- `trajectories/trajectory_level.jsonl`
- `trajectories/step_level.jsonl`
- `trajectories/accepted.jsonl`
- `trajectories/rejected.jsonl`
- `trajectories/dataset_summary.json`

#### 6.3.2 它的任务支持

当前已经支持：

- `vs`
- `ac`
- `pf`
- `e2e`
- `kg`

#### 6.3.3 它的核心职责

这一层做的是“结构性重建”和“任务级质量门”。

它会基于：

- `parsed_answer.json`
- `question.json`
- `run_meta.json`
- 会话事件流

生成轨迹和步骤记录。

#### 6.3.4 reward 已移除

这一步不再写：

- `reward`
- `reward_outcome`

这是当前系统里非常重要的一个稳定化决定。

以前 reward 相关信息会把轨迹、评测、训练目标混在一起；现在统一去掉，后续是否需要 reward 留给单独设计，而不是塞进 raw 导出。

#### 6.3.5 accepted / rejected 的当前语义

`trajectory_exporter` 的 accepted/rejected 是**轨迹结构层面的门**，不是训练层面的最终门。

大体上可以理解成：

- `vs/ac/pf`
  - 仍然走任务规则判定
  - 包括 parse、长度、候选集、正确性等

- `e2e/kg`
  - 更偏执行完成性
  - 重点看任务是否真正跑完、会话是否存在

但要注意：这还不是最终训练样本的 accepted。

最终进入 SFT/RL 的样本，还要经过下一步 `scan_molclaw_usage` 和 `post_process_sft` 的筛选。

#### 6.3.6 你应该把它理解成什么

`trajectory_exporter` 是“从 raw 会话恢复成结构化轨迹”的重建器。

它的输出是后面所有筛选和训练的原料。

---

### 6.4 第 2 段：`scan_molclaw_usage.py`

#### 6.4.1 它做什么

它负责把已经通过轨迹导出的候选样本汇总出来，并按任务复制到统一目录。

当前代码里，它的主要职责是：

- 扫描 `results/**/trajectories/accepted.jsonl`
- 识别任务类型
- 找到对应的 `complete_session.jsonl`
- 汇总到 `output_root/{vs,ac,pf,kg,e2e}/`
- 产出统一总表 CSV：`molclaw_usage_summary.csv`

#### 6.4.2 它当前的输入筛选方式

当前实现里它有一个重要开关：

- `--use-accepted-only`

这意味着它默认围绕已接受的轨迹做候选收集，而不是直接扫全量 raw 会话。

#### 6.4.3 评估指标从哪里来

`scan_molclaw_usage.py` 对单样本指标的来源有两个层级：

1. **优先读取 `trajectory_level.jsonl` 的 `task_metrics`**
   - 这是首选
   - 因为它最接近轨迹级真值

2. **如果轨迹层匹配不到，再回退到 `parsed_answer.json + question.json` 重新计算**

这点很关键，因为它让单样本指标不是依赖某个孤立文件，而是有一个清晰的 fallback 逻辑。

#### 6.4.4 `answer-hit` 的口径

当前语义上，`answer-hit` 不是在 scan 层硬筛，而是在后续 `post_process_sft` 层按任务使用：

- `vs`：`top3_hit_num >= 1`
- `ac`：`is_correct == True`
- `pf`：`acc == 1`

这个设计的原因是：

- 先把候选收集完整
- 再在 SFT 导出阶段决定是否只保留 hit 样本
- 避免把“训练候选”和“候选扫描”绑死

#### 6.4.5 它的输出

输出根目录一般是：

```text
results/postprocess_candidates/
```

里面会有：

- `vs/`
- `ac/`
- `pf/`
- `kg/`
- `e2e/`
- `molclaw_usage_summary.csv`

这一步的结果是后处理的“候选池”。

---

### 6.5 第 3 段：`post_process_sft.py`

这是目前后处理里最关键、也最复杂的一段。

#### 6.5.1 它的目标

它把 `scan_molclaw_usage.py` 选出来的候选样本转成：

- ReAct-style SFT 数据
- RL prompt 数据
- 清洗与验证报告

#### 6.5.2 它现在的 CLI 参数

当前主要参数是：

- `--input-root`
- `--output-dir`
- `--summary-csv`
- `--answer-hit-only`
- `--tool-role-mode user_observation|tool`
- `--split-multi-tool-calls`
- `--max-observation-chars`

#### 6.5.3 当前默认策略

默认策略是：

- `tool_role_mode = user_observation`
  - observation 以 `role=user` 表示
  - 这样更兼容聊天模板

- `split_multi_tool_calls = false`
  - 默认保留原始多 tool call 语义
  - 只有显式加开关才拆分

- `answer-hit-only = false`
  - 默认不强制只保留 answer-hit
  - 这是一个候选级过滤开关，不是 schema 约束

#### 6.5.4 它现在输出什么

主输出位置：

```text
results/postprocess_candidates/sft_outputs/
```

核心产物包括：

- `mcp_sft_all/`
  - 每条样本一个 pretty JSON 文件

- `mcp_sft_all.jsonl`
  - 与上面目录保持兼容的 JSONL 副本

- `mcp_rl_prompts_all.jsonl`

- `rejected_samples.jsonl`

- `cleaning_reports/`

- `cleaning_report_index.jsonl`

- `dataset_manifest.json`

- `schema_validation_report.json`

- `schema_validation_report.md`

#### 6.5.5 为什么现在 `mcp_sft_all/` 是主阅读入口

因为你已经明确提过“不要只看一个长 JSONL，要能逐样本方便阅读”。

所以当前主产物改成：

- `mcp_sft_all/000001__xxx.json`
- `mcp_sft_all/000002__yyy.json`
- ...

每条样本单独一个格式化 JSON 文件，更便于人工检查。

#### 6.5.6 ReAct 清洗到底做了什么

它做的是“把 Claude Code 的执行会话整理成训练时真正想让模型学的行为序列”。

主要规则包括：

- assistant 的思考文本转成 `<thought>...</thought>`
- `mcp__molclaw-scp__*` 工具调用转成 `<tool_call>...</tool_call>`
- 对应 `tool_result` 转成 `<observation tool_name="...">...</observation>`
- 最终回答转成 `<final_answer>{...}</final_answer>`
- 非 `mcp__molclaw-scp__` 的工具调用及其结果会被删掉
- 本地绝对路径会被替换为 `artifact:*`
- `fpocket_toolkit` 的 observation 会被压缩成更小的摘要结构

#### 6.5.7 当前的 `final_answer` 是 task-aware 的

这是一项特别重要的变化。

现在 `final_answer` 不是“所有任务都用同一套模板”，而是按任务类型分别定义：

- `ac`
  - `answer_smiles`
  - `selected_molecule`
  - `short_reason`
  - `evidence`

- `vs`
  - `ranked_smiles` 或 `selected_smiles`
  - `short_reason`
  - `evidence`

- `pf`
  - `selected_smiles` 作为当前主字段
  - `labels` 仅作可选解释字段
  - `prediction` 在解析层可作为兼容别名，但当前落盘 schema 仍以 `selected_smiles` 为主
  - `short_reason`
  - `evidence`

- `kg / e2e`
  - 保留最小任务结果结构
  - 当前不是主线重点，但流程可用

这一点的意义是：训练样本和评测提取逻辑保持一致，不会再出现“训练格式看起来合理，但评测侧提不出来”的问题。

#### 6.5.8 为什么现在训练样本更小

现在每条样本主体已经尽量压缩成：

- `schema_version`
- `id`
- `messages`

而以下信息都被移到旁边的审计文件里：

- 顶层 metadata
- tools 空 schema
- raw_answer
- 各类统计字段
- 路径映射
- 清洗计数

这是训练纯净度提升最明显的一步。

#### 6.5.9 清洗报告保存了什么

每条样本都会有一个独立的 `cleaning_report`，另外还有一个 `cleaning_report_index.jsonl` 方便批量检索。

报告里会保留：

- retained / dropped 的 molclaw tool 调用计数
- orphan tool result 计数
- fence wrapper 去除计数
- observation 截断计数
- 工具名映射
- 清洗动作历史

这意味着：

- 训练样本很干净
- 审计信息没有丢
- 想排查清洗过程时有单独的 sidecar 可看

#### 6.5.10 额外的导出与附属工具

`pipeline/postprocess` 之外，还有两个和后处理强相关、但不属于主三段式核心的辅助工具：

- `scripts/run_llm_clean.sh`
  - 用于对已经输出的顶层 JSON 轨迹做进一步的 LLM clean
  - 适合人工整理、内容缩写、二次清洗
  - 不是 raw pipeline 的必经步骤

- `scripts/build_verl_bundle.sh`（已于 2026-06-08 移除；如需 legacy bundle，请显式调用 Python exporter/validator）
  - 用于把当前的 SFT / RL 输出打包成下游训练可消费的 bundle
  - 相关契约说明在 `docs/mol_pipeline_to_verl_bundle_v0.1.md`
  - 这一步更像“交付层”，不是后处理主链路本身

---

## 7. `molbench` 主流程在当前版本中如何跑

### 7.1 推荐的完整顺序

如果你想跑一轮完整的 `vs/ac/pf` 主流程，推荐按这个顺序：

```bash
bash scripts/run_molbench_workflow.sh --seed 42 --n-cases 30
bash scripts/run_postprocess.sh --results-root results --output-root results/postprocess_candidates
```

### 7.2 第一条命令完成什么

`run_molbench_workflow.sh` 负责：

- 生成 AC / VS / PF 数据
- 合并 PF v0/v1
- tmux 下发 Claude 任务
- 产出 raw 会话
- 对 `vs/ac/pf` 做逐样本评测

### 7.3 第二条命令完成什么

`run_postprocess.sh` 负责：

- 从 raw `complete_session.jsonl` 重新导出轨迹
- 收集 molclaw usage 候选
- 生成 ReAct SFT / RL prompt
- 输出清洗报告和验证报告

### 7.4 这两条命令之间的关系

它们不是重复，而是分工：

- 第一条是“执行 + 评测”
- 第二条是“重建 + 过滤 + 转训练格式”

如果不先跑第一条，就没有 raw 会话可供第二条消费。

---

## 8. E2E 支线：现在怎么接入主 pipeline

### 8.1 E2E 的定位

`pipeline/e2e` 的目标是把 `molbench/MolBench-E2E/questions/*.md` 变成可执行输入，并通过统一执行层跑完整个流程。

### 8.2 入口

```bash
bash pipeline/e2e/run_e2e_pipeline.sh
```

或只跑子集：

```bash
bash pipeline/e2e/run_e2e_pipeline.sh --questions E2E-Q03,E2E-Q05
```

### 8.3 它做什么

- 扫描 `molbench/MolBench-E2E/questions/*.md`
- 生成 E2E 数据集 CSV
- 调用 `pipeline/claude_agent`
- 产出 raw 会话与运行日志

### 8.4 语义上的区别

E2E 任务不是标准 benchmark 的打分任务，因此：

- 默认跳过评测
- 轨迹语义偏“执行完成即 accepted”
- 重点是执行、落盘、导出与审计

### 8.5 产物

典型产物包括：

- `pipeline/e2e/runs/<timestamp>/e2e_dataset.csv`
- `pipeline/e2e/runs/<timestamp>/dataset_manifest.json`
- `pipeline/e2e/runs/<timestamp>/manifest.json`
- `pipeline/e2e/runs/<timestamp>/pipeline.log`

执行结果本体统一还是写入根 `results/`。

---

## 9. KG 支线：`tool-kg` 如何接入主 pipeline

### 9.1 KG 支线的目的

KG 这一支不是替代 MolBench，而是把 `tool-kg` 采样出来的图路径、工具链和自然语言问题，转成 `data-pipe` 可以直接执行和审计的任务。

### 9.2 协议层：`KGTaskSpec v0.2`

当前主协议已经提升为 `kg_task_spec_v0.2`。

它的核心要求是：

- `task_type = kg_sampled`
- 要有 `source.kg_run_id`
- 要有 `toolchain.tools`
- 要有 `toolchain.edges`
- 要有 `expected_trajectory.schema_version = trajectory_v2_graph`
- 不能把 `toolchain` 和 `expected_trajectory` 泄漏进 runtime prompt

### 9.3 输入输出流程

典型流程如下：

1. `pipeline/kg/scripts/inspect_kg_samples.py`
   - 检查 `sample_success_v2.jsonl` / `sample_success.jsonl`
   - 看字段完整性与问题样本

2. `pipeline/kg/scripts/build_kg_task_dataset.py`
   - 把 KG 的 sample 转成 `kg_sampled_tasks.jsonl`
   - 同时生成 `kg_tasks_exec.csv`
   - 输出 `manifest.json`
   - 输出 `schema_validation_report.md`

3. `pipeline/kg/run_kg_pipeline.sh`
   - 把 `kg_sampled_tasks.jsonl` 转成执行 CSV
   - 调用统一执行层
   - 落盘到 `results/kg_sampled/`

4. `pipeline/kg/scripts/scan_kg_rollouts.py`
   - 对 KG 执行结果做审计
   - 生成 `kg_rollout_summary.csv`
   - 生成 `kg_rollout_detailed.jsonl`
   - 生成 `kg_tool_usage_report.md`

5. 反馈文件
   - `pipeline/kg/data/<kg_run_id>/kg_execution_feedback.jsonl`

### 9.4 KG 的执行语义

KG 任务现在采用的是：

- 默认走 `molclaw-scp`
- 不做 benchmark evaluator
- 不引入 reward
- 只做执行与审计

### 9.5 KG 的评估目标是什么

KG 的重点不是传统意义上的分数，而是：

- 是否真的用了 MolClaw 工具
- 实际工具链和 expected toolchain 的重合度
- 是否能稳定生成 complete_session
- 是否能回写审计反馈

这和 `vs/ac/pf` 的目标是不同的。

---

## 10. `tool-kg` 本体在 2026-05-27 之后有哪些实质变化

这部分是很多人容易忽略的：`tool-kg` 也已经从“研究原型”变成了更适合接管管线的前端系统。

### 10.1 Stage 1 / 2 / 3 更清楚了

当前 `tool-kg` 的三段式更加清晰：

- Stage 1：toolcards / snapshot / 基础工具集合
- Stage 2：graph adjudication
- Stage 3：sample question generation

对应脚本：

- `scripts/run_pipeline_stage1_toolcards.sh`
- `scripts/run_pipeline_stage2_graph.sh`
- `scripts/run_sample_questions.sh`

### 10.2 工作区变得“自包含”

这是一个非常重要的工程变化。

现在 Claude Code 的工作目录不再依赖一大堆隐式外部上下文，而是强调：

- workdir 自己带上下文
- Claude 自己从 workdir 读取技能、prompt、说明
- `complete_session.jsonl` 在每个 workdir 内完整保存

这比“只靠外部注入一些参数”稳健得多。

### 10.3 provider 不再被仓库写死

现在 `tool-kg` 里也不再硬编码某个模型 provider。

正确做法变成：

- 运行前由你手动 `cc-switch`
- 仓库只负责按任务把 MCP / prompt / skills 接好

这避免了“代码里写死 qwen，但你当前环境已经切到别的 provider”的问题。

### 10.4 MCP 连接变得更明确

现在更强调：

- 每次 `claude -p` 前要有明确 MCP config
- `complete_session.jsonl` 第一行应能反映 init 信息
- `mcp_servers` 不能永远停留在不就绪状态

这类变化对于后续的稳定采样和审计非常重要。

### 10.5 `tool-kg` 与 `data-pipe` 的接口现在是明确的

接口核心已经变成：

- `tool-kg` 产出 `sample_success_v2.jsonl` / `questions.csv`
- `data-pipe/pipeline/kg` 消费这些输出
- `data-pipe` 再把它们接到统一执行和后处理链路里

也就是说，`tool-kg` 不再只是一个独立研究仓库，而是 `data-pipe` 的上游样本生产器。

---

## 11. 当前版本相对 5/27 之前，最重要的改进表

下面这张表适合拿来做“版本变化摘要”。

| 方面 | 5/27 之前 | 当前版本 | 实际收益 |
|---|---|---|---|
| 架构 | 执行 / 评测 / 后处理容易耦合 | `claude_agent / evaluate / postprocess` 分层明确 | raw 会话可以独立重建与审计 |
| 结果真源 | 中间产物混杂，容易依赖副作用 | `complete_session.jsonl` 成为唯一真源 | 后处理可重跑 |
| reward | 轨迹导出中曾有 reward 语义 | reward 已移除 | 避免不稳健 reward 误导训练 |
| 任务范围 | 主要围绕 `vs/ac/pf` | 增加 `e2e`、`kg` | 管线更通用 |
| provider | 容易写死某个模型 | provider 外部 `cc-switch` 控制 | 更稳、更可移植 |
| MCP | 连接状态不够明确 | 每次运行前都检查 ready | 少“假连接、真失败” |
| 训练数据 | JSON action / 噪声较多 | ReAct + task-aware final answer + 瘦身 | 更适合训练 |
| 后处理输出 | 长 JSONL、不便阅读 | `mcp_sft_all/` 每样本一个 JSON | 更适合人工检查 |
| 审计 | 分散在多个运行文件里 | cleaning report / manifest / validation report | 更适合交接与排错 |
| KG 接入 | 原型化 | `kg_task_spec v0.2` + `pipeline/kg` | 能直接对接统一执行层 |
| E2E 接入 | 较弱 | 独立 `pipeline/e2e` 支线 | 长程任务可以不评测直接跑通 |

---

## 12. 现在如果你要接手，应该先看哪些文件

### 12.1 主链路必看

- `README.md`
- `scripts/README.md`
- `scripts/run_molbench_workflow.sh`
- `scripts/run_postprocess.sh`
- `pipeline/claude_agent/README.md`
- `pipeline/claude_agent/test_flow_claude.sh`
- `pipeline/claude_agent/launch_claude.sh`
- `pipeline/claude_agent/run_claude.py`
- `pipeline/evaluate/readme.md`
- `pipeline/evaluate/eval_runner.py`
- `pipeline/postprocess/README.md`
- `pipeline/postprocess/trajectory_exporter.py`
- `pipeline/postprocess/scan_molclaw_usage.py`
- `pipeline/postprocess/post_process_sft.py`
- `scripts/run_llm_clean.sh`
- `scripts/build_verl_bundle.sh`（已移除）

### 12.2 KG / E2E 必看

- `pipeline/e2e/README.md`
- `pipeline/e2e/run_e2e_pipeline.sh`
- `pipeline/e2e/scripts/build_e2e_dataset.py`
- `pipeline/kg/README.md`
- `pipeline/kg/schemas/kg_task_spec_v0.2.md`
- `pipeline/kg/scripts/build_kg_task_dataset.py`
- `pipeline/kg/scripts/inspect_kg_samples.py`
- `pipeline/kg/scripts/scan_kg_rollouts.py`
- `pipeline/kg/run_kg_pipeline.sh`

### 12.3 `tool-kg` 上游必看

- `tool-kg/README.md`
- `tool-kg/docs/deep-research-plan.md`
- `tool-kg/docs/3-pair-ref.md`
- `tool-kg/src/molclaw_kg/adjudicators/claude_code_runtime.py`
- `tool-kg/src/molclaw_kg/question_sampling/sampler.py`
- `tool-kg/scripts/run_pipeline_stage1_toolcards.sh`
- `tool-kg/scripts/run_pipeline_stage2_graph.sh`
- `tool-kg/scripts/run_sample_questions.sh`

---

## 13. 常见问题定位路径

### 13.1 如果执行阶段根本没跑起来

先看：

- `results/.../run_config.json`
- `results/.../run_summary.jsonl`
- `results/.../completion_report.json`
- `results/.../row*/run_meta.json`
- `results/.../row*/complete_session.jsonl`

如果 `complete_session.jsonl` 没有正常写出，问题通常在：

- provider 没切对
- MCP 没 ready
- `claude` CLI 不兼容
- prompt 或 workdir 配置有问题

### 13.2 如果评测分数很差

优先看：

- `results/.../bench_scores.json`
- `preds/molbench_<task>/molbench_<task>.json`
- 逐样本 `metrics`

然后分任务排查：

- VS：候选长度、重复预测、候选集外
- AC：是否产生有效单答案
- PF：`selected_smiles` / `prediction` 这类集合是否和 GT 一致

### 13.3 如果后处理候选变少了

优先看：

- `results/postprocess_candidates/molclaw_usage_summary.csv`
- `results/postprocess_candidates/{vs,ac,pf,kg,e2e}/`
- `results/postprocess_candidates/sft_outputs/rejected_samples.jsonl`
- `results/postprocess_candidates/sft_outputs/schema_validation_report.json`

常见原因：

- 没有真正的 molclaw usage
- answer-hit 没过
- final_answer schema 不通过
- observation 太大被截断后丢失关键信息
- 工程 chatter / 本地路径未完全清洗

### 13.4 如果 KG 任务跑了但没有价值

优先看：

- `pipeline/kg/data/<kg_run_id>/kg_execution_feedback.jsonl`
- `pipeline/kg/runs/<run_id>/pipeline.log`
- `results/kg_sampled/`

重点不是只看“成功/失败”，而是看：

- 实际用了哪些工具
- 与 expected toolchain 的重合度
- 失败是 prompt 不够驱动，还是工具本身不可执行

---

## 14. 这套系统的当前成熟度判断

### 14.1 已经很稳定的部分

- 样本生成和任务路由已经比较规范
- `complete_session.jsonl` 作为 raw 真源的思路已经固定
- 执行 / 评测 / 后处理已经彻底分层
- `vs/ac/pf/e2e/kg` 的任务边界比之前清楚得多
- 后处理生成的 SFT 数据已经从“杂乱轨迹”收紧成“可训练样本”

### 14.2 仍然需要谨慎的部分

- provider / MCP 的外部依赖仍然是稳定性关键
- `kg` 的样本质量还取决于上游采样质量
- `e2e` 是执行型任务，不应该被误当成可直接打分的 benchmark
- 训练样本虽然更干净了，但仍然需要持续做样本抽查

### 14.3 最后的工程判断

当前版本不是“最终终点”，但已经足够形成一个真正可接手、可扩展、可审计的工程系统。

尤其是这三件事已经到位了：

1. **raw 会话可重建**
2. **训练样本可验证**
3. **上游 KG / 下游训练都能继续接**

---

## 15. 建议的后续工作顺序

如果接手者要继续推进，建议按这个顺序：

1. 先跑一轮标准 MolBench：
   - `bash scripts/run_molbench_workflow.sh --seed <S> --n-cases <N>`

2. 再做后处理：
   - `bash scripts/run_postprocess.sh --results-root results`

3. 检查 `results/postprocess_candidates/sft_outputs/mcp_sft_all/`
   - 看每个样本是否符合 task-aware final answer

4. 如果要做 E2E：
   - 走 `pipeline/e2e/run_e2e_pipeline.sh`

5. 如果要做 KG：
   - 先从 `tool-kg` 的 `sample_success_v2.jsonl` 构建 KG 任务
   - 再走 `pipeline/kg/run_kg_pipeline.sh`

6. 如果要对接训练框架：
   - 如确需 legacy bundle，显式调用相关 Python bundle 导出与验证脚本
   - 再看 `docs/mol_pipeline_to_verl_bundle_v0.1.md`

---

## 16. 最后一句话

如果把当前系统看成一个“生产训练数据的工厂”，那么 5/27 之后最大的变化就是：

- 工厂从“一个大杂烩车间”变成了“有原料、产线、质检、成品和审计报表的工厂”
- `complete_session.jsonl` 是原料
- `trajectory_exporter / scan_molclaw_usage / post_process_sft` 是产线
- `cleaning_report / schema_validation_report / dataset_manifest` 是质检
- `mcp_sft_all/` 和 `mcp_rl_prompts_all.jsonl` 是成品

这也是现在这版 `data-pipe` 最重要的变化。
