# VS Pipeline 全量堆叠式深度研究报告（FULL_DUMP）

> 报告生成时间：2026-05-13
> 研究根目录（PROJECT_ROOT）：`<legacy-vs-pipeline-root>`
> 运行模式（MODE）：`deep`
> 语言（LANGUAGE）：中文
> 堆叠风格（STACK_STYLE）：`FULL_DUMP`
> 说明：用户未提供 `OUTPUT_REPORT` 实值，本文默认写入：`<legacy-vs-pipeline-root>/PROJECT_RESEARCH_REPORT.md`。

---

## 1. 执行摘要（项目一句话 + 当前成熟度 + 总体判断）

一句话定义：该项目是一个面向 MolBench 三任务（VS/AC/PF）的 **Agent 批量执行 + 结果评估 + 轨迹导出** 管线，目标是把 `complete_session.jsonl` 转换为可审计的 trajectory 数据集。
证据：`<legacy-vs-pipeline-root>/readme.md:1-41`、`<legacy-vs-pipeline-root>/claude_agent/run_claude.py:847-867`、`<legacy-vs-pipeline-root>/claude_agent/trajectory_exporter.py:599-639`。

成熟度判断：**研究工程混合型，PoC+工程化过渡阶段（中等成熟）**。
- 工程化已落地：统一入口、完成性检查、评估审计、trajectory 导出。
- 稳定性短板明显：VS 任务工具链可用性与超时问题突出，历史 run 完整率不稳定。
证据：`<legacy-vs-pipeline-root>/readme.md:45-60`，历史 run 检查结果（24 个 run 中仅一部分有 completion/bench/traj 文件）。

总体判断：
- **可用于继续研发与问题定位**（有较完整代码链和可追溯产物）。
- **尚不适合直接作为高稳定生产基线**（尤其 VS 任务的 MCP 连接失败与长时超时会显著拉低可复现性）。
- **可用于论文/交接**，但必须附带“失败模式与过滤规则”，并区分“执行完成”与“样本有效（accepted）”。

---

## 2. 信息源与调查方法（含优先文件阅读顺序）

### 2.1 优先文件处理

用户未给出 `PRIORITY_FILES` 实值（仅给了占位说明），按“空优先列表”处理。
执行策略：先读项目文档入口，再全项目扫描，再下钻核心脚本与产物。

实际阅读顺序：
1. `<legacy-vs-pipeline-root>/readme.md`
2. `<legacy-vs-pipeline-root>/ToFix.md`
3. `<legacy-vs-pipeline-root>/claude_agent/README.md`
4. `<legacy-vs-pipeline-root>/evaluate/readme.md`
5. 核心代码：`run_claude.py`、`trajectory_exporter.py`、`eval_runner.py`、`run_eval_bench.py`、`launch_claude.sh`、`test_flow_claude.sh`、`test_parallel.sh`
6. 数据资产：`molbench/*.csv`
7. 产物资产：`results/` 下 24 个 run + 并行日志 + 代表性样本目录。

### 2.2 方法学

- 静态分析：逐文件读取函数、参数、I/O 协议、异常路径。
- 结构扫描：目录树、文件类型计数、产物完整性检查。
- 运行自检（轻量）：CLI `--help`、对已有结果重跑 `evaluate`/`trajectory_exporter`。
- 历史证据抽样：提取 run 配置、summary、completion、bench、trajectory 汇总。
- 一致性核验：文档↔代码、代码↔产物、输入↔输出、命名↔路径。

---

## 3. 项目全景地图（目录/模块/资产）

### 3.1 项目类型与边界（Phase 1: Requirements Discovery）

研究目标（Research Goal）：
- 将 Agent 执行会话转成可用于 SFT/DPO/RL 的标准轨迹数据（trajectory-level + step-level）。
- 以 MolBench 任务作为样本来源并评估结果质量。

业务目标（Business Goal）：
- 支持 VS/AC/PF 多任务切换与批量运行，形成可审计输出。

技术目标（Technical Goal）：
- 单一执行真源（`run_claude.py`）、多 rollout、完成性检查、评估审计、轨迹导出。

交付目标（Deliverable Goal）：
- 每次 run 输出结构化目录：`run_config.json`、`run_summary.jsonl`、`preds/`、`completion_report.json`、`trajectories/`、`bench_scores.json`（评估后）。

项目类型识别：**研究工程混合型**。
- 研究属性：MolBench scientific task、轨迹数据化。
- 工程属性：脚本编排、质量门、批处理、目录规范。

调查范围：
- 包含：`claude_agent/`、`evaluate/`、`molbench/`、`results/`、`skills/`、`skills_full/`、`ToFix.md`。
- 不包含：外部 MCP 服务内部实现、`claude` CLI 内部实现、远端模型服务运维细节。

### 3.2 目录分层

- `claude_agent/`：执行与导出主链。
- `evaluate/`：评估入口与任务指标实现。
- `molbench/`：本地 CSV 数据集（VS/AC/PF，30/900 规模及样例）。
- `results/`：历史 run 资产（主容量区）。
- `skills/`：VS 任务提示词与技能包装。
- `skills_full/`：AC/PF 使用的 full prompt 与完整技能体系。
- `.mcp.json`：MCP server 配置（含 URL/Header）。

容量与规模：
- 项目总大小：约 `2.3G`。
- `results/`：约 `2.3G`（主占用）。
- 非 results 文件总数：153；全项目文件总数：136,464（大量产物文件）。
证据：`du -sh` 与扩展名统计结果。

---

## 4. 端到端技术链路复原（Phase 2 + Phase 3）

### 4.1 主执行流（工程流）

标准链路：
1. `claude_agent/test_flow_claude.sh` 负责“推理+评估编排”。
2. 调用 `claude_agent/launch_claude.sh --run-dataset`。
3. `launch_claude.sh` 构造任务专用 MCP 配置并转发参数给 `run_claude.py`。
4. `run_claude.py` 逐样本执行、解析答案、落盘 summary/preds/completion。
5. `run_claude.py`（默认）触发 `trajectory_exporter.py`。
6. `test_flow_claude.sh` 拿 `RESULTS_DIR` 后调用 `evaluate/run_eval_bench.py`。

证据：
- `<legacy-vs-pipeline-root>/claude_agent/test_flow_claude.sh:82-124`
- `<legacy-vs-pipeline-root>/claude_agent/launch_claude.sh:194-225`
- `<legacy-vs-pipeline-root>/claude_agent/run_claude.py:951-1125`
- `<legacy-vs-pipeline-root>/evaluate/run_eval_bench.py:18-35`

### 4.2 数据预处理链（Data Flow）

#### VS
- 输入列：`index, questions, answer, n_active`（另含 task_type/answer_score）。
- `questions` 字段为 JSON 字符串，提取 `candidates`。
- GT 使用 `answer` JSON 列表。

证据：`run_claude.py` `_load_samples` VS 分支：`<legacy-vs-pipeline-root>/claude_agent/run_claude.py:459-483`。

#### AC
- 输入列：`question, answer, target, s1,k1,s2,k2`。
- GT 通过 `_parse_pf_gt` 解析后只保留第一项（单标签）。

证据：`<legacy-vs-pipeline-root>/claude_agent/run_claude.py:485-501`。

#### PF
- 输入列：`prompt`（或 `question` 兜底）+ `answer`（可多项集合）。

证据：`<legacy-vs-pipeline-root>/claude_agent/run_claude.py:503-517`。

数据规模核验：
- `molbench-vs-900.csv`: 900x6，候选长度固定 60。
- `molbench-ac-900.csv`: 900x7，answer 经解析长度恒 1。
- `molbench-pf-900.csv`: 900x6，answer 解析长度 1~5。

### 4.3 推理与解析链（Agent/Model Flow）

每样本（每 rollout）执行：
- 写 `question.json`、`prompt.txt`。
- 调 `claude --output-format stream-json`，输出 `complete_session.jsonl`。
- 解析 `<answer>` 或 `<solution>`，并多级 fallback（JSON block / session text / token 抽取）。
- 写 `parsed_answer.json`、`run_meta.json`。

证据：
- 会话执行：`<legacy-vs-pipeline-root>/claude_agent/run_claude.py:537-593`
- 单 rollout 落盘：`<legacy-vs-pipeline-root>/claude_agent/run_claude.py:595-719`
- 解析策略：`<legacy-vs-pipeline-root>/claude_agent/run_claude.py:350-444`

### 4.4 预测与评估链（Model/Eval Flow）

`run_claude.py` 输出：
- `preds/molbench_<task>/molbench_<task>.json`（主评估入口，rollout1）。
- `preds/molbench_<task>/rollouts/rollout_XXXX.json`（多 rollout）。

证据：`<legacy-vs-pipeline-root>/claude_agent/run_claude.py:1089-1106`。

`evaluate/eval_runner.py`：
- VS：top3/top10 命中与一致性审计（长度、候选集外、重复、无候选）。
- AC：准确率（单标签）。
- PF：集合 exact / precision / recall / f1。
- RDKit 可用则 canonical 评估；不可用退化字符串。

证据：
- VS：`<legacy-vs-pipeline-root>/evaluate/eval_runner.py:56-153`
- AC：`<legacy-vs-pipeline-root>/evaluate/eval_runner.py:155-235`
- PF：`<legacy-vs-pipeline-root>/evaluate/eval_runner.py:255-332`

### 4.5 轨迹导出链（Trajectory Flow）

`trajectory_exporter.py` 读取 row/rollout 样本，构建：
- `trajectory_level.jsonl`
- `step_level.jsonl`
- `accepted.jsonl`
- `rejected.jsonl`
- `dataset_summary.json`

核心质量门：
- parse_error、空候选（VS）、长度不等、重复、候选集外等。

证据：`<legacy-vs-pipeline-root>/claude_agent/trajectory_exporter.py:489-560, 599-639`。

---

## 5. 文档-代码-产物一一对应表（重点）

| 文档叙事 | 代码实现锚点 | 产物锚点 | 结论 |
|---|---|---|---|
| 多任务 VS/AC/PF | `run_claude.py --task` `/claude_agent/run_claude.py:849`，`launch_claude.sh --task` `/claude_agent/launch_claude.sh:43-45` | `results/molbench_vs_*`、`results/molbench_ac_*`、`results/molbench_pf_*` | A 已实现 |
| 单一执行真源 | `run_claude.py main` `/claude_agent/run_claude.py:847-1128` | 每个 run 的 `run_config.json`/`run_summary.jsonl` | A 已实现 |
| 多 rollout | `--num-rollouts` + rollout dir `/claude_agent/run_claude.py:65-69, 859, 985-1025` | `.../rowXXXX/rollout0001...`（如 `20260508_192901`） | A 已实现（历史 run 可见） |
| 完整性闸门 | `_check_run_completeness` `/claude_agent/run_claude.py:749-845` | `completion_report.json` | A 已实现 |
| 评估写回 metrics | `eval_runner.py` 写回 entry `/evaluate/eval_runner.py:121-132, 206-215, 303-313` | `preds/molbench_*/molbench_*.json` | A 已实现 |
| 轨迹导出五件套 | `trajectory_exporter.py` 输出 `/claude_agent/trajectory_exporter.py:599-639` | `trajectories/*.jsonl + dataset_summary.json` | A 已实现 |
| VS 使用 `skills/`，AC/PF 使用 `skills_full/` | `test_flow_claude.sh:47-53`，`launch_claude.sh:124-140` | run 日志 route 行（parallel logs） | A 已实现 |
| RDKit 不可用时 trajectory 全拒绝 | 文档宣称 `/readme.md:150`、`/claude_agent/README.md:88` | 代码未添加 `rdkit_unavailable` reject；仅退化字符串 `/trajectory_exporter.py:467-477` | B 文档声称但代码未证实 |
| run_claude 默认数据集可直接运行 | 默认值 `/run_claude.py:850` | 默认路径 `molbench-vs/MolBench-vs-25.csv` 本地不存在 | B 文档/代码默认与目录不一致 |
| 三任务并行可运行 | 文档命令 `/readme.md:25-29` | 并行日志多次失败（database locked / 502）`results/parallel_logs/*/*.log`，`ToFix.md:1` | B 条件成立，稳定性不足 |
| Token 治理/上限策略 | 文档标注未做 `/readme.md:57` | 代码中无 token budget 控制 | C 仅规划 |
| 历史结果批量回填导出 | 文档标注未做 `/readme.md:58` | 无专门批处理回填脚本 | C 仅规划 |
| Step-level dense reward | 文档标注未做 `/readme.md:59` | 代码仅最后一步 reward `/trajectory_exporter.py:349-352` | C 仅规划 |

A=已实现且可证实，B=文档声称但代码/实测不足，C=规划未落地。

---

## 6. 关键实验/关键流程逐步说明

### 6.1 以最新稳定样本组（2026-05-12 14:27:32）为例

#### VS（30 样本）
- run：`results/molbench_vs_qwen-397b_run_20260512_142732`
- completion：`completeness_ok=true`
- bench：`top3_avg_hit_num=0.1667, top10_avg_hit_num=0.6333`
- trajectory：`n_accepted=8, n_rejected=22`
- 解析失败主因：timeout（9/30）+ parse_error（17/30）
- 平均单样本耗时约 870s，最长达 1200s timeout。

证据：
- `.../run_config.json`
- `.../completion_report.json`
- `.../bench_scores.json`
- `.../trajectories/dataset_summary.json`
- `.../run_summary.jsonl`

#### AC（30 样本）
- run：`results/molbench_ac_qwen-397b_run_20260512_142732`
- bench：`accuracy=0.8276 (29 valid)`
- trajectory：`n_accepted=29, n_rejected=1`
- 平均耗时约 387s。

#### PF（30 样本）
- run：`results/molbench_pf_qwen-397b_run_20260512_142732`
- bench：`exact_set_match_rate=0.8667, avg_f1=0.9422`
- trajectory：`n_accepted=30, n_rejected=0`
- 平均耗时约 96s。

### 6.2 历史 run 的“完整性与稳定性”事实

- 总 run：24。
- 部分 run 只有 `run_summary.jsonl`，无 `completion_report/bench_scores/trajectories`，属于中断或未完成后处理。
- 典型：
  - `molbench_vs_qwen-397b_run_20260511_153831`：`selected_rows=900` 但 `run_summary_lines=63`。
  - `molbench_ac_qwen-397b_run_20260511_150242`：`selected_rows=900` 但 `run_summary_lines=0`。

证据：results 批量核验表。

---

## 7. 数据资产核验（文件存在性、shape、标签口径、规模）

### 7.1 原始数据文件

存在文件：
- `molbench/molbench-vs-30.csv`, `molbench/molbench-vs-900.csv`
- `molbench/molbench-ac-30.csv`, `molbench/molbench-ac-900.csv`
- `molbench/molbench-pf-30.csv`, `molbench/molbench-pf-900.csv`
- 兼容样例：`MolBench-vs-25.csv`, `MolBench-vs-sample.csv`

### 7.2 字段与规模

VS（900）：
- 列：`index, questions, task_type, answer, answer_score, n_active`
- `questions` JSON 解析成功率 100%，`candidates` 长度固定 60。
- `n_active` 分布 6~10。

AC（900）：
- 列：`question, answer, target, s1, k1, s2, k2`
- `answer` 为单 SMILES 字符串（非 JSON array），经 `_parse_pf_gt` 后长度恒 1。

PF（900）：
- 列：`prompt, answer, meta, source_variant, task_type, difficulty`
- `task_type/difficulty` 有 600 行缺失。
- `answer` 经 `_parse_pf_gt` 后长度分布 1~5。

### 7.3 标签口径

- VS：排序任务，GT 为 active 分子集合（列表）；候选固定 60。
- AC：二选一（或单标签）任务，GT 单分子。
- PF：集合筛选任务，GT 可多分子集合。

---

## 8. 关键脚本入口清单（按运行顺序）

### 8.1 推荐入口顺序

1. `bash claude_agent/test_claude.sh`（单样本环境健康检查）
2. `bash claude_agent/test_flow_claude.sh ...`（单任务完整链路：推理+评估）
3. `bash claude_agent/test_parallel.sh ...`（三任务并行压测，不建议直接大规模）
4. `python claude_agent/trajectory_exporter.py <results_dir> --task <task>`（轨迹重导）
5. `python evaluate/run_eval_bench.py <results_dir> --task <task>`（评估重跑）
6. `python claude_agent/repair_parsed_answers.py <results_dir> --re-eval`（VS 解析修复）

### 8.2 每个入口职责

- `test_flow_claude.sh`：编排 `launch` + `eval`，并从 stdout 捕获 `RESULTS_DIR`。
- `launch_claude.sh`：provider 切换、MCP 参数生成、运行模式分发。
- `run_claude.py`：唯一批处理执行真源。

证据：
- `<legacy-vs-pipeline-root>/claude_agent/test_flow_claude.sh:82-124`
- `<legacy-vs-pipeline-root>/claude_agent/launch_claude.sh:194-226`
- `<legacy-vs-pipeline-root>/claude_agent/run_claude.py:847-1128`

---

## 9. 复现环境准备（Python版本、依赖、安装命令、环境变量）

### 9.1 代码显式依赖

Python 包（代码直接 import）：
- `tqdm`（`run_claude.py`）
- `rdkit`（评估与轨迹 canonical，可选但强烈建议）

外部可执行依赖：
- `claude` CLI
- `cc-switch`

证据：
- `run_claude.py:20, 84-90, 855`
- `launch_claude.sh:143-146, 179`
- `test_parallel.sh:71-79`

### 9.2 依赖声明现状

- 项目内未发现 `requirements.txt/pyproject.toml/environment.yml`。
- 环境可复现性主要依赖“外部既有运行环境”。

风险：新机器首次部署会出现依赖漂移。

### 9.3 建议的最小安装模板（可执行）

```bash
cd <legacy-vs-pipeline-root>
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install tqdm
# RDKit 建议用 conda 安装更稳：
# conda install -c conda-forge rdkit
```

同时确认：
```bash
which claude
which cc-switch
python -c "from rdkit import Chem; print('rdkit ok')"
```

### 9.4 关键环境变量

- `CC_SWITCH_PROVIDER`（provider 默认）
- `CLAUDE_BIN`（默认 `claude`）
- `TASK/SKILLS_ROOT/SYSTEM_PROMPT_FILE`（可由脚本覆盖）

---

## 10. 复现路线图

### 10.1 Minimal Path（最小可运行路径）

目标：快速验证链路“可执行且可落盘”。

```bash
cd <legacy-vs-pipeline-root>
bash claude_agent/test_claude.sh
bash claude_agent/test_flow_claude.sh qwen-397b claude 1 1 1 vs <legacy-vs-pipeline-root>/molbench/molbench-vs-30.csv 1
```

说明：
- 最后参数 `skip_provider_switch=1` 可减少 provider 并发锁冲突风险。
- 成功标志：stdout 出现 `RESULTS_DIR=...`，目录内有 `run_summary.jsonl` 与 `bench_scores.json`。

### 10.2 Full Path（全量复现路径）

建议按任务串行，不建议直接三任务并发：

```bash
cd <legacy-vs-pipeline-root>

# VS
bash claude_agent/test_flow_claude.sh qwen-397b claude 0 1 1 vs <legacy-vs-pipeline-root>/molbench/molbench-vs-30.csv 1

# AC
bash claude_agent/test_flow_claude.sh qwen-397b claude 0 1 1 ac <legacy-vs-pipeline-root>/molbench/molbench-ac-900.csv 1

# PF
bash claude_agent/test_flow_claude.sh qwen-397b claude 0 1 1 pf <legacy-vs-pipeline-root>/molbench/molbench-pf-900.csv 1
```

若只跑推理：
```bash
bash claude_agent/launch_claude.sh --run-dataset --task vs --dataset-csv <legacy-vs-pipeline-root>/molbench/molbench-vs-900.csv --skills-root <legacy-vs-pipeline-root>/skills --results-root <legacy-vs-pipeline-root>/results --provider qwen-397b --claude-bin claude --num-rollouts 1 --parallel-rollouts 1 --skip-provider-switch
```

后处理：
```bash
python evaluate/run_eval_bench.py <RESULTS_DIR> --task vs
python claude_agent/trajectory_exporter.py <RESULTS_DIR> --task vs
```

### 10.3 关键坑点修复（复现中应先处理）

1. `run_claude.py` 默认 `--dataset-csv=molbench-vs/MolBench-vs-25.csv` 路径在本仓不存在，必须显式传参。
2. VS 的 `molclaw-vs` MCP 在最新 run 中 30/30 初始化失败，需优先修复连接性。
3. 三任务并发时可能出现 `database is locked`（cc-switch 配置锁）与 `API Error: 502`。
4. VS 单样本 timeout 上限 1200s，900 样本全量会非常长。
5. 若 RDKit 缺失，评估会退化字符串匹配（轨迹导出并不会自动拒绝，和文档描述不一致）。

---

## 11. 风险与异常清单（P0/P1/P2）

### P0（会直接导致复现失败或结果失真）

1. **VS MCP 连接失败（高频）**
- 现象：VS run init 中 `mcp_servers=[{"name":"molclaw-vs","status":"failed"}]`。
- 证据：`results/molbench_vs_qwen-397b_run_20260512_142732/row0002_idx2/complete_session.jsonl`（init event）。
- 后果：科学工具链不可用，输出偏向本地文本推理，accepted rate 低。
- 修复：先做 `test_claude.sh` + 单行 VS smoke；检查 URL、header、网络可达性。

2. **服务端 502 失败（API retry 后终止）**
- 证据：`results/molbench_vs_qwen-397b_run_20260512_135001/row0001_idx1/complete_session.jsonl:2-13`（连续 retry + 502 result）。
- 后果：`parse_error=api_error_response`，样本直接无效。
- 修复：重试策略（批次化、降并发、错峰）、失败样本自动重跑队列。

3. **默认数据集路径无效**
- 证据：`run_claude.py:850` 默认 `molbench-vs/MolBench-vs-25.csv`；该目录本地缺失。
- 后果：直接调用 `run_claude.py` 时启动失败。
- 修复：统一改成 `molbench/MolBench-vs-25.csv` 或强制参数必填。

4. **并发模式下 cc-switch 锁冲突**
- 证据：`results/parallel_logs/20260511_150239/*.log` 出现 `database is locked`。
- 后果：任务未启动即失败。
- 修复：provider 切换仅执行一次；并行任务全部 `--skip-provider-switch`。

5. **VS 超时比例高（1200s）**
- 证据：`run_summary.jsonl` 统计：`vs 20260512_142732` 超时 9/30，平均 870s。
- 后果：大量 `parse_error + length_mismatch:0!=60`，有效样本不足。
- 修复：缩短任务范围、优化 prompt/tool 链、加超时分层策略（早停+续跑）。

### P1（不一定失败，但会显著影响质量）

1. 文档与代码不一致：文档称 RDKit 缺失会 trajectory 拒绝，代码实际是字符串退化。
2. 早期 run 配置缺少 `task` 字段（VS 旧 run），依赖推断逻辑。
3. 结果目录规范存在代际差异（部分 run 无 completion/bench/traj）。

### P2（可维护性/可读性问题）

1. trajectory 全为 JSONL 单行，人工可读性差（ToFix 已提）。
2. 缺少统一依赖声明文件（requirements/conda env）。
3. 配置中存在明文凭据（见第 13 节改进建议）。

---

## 12. 已实现 / 未实现 / 规划中（三分法结论）

### A. 已实现且可证实

- 三任务切换（VS/AC/PF）与任务路由。
- run 主链路（launch→run_claude→exporter→eval）。
- 多 rollout 目录结构与预测文件输出。
- completion gate（结构完整性检查）。
- 评估审计与轨迹导出（含 accepted/rejected）。
- fallback 解析机制（answer tag、json block、token 提取）。

### B. 文档声称但代码未证实/部分不一致

- “RDKit 不可用时 trajectory 样本拒绝”——文档写明，代码未实现该 reject reason。
- “三任务并行测试可直接跑”——实际并发失败概率高（502 / database lock）。
- “仅跑推理示例路径”引用旧目录 `molbench-vs/...`，与当前目录结构不一致。

### C. 规划/讨论中未落地

- token 治理/上限策略。
- 历史结果批量回填导出。
- step-level dense reward（当前仅终点 reward）。

---

## 13. 改进建议（工程改造线 + 研究改进线）

### 13.1 工程改造线

1. **先修 VS MCP 可达性（P0）**
- 增加启动前 MCP 健康检查并写入 run_config。
- 若 `molclaw-vs` failed，直接 fail-fast，不进入长时推理。

2. **统一配置与凭据治理（安全+复现）**
- 当前 `.mcp.json` 与 `launch_claude.sh` 含明文 token：
  - `<legacy-vs-pipeline-root>/.mcp.json:7,14`
  - `<legacy-vs-pipeline-root>/claude_agent/launch_claude.sh:130,138`
- 建议迁移到环境变量/密钥管理，不进仓库。

3. **修复默认路径与文档漂移**
- `run_claude.py --dataset-csv` 默认路径改为现有 `molbench/`。
- `readme.md` 示例命令同步更新，避免旧目录误导。

4. **引入失败重跑策略**
- 对 `api_error_response`、`timeout` 样本自动重试 N 次并记录 retry trace。
- 保留失败样本清单，支持断点续跑。

5. **补齐依赖锁定文件**
- 提供 `requirements.txt` 或 `environment.yml`。
- 在 CI 中执行最小 smoke：`test_claude.sh + run_eval_bench.py --help + trajectory_exporter.py --help`。

6. **可读性增强**
- 提供 `trajectory_pretty/` 可选导出（按 task_id 拆分 json）。
- 兼容保留 JSONL 供训练。

### 13.2 研究改进线

1. **VS 任务分段策略**
- 当前 60 候选全排序 + 长工具链造成高超时；可先 coarse-to-fine 两段排序。

2. **解析鲁棒性升级**
- VS 中常见 `answer_n=61`、非字符串项；增加 post-parse sanitizer（截断到候选长度、类型修正）。

3. **奖励定义扩展**
- 在 `step_level` 注入中间奖励（如工具成功率、候选一致性、格式合规度）。

4. **跨 run 可比性协议**
- 固定模型版本、固定 MCP 可用集、固定 prompt 版本 hash、固定数据切片。

5. **论文写作建议**
- 结果章节需同时报告 `n_samples` 与 `n_accepted`，避免“只看均值”掩盖失效率。

---

## 14. 附录（关键文件索引、参数索引、产物索引）

### 14.1 关键文件索引

- 主文档：
  - `<legacy-vs-pipeline-root>/readme.md`
  - `<legacy-vs-pipeline-root>/ToFix.md`
- 执行链：
  - `<legacy-vs-pipeline-root>/claude_agent/run_claude.py`
  - `<legacy-vs-pipeline-root>/claude_agent/launch_claude.sh`
  - `<legacy-vs-pipeline-root>/claude_agent/test_flow_claude.sh`
  - `<legacy-vs-pipeline-root>/claude_agent/test_parallel.sh`
- 导出与评估：
  - `<legacy-vs-pipeline-root>/claude_agent/trajectory_exporter.py`
  - `<legacy-vs-pipeline-root>/evaluate/eval_runner.py`
  - `<legacy-vs-pipeline-root>/evaluate/run_eval_bench.py`
- 数据：
  - `<legacy-vs-pipeline-root>/molbench/*.csv`
- 配置：
  - `<legacy-vs-pipeline-root>/.mcp.json`

### 14.2 参数索引（核心）

`run_claude.py`：
- `--task {vs,ac,pf}`
- `--dataset-csv`
- `--skills-root`
- `--results-root`
- `--system-prompt-file`
- `--provider`
- `--claude-bin`
- `--start-row/--end-row/--limit`
- `--num-rollouts/--parallel-rollouts/--rollout-seed-base`
- `--mcp-config-file/--strict-mcp-config`
- `--skip-provider-switch`
- `--no-export-trajectories`

### 14.3 产物索引（标准 run）

- run 级：
  - `run_config.json`
  - `run_summary.jsonl`
  - `completion_report.json`
  - `bench_scores.json`（若执行评估）
- row 级：
  - `question.json`
  - `complete_session.jsonl`
  - `parsed_answer.json`
  - `run_meta.json`
  - `prompt.txt`
  - 可选 `result.md/run_log.md/step*.json`（取决于 agent 执行）
- 预测与轨迹：
  - `preds/molbench_<task>/molbench_<task>.json`
  - `preds/molbench_<task>/rollouts/rollout_XXXX.json`
  - `trajectories/trajectory_level.jsonl`
  - `trajectories/step_level.jsonl`
  - `trajectories/accepted.jsonl`
  - `trajectories/rejected.jsonl`
  - `trajectories/dataset_summary.json`

### 14.4 代表性 run 索引（建议复查）

- VS（30样本，较新）：
  - `<legacy-vs-pipeline-root>/results/molbench_vs_qwen-397b_run_20260512_142732`
- AC（30样本，较新）：
  - `<legacy-vs-pipeline-root>/results/molbench_ac_qwen-397b_run_20260512_142732`
- PF（30样本，较新）：
  - `<legacy-vs-pipeline-root>/results/molbench_pf_qwen-397b_run_20260512_142732`
- 失败样本（502）：
  - `<legacy-vs-pipeline-root>/results/molbench_vs_qwen-397b_run_20260512_135001/row0001_idx1/complete_session.jsonl`

---

## Phase 3.5 Consolidation（跨模块一致性汇总 + 评分）

### 文档-代码一致性评分：63/100
- 优点：主链路描述与脚本总体一致。
- 扣分：RDKit 拒绝逻辑描述不一致；部分命令路径过时。

### 代码-产物一致性评分：78/100
- 优点：新 run 产物结构与代码定义基本一致。
- 扣分：历史 run 代际差异大；中断 run 较多。

### 输入-输出一致性评分：72/100
- 优点：VS/AC/PF 输入列映射清晰，输出 schema 统一。
- 扣分：VS parse/timeout 导致有效样本低，结果稳定性受影响。

### 命名与路径一致性评分：58/100
- 问题：默认路径与仓库结构不一致（`molbench-vs` vs `molbench`），文档示例存在旧路径。

### 综合质量评分（0-100）：**68/100**

问题分级汇总：
- P0：5 项（MCP failed、502、默认路径、并发锁、VS高超时）
- P1：3 项（文档漂移、旧 run schema、可比性不足）
- P2：3 项（可读性、依赖声明缺失、明文凭据）

---

## Phase 5 Iterative Refinement（报告自检）

自检项 1：陌生读者是否可直接上手？
- 已补充：环境准备、入口顺序、最小路径、全量路径、失败修复清单。

自检项 2：是否列出前5大复现风险？
- 已满足：第 11 节 P0 给出 5 项高概率失败点与修复建议。

自检项 3：是否同时给出 Minimal Path 与 Full Path？
- 已满足：第 10 节。

自检项 4：关键结论是否有证据锚点？
- 已尽可能在每节附路径/函数/参数/产物锚点；历史统计给出具体 run 目录与文件名。

