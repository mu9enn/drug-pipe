# Mol-Pipeline 后处理脚本格式调整与硬清洗 Handoff

> 文档日期：2026-06-08
> 适用仓库：`<data-pipe-root>`
> 核心代码：`pipeline/postprocess/`
> 一键入口：`scripts/run_postprocess.sh`

## 1. 文档目的

Mol-Pipeline 当前将轨迹清洗分成两个阶段：

1. **确定性脚本后处理：格式调整与硬清洗**
2. **LLM 清洗：语义压缩、质量判断和更灵活的内容改写**

本文只描述第一阶段，即 `pipeline/postprocess/` 当前代码真正执行的全部流程、规则、输入输出、边界和已知风险。接手者不需要阅读历史聊天记录，即可安全地重跑、检查或继续改造该阶段。

硬清洗阶段的目标不是判断一条轨迹是否“科学上完美”，而是建立一套可复现、可审计的确定性转换：

```text
raw complete_session.jsonl
        |
        v
trajectory_exporter.py
  - 重建轨迹
  - 计算任务指标
  - 判定 accepted/rejected
        |
        v
scan_molclaw_usage.py
  - 汇总 accepted 会话
  - 复制候选 raw session
  - 汇总指标 CSV
        |
        v
post_process_sft.py
  - ReAct 格式转换
  - 硬过滤与结构清洗
  - 生成 SFT/RL prompt
  - 生成逐样本审计报告
        |
        v
LLM clean（独立第二阶段，不在本文范围内）
```

## 2. 当前工程边界

### 2.1 本阶段负责

- 从 raw run 目录重新构建轨迹文件。
- 根据任务规则、运行完成状态和 MolClaw 使用情况判断 accepted/rejected。
- 为 VS/AC/PF 计算单样本指标。
- 汇总 accepted 会话到统一候选目录。
- 将 Claude Code raw session 转换成 ReAct-style SFT。
- 生成不分 train/valid/test 的统一 SFT 和 RL prompt 文件。
- 硬删除非目标工具调用。
- 配对 tool call 与 observation。
- 清除或替换本地绝对路径。
- 压缩部分特定工具结果，截断超长 observation。
- 生成机器可读和人类可读的校验报告。
- 保留清洗动作 sidecar，支持追踪为什么某段内容被修改或删除。

### 2.2 本阶段不负责

- 不调用 LLM 做语义清洗。
- 不判断复杂科学推理是否合理。
- 不生成 reward。
- 不运行 SFT/RL 训练。
- 不修改 raw `complete_session.jsonl`。
- 不连接在线环境或 MolClaw 服务。
- 不给 KG/E2E 定义质量分数。
- 不保证导出的 RL prompt 已可直接运行在线 RL。

### 2.3 数据真源

**唯一 raw 会话真源是每个样本目录下的 `complete_session.jsonl`。**

其他文件承担辅助职责：

- `question.json`：问题、ground truth、KG provenance 等。
- `prompt.txt`：原始执行 prompt 的文本回退来源。
- `parsed_answer.json`：执行阶段提取出的答案和 parse 状态。
- `run_meta.json`：return code、timeout 等运行信息。
- `run_config.json`：run 级任务类型和配置。
- `bench_scores.json`：run 级评测结果，不是硬清洗主要输入。

历史 `trajectories/accepted.jsonl` 和 `rejected.jsonl` 不是最终真源。Stage 1 会从 raw 文件重建并覆盖它们。

## 3. 目录与脚本地图

```text
data-pipe/
  scripts/
    run_postprocess.sh                 # 硬清洗三阶段一键入口
    run_llm_clean.sh                   # 第二阶段 LLM 清洗入口，不属于本文主流程
    validate_llm_cleaned.py            # 第二阶段 LLM clean 的只读校验器

  pipeline/postprocess/
    trajectory_exporter.py             # Stage 1：重建轨迹和 accepted 判定
    scan_molclaw_usage.py              # Stage 2：汇总候选会话
    post_process_sft.py                 # Stage 3：ReAct SFT/RL 硬清洗和格式转换
    export_verl_training_bundle.py      # 可选：导出 portable verl bundle
    validate_verl_training_bundle.py    # 可选：独立验证 verl bundle
    react_sft_schema_v1.md              # ReAct SFT schema 文档
    README.md                           # 模块简要说明

  results/
    molbench_*_run_*/                   # raw runs 和 Stage 1 轨迹产物
    postprocess_candidates/             # Stage 2、Stage 3 默认输出根目录
```

## 4. 一键硬清洗入口

默认执行全部三阶段：

```bash
cd <data-pipe-root>

bash scripts/run_postprocess.sh \
  --results-root results \
  --output-root results/postprocess_candidates
```

可用参数：

```text
--results-root PATH
  原始 run 根目录。默认 <repo>/results。

--output-root PATH
  Stage 2 候选和 Stage 3 SFT/RL 输出根目录。
  默认 <results-root>/postprocess_candidates。

--answer-hit-only
  只在 Stage 3 转换时过滤 VS/AC/PF。
  KG/E2E 不受影响。

--split-multi-tool-calls
  Stage 3 将同一 assistant 事件内的多个 tool call 拆成多个轮次。
  默认保留原始多工具调用语义，不拆。

--skip-export
  跳过 Stage 1。

--skip-scan
  跳过 Stage 2。

--skip-sft
  跳过 Stage 3。
```

环境变量：

```text
PYTHON_BIN
  指定 Python 可执行文件，默认 python。
```

### 4.1 推荐的安全重跑方式

完整重跑时，建议使用新的或已清空的候选输出根目录：

```bash
bash scripts/run_postprocess.sh \
  --results-root results \
  --output-root results/postprocess_candidates_20260608
```

原因：Stage 2 当前不会清空旧候选，会通过 `__dup2`、`__dup3` 后缀保留重复文件。对同一个 `output-root` 重复跑 Stage 2 会造成候选和 SFT 样本重复累积。

如果只想基于现有候选重新生成 SFT/RL，应跳过前两阶段：

```bash
bash scripts/run_postprocess.sh \
  --results-root results \
  --output-root results/postprocess_candidates \
  --skip-export \
  --skip-scan
```

## 5. Stage 1：`trajectory_exporter.py`

### 5.1 职责

`trajectory_exporter.py` 对单个 run 执行以下工作：

1. 识别任务类型。
2. 发现 run 内的样本和 rollout。
3. 读取 raw 会话和辅助文件。
4. 将 raw session 解析为 step-level 轨迹。
5. 计算任务指标。
6. 根据硬规则判定 accepted/rejected。
7. 覆盖写入 `run_dir/trajectories/`。

单独运行：

```bash
python pipeline/postprocess/trajectory_exporter.py \
  results/molbench_ac_<provider>_run_<timestamp>
```

可显式覆盖任务：

```bash
python pipeline/postprocess/trajectory_exporter.py \
  --task ac \
  results/molbench_ac_<provider>_run_<timestamp>
```

支持任务：

```text
vs, ac, pf, e2e, kg
```

### 5.2 run 发现方式

一键脚本会在 `results-root` 下查找所有 `run_config.json`，将其父目录视为 run 目录。

因此：

- 有 `run_config.json` 的 run 会被 Stage 1 处理。
- 没有 `run_config.json` 的目录不会被 `scripts/run_postprocess.sh` 自动发现。

任务类型推断优先级：

1. CLI `--task`
2. `run_config.json` 中的 `task`
3. 根据 prediction 文件推断

### 5.3 样本与 rollout 发现

脚本寻找形如：

```text
row*_idx*
```

的目录，并支持两种结构：

```text
row*_idx*/parsed_answer.json
```

或：

```text
row*_idx*/rollout*/parsed_answer.json
```

**只有能找到 `parsed_answer.json` 的样本会被 Stage 1 发现。**

每个样本读取：

```text
question.json
parsed_answer.json
run_meta.json
complete_session.jsonl
```

### 5.4 raw session 解析

`complete_session.jsonl` 中：

- 非 JSON 行不会进入事件重建。
- assistant `tool_use` 转为 action step。
- assistant `text` / `thinking` 转为 assistant content。
- user `tool_result` 根据 `tool_use_id` 配对到对应 step。
- 最后一个 step 标记 `done=true`。

脚本还会统计：

- tool use/result 数量
- scientific MCP tool use/result 数量
- docking tool use/result 数量
- assistant 文本块数量
- affinity 相关文本出现次数
- `<answer>` 出现次数

这些是审计字段，不是 reward。

### 5.5 全任务共同 accepted 硬门

所有任务都必须满足：

```text
molclaw_usage_count > 0
```

这里 Stage 1 对 MolClaw 的定义较宽：工具名只要以 `mcp__molclaw` 开头就计数，包括：

```text
mcp__molclaw-scp__*
mcp__molclaw-vs__*
```

否则拒绝：

```text
missing_molclaw_usage
```

所有任务还会检查 `complete_session.jsonl` 最后一条非空行。如果它以：

```text
[runner-error]
```

开头，则拒绝：

```text
runner_error_last_line
```

这一检查独立于 session 前面是否存在合法 JSON 事件和 MolClaw 调用。

### 5.6 VS accepted 规则与指标

硬门：

- `parse_error` 必须为空。
- prediction 非空。
- prediction 数量必须与候选数量一致。
- prediction 不得重复。
- 每个 prediction 必须属于候选集合。
- GT、candidate、prediction SMILES 必须能规范化。
- 必须使用过 MolClaw。
- session 最后一行不能是 runner error。

指标：

```text
top3_hit_num
top10_hit_num
```

### 5.7 AC accepted 规则与指标

硬门：

- `parse_error` 必须为空。
- prediction 必须恰好为一个。
- GT 和 prediction SMILES 必须能规范化。
- 必须使用过 MolClaw。
- session 最后一行不能是 runner error。

指标：

```text
acc
is_correct
```

### 5.8 PF accepted 规则与指标

硬门：

- `parse_error` 必须为空。
- prediction 非空。
- GT 和 prediction SMILES 必须能规范化。
- 必须使用过 MolClaw。
- session 最后一行不能是 runner error。

指标：

```text
precision
recall
f1
acc
```

其中 `acc` 表示 exact set match。

### 5.9 E2E accepted 规则

E2E 不做 VS/AC/PF 质量门，但不是无条件 accepted。必须满足：

```text
return_code == 0
timed_out == false
complete_session.jsonl 存在
molclaw_usage_count > 0
最后一行不是 [runner-error]
```

### 5.10 KG accepted 规则

KG 与 E2E 一样按执行完成性判定：

```text
return_code == 0
timed_out == false
complete_session.jsonl 存在
molclaw_usage_count > 0
最后一行不是 [runner-error]
```

KG 轨迹额外包含：

```text
kg_run_id
kg_task_id
expected_toolchain
expected_trajectory_available
```

### 5.11 RDKit 行为

如果 RDKit 可用，VS/AC/PF 使用 RDKit canonicalization 做 SMILES 比较和合法性检查。

如果 RDKit 不可用，脚本退化为字符串清理和匹配。`dataset_summary.json` 会记录 RDKit 是否可用及错误信息。

### 5.12 Stage 1 输出

每个 run 覆盖生成：

```text
trajectories/
  trajectory_level.jsonl
  step_level.jsonl
  accepted.jsonl
  rejected.jsonl
  dataset_summary.json
```

`dataset_summary.json` 包含：

- 样本数、step 数
- accepted/rejected 数
- reject reason histogram
- task metric averages
- RDKit 状态
- 输出文件路径

### 5.13 Stage 1 的幂等性

Stage 1 每次都覆盖 run 的 `trajectories/*`，因此对同一 raw run 重跑通常是幂等的。

## 6. Stage 2：`scan_molclaw_usage.py`

### 6.1 职责

Stage 2 不再负责重新判断复杂 accept 规则，也不负责 ReAct 清洗。它的职责是：

1. 从 Stage 1 的 accepted 集合找到对应 raw `complete_session.jsonl`。
2. 再做一次 runner-error 防御检查。
3. 将候选会话复制到任务目录。
4. 从 trajectory 读取单样本指标。
5. 写入统一的 `molclaw_usage_summary.csv`。

单独运行：

```bash
python pipeline/postprocess/scan_molclaw_usage.py \
  --results-root results \
  --output-root results/postprocess_candidates \
  --use-accepted-only
```

### 6.2 默认候选来源

默认：

```text
--use-accepted-only
```

Stage 2 会读取所有：

```text
trajectories/accepted.jsonl
```

并解析其对应的 `complete_session.jsonl`。

可以用：

```text
--no-use-accepted-only
```

扫描全部 raw session，但这不是主线推荐方式。

### 6.3 任务识别

优先读取：

```text
run_config.json.task
```

也支持从 run 目录名推断：

```text
molbench_(vs|ac|pf|kg|e2e)_..._run_<timestamp>
```

### 6.4 指标来源

Stage 2 从每个 run 的：

```text
trajectories/trajectory_level.jsonl
```

读取 `task_metrics`。

当前实现不会在 Stage 2 内用 `parsed_answer.json + question.json` 回退重算指标。VS/AC/PF 缺少必需 `task_metrics` 时不会复制候选，并会记录到 `stage2_rejected_candidates.jsonl`。

写入 CSV 的典型字段：

- VS：`vs_top3_hit_num`、`vs_top10_hit_num`
- AC：`ac_is_correct`
- PF：`pf_precision`、`pf_recall`、`pf_f1`、`pf_is_correct`
- 全任务：路径、task、accepted、MolClaw 使用等

### 6.5 `answer_hit_pass`

Stage 2 只计算并记录，不在此处过滤：

```text
VS: top3_hit_num >= 1
AC: is_correct == true
PF: acc/is_correct == true
KG/E2E: null
```

真正的 `--answer-hit-only` 过滤发生在 Stage 3。

### 6.6 runner-error 防御

即使样本出现在历史 accepted 文件中，如果 raw session 最后一条非空行以 `[runner-error]` 开头，Stage 2 仍会跳过该样本。

### 6.7 Stage 2 输出

默认输出：

```text
results/postprocess_candidates/
  vs/*.jsonl
  ac/*.jsonl
  pf/*.jsonl
  kg/*.jsonl
  e2e/*.jsonl
  molclaw_usage_summary.csv
```

复制后的文件名包含来源 run、row 和 rollout 信息。

### 6.8 Stage 2 的非幂等风险

Stage 2 当前不会自动清空：

```text
<output-root>/vs
<output-root>/ac
<output-root>/pf
<output-root>/kg
<output-root>/e2e
```

遇到同名文件时会创建：

```text
__dup2
__dup3
...
```

因此重复运行 Stage 2 会扩大候选数量，并进一步让 Stage 3 生成重复训练样本。

这是当前最重要的操作风险之一。

## 7. Stage 3：`post_process_sft.py`

### 7.1 职责

Stage 3 将候选 raw session 转换为：

- 最小化 ReAct-style SFT 样本
- RL prompt 样本
- rejected 样本记录
- 每样本 cleaning report
- 全局 manifest 和 validation report

单独运行：

```bash
python pipeline/postprocess/post_process_sft.py \
  --input-root results/postprocess_candidates
```

更严格的 answer-hit 子集：

```bash
python pipeline/postprocess/post_process_sft.py \
  --input-root results/postprocess_candidates \
  --answer-hit-only
```

### 7.2 输入

Stage 3 扫描：

```text
<input-root>/{vs,ac,pf,kg,e2e}/*.jsonl
```

并读取：

```text
<input-root>/molclaw_usage_summary.csv
```

它还会根据 copied path 和来源路径尝试读取：

```text
question.json
prompt.txt
parsed_answer.json
```

代码中保留了历史路径回退，包括旧的 `vs_pipeline/results`。这些回退只用于寻找辅助文件，不代表旧目录仍是推荐入口。

### 7.3 当前 SFT schema

```text
drug_agent_sft_react_json_v1
```

每条训练样本正文只保留：

```json
{
  "schema_version": "drug_agent_sft_react_json_v1",
  "id": "mcp_sft_<task>_<hash>",
  "messages": []
}
```

详细来源、清洗动作和计数不放进训练正文，而放入 sidecar cleaning report。

### 7.4 当前目标工具过滤

Stage 3 当前保留：

```text
mcp__molclaw-scp__*
mcp__molclaw-vs__*
```

并将工具名规范化为去掉前缀后的短名。

以下调用会被删除：

- Bash
- Read
- Write
- Glob
- Grep
- 其他 Claude Code workspace 工具
- 其他非目标 MolClaw 工具

其对应 tool result 会因为无法与保留的 tool call 配对而作为 orphan result 删除。

training message 中只保留规范化短工具名；cleaning report 同时保留 raw tool name 和 `molclaw-scp` / `molclaw-vs` namespace。

### 7.5 ReAct 消息协议

输出消息角色：

```text
system
user
assistant
```

默认 observation 使用 `role=user`，提高 Qwen-style chat template 兼容性。

ReAct 标签：

```xml
<thought>科学推理和工具选择理由</thought>

<tool_call>{"tool_name":"...","arguments":{...}}</tool_call>

<observation tool_name="...">{"ok":true,...}</observation>

<final_answer>{"task_type":"...",...}</final_answer>
```

loss mask：

- system：`step_loss_mask=0`
- 初始 user question：`step_loss_mask=0`
- observation：`step_loss_mask=0`
- assistant thought/tool call/final answer：`step_loss_mask=1`

### 7.6 多工具调用

默认保留同一 assistant 事件中的多 tool call 结构，不强行拆分。

只有显式传入：

```text
--split-multi-tool-calls
```

才拆成多个 assistant/user 轮次。

### 7.7 tool call 与 tool result 配对

Stage 3 根据：

```text
tool_use.id
tool_result.tool_use_id
```

配对。

无法配对的 tool result 会：

- 不进入 SFT 正文
- 记录为 orphan
- 写入 cleaning report

### 7.8 文本硬清洗

Stage 3 当前执行：

- 对整段外围 triple-backtick wrapper 做剥离。
- 保留 fenced block 内部内容。
- 保留普通单 backtick。
- 只将 `/root`、`/home`、`/tmp`、`/mnt`、`/workspace` 下的本地绝对路径替换为纯文本 `<artifact:...>`。
- 对字符串、字典和列表递归替换路径。
- 将非 dict 的 tool arguments 归一为空 dict，并记录动作。

路径 placeholder 会按内容分类，例如：

```text
<artifact:fpocket/result>
<artifact:pdbfixer/filename>
<artifact:docking/filename>
<artifact:protein_structures/filename>
<artifact:local/filename>
```

### 7.9 工程 chatter 的真实状态

代码中存在 `_looks_engineering_chatter(...)` 和 `allow_engineering_drop` 参数，但当前 `_clean_text_piece(...)` 并没有根据它们删除工程 chatter。

因此当前硬清洗会清理路径等确定性噪声，但不会可靠删除：

- “接下来读取文件”
- “现在写 result.md”
- “检查工作目录”
- 其他自然语言工程操作描述

这类语义级清洗应由第二阶段 LLM clean 负责，或未来重新启用并严格测试确定性 chatter filter。

### 7.10 observation 标准化

普通 observation 被归一为：

```json
{
  "ok": true,
  "tool_name": "tool_suffix",
  "status": "success",
  "content": {},
  "metadata": {
    "tool_use_id": "...",
    "raw_tool_name": "...",
    "raw_status": "...",
    "raw_is_error": false,
    "raw_event_index": 0
  }
}
```

状态可能是：

```text
success
partial_success
error
timeout
```

脚本会：

- 从 raw error 标志和内容推断 status。
- 提取路径 pointer，并将其安全替换为 artifact 链接。
- 递归清理 observation 内的本地路径。
- 对过长内容进行截断。

默认最大序列化长度：

```text
6000 characters
```

可通过：

```text
--max-observation-chars N
```

修改。

超长 observation 会保存 preview，并在 cleaning report 中记录 `truncate_observation`。原始内容仍可从 raw session 回溯。

### 7.11 `fpocket_toolkit` 特殊压缩

`fpocket_toolkit` 的结果可能非常大，Stage 3 会专门压缩为训练更有用的摘要，例如：

- status
- msg
- pocket_count
- top pocket center
- top pocket size
- score
- chains
- artifact

失败结果也会压缩为短错误摘要。

### 7.12 thought 的处理

assistant `thinking` 和 `text` 会进入 `<thought>`。

当前确定性逻辑主要做：

- fence 清理
- 本地路径替换
- 空文本过滤

当前不会可靠判断哪段 thought 是科学推理、哪段是工程 chatter。该任务属于第二阶段 LLM clean 的重点。

### 7.13 final answer 来源与任务 schema

canonical answer 取值优先级：

1. 附近 `parsed_answer.json` 的结构化字段
2. `parsed_answer.answer_block`
3. 最终 assistant/result 文本

最终答案文本优先取最后一个保留工具调用之后的 assistant 内容；如果缺失，则回退 result event。

AC：

```json
{
  "task_type": "ac",
  "answer_smiles": "...",
  "short_reason": "...",
  "evidence": []
}
```

VS：

```json
{
  "task_type": "vs",
  "ranked_smiles": ["...", "..."],
  "selected_smiles": "...",
  "short_reason": "...",
  "evidence": []
}
```

PF：

```json
{
  "task_type": "pf",
  "selected_smiles": ["...", "..."],
  "short_reason": "...",
  "evidence": []
}
```

KG/E2E 使用轻量通用字段，例如 answer、steps summary 和 evidence。

### 7.14 Stage 3 硬拒绝条件

候选会被写入 `rejected_samples.jsonl` 的典型原因：

- 找不到 question
- 没有保留下来的 `mcp__molclaw-scp__*` 工具调用
- 缺少 final answer
- AC/VS/PF 无法提取任务特定答案
- 消息数量不足
- ReAct 格式验证失败
- `--answer-hit-only` 开启后不满足命中条件

### 7.15 `--answer-hit-only`

该参数只影响 Stage 3 候选过滤：

- VS：保留 `top3_hit_num >= 1`
- AC：保留正确样本
- PF：保留 exact set match 样本
- KG/E2E：不筛

它不改变 ReAct 清洗逻辑，也不改变 accepted 定义。

未开启时，not-hit 但已 accepted 且使用 MolClaw 的轨迹仍可进入训练数据，因为它们可能包含有价值的工具使用过程。

### 7.16 校验规则

Stage 3 对每条 SFT 样本做结构验证，包括：

- 第一条消息必须是 system。
- 第二条消息必须是 user question。
- 角色只能是 system/user/assistant。
- 初始 user 后的 user 消息必须是 observation。
- assistant 内容必须完全由 ReAct tags 包裹。
- tool call 内部必须是可解析 JSON object。
- tool name 必须是规范化短名。
- arguments 必须是 object。
- observation 内部必须是可解析 JSON object。
- final answer 必须符合 task-specific schema。
- 不得残留本地绝对路径。

### 7.17 `--tool-role-mode tool` 的当前限制

CLI 暴露：

```text
--tool-role-mode {user_observation,tool}
```

但当前 validator 只接受：

```text
system/user/assistant
```

因此 `tool` 模式当前与 validator 不兼容，主线应继续使用默认：

```text
user_observation
```

### 7.18 Stage 3 输出

默认：

```text
results/postprocess_candidates/sft_outputs/
  mcp_sft_all/
    000001__<sample_id>.json
    ...
  mcp_sft_all.jsonl
  mcp_rl_prompts_all.jsonl
  rejected_samples.jsonl
  dataset_manifest.json
  schema_validation_report.json
  schema_validation_report.md
  cleaning_reports/
    <sample_id>.json
    cleaning_report_index.jsonl
```

其中：

- `mcp_sft_all/`：格式化分行、逐样本可读 JSON。
- `mcp_sft_all.jsonl`：统一机器消费文件。
- `mcp_rl_prompts_all.jsonl`：RL prompt，不含 reward 字段。
- `rejected_samples.jsonl`：转换阶段拒绝样本和原因。
- `cleaning_reports/`：逐样本确定性清洗动作和审计统计。
- `dataset_manifest.json`：数据集总量和任务分布。
- `schema_validation_report.*`：机器/人工可读校验结果。

### 7.19 RL prompt 当前语义

RL prompt 只保留初始 system/user prompt，并写入：

- data source
- ability
- placeholder reward model 配置
- task payload
- allowed tools
- max steps
- metadata

它不写实际 `reward` 字段，也不定义真实 reward。

### 7.20 Stage 3 的幂等性

- `mcp_sft_all/` 会删除后重建。
- JSONL 和 manifest/report 会覆盖。
- `cleaning_reports/` 目录不会整体清空；不再出现在新 index 中的旧逐样本报告可能残留。

因此如果要求输出目录绝对纯净，建议使用新的 output dir 或先人工清理旧 `sft_outputs`。

## 8. 第二阶段 LLM clean 的接口边界

硬清洗阶段结束后，LLM clean 应消费已经结构化的 SFT 样本，而不是直接重新解释 raw session。

推荐输入：

```text
results/postprocess_candidates/sft_outputs/mcp_sft_all/
```

硬清洗已保证：

- 基本 ReAct 结构
- tool/result 配对
- 目标工具硬过滤
- 本地路径清理
- observation 结构化和截断
- task-specific final answer
- 基本 schema validation

LLM clean 更适合处理：

- thought 中的工程 chatter
- 重复、自我修正和无意义推理
- 科学推理压缩
- observation 摘要是否保留关键信息
- final answer 的语言质量
- 高层语义一致性

LLM clean 不应静默改变：

- 工具调用事实
- 工具参数
- observation 中关键科学数值
- task answer 的 canonical 值
- 来源映射

当前 LLM clean 后还有一个 final hard-clean gate：

```text
脚本预清洗
  -> LLM semantic repair
     - 修 VS ranking
     - 修 observation status 冲突
     - 修 thought/final/evidence
  -> final_hard_clean.py
     - 删除 observation metadata/raw_pointer
     - 再次清理相对路径
     - 只检测 VS ranking/status 冲突，不自动修
  -> validate_llm_cleaned.py
  -> 未通过样本 quarantine
```

`final_hard_clean.py` 不会自动重排 VS，也不会自动改 observation status。它输出 `needs_llm_semantic_repair` 和 `repair_reasons`，修不了的样本不会进入通过 gate 的目录。

第二阶段完成后，可用只读 validator 检查 cleaned 样本：

```bash
python scripts/validate_llm_cleaned.py \
  --input-dir <mcp_sft_all>/cleaned \
  --output-json <output>/llm_clean_validation_report.json \
  --output-md <output>/llm_clean_validation_report.md
```

该 validator 只报告，不移动、删除或修改 cleaned 文件；可检查工程 chatter、artifact/path 污染、exact molecular strings、fpocket 异常、空 evidence、unsupported metric interpretation，以及 VS ranking 与已观察 docking score 的一致性。

## 9. 可选下游：Verl Training Bundle

以下 Python 脚本位于 `pipeline/postprocess/`，但不是三阶段硬清洗主链路：

```text
export_verl_training_bundle.py
validate_verl_training_bundle.py
```

它们负责把 SFT/RL 输出打包为 portable bundle，并做：

- train/valid 稳定切分
- normalized JSON action 转换
- verl-ready RL prompt 转换
- raw reference 收集
- security scan
- manifest 和交付报告
- 可选 preview parquet

`scripts/build_verl_bundle.sh` 已移除。bundle 导出属于可选交付格式转换，不应混同于 raw session 硬清洗。

## 10. 当前已确认事实与已知风险

### 10.1 已确认事实

- 后处理主线已经与执行和评测解耦。
- raw `complete_session.jsonl` 是轨迹重建真源。
- 全任务 accepted 都要求使用 MolClaw。
- E2E/KG 不是无条件 accepted，必须真实完成。
- 最后一行 `[runner-error]` 会导致拒绝。
- 当前不写 reward。
- Stage 3 输出 all-in-one，不分 train/valid/test。
- `--answer-hit-only` 只影响 Stage 3 的 VS/AC/PF。
- SFT 正文和 cleaning audit 已分离。

### 10.2 本次代码核查发现并修正

本次核查修正了三个确定性判定问题：

1. KG 分支虽然记录了 `runner_error_last_line`，但此前没有在追加该原因后重新计算 accepted；现已修复。
2. PF final answer validator 分支缩进错误，导致 PF 任务特定校验没有正确执行；现已修复。
3. 本地绝对路径检测会把 `</tool_call>` 等 ReAct 结束标签误判为路径；现已排除标签场景。

针对性回归结果：

- Python `py_compile` 通过。
- 合法 PF ReAct 样本验证通过。
- 合成 KG session 最后一行包含 `[runner-error]` 时，结果为 rejected，原因正确。

### 10.3 高优先级风险

#### 风险 A：Stage 2 重跑累积重复候选

症状：

```text
__dup2
__dup3
```

影响：

- 候选数量膨胀
- SFT 重复
- 指标和 task distribution 失真

建议：

- 全量重跑使用新 output root。
- 或在重跑 Stage 2 前明确清空候选目录。

#### 风险 B：同 suffix 工具的 namespace 在训练正文中被折叠

Stage 3 同时保留 SCP 和 VS namespace，但训练正文只使用短 tool name。cleaning report 保留 namespace；下游若需要区分同 suffix 工具，必须读取审计 sidecar 或扩展训练 schema。

#### 风险 C：`tool` observation mode 与 validator 不兼容

当前应坚持默认 `user_observation`。

#### 风险 D：工程 chatter 没有在硬清洗阶段真正删除

当前由后续 LLM clean 承担更合理，但文档和使用者不能误以为硬清洗已经完成语义去噪。

#### 风险 E：Stage 2 指标无回退重算

如果 VS/AC/PF trajectory metrics 缺失，候选会被拒绝并写入 `stage2_rejected_candidates.jsonl`。

#### 风险 F：旧 cleaning report 可能残留

Stage 3 会覆盖 index，但不会清空全部逐样本报告。

## 11. 当前已有输出快照

2026-06-08 在全新 output root 上完成了一次完整重跑：

```text
results/postprocess_candidates_20260608_105422/sft_outputs/
```

最终 `schema_validation_report.json` 记录：

```text
ok: true
total_sessions: 381
total_sft_samples: 381
rejected_samples: 0
retained_mcp_tool_calls: 3955
dropped_non_mcp_tool_calls: 5414
orphan_tool_results: 5414
truncated_observations: 28
chat_template_failed: 0

task_counts:
  vs: 26
  ac: 111
  pf: 226
  e2e: 18
  kg: 0
```

该 output root 不包含 `__dup*` 候选；Stage 2 复制 381 条，缺指标拒绝为 0。审计确认 SFT 中 markdown artifact、malformed artifact、严格本地路径残留均为 0；83 个 fpocket observation 中未发现 `size == center`、负 size 或 markdown artifact。

## 12. 检查与排错手册

### 12.1 查看每个 run 的 accepted/rejected

```bash
cat results/<run>/trajectories/dataset_summary.json
```

重点字段：

```text
n_samples
n_accepted
n_rejected
reject_reason_hist
task_metric_averages
```

### 12.2 检查 runner-error 是否被接受

```bash
rg -n 'runner_error_last_line' results/*/trajectories/rejected.jsonl
```

### 12.3 检查候选重复

```bash
find results/postprocess_candidates -type f -name '*__dup*.jsonl' | wc -l
```

### 12.4 查看 Stage 3 总体结果

```bash
cat results/postprocess_candidates/sft_outputs/dataset_manifest.json
cat results/postprocess_candidates/sft_outputs/schema_validation_report.json
```

### 12.5 查看某条样本为何被清洗

```bash
cat results/postprocess_candidates/sft_outputs/cleaning_reports/<sample_id>.json
```

### 12.6 查看某条样本为何被拒绝

```bash
rg -n '<sample_id>|<source filename>' \
  results/postprocess_candidates/sft_outputs/rejected_samples.jsonl
```

### 12.7 检查是否残留本地路径

```bash
rg -n '/home/<user>|/tmp/' \
  results/postprocess_candidates/sft_outputs/mcp_sft_all \
  results/postprocess_candidates/sft_outputs/mcp_sft_all.jsonl
```

### 12.8 检查 Stage 3 是否保留了错误工具前缀

```bash
rg -n 'mcp__molclaw-vs__|mcp__molclaw-scp__' \
  results/postprocess_candidates/sft_outputs/mcp_sft_all
```

规范化后的训练正文中通常不应保留完整 MCP 前缀。

## 13. 接手者下一步执行清单

### P0：消除 Stage 2 重复累积风险

- 推荐给 scanner 增加显式 `--clean-output-task-dirs` 或 `--overwrite`。
- 默认行为应谨慎，避免误删用户数据。
- 在实现前，操作上统一使用新的 timestamped output root。

### P1：让 Stage 3 输出目录完全幂等

- 在安全边界明确后，考虑清理旧 `cleaning_reports/*.json`。
- 保留 index 和实际逐样本报告一一对应。

### P1：补充自动测试

至少覆盖：

- VS/AC/PF/E2E/KG accepted/rejected 规则。
- `[runner-error]` 最后一行。
- `mcp__molclaw-scp__*` 和 `mcp__molclaw-vs__*` 工具过滤。
- tool_use/tool_result 配对和 orphan。
- 多 tool call 默认保留与显式拆分。
- fence wrapper 清理。
- 绝对路径替换但不误判 ReAct 标签。
- observation 截断。
- task-specific final answer。
- Stage 2 重跑重复行为。

### P1：明确硬清洗与 LLM clean 的契约

- 固定 LLM clean 输入 schema。
- 要求 LLM clean 保留 canonical answer 和工具事实。
- 为 LLM clean 输出增加独立 validator。
- 不让 LLM clean 修改 raw session 或硬清洗审计 sidecar。

### P2：清理历史路径回退

- 清理历史路径回退时，先确认旧结果是否仍需支持。

## 14. 最终操作建议

对于一次新的、可审计的全量硬清洗，推荐：

```bash
cd <data-pipe-root>

OUT="results/postprocess_candidates_$(date +%Y%m%d_%H%M%S)"

bash scripts/run_postprocess.sh \
  --results-root results \
  --output-root "$OUT"

cat "$OUT/sft_outputs/dataset_manifest.json"
cat "$OUT/sft_outputs/schema_validation_report.json"
```

只有当 manifest、validation report、任务分布和抽样 SFT 均符合预期后，才将该目录交给第二阶段 LLM clean。
