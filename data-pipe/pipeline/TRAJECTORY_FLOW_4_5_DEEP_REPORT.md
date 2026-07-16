# 4.5 轨迹导出链（Trajectory Flow）深度调查报告

- 项目根目录：`<legacy-vs-pipeline-root>`
- 调查时间：`2026-05-13`
- 调查对象：`claude_agent/trajectory_exporter.py` + 真实 `results/*/trajectories/*` 产物
- 调查重点：
  - Q1. 不同任务（VS/AC/PF）的 `accepted/rejected` 质量门如何定义，何时会被判 `rejected`
  - Q2. 可用于训练的轨迹数据存储在哪里，如何定位与读取

---

## 1. 调查方法与证据来源

1. 静态代码审计（主证据）
- `claude_agent/trajectory_exporter.py:415-655`（导出主流程与质量门）
- `claude_agent/trajectory_exporter.py:254-353`（step-level 记录与 reward 注入）
- `claude_agent/trajectory_exporter.py:51-97`（row/rollout 样本发现逻辑）
- `claude_agent/run_claude.py:722-737,1117-1123`（调用导出器）

2. 文档声明核对（辅助证据）
- `readme.md:135-155`
- `claude_agent/README.md:70-93,169-175`

3. 真实产物抽样（落地证据）
- `results/molbench_vs_qwen-397b_run_20260512_142732/trajectories/*`
- `results/molbench_ac_qwen-397b_run_20260512_142732/trajectories/*`
- `results/molbench_pf_qwen-397b_run_20260512_142732/trajectories/*`
- `results/molbench_pf_qwen-397b_run_20260511_153831/trajectories/*`（PF 全拒批次）

---

## 2. 4.5 轨迹导出链路复原（从入口到落盘）

### 2.1 入口与触发

- `run_claude.py` 在推理完成后默认执行轨迹导出：
  - 参数开关：`--export-trajectories` / `--no-export-trajectories`（`run_claude.py:865-867`）
  - 触发调用：`_run_trajectory_exporter(...)`（`run_claude.py:1117-1123`）
  - 子进程命令：`python claude_agent/trajectory_exporter.py <run_dir> --task <task>`（`run_claude.py:722-737`）

### 2.2 样本发现与输入拼装

- 导出器扫描 `results_dir` 下 `row*_idx*` 目录，识别两种结构：
  - `row_dir/parsed_answer.json`（单 rollout 兼容结构）
  - `row_dir/rolloutXXXX/parsed_answer.json`（多 rollout 结构）
- 证据：`trajectory_exporter.py:51-97`

- 每个样本读取：
  - `question.json`（优先 sample 目录，其次 row 根）
  - `parsed_answer.json`
  - `run_meta.json`
- 证据：`trajectory_exporter.py:440-447`

### 2.3 导出输出

固定写入 `results/<run>/trajectories/`：
- `trajectory_level.jsonl`（轨迹级）
- `step_level.jsonl`（步骤级）
- `accepted.jsonl`（通过质量门）
- `rejected.jsonl`（未通过质量门）
- `dataset_summary.json`（统计+文件索引）
- 证据：`trajectory_exporter.py:599-639`

---

## 3. Q1：accepted/rejected 质量门规则（任务分解）

> 统一判定准则：`accepted = len(reject_reasons) == 0`，否则 `rejected`。
> 证据：`trajectory_exporter.py:557`

### 3.1 通用规则（所有任务）

1. `parse_error`
- 触发条件：`parsed_answer.json` 内 `parse_error` 非空
- 代码：`trajectory_exporter.py:456-466`
- 结果：加入 `reject_reasons=["parse_error", ...]`

2. `invalid_ground_truth_smiles:N`
- 触发条件：ground truth canonicalize 失败数量 `N>0`
- 代码：`trajectory_exporter.py:479-480`

3. `invalid_prediction_smiles:N`
- 触发条件：prediction canonicalize 失败数量 `N>0`
- 代码：`trajectory_exporter.py:481-482`

4. `invalid_candidate_smiles:N`（仅 VS）
- 触发条件：候选 canonicalize 失败数量 `N>0`
- 代码：`trajectory_exporter.py:483-484`

### 3.2 VS（Virtual Screening）专用质量门

1. `empty_candidate_set`
- 条件：候选长度 `expected_n == 0`
- 代码：`trajectory_exporter.py:490-493`

2. `length_mismatch:<pred>!=<cand>`
- 条件：预测长度不等于候选长度
- 代码：`trajectory_exporter.py:493-495`

3. `duplicate_predictions`
- 条件：预测集合去重后长度小于原长度
- 代码：`trajectory_exporter.py:496-499`

4. `outside_candidate_set:<n>`
- 条件：预测中有 `n` 个元素不在候选集
- 代码：`trajectory_exporter.py:500-503`

### 3.3 AC（Activity Cliff）专用质量门

1. `empty_prediction`
- 条件：`len(pred_answers)==0`
- 代码：`trajectory_exporter.py:523-524`

2. `invalid_prediction_count:<n>`
- 条件：预测个数不等于 1
- 代码：`trajectory_exporter.py:525-526`

### 3.4 PF（Property Forecasting）专用质量门

1. `empty_prediction`
- 条件：`len(pred_answers)==0`
- 代码：`trajectory_exporter.py:539-540`

说明：PF 不要求“预测长度=候选长度”，因为 PF 用集合指标（precision/recall/f1/acc）评估。
证据：`trajectory_exporter.py:542-549`

---

## 4. Q1 补充：真实 rejected 触发样例（按任务）

### 4.1 VS 样例

1. 超时导致解析失败 + 长度不匹配
- 文件：`results/molbench_vs_qwen-397b_run_20260512_142732/trajectories/rejected.jsonl`
- 样例：`task_id=vs_row0001_idx1_r0001`
- 字段：
  - `reject_reasons=["parse_error","length_mismatch:0!=60"]`
  - `parse_error="timeout after 1200 seconds"`
  - `final_answer_len=0`, `candidate_len=60`

2. 全部预测不在候选集中
- 样例：`task_id=vs_row0002_idx2_r0001`
- 字段：
  - `reject_reasons=["outside_candidate_set:60"]`
  - `final_answer_len=60`, `candidate_len=60`
  - `task_reject_checks.outside_candidate_count=60`

3. 重复预测
- 样例：`task_id=vs_row0018_idx18_r0001`
- 字段：
  - `reject_reasons=["duplicate_predictions"]`
  - `final_answer_len=60`, `unique_len=56`

4. 预测中含非法 SMILES
- 样例：`task_id=vs_row0026_idx26_r0001`
- 字段：
  - `reject_reasons=["invalid_prediction_smiles:1","duplicate_predictions"]`

### 4.2 AC 样例

- 文件：`results/molbench_ac_qwen-397b_run_20260512_142732/trajectories/rejected.jsonl`
- 样例：`task_id=ac_row0001_idx1_r0001`
- 字段：
  - `reject_reasons=["parse_error","empty_prediction","invalid_prediction_count:0"]`
  - `parse_error="empty answer for AC"`
  - `final_answer=[]`

### 4.3 PF 样例

1. 正常批次（全部通过）
- `results/molbench_pf_qwen-397b_run_20260512_142732/trajectories/dataset_summary.json`
- `n_samples=30, n_accepted=30, n_rejected=0`

2. 失败批次（全部拒绝）
- `results/molbench_pf_qwen-397b_run_20260511_153831/trajectories/dataset_summary.json`
- `n_samples=900, n_accepted=0, n_rejected=900`
- `reject_reason_hist={"parse_error":900,"empty_prediction":900}`

---

## 5. Q2：可用于训练的轨迹存储位置

## 5.1 物理路径规则

每个 run 独立落在：
- `results/molbench_<task>_<provider>_run_<timestamp>/trajectories/`

该目录下核心文件：
- `accepted.jsonl`：通过质量门的轨迹（首选训练源）
- `rejected.jsonl`：未通过质量门的轨迹（错误分析/偏好对/对比学习）
- `trajectory_level.jsonl`：全量轨迹（accepted + rejected）
- `step_level.jsonl`：全量步骤记录（含 `accepted` 标志）
- `dataset_summary.json`：统计 + 绝对路径索引

代码证据：`trajectory_exporter.py:599-639`

## 5.2 真实路径示例

- VS：
  - `results/molbench_vs_qwen-397b_run_20260512_142732/trajectories/accepted.jsonl`
  - `results/molbench_vs_qwen-397b_run_20260512_142732/trajectories/step_level.jsonl`

- AC：
  - `results/molbench_ac_qwen-397b_run_20260512_142732/trajectories/accepted.jsonl`

- PF：
  - `results/molbench_pf_qwen-397b_run_20260512_142732/trajectories/accepted.jsonl`

- 可由 `dataset_summary.json.files` 程序化发现绝对路径：
  - 证据文件：
    - `results/molbench_vs_qwen-397b_run_20260512_142732/trajectories/dataset_summary.json`
    - `results/molbench_ac_qwen-397b_run_20260512_142732/trajectories/dataset_summary.json`
    - `results/molbench_pf_qwen-397b_run_20260512_142732/trajectories/dataset_summary.json`

## 5.3 训练用途建议口径（基于当前实现）

1. SFT（轨迹级）
- 推荐输入：`accepted.jsonl`
- 使用字段：`task`, `task_id`, `candidates`, `ground_truth`, `final_answer`, `artifact_audit`, `task_metrics`

2. RL / Process 数据（步骤级）
- 推荐输入：`step_level.jsonl`
- 关键字段：`task_id`, `step_id`, `action_type`, `tool_name`, `observation`, `done`, `reward`, `accepted`
- 注意：当前 reward 只在最后一步注入，前面步骤 `reward=0.0`。
  - 证据：`trajectory_exporter.py:349-352`

3. DPO / 偏好构造
- 可用 `accepted.jsonl` 与 `rejected.jsonl` 构造正负样本。
- 若希望“同题 pair”，需要多 rollout 或额外按 `dataset_index/row_number` 对齐策略。

---

## 6. 任务级统计快照（真实 run）

数据源均为 `dataset_summary.json`：

1. VS（`molbench_vs_qwen-397b_run_20260512_142732`）
- `n_samples=30`
- `n_accepted=8`
- `n_rejected=22`
- 高频拒因：`parse_error`、`length_mismatch:0!=60`

2. AC（`molbench_ac_qwen-397b_run_20260512_142732`）
- `n_samples=30`
- `n_accepted=29`
- `n_rejected=1`
- 拒因集中在空输出与个数不为 1

3. PF（`molbench_pf_qwen-397b_run_20260512_142732`）
- `n_samples=30`
- `n_accepted=30`
- `n_rejected=0`

4. PF 异常批次（`molbench_pf_qwen-397b_run_20260511_153831`）
- `n_samples=900`
- `n_accepted=0`
- `n_rejected=900`
- 全部由 `parse_error + empty_prediction` 触发

---

## 7. 一致性核验与不一致项

## 7.1 已实现且可证实（A）

1. 质量门切分 `accepted/rejected`
- 证据：`trajectory_exporter.py:557,609-617`

2. 轨迹五件套落盘
- 证据：`trajectory_exporter.py:599-639`

3. 默认主流程会触发导出
- 证据：`run_claude.py:1117-1123`

## 7.2 文档声称但代码未证实（B）

1. “RDKit 不可用会拒绝样本（rdkit_unavailable）”
- 文档声明：`readme.md:150`、`claude_agent/README.md:88,171`
- 代码实际：RDKit 不可用时仅退化为字符串 canonical（不自动拒绝）
  - 证据：`trajectory_exporter.py:467-474`
- 结论：文档与实现存在偏差。

## 7.3 仅规划/未落地（C）

1. 过程稠密奖励（非终点 reward）
- 代码仅在最后一步设置 reward/done：`trajectory_exporter.py:349-352`
- 文档也承认“过程奖励未做”：`readme.md:59,270`

---

## 8. 复现与排障指引（聚焦 4.5）

## 8.1 仅重导出轨迹

```bash
cd <legacy-vs-pipeline-root>
python claude_agent/trajectory_exporter.py /absolute/path/to/results/molbench_vs_xxx_run_xxx --task vs
```

证据：`claude_agent/README.md:132-136`

## 8.2 快速检查质量门结果

```bash
jq '{task,n_samples,n_accepted,n_rejected,reject_reason_hist}' \
  /absolute/path/to/results/.../trajectories/dataset_summary.json
```

## 8.3 高概率失败点（4.5 相关）

1. 模型输出未满足 answer 解析格式，导致 `parse_error` 激增。
- 表现：`reject_reason_hist` 中 `parse_error` 高。

2. VS 输出数量不是候选长度。
- 表现：`length_mismatch:<pred>!=<cand>`。

3. VS 预测重复或越界。
- 表现：`duplicate_predictions` / `outside_candidate_set:<n>`。

4. AC 输出条数不是 1。
- 表现：`invalid_prediction_count:<n>`。

5. 文档误导 RDKit 行为，导致错误预期。
- 现状：代码不是“RDKit 不可用就全拒”。

---

## 9. 对两个问题的直接结论

1. 不同任务的 `accepted/rejected` 质量门如何规定，何时 `rejected`？
- 规则由 `trajectory_exporter.py` 的 `reject_reasons` 生成逻辑定义。
- 只要 `reject_reasons` 非空，就判 `rejected`。
- VS 关注“候选一致性与集合约束”；AC 关注“必须恰好 1 个预测”；PF 关注“不能空预测”。
- 具体触发案例已在第 4 节逐条给出并附真实 `task_id`。

2. 可用于训练的轨迹存储在哪里？
- 统一在每个 run 的 `trajectories/` 子目录。
- `accepted.jsonl` 是默认训练主源；`step_level.jsonl` 供过程级训练；`dataset_summary.json` 提供可编程定位。
- 真实路径示例见第 5.2 节。

