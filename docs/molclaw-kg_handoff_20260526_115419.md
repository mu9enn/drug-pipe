# MolClaw-KG Handoff (2026-05-26 11:54:19 Asia/Shanghai)

## 0. 文档用途
本文件用于把 `molclaw-kg` 项目完整交接给新的高能力协作者/AI agent。读者无需访问历史聊天记录，即可继续推进。

---

## 1. Executive Summary

### 1.1 核心目标（已确认事实）
- 构建 MolClaw MCP 81 工具的 **tool-only directed typed graph**。
- 图节点只保留工具；边表达工具间真实任务流可邻接/协作关系。
- 下游用途有两个：
  1. 由图抽取 tool-chains，生成 QA 对。
  2. 作为替代 L1/L2/L3 skills 的知识底座，指导 agent 正确用工具。

### 1.2 当前最终方向（当前采用决策）
- 采用 **固定 taxonomy + agent 裁决 + validator 守门 + provenance** 的工程路线。
- 主流程分三段：
  - Stage1：`snapshot -> doc-chunks -> tool-cards`
  - Stage2：`candidates -> adjudicate -> validate -> score -> views -> export ...`
  - Stage3（新增）：`基于 graph 随机 walk 采样 toolchain -> agent 生成问题+期望轨迹`
- 判定语义收敛为：
  - `relation_status = valid | negative | uncertain | alternative`
  - `edge_types[*].support_scope = full | partial | n/a`
  - 置信度主分：`confidence_raw = agent_conf`

### 1.3 当前状态一句话
- Stage1/2 主线可跑通；Stage3 已实现并能落盘，但目前样本成功率受上游 502 与 agent JSON 解析失败影响，需继续稳态化。

---

## 2. 项目背景与关键术语

## 2.1 数据与证据源（已确认事实）
- MCP 工具快照：`runs/<run_id>/tool_snapshot.jsonl`（81 tools）。
- skills 文档：项目内置 `skills_full/.claude/skills`（已复制进仓库，不依赖外部路径）。
- taxonomy 真源：`configs/stage_taxonomy.json`（v3，81 工具全映射）。

## 2.2 关键术语
- **pruning taxonomy**：用于边候选剪枝，不是通用学科分类。
- **relation_aware pruning**：根据 `allowed_stage_transitions` 与 same-stage policy 过滤 pair。
- **bidirectional adjudication**：单次 agent 调用同时判断 `A->B` 与 `B->A`，输出再拆分为两条有向记录。
- **canonical-evidence-first**：优先读取 `.claude/skills` 原始证据，派生文件仅辅助。
- **support_scope**：`full/partial` 只承载“输入覆盖程度”，不再用旧 `coverage_level`。

---

## 3. 决策演化与关键 Pivot（重要）

## 3.1 从“文档/规则混合猜边”转为“固定约束+局部裁决”
- 早期问题：边构建过度依赖启发式和宽松文本共现，质量差。
- 最终采用：`schema candidate -> pairwise agent adjudication -> validators -> score/view`。
- 原因：减少幻觉、可追溯、可审计。

## 3.2 Stage 从打分项改为剪枝项
- 早期尝试里 stage 参与评分，逻辑混杂。
- 最终采用：stage **只用于剪枝**；评分仅保留 agent 主导（简化审美 + 减少硬匹配误判）。

## 3.3 决策协议升级
- 旧：`valid_full/valid_partial/invalid/...` 与 coverage 交织。
- 新：`relation_status + support_scope + confidence`。
- 影响：下游视图、导出、审计均重写；`rejected` 被并入 `uncertain` 逻辑。

## 3.4 Pairwise 输入文件重构
- 被放弃路线：`doc_context.jsonl + heavy pair_payload.json` 作为主要证据。
- 放弃原因：agent倾向读派生摘要，弱化对原始 skills 的深入读取。
- 最终采用：`task_context.json + pair_spec.json + source_manifest.json` 轻量导航，证据回读 canonical skills。

## 3.5 Stage3 新增（本轮）
- 新目标：KG 后处理自动采样工具链并生成问题/期望轨迹数据。
- 已实现：`sample-questions` CLI + 一键脚本 + 独立 `sample_workdir/sample_results` 落盘规范。

---

## 4. 当前架构与实现边界

## 4.1 目录与入口
- 代码根：`<molclaw-kg-root>`
- 主包：`src/molclaw_kg`
- 脚本：
  - `scripts/run_pipeline_stage1_toolcards.sh`
  - `scripts/run_pipeline_stage2_graph.sh`
  - `scripts/run_full_pipeline.sh`（串行 stage1+2，含 alert-rerun）
  - `scripts/run_sample_questions.sh`（Stage3）

## 4.2 Stage1（Tool-card）
- 固定 stage 映射来自 `configs/stage_taxonomy.json`，默认 fail-fast 覆盖校验。
- tool-card 由 deterministic MCP schema + agent enrich 构建。
- 异常策略：不阻断整体，fallback 到 deterministic base card 并打 `needs_review`，写 alerts 与 rerun targets。

## 4.3 Stage2（Graph）
- 候选边生成 -> 无序对聚合 -> 单次双向判定 -> 拆分为有向记录。
- relation-aware 剪枝 + deterministic alternative cluster 生成。
- validate 后打分：`raw=agent_conf`；硬失败降 `uncertain`。
- 视图输出：`core / expanded / uncertain / negative`。

## 4.4 Stage3（Question Sampling）
- 模块：`src/molclaw_kg/question_sampling/`
- 入口命令：`sample-questions`
- 逻辑：
  - 读取 `graph_all.jsonl`，过滤边池：
    - `view in {core,expanded,uncertain}`
    - `relation_status=valid`
    - `edge_type in TRANSITION_EDGE_TYPES`
    - `direct_transition=true`
  - 随机 walk（2~4 hops，允许重复节点）
  - 独立 workdir 调 Claude 生成问题+expected trajectory
  - 强校验 JSON + 强防泄露（工具名与顺序提示）
  - 输出 `sample_results/questions.csv` 等

---

## 5. 对外契约与数据格式（当前采用）

## 5.1 Adjudication 核心字段
- `relation_status`: `valid|negative|uncertain|alternative`
- `edge_types[*].support_scope`: `full|partial|n/a`
- `context`: 字符串（替代旧 coverage/additional 等多字段）
- 保留兼容读取旧 `decision`（写新读旧）

## 5.2 导出
- edge-level：`graph_all.csv`（调试向）
- pair-level官方汇报：`runs/<run_id>/<run_id>.csv`
  - `edge_types` / `edge_support_scopes` / `edge_confidences` 为 JSON 列表
  - 已去除 `Weight=1/2` 旧表达

## 5.3 Stage3 输出
- `runs/<run_id>/sample_results/`:
  - `sample_attempts.jsonl`（全部尝试）
  - `sample_success.jsonl`（成功子集）
  - `questions.csv`（汇总主文件）
  - `sampling_meta.json`

---

## 6. 当前已验证结果（基于 run_20260519_230652）

## 6.1 主流程状态（已确认事实）
- 快照工具数：81
- doc chunks：920（doc_count=76）
- graph views 统计：
  - core: 39
  - expanded: 17
  - uncertain: 626
  - negative: 220
- 官方 pair CSV 已生成：`runs/run_20260519_230652/run_20260519_230652.csv`

## 6.2 Stage3 状态（已确认事实）
- 已能完整执行与落盘。
- 当前测试样本出现失败：
  - `dead_end_before_target_hops`（walk 死路）
  - `agent_output_parse_failed`（本次实际由上游 502 导致，见 `complete_session.jsonl`）
- 结论：流程可运行，但成功率仍需工程稳态优化。

---

## 7. 已放弃/弱化路线（务必不要回滚）

1. **把 stage 当置信评分项**：已放弃。stage 仅用于 pruning。
2. **每次只判 A->B**：已放弃。改为单次双向判定。
3. **`doc_context.jsonl`/heavy `pair_payload.json` 驱动判边**：已放弃主导地位。
4. **`rejected` 作为独立视图**：已并入 `uncertain` 的硬失败降级逻辑。
5. **依赖外部 skills 路径**：已改为项目内置 `skills_full`。

---

## 8. 风险、待验证与已知问题

## 8.1 已知问题（高优先级）
1. **Stage3 真实可用性不足**：在小样本中出现 502/API error 导致 parse fail。
2. **`question_sampling/sampler.py` 存在实现瑕疵**：
   - `hops = rng.randint(...)` 重复赋值两次（行为无致命影响但属代码噪声）。
   - `card_by_tool` 当前未使用。
3. **配置与实现有“语义漂移”**：`configs/rules_v1.yaml` 仍保留旧 `weights` 字段，但 `confidence.py` 实际已改为 `raw=agent_conf`。
4. **provider switch 机制不统一**：
   - `ClaudeCodeRuntime.switch_provider()` 在 runtime 内被注释禁用；
   - Stage1/2 主要依赖外部环境；
   - Stage3 脚本显式执行了 `cc-switch` 一次。
   建议统一策略并写入文档。

## 8.2 待验证假设
- “让 agent 直接回读 `.claude/skills` 比读派生摘要质量更高”是当前设计假设，需在多 run 对比中量化验证。
- `relation_status + support_scope + agent_conf` 是否足以支撑下游 QA 质量，还需系统评估。

## 8.3 数据与安全注意事项
- `.env` 当前含明文 API Key（历史上按需求写入）。
- 对外共享代码或报告前应脱敏；内部运行可按当前配置继续。

---

## 9. 关键文件索引（接手必读）

## 9.1 配置
- `configs/stage_taxonomy.json`（v3 真源）
- `configs/rules_v1.yaml`
- `configs/edge_ontology_v1.yaml`
- `configs/prompts/pairwise_adjudication_v1.md`
- `configs/prompts/tool_card_agent_v1.md`
- `configs/prompts/toolchain_question_sampler_v1.md`

## 9.2 核心实现
- `src/molclaw_kg/tool_card_builder.py`
- `src/molclaw_kg/pairwise_runner.py`
- `src/molclaw_kg/validators.py`
- `src/molclaw_kg/confidence.py`
- `src/molclaw_kg/graph_views.py`
- `src/molclaw_kg/exporters.py`
- `src/molclaw_kg/adjudicators/claude_code_runtime.py`
- `src/molclaw_kg/question_sampling/sampler.py`
- `src/molclaw_kg/cli.py`

## 9.3 参考运行样本
- 推荐先看：`runs/run_20260519_230652/`
  - 含 stage1/2 主要产物与 stage3 sample 产物。

---

## 10. 接手者下一步执行清单（按优先级）

## P0（先做）
1. **修复 Stage3 稳态**
   - 在 `sample-questions` 增加失败重试策略（至少针对 `502` / `agent_output_parse_failed`）。
   - 增加 `--max-retries-per-sample` 与 `--api-error-rerun`。
2. **统一 provider switch 策略**
   - 决定“脚本层统一切换”或“runtime 统一切换”，避免 stage 间行为不一致。
3. **消除已知代码瑕疵**
   - 去掉 `sampler.py` 重复 `hops` 赋值。
   - 清理未使用变量。

## P1（紧接着）
1. **做一次可复现的端到端回归 run**
   - 运行 `run_full_pipeline.sh`（stage1+2）+ `run_sample_questions.sh`（stage3，建议 `sample-size=10` 起）。
   - 输出成功率、失败分布、平均调用时长、API 错误率。
2. **质量对比实验**
   - 对比“是否使用派生摘要上下文”的效果（命中率、uncertain比例、人工审计一致性）。

## P2（中期）
1. **Stage3 数据产品化**
   - 增加样本分层（按 stage 覆盖、edge_type 覆盖、hops 分布）采样。
   - 增加 QA 可执行性校验器（检查 expected_trajectory 与 toolchain 一致性更严格）。
2. **配置清理**
   - 更新 `rules_v1.yaml` 与 README，删除已失效权重语义，避免误导。

---

## 11. 面向不同 Agent 的分工建议

- **Coding Agent**
  - 负责 P0/P1 的代码改造与回归脚本化；优先 Stage3 稳态、重试与日志结构化。
- **Research Agent**
  - 负责质量评估设计：采样策略、审计集、指标定义（edge validity、chain validity、question leak rate、trajectory executability）。
- **Writing Agent**
  - 负责论文/汇报材料对齐：把当前“固定 taxonomy + 双向裁决 + relation_status 语义瘦身 + stage3 数据生成”写成方法章节与实验章节。

---

## 12. 快速启动命令

```bash
cd <molclaw-kg-root>

# Stage1 + Stage2（含 alert-rerun）
bash scripts/run_full_pipeline.sh run_$(date +%Y%m%d_%H%M%S)

# Stage3（采样）
bash scripts/run_sample_questions.sh <run_id> --sample-size 10 --min-hops 2 --max-hops 4 --seed 42
```

