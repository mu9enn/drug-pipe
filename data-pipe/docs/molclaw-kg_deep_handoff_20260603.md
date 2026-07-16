# MolClaw-KG 深度交接文档（截至 2026-06-03）

这是一份面向**完全没有读过历史聊天记录**的接手文档。目标不是概述，而是让新的高能力协作者、AI agent 或 coding agent 在**只看这一份文档**的情况下，也能继续理解、修改、验证并扩展 `tool-kg` 项目。

本文件只描述 `tool-kg`，不描述 `data-pipe` 的其他部分；但会在最后说明两者的接口关系，因为 `tool-kg` 的 Stage3 产物已经成为 `data-pipe/pipeline/kg` 的上游输入。

---

## 0. 一句话定位

`tool-kg` 的任务，是把 MolClaw MCP server 暴露的 81 个工具，构建成一张**tool-only directed typed graph**，再基于这张图自动抽取可执行 toolchain / workflow DAG，最终生成可用于问答、评测和下游 `data-pipe` KG-sampled task 构建的数据资产。

核心目标可以概括为三层：

1. 构建高质量 Tool Knowledge Graph。
2. 从图中采样出依赖闭合的工具链 / workflow。
3. 让这些 workflow 变成可供下游训练、评测和论文分析的数据产品。

---

## 1. 项目边界与当前阶段

### 1.1 项目现在做什么

`tool-kg` 不是一个简单的图抽取器，而是一个三阶段系统：

- **Stage1**：从 MCP server 抓取工具快照，结合 `skills_full` 文档为每个工具构建 tool-card。
- **Stage2**：基于 tool-card、taxonomy 和 canonical skills 文档，判定工具之间的有向关系，生成 tool graph。
- **Stage3**：在图上做 dependency-closed DAG 采样，自动生成不泄露工具链的科学问题及 expected trajectory。

它最终服务两个方向：

1. 为 QA / benchmark / training 生成 toolchain 级数据。
2. 为 `data-pipe` 的 `pipeline/kg` 提供原生 KG-sampled tasks。

### 1.2 当前版本的核心特征

截至 2026-06-03，当前版本已经不是早期的“文档摘要判边”系统，而是一个更固定、更工程化的协议系统，关键特征如下：

- 固定 stage taxonomy 是真源。
- tool-card 主路径变瘦，重解释字段进入 sidecar。
- pairwise adjudication 改成**单向 directed candidate-first**。
- `validate` 已从主链路移除，不再参与图构建决策。
- `confidence_raw` 直接采用 `agent_confidence`。
- `generates_input_for` 被拆成：
  - `generates_full_input_for`
  - `generates_partial_input_for`
- `requires_intermediate` 不再作为 edge type，只作为 `negative_reason`。
- Stage3 默认是 `dag_closure`，允许 partial edge，但必须先闭合依赖。
- `trajectory_v2_graph` 已成为 Stage3 主协议，并且被 `data-pipe/pipeline/kg` 原生消费。

### 1.3 当前最重要的现实状态

当前系统的协议、文件、脚本、workdir 结构已经较完整，但仍有两个现实问题必须清楚认识：

1. **Stage2 仍然是最慢的瓶颈**，即使支持 `max-workers` 并行，Claude Code / MCP 的实际耗时仍然高。
2. **Stage3 的成功率仍受外部 agent runtime 稳定性影响**，最近的长跑中出现过 `API Error 502`、parse failed 等失败原因。

也就是说：当前系统是“协议上收敛了、工程上可跑了”，但“生产级稳态还在持续优化中”。

---

## 2. 项目与 `data-pipe` 的关系

### 2.1 两个项目的分工

- `tool-kg`：构建 ToolKG 与 Stage3 sampled workflow。
- `data-pipe`：消费 sampled workflow，把它转成可执行任务与训练 / 评测数据。

### 2.2 `tool-kg` 的上游数据源

`tool-kg` 主要依赖四类真源：

1. **MCP 工具快照**
   - 来自 MolClaw MCP server 的 `tools/list`
   - 落盘为 `tool_snapshot.jsonl`

2. **canonical skills 文档**
   - 仓库内置 `skills_full`
   - 运行时会被复制到 workdir 的 `.claude/skills`

3. **固定 stage taxonomy**
   - 真源文件：`configs/stage_taxonomy.json`

4. **规则 / ontology / prompt**
   - `configs/edge_ontology_v1.yaml`
   - `configs/rules_v1.yaml`
   - `configs/prompts/*.md`

### 2.3 `data-pipe` 如何消费 `tool-kg`

`data-pipe/pipeline/kg` 当前原生消费：

- `runs/<run_id>/sample_results/sample_success_v2.jsonl`
- `runs/<run_id>/sample_results/questions.csv`

主协议是：

- `kg_task_spec_v0.2`
- `expected_trajectory.schema_version = trajectory_v2_graph`

也就是说，`tool-kg` 已经不是单纯“生产图”，而是直接产出可被下游执行链消费的 KG 任务。

---

## 3. 当前仓库结构与应该信任的文件

### 3.1 代码根目录

`<tool-kg-root>`

### 3.2 现在应该把哪些文件视为主协议

#### 配置与契约

- `configs/stage_taxonomy.json`
- `configs/edge_ontology_v1.yaml`
- `configs/rules_v1.yaml`
- `configs/semantic_types_v1.yaml`
- `configs/prompts/tool_card_agent_v1.md`
- `configs/prompts/pairwise_adjudication_v1.md`
- `configs/prompts/toolchain_question_sampler_v1.md`

#### 核心代码

- `src/molclaw_kg/tool_card_builder.py`
- `src/molclaw_kg/candidate_generation.py`
- `src/molclaw_kg/pairwise_runner.py`
- `src/molclaw_kg/confidence.py`
- `src/molclaw_kg/graph_views.py`
- `src/molclaw_kg/exporters.py`
- `src/molclaw_kg/provenance.py`
- `src/molclaw_kg/audit_sampler.py`
- `src/molclaw_kg/evaluate_logs.py`
- `src/molclaw_kg/question_sampling/sampler.py`
- `src/molclaw_kg/adjudicators/claude_code_runtime.py`
- `src/molclaw_kg/adjudicators/agent_cc.py`
- `src/molclaw_kg/stage_taxonomy.py`
- `src/molclaw_kg/relation_utils.py`
- `src/molclaw_kg/models.py`
- `src/molclaw_kg/schemas.py`
- `src/molclaw_kg/validators.py`

#### 运行脚本

- `scripts/run_pipeline_stage1_toolcards.sh`
- `scripts/run_pipeline_stage2_graph.sh`
- `scripts/run_full_pipeline.sh`
- `scripts/run_sample_questions.sh`

#### 下游消费

- `../data-pipe/pipeline/kg/README.md`
- `../data-pipe/pipeline/kg/schemas/kg_task_spec_v0.2.md`
- `../data-pipe/pipeline/kg/scripts/build_kg_task_dataset.py`
- `../data-pipe/pipeline/kg/scripts/inspect_kg_samples.py`
- `../data-pipe/pipeline/kg/scripts/scan_kg_rollouts.py`

### 3.3 不应再把哪些东西当成主合同

以下内容现在主要属于历史文档或兼容层，不应覆盖当前代码事实：

- `coverage_level` 作为主判定字段
- `support_scope` 作为主语义层
- `validator_status` 作为主图门禁
- `requires_intermediate` 作为正式 edge type
- `pair_payload.json` 作为 pairwise 主证据入口
- `doc_context.jsonl` 作为 pairwise 主证据入口

这些字段、文件和概念有的仍在兼容层或历史文档里出现，但**当前主线不再以它们作为核心协议**。

---

## 4. 端到端总览：三阶段工作流

可以把当前 `tool-kg` 看成下面这条主链路：

```text
MCP tools/list snapshot
        ↓
skills_full chunking
        ↓
Stage1: tool-card enrich (fixed taxonomy + agent)
        ↓
Stage2: candidate generation → stage pruning → pairwise adjudication → scoring → views → export
        ↓
Stage3: graph filtering → DAG closure sampling → question generation → v2 trajectory export
        ↓
data-pipe/pipeline/kg consumption
```

每一段都不是“简单拼接”，而是各自有单独的输入、workdir、agent prompt、校验器和输出文件。

---

## 5. Source of Truth：输入、规则与技能文档

### 5.1 MCP 工具快照

#### 作用

`tool_snapshot.jsonl` 是所有工具节点的原始快照，来自 MCP server 的 `list_tools`。

#### 当前落盘内容

每条工具快照记录包含：

- `tool_id`
- `title`
- `name`
- `description`
- `inputSchema`
- `outputSchema`
- `annotations`

#### 代码来源

- `src/molclaw_kg/mcp_snapshot.py`

### 5.2 canonical skills 文档

#### 作用

`skills_full` 是本项目自己的技能文档真源。它既用于 tool-card enrich，也用于 pairwise adjudication 和 Stage3 的“canonical evidence first”原则。

#### 目录结构

代码支持两种布局：

- `skills_full/L1_tools/...`
- `skills_full/.claude/skills/L1_tools/...`

#### chunking 逻辑

- `doc_chunker.py` 会按标题结构切块。
- 每个 chunk 保留：
  - `doc_id`
  - `path`
  - `skill_level`
  - `section_id`
  - `heading_path`
  - `block_type`
  - `chunk_id`
  - `char_start`
  - `char_end`
  - `text`

### 5.3 stage taxonomy

#### 真源文件

- `configs/stage_taxonomy.json`

#### 当前用途

它不只是“工具阶段标签表”，而是整个 graph 构建的**剪枝与合法转移真源**：

- 定义 `primary_stage`
- 定义 `secondary_stages`
- 定义 `allowed_stage_transitions`
- 定义 `same_pruning_stage_transition_policy`
- 定义 `alternative_clusters`
- 定义 `edge_type_stage_policy`
- 定义 `stage_pruning_policy`
- 定义 `coverage_policy`

#### 强约束

- 81 工具必须全覆盖
- 未映射工具 fail-fast
- 不能发明新 stage
- `primary_stage` 必须来自映射表

### 5.4 rules 与 edge ontology

#### `rules_v1.yaml`

当前实际读取到的关键项主要是：

- `thresholds`
  - `schema_candidate_min`
  - `schema_high_priority`
  - `positive_decision_min`
  - `uncertain_lower`
  - `expanded_min`
  - `core_min`
  - `uncertain_min`
- `weights`
  - 现在主代码里真正使用的是 `agent: 1.0`
- `stage_pruning.mode`
  - 当前默认 `relation_aware`

#### `edge_ontology_v1.yaml`

当前 canonical edge types：

- `generates_full_input_for`
- `generates_partial_input_for`
- `preprocesses_for`
- `converts_format_for`
- `parameterizes_for`
- `filters_candidates_for`
- `ranks_or_scores_for`
- `validates_output_of`
- `refines_output_of`
- `reports_or_summarizes`
- `alternative_to`

其中：

- 前 10 个属于 transition / chain edge。
- `alternative_to` 是非转移关系。

`requires_intermediate` 已经不再是正式 edge type，而是作为 `negative_reason` 存在。

---

## 6. 当前协议：ToolCard / Edge / Trajectory 的核心数据模型

这一节是整个系统最关键的部分。接手者应该把它当成主协议而不是“字段列表”。

### 6.1 ToolCard：工具卡

`ToolCard` 是 Stage1 的主中间表示，也是 Stage2 candidate generation 的基础。

#### 主路径字段

- `tool_id`
- `title`
- `description_summary`
- `primary_stage`
- `secondary_stages`
- `aliases`
- `inputs`
- `outputs`
- `connectable_inputs`
- `connectable_outputs`
- `input_requirement_sets`
- `preconditions`
- `side_effects`
- `needs_review`

#### Slot 结构

每个 `Slot` 记录：

- `name`
- `raw_type`
- `semantic_type`
- `format`
- `unit`
- `cardinality`
- `parameter_kind`
- `requirement_status`
- `required`
- `description`
- `source`
- `confidence`

#### 当前设计含义

- `inputs` / `outputs`：从 raw MCP schema 和 outputSchema 结构化得来。
- `connectable_inputs` / `connectable_outputs`：可被 graph 邻接关系直接使用的“连接槽位”。
- `input_requirement_sets`：表达不同调用模式下的 required / optional / defaulted 组合。
- `needs_review`：当 agent 输出不稳定、验证失败、stage 约束冲突时，fallback 到 deterministic base card 的标记。

#### debug / audit 信息

这些字段不再是主卡核心，而是进入 sidecar：

- `capability_type`
- `domain_entities`
- `typical_upstream_roles`
- `typical_downstream_roles`
- `quality_checks`
- `negative_constraints`
- `stage_rationale`
- `evidence_refs`
- `provenance_refs`
- `extraction_confidence`
- `notes`

它们仍然存在于某些 agent 输出或 debug 输出中，但不应被当成主路径必需字段。

### 6.2 CandidatePair：候选边

候选边由 `candidate_generation.py` 产生。

#### 主字段

- `pair_id`
- `source_tool`
- `target_tool`
- `source_stage`
- `target_stage`
- `source`
- `schema_score`
- `suggested_edge_types`
- `negative_reason`

#### 含义

- `schema_score` 是 schema/semantic/format/name 的组合分数，用于候选召回。
- `suggested_edge_types` 是给 pairwise agent 的候选 edge type 提示，不是最终边类型。
- `negative_reason` 表示被判定为负候选或需要中间步骤的原因。

### 6.3 AdjudicationRecord：pairwise 判边结果

Stage2 的 agent 输出必须符合这个协议。

#### 主字段

- `pair_id`
- `relation_status`
- `direct_transition`
- `edge_types`
- `negative_reason`
- `satisfied_mappings`
- `unsatisfied_required_inputs`
- `context`
- `evidence_refs`
- `rationale`
- `agent_model`
- `agent_confidence`
- `raw_payload_hash`

#### `relation_status` 当前取值

- `valid`
- `negative`
- `uncertain`
- `alternative`

#### 语义解释

- `valid`：该方向是可执行 / 可邻接 / 可转移的正向边。
- `negative`：该方向明确不成立，或明确需要中间步骤而不能直接相邻。
- `uncertain`：证据不足或格式/解析问题，不足以作为强正边或强负边。
- `alternative`：两个工具在功能上可替代，但这不是直接转移语义。

#### `edge_types` 的结构

每个 `edge_types[*]` 是对象，至少包含：

- `type`
- `source_slot`
- `target_slot_or_precondition`
- `confidence`
- `evidence_ids`

#### 当前 edge type 语义

- `generates_full_input_for`：源工具输出可直接满足目标工具的完整/主要输入集合。
- `generates_partial_input_for`：源工具只能满足一部分输入，后续必须通过 provider 或 user_given 闭合依赖。
- `preprocesses_for`：源工具为目标工具准备、净化、修复、标准化对象。
- `converts_format_for`：格式或编码转换。
- `parameterizes_for`：提供参数、阈值、网格、口袋、配置等。
- `filters_candidates_for`：生成候选子集。
- `ranks_or_scores_for`：排序 / 打分 / 重排序。
- `validates_output_of`：验证前序工具输出。
- `refines_output_of`：精修或修复前序结果。
- `reports_or_summarizes`：报告 / 汇总 / 可视化 / 导出。
- `alternative_to`：功能替代关系，不是转移边。

### 6.4 FinalEdge：最终图边

Stage2 评分与图视图构建后，输出最终边。

#### 主字段

- `edge_id`
- `pair_id`
- `source_tool`
- `target_tool`
- `edge_type`
- `direct_transition`
- `source_slot`
- `target_slot`
- `stage_src`
- `stage_tgt`
- `relation_status`
- `confidence_raw`
- `confidence_calibrated`
- `view`
- `evidence_ids`
- `negative_reason`
- `created_at`
- `run_id`

#### 当前核心语义

- `confidence_raw = agent_confidence`
- `confidence_calibrated` 默认等于 `confidence_raw`
- `view` 由 `relation_status + confidence` 共同决定

#### 当前视图集合

- `core`
- `expanded`
- `uncertain`
- `negative`

旧文档里提到的 `rejected` 已并入 `uncertain` 的硬失败降级逻辑，不再作为独立主视图。

### 6.5 置信度和视图阈值

当前主路径非常简化：

- `confidence_raw = agent_confidence`
- `confidence_calibrated = confidence_raw`

视图阈值：

- `core_min = 0.80`
- `expanded_min = 0.55`

映射逻辑：

- `valid`
  - `conf >= 0.80` → `core`
  - `0.55 <= conf < 0.80` → `expanded`
  - `< 0.55` → `uncertain`
- `alternative`
  - `conf >= 0.55` → `expanded`
  - `< 0.55` → `uncertain`
- `negative`
  - → `negative`
- `uncertain`
  - → `uncertain`

---

## 7. Stage1：MCP snapshot + doc chunking + tool-card enrich

Stage1 的目标不是做 graph，而是做高质量 tool-card。

### 7.1 输入

Stage1 依赖两个最重要的输入：

1. `tool_snapshot.jsonl`
2. `doc_chunks.jsonl`

其中：

- `tool_snapshot.jsonl` 来自 MCP `tools/list`
- `doc_chunks.jsonl` 来自 `skills_full` chunking

### 7.2 运行顺序

Stage1 的主顺序是：

```text
snapshot -> doc-chunks -> tool-cards
```

#### 脚本入口

- `scripts/run_pipeline_stage1_toolcards.sh`

支持参数：

- `run_id`
- `--alert-rerun`
- `--max-alert-rerun-rounds`
- `--max-workers`
- `--resume`
- `--tool-ids-file`

### 7.3 MCP snapshot 的细节

`mcp_snapshot.py` 通过 streamable HTTP 连接 MCP server，抓取工具并规范化为 JSONL。

每条 snapshot 会包含：

- `tool_id`
- `title`
- `name`
- `description`
- `inputSchema`
- `outputSchema`
- `annotations`

### 7.4 doc chunking 的细节

`doc_chunker.py` 会把技能文档切成 chunk，并保留标题层级信息。

#### chunk 规则

- 按 `#` / `##` / `###` 标题切块
- 保留：
  - `heading_path`
  - `block_type`
  - `chunk_id`
  - `char_start`
  - `char_end`

#### doc level

- `L1`
- `L2`
- `L3`

### 7.5 deterministic base tool-card

在 agent 之前，Stage1 先基于 raw schema 构造一个 deterministic base card。

#### base card 的来源

- 输入 schema 的 `properties` / `required`
- outputSchema 的 `properties`
- 对嵌套 object / array object 做 flatten

#### base card 的作用

- 作为 agent 的先验结构
- 作为 validation 失败时的 fallback
- 保证每个工具至少有一个可用主卡

### 7.6 Stage1 workdir 的具体内容

每个 tool-card attempt 会在：

`runs/<run_id>/cc_workdir/toolcard__<tool_id>/attempt_000x/`

写入：

- `.claude/`
- `CLAUDE.md`
- `tool_snapshot_row.json`
- `deterministic_base_tool_card.json`
- `stage_taxonomy.json`
- `task_context.json`
- `doc_context.jsonl`
- `prompt.txt`
- `complete_session.jsonl`
- `repair_prompt.txt`（如果 JSON 解析失败并尝试修复）

#### `task_context.json` 的含义

它是轻量任务清单和约束索引，告诉 Claude Code：

- 当前 tool_id
- 固定 primary stage
- allowed stages
- 文件名索引
- 该 workdir 下有哪些输入文件

#### `doc_context.jsonl` 的含义

这是从 `doc_chunks.jsonl` 里按工具词条命中的派生上下文片段，字段通常包括：

- `chunk_id`
- `doc_id`
- `path`
- `heading_path`
- `text`

它的作用是：给 agent 提供**局部、聚焦的文档证据**，但它本质上是派生摘要，不是 canonical 真源。

### 7.7 tool-card prompt 的真实约束

tool-card prompt 强调：

- `fixed_primary_stage` 必须严格使用
- `secondary_stages` 只能从 allowed stages 里选
- 不得发明 stage
- 输出必须是严格 JSON
- `format` 必须是字符串，未知填 `"unknown"`
- nested output 要 flatten 成可连接 slot

### 7.8 agent 输出与校验

Stage1 agent 输出后，代码会：

1. 尝试从 `result_text` / `assistant_text` / `raw_stream` 中提取 JSON。
2. 运行 Pydantic 校验。
3. 校验 `primary_stage` 是否与 fixed taxonomy 一致。
4. 校验 `secondary_stages` 是否在 allowed stages 里。
5. 若失败，则：
   - 记 alert
   - fallback 到 deterministic base card
   - 强制 `needs_review = true`

### 7.9 Stage1 的异常分类

当前常见状态包括：

- `ok`
- `parse_failed`
- `stage_mismatch`
- `secondary_stage_invalid`
- `validation_failed`
- `worker_exception`

这些状态会进入：

- `tool_card_progress.jsonl`
- `tool_cards_debug.jsonl`
- `tool_card_alerts.jsonl`
- `tool_card_alerts_meta.json`

### 7.10 Stage1 的续跑和 alert-rerun

Stage1 已实现：

- `--resume`
- `--max-workers`
- `--alert-rerun`
- `--max-alert-rerun-rounds`
- `--tool-ids-file`
- `--merge-into-existing`
- `--rerun-round`

#### 续跑逻辑

- 如果 `tool_card_progress.jsonl` 和 `tool_cards.jsonl` 已存在，resume 会跳过已完成工具。
- alert rerun 会读取 `tool_card_rerun_targets.txt`，只重跑失败工具。
- 失败重跑时会 merge 回全量 `tool_cards.jsonl`，不会丢历史结果。

#### progress ledger

`tool_card_progress.jsonl` 是 Stage1 的增量账本，记录每个 tool 的：

- `status`
- `error`
- `fallback_applied`
- `workdir`
- `session_file`
- `prompt_file`
- `doc_context_hits`
- `attempt_dir`
- `rerun_round`
- `card`
- `debug_row`
- `alert_row`

### 7.11 Stage1 输出产物

主产物：

- `tool_cards.jsonl`
- `tool_cards_debug.jsonl`
- `tool_cards_meta.json`

警报产物：

- `tool_card_alerts.jsonl`
- `tool_card_alerts_meta.json`
- `tool_card_rerun_targets.txt`

运行辅助：

- `tool_card_progress.jsonl`

### 7.12 Stage1 当前的设计张力

当前 Stage1 仍然存在两层重叠上下文：

1. canonical skills / `.claude/skills`
2. `doc_context.jsonl` 这种派生命中片段

这意味着 agent 既能看到原始技能文档，也能看到集中命中的摘要片段。
这是当前设计上的折中：它提高了局部命中率，但也可能让 agent 过度依赖派生摘要。
这不是 bug，而是当前版本仍需持续评估的设计张力。

---

## 8. Stage2：candidate generation + pairwise adjudication + graph export

Stage2 是当前图构建的核心。

### 8.1 总体顺序

```text
candidates -> adjudicate -> score -> views -> provenance -> export -> audit -> eval-logs -> manifest
```

#### 脚本入口

- `scripts/run_pipeline_stage2_graph.sh`

支持参数：

- `run_id`
- `--alert-rerun`
- `--max-alert-rerun-rounds`
- `--max-workers`
- `--resume`
- `--pair-ids-file`
- `--merge-into-existing`
- `--bypass-cache-for-targets`
- `--rerun-round`

### 8.2 candidate generation：全量有向 pair 的 schema 召回

`candidate_generation.py` 对 81 个工具做全量有向组合：

- ordered pair 总数理论上是 `81 * 80 = 6480`

#### 召回依据

对每个 ordered pair `(A, B)` 计算 schema score，主要使用：

- semantic type compatibility
- format compatibility
- name overlap

#### 召回阈值

来自 `rules_v1.yaml`：

- `schema_candidate_min = 0.45`
- `schema_high_priority = 0.70`

#### 候选记录字段

每个候选边记录：

- `source_tool`
- `target_tool`
- `source_stage`
- `target_stage`
- `source`
- `schema_score`
- `suggested_edge_types`
- `negative_reason`

#### edge type 建议逻辑

候选阶段会根据 tool card / stage / schema 得到一个 `suggested_edge_types`，但这只是提示，不是最终判定。

### 8.3 stage pruning：把不该判的 pair 先剪掉

Stage2 的 pruning 是当前图构建的重要前置步骤。

#### 默认模式

- `relation_aware`

#### 兼容模式

- `cross_stage_only`
- `all`

#### pruning 依据

依赖 `stage_taxonomy.json` 的：

- `allowed_stage_transitions`
- `same_pruning_stage_transition_policy`
- `alternative_clusters`
- `edge_type_stage_policy`

#### 产物

- `pair_pruned_by_stage.jsonl`
- `pair_pruned_by_stage_meta.json`

### 8.4 alternative_to 的特殊处理

`alternative_to` 不走普通 transition 判边逻辑。

它是由 taxonomy 中的 `alternative_clusters` 确定的确定性关系，典型例子包括：

- `pred_protein_structure_esmfold` ↔ `chai1_predict`
- `pred_pocket_prank` ↔ `fpocket_toolkit`

这些边是“替代关系”，不是“输出传递关系”。

### 8.5 pairwise adjudication：单向 directed 调用

这是最核心的变化之一。

#### 旧思路

曾经有过“一个调用里同时判断 A->B 和 B->A”的设计讨论，但当前主实现已经改成：

- **每个 allowed direction 单独判**

#### 当前 pairwise 主流程

对于每个 stage-pruning 后保留的 directed pair：

1. 准备 pair spec。
2. 准备 source manifest。
3. 复制 canonical skills bundle 到独立 workdir。
4. 构造 Claude Code prompt。
5. 调用 agent。
6. 严格 JSON 解析。
7. 运行 schema 校验。
8. 修复 legacy / invalid 字段。
9. 写入 pair adjudication 记录。

### 8.6 pairwise workdir 的真实结构

每个 pair attempt 的 workdir 目录通常是：

`runs/<run_id>/cc_workdir/<source_tool>__to__<target_tool>/attempt_000x/`

#### 其中会写入：

- `.claude/`
- `CLAUDE.md`
- `pair_spec.json`
- `source_manifest.json`
- `source_tool_card.json`
- `target_tool_card.json`
- `stage_taxonomy.json`
- `output_schema.json`
- `task_context.json`
- `prompt.txt`
- `complete_session.jsonl`

#### 各文件含义

- `pair_spec.json`
  - 当前 directed pair 的结构化描述
  - 包含 `pair_meta`、source/target 工具信息、候选技能路径等

- `source_manifest.json`
  - 不是证据正文，而是 canonical skills 的可搜索路径清单
  - 目的是强迫 agent 回读 `.claude/skills`

- `source_tool_card.json` / `target_tool_card.json`
  - 辅助工具卡，只作结构化提示，不是主证据

- `task_context.json`
  - 轻量任务索引
  - 指出当前 pair 的核心文件名和任务类型

### 8.7 pairwise prompt 的核心规则

当前 pairwise prompt 强调：

- 只能输出严格 JSON
- 只读当前 directed pair
- 先读 `task_context.json`
- 再读 `pair_spec.json`
- 再读 `stage_taxonomy.json`
- 再读 `source_manifest.json`
- 再回读 canonical skills
- 不得仅根据 `pair_spec` / `task_context` 作判断
- `requires_intermediate` 不能作为 edge type，只能作为 `negative_reason`

### 8.8 pairwise 输出契约

输出对象必须包含：

- `pair_id`
- `relation_status`
- `direct_transition`
- `edge_types`
- `negative_reason`
- `context`
- `satisfied_mappings`
- `unsatisfied_required_inputs`
- `evidence_refs`
- `rationale`
- `agent_confidence`
- `agent_model`

### 8.9 pairwise 的 cache / resume / rerun

Stage2 当前支持：

- `pairwise_cache.jsonl` 缓存
- `--resume`
- `--merge-into-existing`
- `--pair-ids-file`
- `--bypass-cache-for-targets`
- `--rerun-round`

#### 典型语义

- `resume`
  - 不重跑已存在的 pair adjudication 结果

- `merge-into-existing`
  - 把 subset rerun 的结果合并回原始全量文件

- `bypass-cache-for-targets`
  - 对 rerun 目标强制重新调用 agent，不走旧 cache

### 8.10 pairwise alert 与局部重跑

Stage2 会把以下情况写成 alert：

- response schema 不合法
- JSON 解析失败
- `negative_reason=agent_output_parse_failed_directional`

然后把对应 pair_id 写到：

- `pair_adjudication_rerun_targets.txt`

脚本在 `--alert-rerun` 模式下会自动重跑这些目标，最多 `3` 轮。

### 8.11 pairwise 的错误修复逻辑

在 code 层，`agent_cc.py` 和 `pairwise_runner.py` 都对 legacy 字段做过修复：

- `valid_full / valid_partial / positive` → `valid`
- `invalid / negative` → `negative`
- `requires_intermediate` 不能再作为正式 edge type，若 agent 输出则转成：
  - `relation_status=negative`
  - `negative_reason=requires_intermediate`

### 8.12 Stage2 的 scoring

当前的 confidence 逻辑已经被极度简化：

- `confidence_raw = agent_confidence`
- `confidence_calibrated = confidence_raw`

历史上曾考虑过 `schema + agent + consistency - negative` 的加权设计，但现在主路径不再使用那套复杂分解。

### 8.13 Graph views

`graph_views.py` 会根据 `relation_status` 和 `confidence` 生成：

- `graph_core.jsonl`
- `graph_expanded.jsonl`
- `graph_uncertain.jsonl`
- `graph_negative.jsonl`
- `graph_all.jsonl`
- `edge_debug_sidecar.jsonl`

#### view 的主要含义

- `core`
  - 强正边，高置信度
- `expanded`
  - 可接受正边，置信度略低，但仍有下游价值
- `uncertain`
  - 证据不足、置信度低、或者硬失败降级
- `negative`
  - 明确负边

### 8.14 edge_debug_sidecar 的意义

`edge_debug_sidecar.jsonl` 保留了主图不想携带的重解释信息，例如：

- `context`
- `satisfied_mappings`
- `unsatisfied_required_inputs`
- `evidence_refs`
- `agent_conf`
- `rationale`

这是一个典型 sidecar：主图保持轻，解释层单独保留。

### 8.15 导出层

当前导出产物包括：

- `graph_all.csv`
- `<run_id>.csv`（pair-level 官方汇总表）
- `graph_all.graphml`
- `export_meta.json`

#### pair CSV 的字段特点

pair-level CSV 不再用 `Weight=1/2` 这种老表达，而是聚合成一行一个 `(Source, Target, pair_id)`，其中包含：

- `edge_types`（JSON list）
- `edge_confidences`（JSON list）
- `max_confidence`
- `min_confidence`
- `view`
- `metadata`

#### `metadata`

当前主要包含：

- `pair_id`
- `context`
- `relation_statuses`

### 8.16 provenance

`provenance.py` 会把每条边映射成一个 PROV 风格的 sidecar：

- `prov:Entity`
- `prov:Activity`
- `prov:Agent`
- `prov:Plan`

它的目标不是替代主图，而是让边的生成来源可追溯、可复盘。

### 8.17 audit / log evaluation / manifest

Stage2 还会生成三个辅助视图：

- `audit_sample.csv`
- `log_evaluation.json`
- `repro_manifest.json`

#### audit sampler

从 `core / expanded / uncertain / negative` 里抽固定配额样本，供人工审阅：

- core: 120
- expanded: 80
- uncertain: 60
- negative: 40

#### log evaluation

`evaluate_logs.py` 只做 coverage 评估，不做图真值构建。

它会看历史 `logs_root` 下的 `complete_session.jsonl` 或类似 JSONL，计算：

- log pair coverage
- graph pair coverage
- missed pair 列表

#### repro manifest

`repro_manifest.json` 会记录 run 目录里所有文件的 SHA256 和大小，方便复现和比对。

### 8.18 当前 validator 的地位

`validators.py` 仍然存在，但它**不再是主流程的一部分**。

它包含的检查器主要有：

- `ShapeEdgeValidator`
- `TypedIOValidator`
- `DirectnessValidator`
- `StageValidator`
- `EvidenceSourceValidator`
- `ConflictResolver`

它可以在兼容或手工审计场景下生成：

- `validation_results.jsonl`
- `pair_adjudications_validated.jsonl`
- `validation_meta.json`

但当前主 pipeline 的 `run_all()` 和 CLI 主链路里**不再调用 validate**。

这意味着：

- validator 是兼容层 / 审计层
- 不是主图生成真源

---

## 9. Stage3：图上采样、闭包检查、问题生成与 trajectory_v2_graph

这是当前 `tool-kg` 与 `data-pipe` 的关键接口。

### 9.1 Stage3 的目标

Stage3 不再只做“线性 random walk 生成问题”，而是做：

1. 从 `graph_all.jsonl` 里采样 anchor toolchain。
2. 检查 toolchain 是否依赖闭合。
3. 如果有 partial edge，则寻找 provider candidate 或 user_given 闭合缺口。
4. 让 Claude Code 生成**不泄露工具名**的问题和 `trajectory_v2_graph`。
5. 把成功样本写成可供 `data-pipe` 消费的 KG sampled tasks。

### 9.2 Stage3 的输入条件

必须有：

- `graph_all.jsonl`
- `tool_cards.jsonl`
- `tool_snapshot.jsonl`

### 9.3 Stage3 的主模式

当前默认：

- `sampling_mode = dag_closure`

调试模式：

- `sampling_mode = linear_debug`

### 9.4 graph edge 过滤

Stage3 只保留符合以下条件的边作为可走边池：

- `view in {core, expanded, uncertain}`
- `relation_status = valid`
- `direct_transition = true`
- `edge_type in TRANSITION_EDGE_TYPES`

#### `linear_debug` 的额外限制

- 会排除 `generates_partial_input_for`

#### `dag_closure` 的默认语义

- 允许 `generates_partial_input_for`
- 但必须通过 closure engine 后续闭合依赖

### 9.5 start node 选择

Stage3 不是随便挑起点，而是先判断节点是否能形成长度至少为 `min_hops` 的 walk。

#### 选择策略

- 先对每个 hop 数 `k` 预计算可行 starts
- start 节点按出度加权抽样
- 允许重复节点
- 若某个 hop 数没有可行 start，则该样本失败

### 9.6 walk 规则

walk 的 hop 范围默认是：

- `min_hops = 2`
- `max_hops = 4`

每次 sample：

1. 随机取 hop 数 `k`
2. 在可行 starts 中抽 start
3. 逐步向前走 `k` 步
4. 若中途死路，记失败

### 9.7 input closure engine

这是 Stage3 的关键创新点。

它的目标不是直接“能不能走图”，而是判断：

- 每个 tool 的 required inputs 是否被覆盖
- 覆盖来自哪里
  - upstream tool
  - user_given
  - unknown

#### 输入来源

closure engine 会读取：

- `ToolCard.input_requirement_sets`
- `connectable_inputs`
- `connectable_outputs`
- graph edges
- initial user-given inputs

#### 输出

每个 sample 都会有一个 closure report，主要字段：

- `closure_status`
  - `closed`
  - `open`
  - `unknown`

- `per_tool_requirements`

- `open_requirements`

#### open requirement 的字段

每个缺口会记录：

- `requirement_id`
- `tool_id`
- `input_name`
- `semantic_type`
- `format`
- `reason`
- `can_be_user_given`

### 9.8 provider candidate retrieval

对于每个 open requirement，Stage3 会从 KG 中找可能的补充 provider。

#### 候选标准

- provider 与 target 之间存在边
- provider 的 output / side_effect 与缺失输入在 semantic / format / name 上相容
- provider 不会引入循环
- provider 排名要兼顾语义匹配与边置信度

#### 候选输出

`provider_candidates.jsonl` 每条记录包含：

- `sample_id`
- `requirement_id`
- `candidates`

候选 item 包含：

- `provider_tool_id`
- `target_tool_id`
- `matched_output`
- `semantic_score`
- `edge_confidence`
- `edge_type`
- `view`
- `already_in_workflow`
- `rank_score`

### 9.9 agent resolution decisions

Stage3 不是让 agent 自由编图，而是让它在每个 requirement 上做有限决策：

- `user_given`
- `provider_tool`
- `reject`
- `unknown`

#### 约束

- 如果选择 `provider_tool`，必须来自 Python 提供的 candidate list。
- 如果选择 `user_given`，public question 中必须能找到对应输入。
- 如果选择 `reject`，表示这个样本科学上不自然或无法闭合。
- 如果选择 `unknown`，表示证据不足但暂不拒绝。

### 9.10 expected_trajectory = trajectory_v2_graph

这是 Stage3 的主协议。

#### 图节点类型

- `input`
- `llm`
- `tool`
- `output`

#### LLM role

- `plan`
- `parameterize`
- `interpret`
- `route`
- `summarize`
- `repair`（可选）

#### 图边关系

- `provides_context`
- `selects_tool`
- `parameterizes_tool`
- `tool_observation`
- `routes_to_next`
- `summarizes_result`
- `provides_input_to`

#### 关键原则

- LLM 节点**只存在于 trajectory 层**
- 不进入 ToolKG 本体图
- ToolKG 仍然是 tool-only graph

### 9.11 Stage3 的 output schema

`QUESTION_SAMPLER_OUTPUT_SCHEMA` 强制要求：

- `status`
- `reject_reason`
- `question_payload`
- `public_question_text`
- `resolution_decisions`
- `expected_trajectory`
- `quality_checks`

#### 强约束

- `public_question_text` 不能泄露工具名
- 不能写“先用 A 再用 B”这种显式顺序
- `expected_trajectory.schema_version` 必须是 `trajectory_v2_graph`
- `quality_checks` 里要说明输入闭合、工具链不可泄露、final deliverable 可验证

### 9.12 Stage3 workdir 的真实结构

每个样本会落到：

`runs/<run_id>/sample_workdir/sample_XXXX__<start>__<end>/`

#### workdir 文件

- `.claude/`
- `CLAUDE.md`
- `task_context.json`
- `toolchain_spec.json`
- `output_schema.json`
- `prompt.txt`
- `complete_session.jsonl`
- `agent_trace.json`

#### `toolchain_spec.json` 的含义

这是给 agent 的完整任务图上下文，包含：

- `sample_id`
- `sampling_mode`
- `walk_hops`
- `anchor_toolchain_nodes`
- `anchor_toolchain_edges`
- `dag_tools`
- `dag_edges`
- `closure_report`
- `open_requirements`
- `candidate_provider_tools`
- `deterministic_resolution_suggestions`
- `workflow_skeleton`
- `source_filters`

这份 spec 是 Stage3 的核心任务输入。

### 9.13 Stage3 的 success / failure 规则

样本只有在以下条件都满足时才算 success：

1. JSON 格式正确。
2. schema 校验通过。
3. public question 没有工具泄露。
4. 没有显式 sequence hint。
5. resolution decisions 与 provider candidate 约束一致。
6. 经过 resolution 后的 DAG 仍然闭合，或者允许 unknown soft 通过时满足对应条件。
7. `trajectory_v2_graph` 结构合法。
8. trajectory 中包含所有 resolved tool nodes。

#### 常见 failure reason

Stage3 会把失败原因细分并写入 `sample_attempts.jsonl`，常见包括：

- `no_feasible_start_for_hops:<k>`
- `dead_end_before_target_hops:...`
- `agent_api_error_502`
- `agent_timeout`
- `agent_runtime_nonzero_exit`
- `agent_output_parse_failed`
- `response_schema_invalid:...`
- `agent_reject:...`
- `question_leak_detected`
- `question_sequence_hint_detected`
- `resolution_decisions_invalid`
- `resolution_validation_failed:...`
- `closure_open_after_expansion`
- `trajectory_graph_invalid:...`
- `trajectory_missing_resolved_tools`

### 9.14 Stage3 输出文件

主要输出写在：

`runs/<run_id>/sample_results/`

#### 核心文件

- `sample_attempts.jsonl`
- `sample_success.jsonl`
- `sample_success_v2.jsonl`
- `questions.csv`
- `sampling_meta.json`
- `workflow_quality_report.json`
- `input_closure_report.jsonl`
- `provider_candidates.jsonl`

#### questions.csv 的字段

- `index`
- `sample_id`
- `question`
- `task_type`
- `public_question_text`
- `expected_trajectory`
- `toolchain_nodes`
- `toolchain_edges`
- `walk_hops`
- `start_tool`
- `end_tool`
- `status`
- `failure_reason`
- `closure_status`
- `open_requirements_count`
- `llm_node_count`
- `workflow_node_count`
- `workflow_edge_count`
- `workdir`
- `session_file`

### 9.15 workflow_quality_report 的含义

这个文件不是“评测分数”，而是 Stage3 质量画像，包含：

- attempt_count
- success_count
- failure_count
- failure_breakdown
- closure_status_breakdown
- llm_role_distribution
- provider_candidate_records
- closure_report_records

它更像 run 级诊断报告。

---

## 10. Claude Code Runtime：agent 是怎么被包起来的

### 10.1 Claude Code 的调用框架

`tool-kg` 不是直接调用裸 LLM，而是通过 `ClaudeCodeRuntime` 调 Claude Code CLI。

#### 标准调用形态

```text
claude --dangerously-skip-permissions --verbose --output-format stream-json \
  --mcp-config <tmp/mcp_config.json> \
  --strict-mcp-config \
  [--add-dir <workdir>] \
  [--allowedTools Read,Glob,mcp__<server>] \
  -p
```

### 10.2 MCP config

每次调用前都会生成临时 `mcp_config.json`，其中包含：

- MCP server name
- MCP server URL
- auth header
- auth token

### 10.3 stream-json 处理

Claude Code 的输出是 stream-json，runtime 会：

1. 读取完整 `complete_session.jsonl`
2. 解析 assistant chunk
3. 从 raw stream / assistant / result 中提取 JSON
4. 校验 MCP init event
5. 检查 mcp server 是否 connected
6. 如果需要，重试 MCP ready 检查

### 10.4 readiness 检查

runtime 里有一层 MCP ready 检查逻辑：

- 先看 system init 事件
- 再看 MCP server status
- 如果 server 未 connected，而返回码却是 0，会被修正为非零状态

这意味着：即使 Claude CLI 表面执行成功，如果 MCP 没连上，也不会被当成真正成功。

### 10.5 provider switch 现状

当前代码里 provider switch 是一个“外部预设”而不是 runtime 内部自动切换流程。

也就是说：

- `MOLCLAW_AGENT_PROVIDER` / `CC_SWITCH_PROVIDER` 可以设置 provider 名称
- 但 runtime 内的 `switch_provider()` 当前是禁用的 / no-op
- 运行前需要确保外部环境已经把 provider 切好

这是一个当前工程现实，而不是理想状态。

### 10.6 运行中会记录什么

每次 Claude 调用都会写：

- `complete_session.jsonl`
- `agent_trace.json`（Stage3）
- runtime trace 信息：
  - provider
  - command
  - return_code
  - timed_out
  - latency_sec
  - prompt_sha256
  - mcp_config_sha256
  - mcp_server_name
  - mcp_server_url
  - workdir
  - session_file
  - parsed_ok

---

## 11. 输出目录：run 目录里到底会出现什么

一个标准 run 目录大致是：

`runs/<run_id>/`

### 11.1 Stage0 / Stage1 / Stage2 输出

- `tool_snapshot.jsonl`
- `tool_snapshot_meta.json`
- `doc_chunks.jsonl`
- `doc_chunks_meta.json`
- `tool_cards.jsonl`
- `tool_cards_debug.jsonl`
- `tool_cards_meta.json`
- `tool_card_progress.jsonl`
- `tool_card_alerts.jsonl`
- `tool_card_alerts_meta.json`
- `tool_card_rerun_targets.txt`
- `candidate_pairs.jsonl`
- `candidate_meta.json`
- `pair_pruned_by_stage.jsonl`
- `pair_pruned_by_stage_meta.json`
- `pairwise_cache.jsonl`
- `pair_adjudications.jsonl`
- `pair_adjudication_meta.json`
- `pair_adjudication_alerts.jsonl`
- `pair_adjudication_alerts_meta.json`
- `pair_adjudication_rerun_targets.txt`
- `scored_edges.jsonl`
- `scoring_meta.json`
- `graph_all.jsonl`
- `graph_core.jsonl`
- `graph_expanded.jsonl`
- `graph_uncertain.jsonl`
- `graph_negative.jsonl`
- `graph_views_meta.json`
- `edge_debug_sidecar.jsonl`
- `provenance_sidecar.jsonl`
- `export_meta.json`
- `graph_all.csv`
- `<run_id>.csv`
- `graph_all.graphml`
- `audit_sample.csv`
- `audit_sample_meta.json`
- `log_evaluation.json`
- `repro_manifest.json`
- `pipeline_status.json`

### 11.2 Stage3 输出

`runs/<run_id>/sample_workdir/`

以及：

`runs/<run_id>/sample_results/`

其中 `sample_results` 里至少有：

- `sample_attempts.jsonl`
- `sample_success.jsonl`
- `sample_success_v2.jsonl`
- `questions.csv`
- `sampling_meta.json`
- `input_closure_report.jsonl`
- `provider_candidates.jsonl`
- `workflow_quality_report.json`

### 11.3 与 workdir 相关的目录命名

Stage1 / Stage2 的 Claude workdir 采用 attempt 级目录：

- `cc_workdir/toolcard__<tool_id>/attempt_000x/`
- `cc_workdir/<source_tool>__to__<target_tool>/attempt_000x/`

Stage3 的 sample workdir 则是每个 sample 一个目录：

- `sample_workdir/sample_0001__<start>__<end>/`

---

## 12. 当前代码里仍然保留的兼容层

这一节很重要，因为它决定了“哪些是主线，哪些只是老代码兼容”。

### 12.1 validators.py 仍然存在

它会在手动调用时输出：

- `validation_results.jsonl`
- `pair_adjudications_validated.jsonl`
- `validation_meta.json`

#### 其内部检查内容

- shape
- typed IO
- directness
- stage transition legality
- evidence source
- conflict resolution

但它**不再属于主 pipeline 的主决策链路**。

### 12.2 旧字段的兼容读取

当前代码仍能兼容读取一些旧字段，例如：

- `decision` → `relation_status`
- `valid_full / valid_partial / positive` → `valid`
- `invalid / negative` → `negative`

这只是为了兼容历史 run / 旧数据，不代表旧协议仍然是主协议。

### 12.3 历史文档中的旧术语

以下术语在历史文档中出现过，但如果和当前代码冲突，应以当前代码为准：

- `coverage_level`
- `support_scope`
- `rejected`
- `bidirectional adjudication`
- `pair_payload.json`
- `doc_context.jsonl` 作为 pairwise 主证据入口

### 12.4 `doc_context.jsonl` 的当前位置

它**仍然存在于 Stage1 tool-card workdir**，但**不是 pairwise 的主输入**。

换句话说：

- Stage1 仍会把 doc chunks 命中后写成 `doc_context.jsonl`
- Stage2 不再使用 `pair_payload.jsonl`
- Stage2 的主证据入口是 canonical skills + pair spec + source manifest

这点很容易和老 run 混淆，接手时要特别注意。

---

## 13. 当前的设计选择：为什么这样做

### 13.1 为什么要固定 taxonomy

因为如果 stage 由 agent 自由发明，图的稳定性会差，且 81 工具的全覆盖无法审计。
固定 taxonomy 的目的不是“限制研究空间”，而是让剪枝和关系判断有稳定边界。

### 13.2 为什么要把 validator 从主链路移除

因为脚本 validator 太容易把“形式检查”误认为“图质量真值”。
当前项目更相信：

- 固定 taxonomy
- canonical evidence
- 局部 agent 判定
- sidecar 审计

而不是让一套 hard rule validator 决定图是否成立。

### 13.3 为什么要把 full / partial 放进 edge type

因为 partial 语义不是 metadata 装饰，而是 downstream walk 的核心决策因素。
如果只把 partial 放在 metadata 里，Stage3 很容易把“只提供部分输入”的边误当成“可直接执行”的边。

### 13.4 为什么要做 DAG closure

因为线性 walk 无法自然处理 partial input 依赖闭合。
一旦引入 partial edge，就必须有：

- input closure checker
- provider retrieval
- user_given 建模
- dependency-closed DAG sampler

### 13.5 为什么要保留 LLM 节点在 trajectory，而不是 ToolKG

因为真实 agent 执行不只是 tool→tool：

- 有 planning
- 有 parameterization
- 有 route / interpret / summarize
- 有 tool observation

这些应当出现在 trajectory 层，而不应污染 base graph。

---

## 14. 当前已知风险与未决问题

### 14.1 Stage2 仍然非常慢

即便有：

- `--max-workers`
- `--resume`
- `--alert-rerun`

Stage2 仍是长耗时阶段。
这不是逻辑错误，而是 Claude Code + MCP 任务本身重。

### 14.2 agent runtime 仍会受外部服务影响

当前最明显的失败模式包括：

- `API Error 502`
- timeout
- JSON parse failed

这些属于 runtime 稳定性问题，不一定是协议错。

### 14.3 Stage3 仍然是新协议，质量要继续验证

虽然 DAG closure 已经能跑，但还要继续看：

- closure 成功率
- provider 召回是否足够
- `trajectory_v2_graph` 是否稳定覆盖所有 resolved tools
- 是否能避免 public question 泄露工具链

### 14.4 仍然存在轻度上下文重复

Stage1 的 `doc_context.jsonl`、`task_context.json`、`deterministic_base_tool_card.json` 与 canonical skills 之间存在重复信息。
这是当前工程上的折中，不是尚未发现的 bug，但确实可能继续简化。

### 14.5 历史文档可能误导新接手者

`docs/` 里仍有一些旧文档，例如：

- `deep-research-plan.md`
- `2-edge-val.md`
- `3-pair-ref.md`

这些文档对理解演进很有价值，但不能直接替代当前代码。

---

## 15. 已经做过的重要 pivot（按结果而不是时间线）

### 15.1 从“启发式图”转为“固定协议图”

最终采用：固定 taxonomy + canonical skills + agent adjudication + sidecar 审计。
放弃了纯启发式 stage 归类和随意的摘要判边方式。

### 15.2 从“多字段覆盖解释”转为“语义瘦身”

最终采用：

- `relation_status`
- `edge_type`
- `confidence`
- `context`

而不是把一堆 coverage / validator / source-context 字段塞进主图。

### 15.3 从“pair_payload 重摘要”转为“轻量 pair_spec + canonical skills”

pairwise 不再依赖重型 `pair_payload.json`，而是使用：

- `pair_spec.json`
- `source_manifest.json`
- `source_tool_card.json`
- `target_tool_card.json`
- canonical `.claude/skills`

### 15.4 从“线性 walk”转为“dependency-closed DAG”

Stage3 的核心变化就是这一条。
它是当前版本里最重要、也是最能影响下游质量的 pivot。

### 15.5 从“validate 主守门”转为“validator 兼容层”

这也是一个非常重要的 pivot。
现在主流程更依赖：

- prompt
- evidence
- taxonomy
- runtime sanity
- downstream audit

而不是 validator。

---

## 16. 运行命令速查

### 16.1 Stage1

```bash
cd <tool-kg-root>
bash scripts/run_pipeline_stage1_toolcards.sh run_x --alert-rerun --max-alert-rerun-rounds 3 --max-workers 3
```

### 16.2 Stage2

```bash
cd <tool-kg-root>
bash scripts/run_pipeline_stage2_graph.sh run_x --alert-rerun --max-alert-rerun-rounds 3 --max-workers 3
```

### 16.3 一键全跑

```bash
cd <tool-kg-root>
bash scripts/run_full_pipeline.sh run_x --max-alert-rerun-rounds 3 --max-workers 1
```

### 16.4 Stage3

```bash
cd <tool-kg-root>
bash scripts/run_sample_questions.sh run_x --sample-size 10 --min-hops 2 --max-hops 4 --sampling-mode dag_closure
```

线性 debug：

```bash
bash scripts/run_sample_questions.sh run_x --sample-size 5 --sampling-mode linear_debug
```

### 16.5 续跑已有 run

```bash
bash scripts/run_pipeline_stage2_graph.sh run_20260601_123052 --resume --max-workers 3 --alert-rerun --max-alert-rerun-rounds 3
```

### 16.6 data-pipe 的消费命令

```bash
cd <data-pipe-root>
python3 pipeline/kg/scripts/inspect_kg_samples.py --kg-run-dir <tool-kg-root>/runs/<run_id>
python3 pipeline/kg/scripts/build_kg_task_dataset.py --kg-run-dir <tool-kg-root>/runs/<run_id> --output-dir pipeline/kg/data/<run_id> --max-samples 10
```

---

## 17. 接手者下一步应该怎么做

### P0：先确认稳定性

1. 用一个真实 `run_id` 跑通 Stage1 + Stage2 + Stage3。
2. 看 `tool_card_alerts` 和 `pair_adjudication_alerts` 的残留数量。
3. 看 Stage3 的 `failure_breakdown` 是否主要是 runtime error，还是 closure / schema 问题。

### P1：再看图质量

1. 抽样审查 `graph_core.jsonl`、`graph_expanded.jsonl`、`graph_uncertain.jsonl`。
2. 检查 partial edge 是否真的在 Stage3 中起作用。
3. 检查 `edge_debug_sidecar.jsonl` 里的 context 是否足够解释边。

### P2：再做协议清理

1. 评估是否还需要 Stage1 的 `doc_context.jsonl` 作为派生辅助。
2. 评估 validator 是否需要继续保持兼容层。
3. 评估 `relation_status + edge_type + confidence + context` 是否足够支撑未来论文和产品化。

### P3：最后做论文/产品对齐

1. 把现有协议写成方法章节。
2. 把 Stage3 的 DAG closure 写成数据生成章节。
3. 把 `data-pipe/pipeline/kg` 的联调写成 downstream consumption 章节。

---

## 18. 总结

截至 2026-06-03，`tool-kg` 已经从“工具文档 + 启发式边 + 线性采样”的研究型原型，演进为一个有明确协议边界的三阶段系统：

1. **Stage1** 构建 lean ToolCard。
2. **Stage2** 构建 directed tool-only KG。
3. **Stage3** 采样 dependency-closed DAG workflow，并输出 `trajectory_v2_graph` 供 `data-pipe` 消费。

当前系统最值得记住的不是某个单独字段，而是这几个最终共识：

- taxonomy 是真源。
- edge 语义比 metadata 更重要。
- validator 不再决定主图。
- partial 依赖必须通过 closure 处理。
- LLM 节点只存在于 trajectory，不污染 ToolKG。
- `tool-kg` 已经和 `data-pipe` 形成了清晰的上下游接口。

如果接手者只记住一句话，那应该是：

> 现在的 `tool-kg` 已经不是“从文档猜边”的工具，而是一个“固定协议 + 局部 agent 裁决 + DAG 级采样 + 下游任务化输出”的完整 KG 生成系统。

