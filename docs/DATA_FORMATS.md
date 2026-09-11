# Data Formats

下列字段只列稳定边界；debug、provenance 和派生统计不构成新的 authority。

## Tool Catalog

正式产物 `results/tool_catalog.jsonl` 每行一个工具；run 根目录的 `tool_cards.jsonl` 是可恢复的 Stage1 中间状态。

```json
{
  "tool_id": "tool_name",
  "description_summary": "...",
  "primary_stage": "...",
  "scheduling_stages": ["..."],
  "schema_slots": [{
    "slot_path": "input.protein_file",
    "direction": "input",
    "raw_type": "string",
    "required": true,
    "source": "input_schema"
  }],
  "slot_annotations": {
    "input.protein_file": {
      "semantic_type": "protein_structure",
      "format": "pdb",
      "connectable": true,
      "evidence_refs": ["..."]
    }
  },
  "skill_derived_slots": [],
  "connectable_inputs": [{"name": "...", "raw_type": "...", "required": true}],
  "connectable_outputs": [{"name": "...", "raw_type": "..."}],
  "preconditions": []
}
```

MCP schema 的参数名、raw type、required/default/enum 是不可覆盖的确定性事实；skills
语义由 tool-card agent 以 annotation patch 补充。真实但暂时无法解释的 schema slot
仍以 `connectable_state=unknown` 保留。旧 `doc_chunks` 索引及其模型已删除；Tool Card
agent 直接读取 canonical skills。

## Edge Decisions 与 Graph

正式产物 `results/edge_decisions.jsonl` 中每个 directed pair 最多一条标准 decision：

```json
{
  "schema_version": "tool_kg_edge_decision_v1",
  "pair_id": "...",
  "source_tool": "...",
  "target_tool": "...",
  "relation_status": "valid",
  "direct_transition": true,
  "edge_type": "generates_full_input_for",
  "edge_types": [{
    "type": "generates_full_input_for",
    "source_slot": "...",
    "target_slot_or_precondition": "...",
    "confidence": 0.86,
    "evidence_ids": ["..."]
  }],
  "satisfied_inputs": [],
  "unsatisfied_inputs": [],
  "negative_reason": null,
  "evidence": [],
  "rationale": "...",
  "confidence_raw": 0.86,
  "source_authority": "claude_adjudication",
  "eligible_for_sampling": true
}
```

`source_authority` 必须是 `claude_adjudication`；其他 authority 会被明确拒绝。`results/graph.jsonl` 只投影 `valid + eligible_for_sampling` 的边，不修改 relation/type/confidence；其中 `confidence` 直接复制 Claude decision 的 `confidence_raw`，主线没有第二个 calibration authority。旧 core/expanded/negative/uncertain、CSV、GraphML 不属于当前主线。

## Canonical Task

Tool-KG Stage3 先写 `results/tasks.jsonl`（`tool_kg_task_v1`）；Data-Pipe adapter 再输出 `kg_task_spec_v0.2`：

```json
{
  "task_id": "...",
  "task_type": "kg_sampled",
  "question": "...",
  "source": {"type": "tool_kg", "kg_run_id": "...", "sample_id": "..."},
  "toolchain": {"tools": [], "edges": [], "hops": 0},
  "expected_trajectory": {},
  "execution": {
    "allowed_tools": "all_molclaw",
    "must_follow_expected_trajectory": false,
    "leak_toolchain_to_agent": false
  },
  "evaluation": {"mode": "none"},
  "metadata": {}
}
```

每个 workflow transition 必须引用 canonical ToolKG `pair_id`。skills 可帮助理解工具，但不能独立创建关系。
Stage3 只读取 `graph.jsonl`、`edge_decisions.jsonl` 与 `tool_catalog.jsonl`，并按
`pair_id` join mapping/evidence；不读取 debug sidecar、legacy graph view 或 Claude
intermediate。采样使用 `question_sampling.yaml` 的 named profile，resolved values、
profile/config hash 与 prompt hash 写入 manifest。

## Raw Trace 与执行目录

Claude Code invocation 以 `--verbose --output-format stream-json` 运行；DeepSeek Harness invocation
保存未压缩、`packChunks=false` 的 canonical session JSONL。两种不可变 raw trajectory 都
保存在外层 rollout metadata 目录的 `attempts/attempt_NNNN/complete_session.jsonl`；选中的
attempt 再按字节复制到该 sample 的顶层 `complete_session.jsonl`。`question.json`、`prompt.txt`、
`run_meta.json`、selected-attempt manifest 等采集文件也只在外层目录。
`run_meta.json` 与 selected-attempt manifest 都记录 question、user prompt、system prompt、selected raw
session 和 source dataset 的 SHA-256；新采集样本进入 semantic 前必须全部互相匹配。DSH raw
事件使用 `assistant/message`、`tool/call`、`tool/result`、`turn/end`；cleaning parser 只在内存中
投影到既有事件模型，不重写 raw 文件。
任一 hash 或 selected-attempt 绑定缺失时同样 fail closed，不自动降级为旧采集格式。

每个 `complete_session.jsonl` 都有 `complete_session.pretty.json` 阅读副本。raw 中若混有 Claude CLI
runtime diagnostic，pretty 文件会将该行明确包装为 `raw_stream_diagnostic`，并保留原行号和文本；
raw JSONL 本身不被修改，且仍是唯一 raw authority。

Claude 启动时的 cwd 是 attempt 下的独立 `workdir/`。启动前其中只能有
`workdir-skills/molclaw-trajectory-execution/` 明确投放的运行材料；题目通过正常 user prompt
传入，collector 不向 cwd 写 question、prompt 或采集 sidecar。agent 运行期间产生的科学产物可在
该 cwd 中出现。

## Pre-clean Qwen3.5 Native Projection（审计中间态）

`qwen35_native_raw.jsonl` 将 immutable Claude raw events 直接投影为 Qwen3.5 的 structured
message 形状：thinking 进入 `assistant.reasoning_content`，同一 `message.id` 中的 `tool_use`
保持在同一 `assistant.tool_calls[]`，`tool_result` 按 ID 进入 `role=tool`，最终文本进入
`assistant.content`。

该产物的 schema 是 `claude_raw_qwen35_native_projection_v1`，明确标记
`training_source=false`，并将 skill/runtime filtering、path sanitation、observation compaction
和 reasoning cleaning 全部记录为未执行。它因此仍包含 teacher runtime 调用、原始本机路径和未压缩
observation，仅用于逐阶段审计，不是 semantic authority，也不是训练输入。正式下游始终从 raw
构建 semantic，不从这个 Qwen projection 反向解析。此阶段也不伪造 raw stream 中不存在的
deployment tool JSON schemas；完整 `tools` 只由最终 Qwen adapter 按 student visibility 注入。

每个主要 trajectory JSONL 同目录都有同名 `.pretty.json` companion。JSONL 保持一行一个 record，
供程序读取；pretty 文件是缩进后的 JSON array，只供人工检查，不进入 loader 或后续转换。

## Semantic Canonical（母数据）

`semantic_trajectories.jsonl` 是 SFT 和 ToolRL 的共同 authority：

```json
{
  "schema_version": "drug_agent_semantic_trajectory_v1",
  "id": "...",
  "user_task": "...",
  "events": [
    {
      "type": "assistant_decision",
      "source_message_id": "...",
      "reasoning": "...",
      "tool_calls": [{"name": "...", "arguments": {}, "source_tool_use_id": "..."}],
      "final_answer": null
    },
    {
      "type": "tool_observation",
      "name": "...",
      "source_tool_use_id": "...",
      "status": "success",
      "is_error": false,
      "content": "..."
    }
  ],
  "metadata": {"task_type": "kg", "source_session_sha256": "..."}
}
```

一个 Claude `message.id` 始终对应一个 assistant decision；同一 response 的并行 calls 保持在同一
`tool_calls[]`，结果按 `tool_use_id` 配对并按 call 顺序落盘。删除 teacher-runtime 交互时，
整组删除 decision 与 observation，绝不跨 observation 合并前后 decisions。

Semantic 正文只含 user task、assistant decisions、calls、observations 和 final response；Claude
system/runtime instructions 只可进入 audit，不进入正文。它不含 Qwen system prompt、tool catalog
或任何模型 wire-format 标签。

Raw → Semantic canonicalization 同时完成 Python 确定性清洗：去除 L2/L3、CLAUDE/system/runtime 和
collector sidecar 动作；把只读 mixed L1/L2 Bash 在原 decision 内投影成一个或多个 `Read`；压缩
base64/大数组/超长日志并保留下游实际复用的路径与标量证据。

路径按来源处理：attempt workdir 内文件写成普通 cwd-relative 路径；Claude L1 路径写成
`skills/L1_tools/...`；user task 与 MCP observation 引入的服务器绝对路径保持原值并在后续原样复用。
不生成 `resource://` 或 `<artifact:...>`。若 raw 确实执行过 `Bash pwd`，该 decision 保留，workdir
及其依赖路径统一映射到 trajectory-specific absolute root；Python 不事后伪造调用。raw mapping
只进入 audit。

正式 SFT 主线强制 LLM clean。它只返回 `semantic_reasoning_patch_v1`，允许清理 reasoning 中的 skill prose、连续
重复 thoughts，并在第一条 decision reasoning 前补充任务级 high-level plan。tool calls、arguments、
observations、final response、顺序和 provenance 必须通过 immutable check。provider/patch 失败不会
使 semantic sample 失效：母数据保留，失败项进入 `llm_pending.jsonl`；正式 cleaned view 只包含成功
patch 的样本。

## Qwen3.5 Structured SFT View

Qwen adapter 注入训练 system prompt 和 deployment-visible tool set，生成：

```json
{
  "schema_version": "drug_agent_qwen35_sft_v1",
  "id": "...",
  "messages": [
    {"role": "system", "content": "...", "step_loss_mask": 0},
    {"role": "user", "content": "...", "step_loss_mask": 0},
    {"role": "assistant", "reasoning_content": "...", "content": "", "tool_calls": [], "step_loss_mask": 1},
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "...", "step_loss_mask": 0},
    {"role": "assistant", "reasoning_content": "...", "content": "...", "step_loss_mask": 1}
  ],
  "tools": [{"type": "function", "function": {"name": "...", "parameters": {}}}]
}
```

`tools` 的 authority 是 student 在未来 deployment/rollout 时实际可见的 deployment manifest，不是
teacher runtime。默认 `all` 为每条样本提供 manifest 全部工具；
`trajectory-plus-distractors` 始终包含六个本地工具、trajectory 实际使用的全部 MolClaw 工具，并按
sample ID 确定性加入与已用 MolClaw 工具等量的 distractors。adapter 对缺失的已调用工具 fail closed。
选择策略与 manifest hash 写入 materialization manifest，semantic 不记录它们。source dataset 始终保存 structured messages；Qwen wire-format 仅由
`tokenizer.apply_chat_template(messages, tools=tools)` 在加载/验证时产生。Slime 使用
`--loss-mask-type qwen3_5 --tool-key tools`，只监督 assistant reasoning/action/final。

## Structured ToolRL View

ToolRL 每个 semantic assistant decision 派生一行 `drug_agent_qwen35_toolrl_decision_v8`，包含历史
`prompt`、同一 deployment-visible `tools`、结构化 `label.target_assistant`、
`target_tool_calls|target_final_answer` 和 provenance metadata。它不从 rendered text 反向解析 teacher
decision。

筛选后的 decision 按 `trajectory_index, decision_ordinal` 恢复原始顺序。允许 ordinal 缺号，也允许一个
trajectory 跨 rollout batch 边界；读取顺序是 `A1,A3,A6,B2,B5,...`，不能全局 shuffle 或按 selector
分数排序。Slime 要求固定 RBS，因此尾批由 data source 在 epoch 边界继续读取下一 epoch 的确定性前缀；
任何 selected decision 都不会因整除问题被永久删除。每个 decision 独立生成 4 个候选并形成自己的
GRPO 比较组，不与其他 decision 的候选混组。后续 decision 始终读取母数据中的真实历史，而不是此前
训练时临时生成的回答。

默认 selector 见 [NeMo 三步流程](NEMO_SELECTOR.md)。比较副本保留任务、当前完整决策和近期完整助手消息；
公共系统消息单独保存并校验版本，所有工具返回（含 skill 返回）不进入比较文本，
用 Qwen3-Embedding-8B 编码，再运行 NeMo 官方语义筛选。比较条件只从原生消息生成一次，工具决策可跨
任务类型比较，最终回答仍按任务类别区分。一批只接受一个完整工具目录版本。
比较副本不替换训练 prompt/tools/label；未入选不等于严格重复或坏数据。
没有每组配额、固定 20% 预算或最终回答最低名额。旧 0.6B/complete-linkage 方法仅作为独立历史实验保留。
训练长度使用独立绑定的原生 Qwen template/tokenizer 审计，不能用窗口长度代替；不截断工具参数或教师
回答，也不以固定 16K 阈值直接删除 decision。现有 32K/64K 分档仍检查完整教师回答，超长记录单列。

rollout/reward 使用当前 checkpoint 与当前 SGLang 版本实际支持的 native reasoning/tool parser。
parser 名称是 launcher/serving 配置，不写入数据 schema；正式启动前必须通过 tokenizer-rendered
target 的 native parser round-trip gate。Drug-Pipe 不提供旧 XML fallback parser。

## Online MolBench Evaluation

评测启动时调用唯一 MCP server `molclaw-scp` 的 `list_tools`，并将完整名称、description 和
JSON Schema 固化在当次 `tool_catalog.json`。其 hash 在 preflight 与 rollout worker 间必须
一致；旧工具名不会通过 alias 静默转换。模型可见 observation 中的服务器绝对路径必须通过
当前统一 path-sanitization contract 变成稳定匿名引用；数据层不硬编码其文本形式，raw mapping
只保存在 `artifact_audit.jsonl`。

一次正式评测目录为：

```text
slime-wd/outputs/slime_drug_agent_evals/<run_name>/
├── run_manifest.json
├── benchmark_manifest.json
├── tool_catalog.json
├── molbench_eval.jsonl
├── overlap_audit.jsonl
├── task_results/<task-id-hash>.json   # 每题完成后原子写入的 resume authority
├── partial_results.jsonl              # 已 checkpoint 题目的轻量状态/预测快照
├── progress.json                      # expected/checkpointed/remaining 计数
├── predictions.jsonl
├── traces.jsonl
├── metrics.json
├── failures.jsonl
├── artifact_audit.jsonl
└── workspaces/<task_id>__<sample_index>/
```

`task_results/` 在每题返回后立即写入，任务级 trace、prediction、status 和 artifact audit 不依赖
全量 evaluation logger 才能保存。同一 run 使用 `RESUME_EVAL=1` 时，preflight 必须验证 checkpoint、
输入文件、实时 tool catalog、L1 skills、模型拓扑及生成设置的联合 fingerprint；匹配的题目直接恢复，
不重新生成或调用工具。正式 `predictions/traces/metrics` 仍只在全部题目齐全后发布。

当前 held-out adapter 固定选择 MS-1 50、MS-2 33、MS-3 25 和 MO 78，共 186 题。
MS-2 的4个 exact normalized prompt overlap 单独进入 audit；MO 源数据缺少的41条 target
optimization 只记入 manifest。`metrics.json` 直接由外部 MolClaw 仓库现有 evaluator
产生，Drug-Pipe 不复制或重写指标公式。

未来训练数据针对新的 deployment-visible catalog 应从 semantic mother dataset 重新运行 adapter，
并由 manifest hash 显式审计。未知 name/schema 关系必须拒绝，不得通过 runtime alias 或文本迁移猜测。
