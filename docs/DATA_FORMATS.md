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

## Raw Trace

每次 Claude CLI invocation 都以 `--verbose --output-format stream-json` 运行，并把
stdout 与 stderr 按 `2>&1` 语义直接写入独立的不可变归档：

```text
<workdir>/attempts/attempt_0001/complete_session.jsonl
<workdir>/attempts/attempt_0002/complete_session.jsonl
...
```

执行层随后把当前流程最终采用的 attempt 按字节复制到稳定兼容路径
`<workdir>/complete_session.jsonl`，并校验两者 SHA256 一致。`run_meta.json` 或对应
trace 记录所有 attempt 的 index、路径、return code、字节数、SHA256 与最终选择；
MCP-ready retry 不覆盖之前的 attempt。timeout、非零退出和 CLI stderr 留在原始文件，
runner 自己的 timeout/MCP-ready 诊断只写 metadata，禁止追加到 raw stream。

下游仍只读取顶层 `complete_session.jsonl`。空文件或完全没有可解析 stream-json event
的文件标记为 `raw_session_invalid`；Data-Pipe rollout/Tool-KG 视为执行失败，LLM clean
回退 Python draft。该契约只适用于新运行，不为历史目录伪造 session。

执行层另保存 `run_config.json`/run meta、task input snapshot 和 artifacts。raw event
顺序、observation、工具调用与科学结论不得由后处理修改。

## Canonical ReAct

Step 1 的内部接口为 `python_drafts.jsonl`。其中每条已经通过确定性构造和验证，状态仅为
`python_valid`，不是最终 accepted。Step 2 完成 restricted LLM patch 和 final gate 后，训练
唯一接口才是 `react_trajectories.jsonl`：

```json
{
  "schema_version": "drug_agent_sft_react_json_v1",
  "id": "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<thought>...</thought><tool_call>...</tool_call>"},
    {"role": "user", "content": "<observation tool_name=\"...\">...</observation>"},
    {"role": "assistant", "content": "<thought>final analysis</thought><final_answer>...</final_answer>"}
  ]
}
```

`<final_answer>` 是 Drug-Pipe 主动选择的 canonical terminal-decision 表示，不是 Claude Code
stream-json 原始协议。最终 reasoning 与结构化 final 必须属于同一个 assistant generation；
没有 user/observation 分隔的连续 assistant turn 非法。Final 的 task-specific result 和 evidence
由 Python authority 构造，`summary` 可选且不得复制完整 thought。

训练文件不含 source path、return code、ground truth、benchmark metrics、evaluator
validity 或 rejection reasons。rich final 只能来自 agent prediction、raw assistant final
和真实 observation evidence。reference labels 与 evaluator 输出按相同 `id` 写入
`curation_audit.jsonl`。默认最终文件是：

```text
react_trajectories.jsonl
curation_audit.jsonl
rejected.jsonl
run_manifest.json
```

`task_answer_valid` 只表示 `parsed_answer.parse_error` 是否为空。Evaluator 仍是 benchmark
metrics 和 `aggregate_eligible` 的唯一 owner，但其 invalid SMILES、重复、长度或空预测等
finding 只进入 audit，不参与清洗准入。`execution_valid`、`task_answer_valid` 和
`training_trace_valid` 都由 `python_clean` 产生；final/observation consistency 由
`invariants.py` 只读记录。LLM clean 内部的 final acceptance gate 只把这三个 gate 投影为
accepted/rejected，不存在 quarantine 或第二个准入 authority。

LLM clean 不再消费 Python 生成的逐段 repair target。它检查每条 Python-valid trajectory
的全部 thought 与 final summary；空 patch 记为 `not_required`，失败或不安全 patch 回退
Python draft。残留的 L2/L3 或 teacher-sidecar prose 只写 audit，不改变 A/B/C 准入。

VS 的 QuickVina 排序以每个 SMILES 在整条轨迹中的成功
`docking_affinity_value` 最小值（最负、最佳 pose）为准，不以 pocket/protein context
一致作为硬门槛。已有分数的分子先按最佳分数升序排列；缺少成功分数的分子保持原相对
顺序并置于末尾；重复 SMILES 保留。context、全部重试分数和选中的最佳分数写入 audit。

Canonical artifact token 仅允许 `<artifact:[A-Za-z0-9._/-]+>`。Observation compaction
不得截断 token；final 中不被保留 call argument 或 observation 支持的 artifact ref 会在
Python 结构化时替换为中性 unavailable-path 文本。

默认 canonical ReAct 保留 MolClaw 与受支持的本地工具
`Read/Write/Edit/Bash/Grep/Glob`；其中 `Read/Grep/Glob` 可只读访问 L1 tool-level skill，Bash
受任务 workspace 限制。Teacher runtime sidecar（如 `question.json`、`run_meta.json`、
`complete_session.jsonl`、`CLAUDE.md`）和非 L1 skills catalog 访问会成对移除，避免
benchmark label 与层级脚手架通过 observation 回流。显式 `--only-molclaw-tool` 会成对
移除所有非 MolClaw call 和 observation，但不改变 A/B/C gate 或 record ID。

## Shared Decision State

`iter_react_decisions(messages)` 对每个 assistant decision 产生：

```json
{
  "assistant_index": 2,
  "state_messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "target_assistant": {"role": "assistant", "content": "..."},
  "decision_type": "tool_call",
  "tool_calls": [],
  "final_answer": null
}
```

`state_messages` 只含当前 assistant 之前的 `role/content/name`。当前 target 和后续 observation 永不进入 state。

## Training Views

SFT 直接消费 canonical ReAct `messages`，使用 `--loss-mask-type qwen3_5` 实现 assistant-only loss。

ToolRL v2 每行包含 `decision_type=tool_call|final_answer`、`prompt`、`label`、`metadata`、
`target_assistant`，以及对应的 `target_tool_calls` 或 `target_final_answer`。默认
`TOOLRL_REWARD_MODE=official`；`molclaw` 是同一 trainer 的领域适配模式。

GAD 每行包含 `prompt/state_messages`、`teacher_response`、`label`、`metadata`。GAD 保留
tool-call 与 final-answer decisions；`GAD_REWARD_MODE=pure|rule|hybrid`，默认 pure。

## Online MolBench Evaluation

评测启动时调用唯一 MCP server `molclaw-scp` 的 `list_tools`，并将完整名称、description 和
JSON Schema 固化在当次 `tool_catalog.json`。其 hash 在 preflight 与 rollout worker 间必须
一致；旧工具名不会通过 alias 静默转换。模型可见 observation 中的服务器绝对路径会变成
稳定 `<artifact:namespace/name>`，只有 `artifact_audit.jsonl` 保存 raw path 映射。

一次正式评测目录为：

```text
slime-wd/outputs/slime_drug_agent_evals/<run_name>/
├── run_manifest.json
├── benchmark_manifest.json
├── tool_catalog.json
├── molbench_eval.jsonl
├── overlap_audit.jsonl
├── predictions.jsonl
├── traces.jsonl
├── metrics.json
├── failures.jsonl
├── artifact_audit.jsonl
└── workspaces/<task_id>__<sample_index>/
```

当前 held-out adapter 固定选择 MS-1 50、MS-2 33、MS-3 25 和 MO 78，共 186 题。
MS-2 的4个 exact normalized prompt overlap 单独进入 audit；MO 源数据缺少的41条 target
optimization 只记入 manifest。`metrics.json` 直接由外部 MolClaw 仓库现有 evaluator
产生，Drug-Pipe 不复制或重写指标公式。

未来训练数据针对实时 catalog 的确定性迁移输出 canonical ReAct、ToolRL、GAD、format
examples、逐条 migration audit/rejected sidecar 和 `derived_data_manifest.json`。迁移只允许
结构化且可验证的 name/argument/schema 适配；未知等价关系整条拒绝，不进入 runtime alias。
