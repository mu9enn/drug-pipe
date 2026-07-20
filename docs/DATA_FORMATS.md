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
仍以 `connectable_state=unknown` 保留。`doc_chunks` 不属于默认构建依赖，只能显式用于
debug/index。

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

`legacy_scored_supplement` 默认 `eligible_for_sampling=false`。`results/graph.jsonl` 只投影 `valid + eligible_for_sampling` 的边，不修改 relation/type/confidence。旧 core/expanded/negative/uncertain、CSV、GraphML 都是按需 compatibility export，不是默认正式产物。

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
intermediate。采样使用 `question_sampling_v2.yaml` 的 named profile，resolved values、
profile/config hash 与 prompt hash 写入 manifest。

## Raw Trace

执行层保存不可改写的 `complete_session.jsonl`、`run_config.json`/run meta、task input snapshot 和 artifacts。raw event 顺序、observation、工具调用与科学结论不得由后处理修改。

## Canonical ReAct

训练唯一接口为 `react_trajectories.jsonl`：

```json
{
  "schema_version": "drug_agent_sft_react_json_v1",
  "id": "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<thought>...</thought><tool_call>...</tool_call>"},
    {"role": "user", "content": "<observation tool_name=\"...\">...</observation>"},
    {"role": "assistant", "content": "<final_answer>...</final_answer>"}
  ]
}
```

训练文件不含 source path、return code、ground truth、benchmark metrics、evaluator
validity 或 rejection reasons。rich final 只能来自 agent prediction、raw assistant final
和真实 observation evidence。reference labels 与 evaluator 输出按相同 `id` 写入
`curation_audit.jsonl`。默认最终文件是：

```text
react_trajectories.jsonl
curation_audit.jsonl
rejected.jsonl
quarantine.jsonl          # 仅有内容时
curation_summary.json
```

`task_answer_valid` 只来自 evaluator；KG/E2E evaluator 仅接收 prediction、parse error
与 task contract，不接收 raw session、return code、timeout 或 tool/observation counts。
`execution_valid`/`training_trace_valid` 来自 curator；final/observation consistency 由 hard
clean 报告；只有 final acceptance gate 决定 accepted/rejected/quarantine。聚合器只去重和汇总。

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

ToolRL 每行包含 `prompt`、`label`、`metadata`、`target_assistant`、`target_tool_calls`；reward 只对生成文本与 reference calls 做 format/schema/reference match。

GAD 每行包含 `prompt/state_messages`、`teacher_response`、`label`、`metadata`。GAD 保留 tool-call 与 final-answer decisions，并拥有 negative cache、discriminator 与 reward 组合。
