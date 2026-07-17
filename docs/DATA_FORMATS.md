# Data Formats

下列字段只列稳定边界；debug、provenance 和派生统计不构成新的 authority。

## Tool Catalog

正式产物 `results/tool_catalog.jsonl` 每行一个工具；run 根目录的 `tool_cards.jsonl` 是可恢复的 Stage1 中间状态。

```json
{
  "tool_id": "tool_name",
  "description_summary": "...",
  "primary_stage": "...",
  "connectable_inputs": [{"name": "...", "raw_type": "...", "required": true}],
  "connectable_outputs": [{"name": "...", "raw_type": "..."}],
  "preconditions": []
}
```

MCP schema 字段为确定性事实；skills 语义由 tool-card agent 摘要。`doc_chunks` 只是 tool-card 构建缓存。

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

训练文件不含 source path、return code、metrics 或 rejection reasons。它们按相同 `id` 写入 `curation_audit.jsonl`。默认最终文件是：

```text
react_trajectories.jsonl
curation_audit.jsonl
rejected.jsonl
quarantine.jsonl          # 仅有内容时
curation_summary.json
```

`task_answer_valid` 只来自 evaluator；`execution_valid`/`training_trace_valid` 来自 curator；只有 final acceptance gate 决定 accepted/rejected/quarantine。聚合器只去重和汇总。

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
