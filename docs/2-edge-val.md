ok.第一个项目代码的修改意见我已经记下了，接下来是tool-card的建立。 这里粘贴了一个现在代码产生的tool-card。现在的tool-card似乎也有完整的这个工具的输入输出标注？我的想法是tool-card中清晰地标注工具的输入输出是很关键的，这事关后续再产生边的时候： 设想这样一种情景，就是工具B的输入有两个关键文件B1和B2，这里B1恰是工具A的输出，但工具A不能提供B2.这种情景也是要有边的建立的。现在就是说LLM（agent）经常在判断一条边是否成立的时候认为A不能给B提供所需的全部参数而否定这条边，实际上是需要记录这个关系的。 请你从我这个提示上思考一下，继续给出这个点的handoff修改意见

你的提示非常关键。这里需要把 Tool-Card 和 Edge 判断的逻辑从 **“A 是否能让 B 独立完整执行”** 改成 **“A 是否能为 B 提供一个可连接的关键输入 / 前置条件，并显式记录 B 还缺什么”**。

当前 tool-card 已经有比较完整的 `inputs`、`outputs`、`preconditions`、`side_effects`、`typical_upstream_roles`、`typical_downstream_roles`、`negative_constraints` 等字段，例如 `chai1_predict` 里已经标注了 `seq`、`fasta_path` 等输入，以及 `output_dir`、`model_scores`、`best_model`、`model_cif_generation` 等输出 / 副作用产物。 但它还不够支持“部分输入满足”的边判断。

---

## 1. 当前问题的本质

现在 pairwise edge 判断里容易出现这个错误逻辑：

```text
A 只能提供 B 的一个必要输入；
B 还有其他必要输入 A 不能提供；
所以 A -> B 不成立。
```

但在科学工具链里，这个判断是错的。

更合理的逻辑应该是：

```text
A -> B 是否成立，不要求 A 独立提供 B 的全部输入。
只要 A 的输出能直接满足 B 的某个关键输入、前置条件或数据槽位，
并且这个连接不违反格式、语义、阶段和负约束，
就应该记录一条边。

同时，B 仍然缺失的其他输入必须被记录下来。
```

例如你说的情景：

```text
B requires: B1 + B2
A outputs: B1
A cannot provide: B2
```

这时不应该拒绝：

```text
A -> B
```

而应该记录：

```json
{
  "source": "A",
  "target": "B",
  "edge_valid": true,
  "coverage": "partial",
  "satisfied_target_inputs": ["B1"],
  "unsatisfied_required_inputs": ["B2"],
  "requires_additional_inputs": true
}
```

这条边的语义不是：

```text
A alone fully enables B.
```

而是：

```text
A contributes a required or useful input to B.
```

---

## 2. 需要修改的核心定义

建议把边的语义从原来的：

```text
A 的输出能直接作为 B 的输入或前置条件。
```

进一步精确定义为：

```text
A -> B is valid if at least one connectable output, side-effect artifact, or state produced by A can directly satisfy one semantically meaningful input slot or precondition slot of B, even if B still requires additional inputs from task context, user input, or other upstream tools.
```

中文就是：

> 只要 A 能为 B 提供至少一个语义上有意义、格式上可直接连接的输入槽位或前置条件槽位，就可以建立 A -> B 边；A 不需要独立满足 B 的全部调用需求，但必须显式记录 B 仍缺哪些输入。

这会避免 LLM 把“partial input provider”误判为 invalid edge。

---

## 3. Tool-card 需要新增的字段

当前 tool-card 有 `inputs` 和 `outputs`，但它们还偏“schema 参数表”。下一版需要补充更适合边判断的字段。

### 3.1 新增 `connectable_inputs`

这是从 `inputs` 和 `preconditions` 中整理出来的“可被上游工具满足的输入槽位”。

```json
{
  "connectable_inputs": [
    {
      "slot_id": "protein_structure_file",
      "source_fields": ["protein_path", "pdb_path"],
      "semantic_type": "protein_structure",
      "accepted_formats": ["pdb", "cif"],
      "requirement_status": "required",
      "parameter_kind": "data",
      "can_be_user_provided": true,
      "can_be_upstream_generated": true,
      "description": "Protein structure file required for downstream analysis."
    }
  ]
}
```

重点是区分：

```text
data input：真正需要上游工具提供的数据
config input：用户配置参数，如 samples, mode
control input：执行控制参数，如 dry_run
```

后续建边时，主要关注 `parameter_kind = data` 或 `precondition` 的槽位，而不是把 `samples`、`dry_run` 这种配置参数当成必须由上游工具提供的东西。

---

### 3.2 新增 `connectable_outputs`

这是从 `outputs` 和 `side_effects` 中整理出来的“可以喂给下游工具的输出槽位”。

当前 `chai1_predict` 的 `best_model.cif_path`、`model_scores[*].cif_path`、`model_cif_generation` 都是重要连接点，但它们现在有的藏在 nested object 里，有的藏在 side effects 里。

建议展开为：

```json
{
  "connectable_outputs": [
    {
      "slot_id": "best_model_cif",
      "source_field": "best_model.cif_path",
      "semantic_type": "predicted_protein_structure_file",
      "format": "mmCIF",
      "cardinality": "single",
      "downstream_connectable": true,
      "description": "CIF file path of the top-ranked predicted model."
    },
    {
      "slot_id": "all_model_cifs",
      "source_field": "model_scores[*].cif_path",
      "semantic_type": "predicted_protein_structure_file",
      "format": "mmCIF",
      "cardinality": "list",
      "downstream_connectable": true
    },
    {
      "slot_id": "structure_confidence_scores",
      "source_field": "model_scores[*].scores",
      "semantic_type": "structure_confidence_score",
      "format": "dict",
      "warning": "Do not interpret ipTM as binding affinity."
    }
  ]
}
```

这样 pairwise edge 判断时，LLM 不需要自己从复杂 JSON 里猜哪个输出能连接下游。

---

### 3.3 新增 `input_requirement_sets`

很多工具不是简单“所有 required 参数都必须由上游提供”，而是存在模式、条件和替代输入。

例如 `chai1_predict`：

```text
mode='sequence' 时需要 seq
mode='fasta' 时需要 fasta_path
mode='info' 时不做实际预测
samples 有默认值
dry_run 是控制参数
```

所以应该显式建模：

```json
{
  "input_requirement_sets": [
    {
      "set_id": "sequence_mode_execution",
      "condition": "mode == 'sequence' or mode omitted",
      "required_slots": ["protein_sequence"],
      "optional_slots": ["chain_identifier", "sample_count"],
      "defaulted_slots": ["mode", "samples", "dry_run"],
      "execution_meaning": "actual_prediction"
    },
    {
      "set_id": "fasta_mode_execution",
      "condition": "mode == 'fasta'",
      "required_slots": ["fasta_file_path"],
      "optional_slots": ["sample_count"],
      "defaulted_slots": ["samples", "dry_run"],
      "execution_meaning": "actual_prediction"
    }
  ]
}
```

这样 LLM 在判断边时不会错误地认为：

```text
A 没有提供 mode / samples / dry_run，所以 A -> chai1_predict 不成立。
```

这些参数可以由默认值或用户上下文提供，不应该作为拒绝边的理由。

---

## 4. Edge schema 也要改

光改 tool-card 不够。候选边 / 最终边也需要记录“输入覆盖情况”。

建议每条 edge 增加：

```json
{
  "source_tool": "A",
  "target_tool": "B",
  "edge_type": "generates_input_for",
  "coverage_level": "partial",
  "satisfied_mappings": [
    {
      "source_output_slot": "A.output.B1",
      "target_input_slot": "B.input.B1",
      "semantic_match": "exact",
      "format_match": "exact",
      "evidence": "..."
    }
  ],
  "unsatisfied_required_inputs": [
    {
      "target_input_slot": "B.input.B2",
      "reason": "not produced by source tool",
      "can_be_user_provided": true,
      "can_be_satisfied_by_other_upstream_tool": true
    }
  ],
  "requires_additional_context": true,
  "single_source_executable": false,
  "edge_validity": "valid_partial"
}
```

这里最关键的是新增两个概念：

```text
coverage_level
single_source_executable
```

### `coverage_level`

可以取：

```text
full
partial
context_dependent
none
```

含义：

```text
full:
  A 提供了 B 的所有关键 data inputs，B 只剩默认参数 / config 参数。

partial:
  A 提供了 B 的一个或多个关键 data inputs，但 B 还需要其他 data inputs。

context_dependent:
  A 提供了关键输入，但还需要用户上下文或任务初始条件。

none:
  A 没有提供任何 meaningful input，不应成边。
```

### `single_source_executable`

表示：

```text
仅凭 A 的输出 + B 默认参数，是否足以调用 B。
```

但注意：这不应该决定边是否成立。

```text
single_source_executable = false
```

仍然可以是有效边。

---

## 5. LLM pairwise prompt 必须明确这一点

当前 LLM 可能否定边，是因为 prompt 很可能让它判断：

```text
Can A provide all required inputs for B?
```

这个问题本身就错了。

应该改成：

```text
Does A provide at least one meaningful connectable input or precondition for B?
Do not reject the edge merely because A cannot provide all required inputs of B.
Instead, record which target inputs are satisfied and which remain unsatisfied.
```

建议在 prompt 里加入硬规则：

```text
A valid edge does NOT require the source tool to fully instantiate the target tool call.
If the source tool satisfies one required or useful data/precondition slot of the target tool,
mark the edge as valid_partial unless there is an explicit semantic, format, or workflow incompatibility.
```

同时要求 LLM 输出：

```json
{
  "decision": "valid_full | valid_partial | invalid | uncertain",
  "do_not_reject_due_to_missing_other_inputs": true,
  "satisfied_target_inputs": [],
  "unsatisfied_target_inputs": [],
  "required_external_context": [],
  "reason_for_invalid_if_any": ""
}
```

---

## 6. 负边判断也要更严格

以前 LLM 可能因为：

```text
A 不能提供 B 的全部输入
```

就标成 negative edge。

下一版要明确：

```text
missing_other_required_inputs ≠ negative edge
```

真正的 negative edge 应该是：

```text
1. A 没有提供 B 的任何 meaningful input / precondition。
2. A 的输出和 B 的输入语义类型不匹配。
3. A 的输出格式不能被 B 接受，且没有直接转换关系。
4. A 和 B 之间只是 broad collaboration，不是直接输入依赖。
5. 有明确 negative constraint。
6. A 的输出是 B 不应使用的值，例如把 ipTM 当 binding affinity。
```

所以要把：

```text
partial coverage
```

和：

```text
invalid edge
```

彻底分开。

---

## 7. 这条修改意见可以这样 handoff 给代码实现

下面这段可以直接放进项目修改意见里。

```text
Tool-card / edge-construction handoff modification:

The current edge adjudication appears to treat a source-to-target edge as valid only when the source tool can provide all required inputs for the target tool. This is too strict and causes false negatives in multi-input scientific workflows. In MolClaw Tool-KG, a directed edge A -> B should mean that A can provide at least one semantically meaningful connectable input, artifact, or precondition for B. A does not need to fully instantiate B's complete invocation. Missing target inputs should be recorded as unsatisfied requirements, not used as a rejection reason.

Please update the tool-card schema and pairwise edge adjudication accordingly:

1. Extend each tool-card with `connectable_inputs` and `connectable_outputs`.
   - `connectable_inputs` should be derived from `inputs` and `preconditions`.
   - `connectable_outputs` should be derived from `outputs` and `side_effects`.
   - Nested outputs such as `best_model.cif_path` or `model_scores[*].cif_path` must be flattened into explicit connectable output ports.
   - Each connectable port should include semantic_type, accepted/produced format, cardinality, requirement_status, parameter_kind, and provenance.

2. Add `input_requirement_sets` to each tool-card.
   - This should represent alternative or conditional execution modes.
   - Example: a prediction tool may require either `seq` in sequence mode or `fasta_path` in fasta mode.
   - Config/control/defaulted parameters such as `mode`, `samples`, or `dry_run` should not be treated as upstream-generated data requirements unless the documentation explicitly says so.

3. Modify pairwise LLM adjudication.
   - The LLM should judge whether source tool A satisfies at least one target input/precondition slot of B.
   - It must not reject A -> B simply because A cannot provide all required inputs of B.
   - It must output satisfied input mappings, unsatisfied required inputs, whether additional context/upstream tools are needed, and whether the edge is full or partial.

4. Modify edge schema.
   Add fields:
   - `coverage_level`: full | partial | context_dependent | none
   - `satisfied_mappings`
   - `unsatisfied_required_inputs`
   - `requires_additional_context`
   - `single_source_executable`
   - `edge_validity`: valid_full | valid_partial | invalid | uncertain

5. Modify negative-edge logic.
   - Missing other required inputs is not a negative-edge reason.
   - A negative edge should require semantic incompatibility, format incompatibility, wrong direction, broad-only collaboration, explicit negative constraint, or no meaningful satisfied target slot.

6. Modify confidence scoring.
   - Full coverage edges can receive a bonus.
   - Partial coverage edges should remain valid but receive a lower coverage score.
   - Edges with high-value satisfied slots, such as receptor structure, ligand file, docking pose, trajectory, or candidate set, should not be over-penalized merely because other inputs come from user context or separate upstream tools.
```

---

## 8. 推荐增加一个边类型或边属性

我不建议为这个问题专门增加太多 edge type。更好的方式是保留原来的 edge type，然后增加 coverage 属性。

例如：

```json
{
  "edge_type": "generates_input_for",
  "coverage_level": "partial"
}
```

而不是新增很多类型：

```text
partially_generates_input_for
fully_generates_input_for
contextually_generates_input_for
```

否则 edge ontology 会膨胀。

不过可以增加一个统一属性：

```text
input_coverage_semantics
```

取值：

```text
single_source_enables_target
source_contributes_required_input
source_contributes_optional_input
source_satisfies_precondition
source_requires_context_to_enable_target
```

这比增加新 edge type 更清晰。

---

## 9. 对当前 chai1_predict tool-card 的具体启发

以你贴的 `chai1_predict` 为例，它的输出里有：

```text
best_model.cif_path
model_scores[*].cif_path
output_dir
model_scores[*].scores
```

但这些现在没有被完全展开成 `connectable_outputs`。

如果下游 B 是一个需要：

```text
B1 = protein structure file
B2 = ligand file
```

的 docking 或 interaction 工具，那么：

```text
chai1_predict
```

只能提供：

```text
B1 = predicted structure / CIF file
```

不能提供：

```text
B2 = ligand file
```

但这条边仍应存在：

```text
chai1_predict -> B
```

边上记录：

```json
{
  "coverage_level": "partial",
  "satisfied_mappings": [
    {
      "source_output_slot": "best_model.cif_path",
      "target_input_slot": "protein_structure_file"
    }
  ],
  "unsatisfied_required_inputs": [
    "ligand_file"
  ],
  "single_source_executable": false
}
```

这样图才能反映真实科学 workflow：一个工具链通常是多源汇合，而不是严格单输入单输出线性链。

---

一句话 handoff：

> **Tool-card 和 edge adjudication 必须从“source 是否完全满足 target 调用”改成“source 是否满足 target 的至少一个关键 connectable slot”。多输入工具中的未满足输入应被记录为 `unsatisfied_required_inputs`，而不是作为否定边的理由。最终边应区分 `valid_full`、`valid_partial`、`context_dependent`、`invalid`，并显式保存 satisfied mappings 和 missing requirements。**
