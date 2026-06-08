ok, third. 附件是一个agnet在判别边的时候的一个完整的stream-json记录。 实际上，(nanobot) sunxiangyu@PJNL24107C0098:~/sunxiangyu/molclaw-kg$ ls <molclaw-kg-root>/runs/run_20260518_190716/cc_workdir/analyze_mmpbsa_with_pepinvent_peptide_sampling_by_peptide .claude CLAUDE.md doc_context.jsonl prompt.txt system_prompt_FULL.md complete_session.jsonl pair_payload.json stage_taxonomy.json task_context.json 这里.claude CLAUDE.md system_prompt_FULL.md stage_taxonomy.json 这四项是我要求拷贝到这每一个workdir里的， 其他的doc_context.jsonl pair_payload.json task_context.json： 这 3 个文件是在 cc_workdir 里给 Claude Code Agent 提供“本地任务上下文”的。 doc_context.jsonl 存什么：与当前任务相关的文档分块（每行一个 chunk），字段包括 chunk_id/doc_id/path/heading_path/text。 在哪一步生成： tool-card 阶段生成（stage1 的 tool_cards 步） pairwise adjudication 阶段生成（stage2 的 adjudicate 步） 代码位置： tool_card_builder.py (line 253) pairwise_runner.py (line 254) pair_payload.json 存什么：只在 pairwise 场景有，包含该无序工具对的一次双向判定 payload：pair_key、pair_meta(A_TO_B/B_TO_A)、tool_A_card、tool_B_card、doc_context、规则配置等。 在哪一步生成：pairwise adjudication 的每个 group 调用前写入。 代码位置： pairwise_runner.py (line 241) 调用点 pairwise_runner.py (line 349) task_context.json 存什么：轻量“索引/约束”信息，告诉 agent 当前任务关键参数和本地文件名。 tool-card 下：tool_id/fixed_primary_stage/allowed_stages + 各输入文件名 pairwise 下：pair_key/template_version + payload/taxonomy/doc 文件名 在哪一步生成： tool-card 阶段：_prepare_toolcard_workdir pairwise 阶段：_prepare_pair_workdir 代码位置： tool_card_builder.py (line 241) pairwise_runner.py (line 244) 补充：pair_payload.json 不会在 tool-card workdir 出现；另外两个文件在两种 workdir 都会有，但内容结构不同。 导致了现在的agent在对话的时候没有深入阅读skills而是直接阅读了这些“doc_context.jsonl pair_payload.json task_context.json”文件。 实际上这三个文件也是从“.claude CLAUDE.md system_prompt_FULL.md stage_taxonomy.json”的文档里产生的。也就是说现在项目里存在两个意义重叠的两套东西并存着，这是比较浪费的。 我的偏向是现在的这个run流程是不需要这些“doc_context.jsonl pair_payload.json task_context.json”文件的，让claude code直接在自己的框架里去读.claude下的skills等可能会效果更好？如果要删掉“doc_context.jsonl pair_payload.json task_context.json”的话不仅要去掉这三个文件在agent判断边的时候的参与，就连这三个文件的生成逻辑也不需要保留。 我还是比较拿不准主意，你有什么建议


我的建议：**不要把三类文件全部简单删除；但要把它们从“证据内容载体”降级为“任务导航/索引文件”。**

现在的问题不是“有本地上下文文件”本身，而是：

> `doc_context.jsonl` / `pair_payload.json` / `task_context.json` 里装了太多已经加工过的内容，导致 Claude Code Agent 选择读取这些派生材料，而不是回到 `.claude/skills_full` 里的原始 skills 证据。

你给的 `complete_session` 里也能看到这一点：agent 一开始就读 `pair_payload.json`、`stage_taxonomy.json`、`task_context.json`、`doc_context.jsonl`，而不是先系统性搜索 `.claude` / `skills_full`。这会让整个 adjudication 变成“读摘要判边”，而不是“读原始 skills 判边”。

---

## 1. 先判断你的想法：是否应该删掉这三个文件？

我的判断是：

```text
doc_context.jsonl：建议删除或彻底降级，不再作为 agent 的主要阅读材料。
pair_payload.json：不建议保留现在这种大 payload，应改成极简 pair_spec.json。
task_context.json：建议保留，但只作为轻量任务清单。
```

也就是说，我不建议保留当前三件套；但也不建议彻底没有任何任务文件。

最优方案是：

```text
删除：doc_context.jsonl
替换：pair_payload.json -> pair_spec.json
保留并缩小：task_context.json
```

---

# 2. 为什么不能简单全删？

因为 pairwise adjudication 是批处理任务。每个 workdir 必须明确告诉 agent：

```text
这次判断哪两个工具？
方向是什么？
输出写到哪里？
使用哪个 taxonomy？
使用哪个输出 schema？
必须读哪些 canonical source？
```

如果连 `task_context.json` 都删掉，只靠 prompt 和 `.claude`，会有几个风险：

1. agent 可能搞错当前 pair；
2. 输出文件路径不稳定；
3. 不同 run 之间不可复现；
4. 失败时不好追踪；
5. 后续聚合器不知道这个结果对应哪个 pair；
6. prompt 太长，容易把任务约束和证据阅读混在一起。

所以，**最小任务规格文件是需要的**。但它只能告诉 agent “任务是什么”，不能替 agent “总结证据是什么”。

---

# 3. 现在三类文件的问题

## 3.1 `doc_context.jsonl` 的问题最大

你描述它存的是相关文档分块，包括：

```text
chunk_id / doc_id / path / heading_path / text
```

这看似是 RAG，但在 Claude Code 场景下会产生副作用：

```text
agent 读 doc_context.jsonl
→ 认为这些 chunk 已经足够
→ 不再去 .claude/skills_full 里搜索
→ 错过未被检索进来的上下文
→ 被 chunk selection bias 影响
```

这和我们前面说的“高质量 KG 需要 evidence-grounded”其实是冲突的。证据应该来自原始 skills，而不是来自一个被预筛选过的 chunk 文件。

尤其是当前项目里，核心文档源本来就是 `skills_full/{L1_tools,L2_workflows,L3_methodology}`，它们分别承担工具级、workflow 级和方法论级证据来源。

所以：

> `doc_context.jsonl` 不应该再作为 pairwise agent 的阅读入口。

可以完全删除它的生成逻辑，或者改成 **不含正文 text 的 evidence locator**。

---

## 3.2 `pair_payload.json` 太重，容易变成“二手世界模型”

你说它包含：

```text
pair_key
pair_meta(A_TO_B/B_TO_A)
tool_A_card
tool_B_card
doc_context
规则配置
```

这个文件现在的问题是：它把所有东西都塞进一个 payload 里，导致 agent 会把它当成最权威上下文。

尤其是 `tool_A_card` / `tool_B_card` 本身已经是生成结果，而不是原始证据。当前 tool-card 里确实已经包含 inputs、outputs、preconditions、side_effects、typical roles、negative constraints、provenance_refs、evidence_refs 等信息，例如 `chai1_predict` 的 outputs 里有 `output_dir`、`model_scores`、`best_model`，side effects 里有 `model_cif_generation`，并且 provenance 指向 L1/L2 文档和 snapshot。

但这类 tool-card 应该是：

```text
辅助结构化视图
```

不是：

```text
替代原始 skills 的唯一证据源
```

所以 `pair_payload.json` 不应该内联完整 tool-card 和 doc_context。否则 agent 会自然地“读 payload -> 判边”，而不是“读 canonical source -> 判边”。

---

## 3.3 `task_context.json` 可以保留，但要极简

`task_context.json` 的定位应该是：

```text
告诉 agent 当前任务是什么，以及哪些文件是 canonical sources。
```

而不是：

```text
把证据提前喂给 agent。
```

所以它可以保留，但字段应很少。

---

# 4. 推荐的新 workdir 文件结构

建议每个 pairwise workdir 里保留：

```text
.claude/
CLAUDE.md
system_prompt_FULL.md
stage_taxonomy.json

task_context.json
pair_spec.json
output_schema.json
```

删除：

```text
doc_context.jsonl
pair_payload.json
```

可选保留，但不放正文：

```text
source_manifest.json
```

---

## 4.1 新版 `task_context.json`

只放任务导航信息：

```json
{
  "task_type": "pairwise_edge_adjudication",
  "pair_key": "pairgroup::analyze_mmpbsa__AND__pepinvent_peptide_sampling_by_peptide",
  "directions": ["A_TO_B", "B_TO_A"],
  "pair_spec_file": "pair_spec.json",
  "output_schema_file": "output_schema.json",
  "taxonomy_file": "stage_taxonomy.json",
  "canonical_skill_root": ".claude/skills",
  "required_behavior": [
    "read canonical skills before judging",
    "do not rely on generated summaries as evidence",
    "cite source skill paths and headings in final JSON"
  ]
}
```

这里没有 doc chunk，也没有 tool-card 正文。

---

## 4.2 新版 `pair_spec.json`

只放 pair 定义，不放证据正文：

```json
{
  "pair_key": "pairgroup::analyze_mmpbsa__AND__pepinvent_peptide_sampling_by_peptide",
  "tools": {
    "A": {
      "tool_id": "analyze_mmpbsa",
      "stage": "free_energy_result_analysis",
      "canonical_skill_globs": [
        ".claude/skills/L1_tools/*mmpbsa*/SKILL.md",
        ".claude/skills/L2_workflows/*mmpbsa*.md"
      ],
      "tool_card_path": "../../artifacts/tool_cards/analyze_mmpbsa.json"
    },
    "B": {
      "tool_id": "pepinvent_peptide_sampling_by_peptide",
      "stage": "ligand_de_novo_or_template_generation",
      "canonical_skill_globs": [
        ".claude/skills/L1_tools/*pepinvent*/SKILL.md",
        ".claude/skills/L2_workflows/*peptide*.md"
      ],
      "tool_card_path": "../../artifacts/tool_cards/pepinvent_peptide_sampling_by_peptide.json"
    }
  },
  "adjudication_goal": "Judge whether each ordered direction has a direct typed edge, partial edge, negative edge, alternative relation, or uncertain relation.",
  "must_not_do": [
    "Do not decide based only on tool-card summaries.",
    "Do not treat missing unrelated required inputs as invalid edge reason.",
    "Do not hallucinate evidence not found in canonical skills."
  ]
}
```

重点：这里只给路径，不内联 tool-card 内容。agent 可以读 tool-card，但必须回读 skills。

---

## 4.3 新增 `source_manifest.json`，替代 `doc_context.jsonl`

如果你担心完全没有索引会让 agent 找不到文档，可以保留一个不含正文的 manifest：

```json
{
  "candidate_sources": [
    {
      "path": ".claude/skills/L1_tools/molclaw-protein-ligand-mmpbsa/SKILL.md",
      "reason": "source tool L1 skill"
    },
    {
      "path": ".claude/skills/L1_tools/molclaw-pepinvent/SKILL.md",
      "reason": "target tool L1 skill"
    },
    {
      "path": ".claude/skills/L2_workflows/09-peptide-protein-binder-design.md",
      "reason": "possible workflow-level relation"
    }
  ]
}
```

它只告诉 agent **去哪里读**，不把 chunk text 提前抽出来。这样可以减少 RAG bias。

---

# 5. 是否让 Claude Code 直接读 `.claude` 下的 skills？

是，应该这样做。

但要注意，不是“完全自由地读”，而是：

```text
给它 pair_spec 规定当前 pair；
给它 source_manifest 指向候选文档；
要求它用 Glob/Grep/Read 去 canonical skills 里找证据；
最终输出必须包含 evidence_refs。
```

这样比现在更好，因为 `.claude/skills_full` 本来就是核心证据源。交接报告中也明确，L1 是工具级说明，L2 是 workflow 级说明，L3 是方法论原则，这些都是边与约束的证据来源。

同时，`stage_taxonomy.json` 可以继续作为固定约束，因为它本来就说明 tool-card agent 必须从 taxonomy 中选择 stage、不能发明新 stage。

---

# 6. 新的 agent 读取顺序应该强制规定

现在 agent 的自然读取顺序是：

```text
pair_payload.json
stage_taxonomy.json
task_context.json
doc_context.jsonl
```

这会强化“读派生上下文”的行为。

应该改成：

```text
1. Read task_context.json
2. Read pair_spec.json
3. Read stage_taxonomy.json
4. Glob/Grep canonical_skill_root for both tools
5. Read relevant L1 skill files
6. Search L2 workflow files for both tool names / aliases / stage terms
7. Search L3 methodology only if relation type involves validation, reporting, negative constraints, or quality control
8. Optionally read tool_card_path as structured aid
9. Produce adjudication JSON with evidence_refs pointing to canonical skill files
```

注意第 8 步：tool-card 只能是辅助，不是第一证据。

---

# 7. 代码层面的 handoff 修改意见

下面这段可以直接给 Codex / Claude Code 做项目修改。

```text
Pairwise workdir context refactor:

Current issue:
Each pairwise workdir writes doc_context.jsonl, pair_payload.json, and task_context.json. The Claude Code agent tends to read these generated files first and make edge decisions from derived context, instead of reading canonical `.claude/skills` documents. This creates duplicated context sources and can degrade edge quality because the generated context may omit important L1/L2/L3 evidence or bias the agent toward preselected chunks.

Required refactor:
1. Remove `doc_context.jsonl` from pairwise adjudication workdirs.
   - Do not generate document chunks with full text for pairwise agent consumption.
   - Remove or disable the generation logic in `pairwise_runner.py`.
   - If source guidance is needed, replace it with `source_manifest.json` containing only file paths, tool aliases, and reasons, but no document text.

2. Replace `pair_payload.json` with a minimal `pair_spec.json`.
   - Do not inline full tool cards.
   - Do not inline `doc_context`.
   - Include only:
     - pair_key
     - source/target tool IDs for both directions
     - stage labels
     - tool-card file paths
     - canonical skill root
     - candidate source globs
     - output file path
     - edge schema version
     - key adjudication rules

3. Keep `task_context.json`, but reduce it to a lightweight task manifest.
   - It should tell the agent what task it is solving and where to find:
     - pair_spec.json
     - stage_taxonomy.json
     - output_schema.json
     - canonical skill root
   - It must not contain evidence text or derived summaries.

4. Modify the pairwise system prompt / CLAUDE.md.
   - Require the agent to read canonical `.claude/skills` files before judging edges.
   - Require at least one explicit `evidence_ref` from canonical skills or tool schema for any positive edge, unless the edge is schema-only and explicitly marked as such.
   - Forbid treating generated summaries or tool cards as primary evidence.
   - Tool cards may be used only as structured aids; conflicts must be resolved by canonical skill files and MCP schema.

5. Modify validation.
   - After each pairwise output, check that positive/negative/uncertain decisions include evidence_refs.
   - Evidence refs must point to canonical files such as:
     - `.claude/skills/L1_tools/.../SKILL.md`
     - `.claude/skills/L2_workflows/...`
     - `.claude/skills/L3_methodology/...`
     - `snapshot::<tool_id>`
   - If an output cites only `pair_spec.json`, `task_context.json`, or generated tool-card summaries, mark it as invalid or needs_review.

6. Remove stale generation logic.
   - Remove `doc_context.jsonl` writing from `pairwise_runner.py`.
   - Remove the `doc_context` field from pair payload construction.
   - Rename `_prepare_pair_workdir` outputs accordingly.
   - Update downstream readers to consume pairwise result JSON only, not pair_payload/doc_context.

Goal:
The pairwise Claude Code agent should use generated files only to understand the task, not as the evidence base. The evidence base must be the canonical `.claude/skills` tree plus MCP schema/tool snapshots.
```

---

# 8. 对 tool-card 阶段是否也要删 doc_context？

这里要分开看。

## Tool-card 阶段

tool-card 构建本身可以有一个 retrieval / chunking 步骤，因为它要汇总一个工具的输入、输出、上下游、负约束。
但如果它也使用 Claude Code，最好同样不要让 `doc_context.jsonl` 成为最终证据来源，而是让 agent 读取 raw skills。

可以保留一个中间工程用的检索索引，但最终 tool-card 必须保存：

```text
provenance_refs
evidence_refs
source paths
```

你贴的 `chai1_predict` tool-card 已经有 `provenance_refs` 和 `evidence_refs`，这是好的方向。

## Pairwise adjudication 阶段

更应该删除 `doc_context.jsonl`，因为 pairwise 判断最怕被预选 chunk 误导。它必须重新面向两个工具和 workflow 文档做局部证据搜索。

---

# 9. 对 `pair_payload.json` 的进一步判断

如果你问“能不能完全删掉 pair_payload 生成逻辑”，我的建议是：

```text
可以删掉当前 pair_payload.json 的生成逻辑；
但要用 pair_spec.json 替代。
```

不能完全没有替代物。

因为 pairwise runner 仍然需要告诉 agent：

```text
这次判断 analyze_mmpbsa 与 pepinvent_peptide_sampling_by_peptide；
需要双向判断；
输出 schema 是什么；
taxonomy 是什么；
结果写到哪里。
```

所以重构方向不是：

```text
pair_payload.json -> nothing
```

而是：

```text
pair_payload.json -> minimal pair_spec.json
```

---

# 10. 最终建议

我的建议可以浓缩为：

> **删除 `doc_context.jsonl`；废弃当前重型 `pair_payload.json`；保留极简 `task_context.json`；新增极简 `pair_spec.json` 和可选 `source_manifest.json`。让 Claude Code Agent 必须回读 `.claude/skills` 原始文档，并把 generated tool-card 只当结构化辅助，不当证据来源。**

这样做的好处是：

```text
1. 避免两套上下文体系并存。
2. 避免 agent 偷懒只读派生摘要。
3. 保留批处理任务的可复现性。
4. 保留 pairwise adjudication 的稳定输入。
5. 强化 evidence-grounded 输出。
6. 降低 doc chunk selection bias。
```

不建议做的是：

```text
完全删除所有任务上下文文件，只让 agent 自由读 .claude。
```

那会让系统变得不稳定、不可控，也不利于结果聚合和复现实验。
