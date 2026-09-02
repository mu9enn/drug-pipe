# v6-final 训练的样本因果生命线：从 365 条历史轨迹到 compatibility 评测

> 本文与 [`v7_release_sample_causal_lifecycle_zh.md`](v7_release_sample_causal_lifecycle_zh.md) 定位相同：既供具备 SFT、GRPO/ToolRL 和 agent 后训练经验的专家逐层诊断，也帮助我们自己重新 own 这条训练链。本文不把数据、SFT、ToolRL、恢复和评测当成互不相干的模块，而是固定追踪一条真实样本，逐步说明每次变换后模型看见什么、优化什么、没有看见什么，以及最终 58% 的 MS-1 结果究竟能够和不能够证明什么。

## 先给出准确结论

v6-final 是一次真实完成的 **Qwen3.5-9B 全参数 SFT + 离线 decision-level GRPO**。365 条 MolBench-style 历史完整轨迹先被统一成严格 XML ReAct：SFT 对整条 teacher 轨迹中的全部 assistant decision 做 teacher forcing；随后同一批轨迹被拆成 5,329 个 `teacher-prefix → next assistant decision` 样本，静态选择其中 2,252 个，每个 prompt 采样四次做 hierarchical GRPO。实际 ToolRL 从 SFT checkpoint warm-start，使用冻结的 SFT reference、KL loss 和固定数据视图，最终得到 iter 562 checkpoint。它不是只有配置文件而没有执行的“纸面方案”。

但 v6 同样不是 full-episode、environment-interactive ToolRL。ToolRL rollout 只生成静态 teacher prefix 之后的一段 assistant completion；预测的工具调用不被执行，不产生新的 observation，也不会沿模型自己的分支继续到 final。工具 reward 衡量的是局部 action 与 teacher/tool schema 的匹配；终局 reward 衡量的是结构化 final 是否与 teacher target 精确一致，而不是由官方 MolBench scorer 对真实 episode 成败评分。因此，v6 的正确命名仍应是：

> **full-trajectory teacher-forced SFT + teacher-prefix offline next-decision GRPO**

v6 在旧的 `v6_mol_eval_compat` 在线评测中，50 道 MS-1 每一道都至少调用一次工具，总计发起 210 个工具调用，其中 208 个执行成功；官方 exact score 为 29/50（58%）。这证明该 checkpoint 在那套 prompt、parser compatibility、skill/tool 暴露和执行 harness 下，能够稳定进入确定性的属性计算链，也支持“先真正计算，MS-1 才有较高答对机会”这一机制判断。但它不能单独证明是 v6 ToolRL 使工具调用率达到 100%，更不能与后来标准 DSH 下的 v7 分数做 checkpoint-only 因果比较：v6 的评测 harness 带有显式 compatibility 行为，且没有 base、SFT-only、v6 ToolRL 三个 checkpoint 在同一 harness 下的对照。

因此，v6 最准确的定位是：它已经是一个工程上较完整、协议经过审计的 **offline decision-training baseline**，其 terminal objective 比 v7 的纯格式 reward 更接近答案内容；但它仍没有把训练的因果单位从“局部 teacher-prefix decision”升级为评测真正要求的“从原题到正确 final 的自主 episode”。

## 这份判断依据的边界与证据优先级

本文所说的 v6，是实际 run：

```text
outputs/slime_drug_agent_runs/
Qwen3.5-9B_v6_final_mol_production_20260817
```

它在 2026-08-17 启动，SFT 当天完成；ToolRL 中途发生一次 checkpoint 保存失败，随后在 08-18 恢复并完成。对应的 MolBench-MS compatibility 评测在 08-19 运行。因此 7、8 月汇报 PDF 能覆盖 v6 的许多直接背景，但本文仍以 frozen dataset manifest、run-local config、实际命令日志、checkpoint、固定遍历记录和评测 trace 为事实权威。

必须向专家主动披露两个 provenance 限制。第一，run-local `serial_config.env` 的 `CODE_COMMIT=unknown`，虽然数据文件有 SHA-256、`CODE_DIRTY=0`，但实际训练代码没有由 commit hash 完整冻结。第二，当前工作树里的 reward 代码已经演化到 v7 语义，不能用今天的 `molclaw_reward.py` 反推 v6 当时的 final reward；v6 的 `structured_final_exact_enabled=True` 及逐样本 component 只能以当时 `logs/toolrl.log` 为准。本文凡涉及“实际跑了什么”，优先采用运行产物，而不是新代码或后写报告。

## 这 365 条轨迹从哪里来

v6-final 并不是重新在线采集的 365 个 episode。其上游完整集合 `live_tool_catalog_v5-sftnrl` 有 605 条由历史 agent/teacher 生成并清理的完整轨迹，teacher provider 记录为 `deepseek-v4-flash-0731`；v5-mol 再按照已有的 MolBench 训练成员审计，从中保留 365 条，排除 240 条非目标父轨迹。v6-final 以这 365 条 v5-mol 轨迹为源，做协议正规化、决策切分、静态选择和审计，而不是重新让当前 policy 在当前 MolClaw/DSH 环境中执行并验证。

365 条也不是“365 道 MS-1”。按历史任务类型，它们由 145 条 PF、104 条 AC、98 条 VS、18 条 E2E 组成：

| task type | 轨迹数 | canonical decision 数 |
|---|---:|---:|
| PF / property filtering | 145 | 778 |
| AC / activity classification | 104 | 1,690 |
| VS / virtual screening | 98 | 2,371 |
| E2E / end-to-end | 18 | 490 |
| 合计 | 365 | 5,329 |

一条轨迹不是一个短问答，而是一整段 `system → user → assistant action → observation → ... → final` 的 JSONL 记录。全体记录的 message 数从 5 到 141、平均约 30.2；字符数从约 1.0 万到 31.7 万、平均约 6.1 万。这里是字符统计，不应当误报成 tokenizer token。长轨迹主要来自 VS/E2E 的文件操作、结构处理、docking 和多轮工具 observation。

v5 清理阶段还生成了 365 条 planning annotation sidecar，并将历史 thought block 做了合并；v6 manifest 记录 `merged_thought_blocks=4532`。但 planning sidecar 没有作为独立 planner role 或额外 RL target 进入实际 v6 run。模型训练的是 canonical assistant 文本中的 `<thought> + action`，而不是另有一个“先学 plan、再学 execution”的训练阶段。

## 固定追踪的一条真实样本

为与 v7 文档形成逐项可比的因果对照，本文也追踪 `react_pf_303eea476077228e`。它要求从 10 个候选 SMILES 中选出同时满足 Lipinski 条件、`MolLogP <= 3.55` 和 `MolWt <= 285.37` 的分子。

v6 canonical/SFT JSONL 中，这条完整轨迹是五个 message；压缩长字段后形状如下：

```json
{
  "id": "react_pf_303eea476077228e",
  "messages": [
    {"role": "system",    "content": "统一 XML ReAct 协议", "step_loss_mask": 0},
    {"role": "user",      "content": "10 个 SMILES 与过滤条件", "step_loss_mask": 0},
    {"role": "assistant", "content": "<thought>...</thought><tool_call>...</tool_call>", "step_loss_mask": 1},
    {"role": "user",      "content": "<observation ...>...</observation>", "step_loss_mask": 0},
    {"role": "assistant", "content": "<thought>...</thought><final_answer>...</final_answer>", "step_loss_mask": 1}
  ]
}
```

observation 在 chat schema 中作为 `role="user"` 保存，这是为了进入 Qwen 支持的 role 集合；其内容仍由 XML `<observation tool_name="...">` 明确标识，不是自然语言用户追问。mask 为 0，意味着它只作为 assistant 下一步的条件，不承担语言模型 loss。

这条 teacher 轨迹的实际因果顺序是：

```text
system: XML ReAct 规则
user:   10 个候选 SMILES + 属性阈值
assistant:
  <thought>需要先计算属性</thought>
  <tool_call>
    {calculate_mol_basic_info, 10 个 smiles}
    {calculate_mol_hbond, 10 个 smiles}
    {calculate_mol_hydrophobicity, 10 个 smiles}
  </tool_call>
user:
  三个工具的真实 observation
assistant:
  <thought>根据 MW/HBD/HBA/LogP 逐个过滤</thought>
  <final_answer>{
    "task_type": "pf",
    "selected_smiles": [
      "Clc1cccc(Cl)c1N=C1NCCN1",
      "CCc1nc(N)nc(N)c1-c1ccc(Cl)cc1"
    ],
    "evidence": [...]
  }</final_answer>
```

第一段 assistant action 在一个 `<tool_call>` 容器里放三个以空白分隔的 JSON object，表示“一次 assistant decision 中并行/成组发起三次调用”，而不是 JSON array，也不是三个被 observation 隔开的 assistant turn。这种 multi-call 表示在 v6 已经正规化：365 条轨迹共有 1,527 个 multi-call decision。runtime parser 与 reward parser 的审计表明，二者都接受单容器内的空白分隔对象，并一致拒绝逗号分隔对象、JSON array 和多个 `<tool_call>` 容器。

这点对理解 v6→v7 很重要：XML、multi-call turn 边界、SFT/ToolRL serializer parity 并不是 v7 才第一次建立。v6 已经做到 5,329 个 SFT target 与 5,329 个 ToolRL gold action 逐字节一致，保留 4,964 条 observation，不跨 observation 合并 action，5,329 个 assistant turn 全部含可监督 action。v7 继承了这套基础，新增的核心数据升级主要是 evidence 去重、artifact 解包、path/resource contract 修复与错误 path action masking。

## v6 正规化解决了什么，又留下了什么

v6 将 365 条上游轨迹正规化为 5,329 个 assistant decision：4,964 个 tool decision、365 个 final decision；其中 428 个 action 没有显式 thought，仍按原始 action 保留。正规化没有发现 interleaved assistant turn、thought-only incomplete tail 或跨 observation 合并。数据 audit 因而能够回答三个机械问题：同一历史 action 在 SFT 与 ToolRL 中是否相同；multi-call 的顺序和边界是否保留；runtime parser 与 reward parser 是否对同一文本作同一解释。答案都是肯定的。

但是，“serializer 正确”不等于“轨迹在当前环境可执行”。v6 仍保留大量 legacy `<artifact:...>` literal，也存在把 server-side artifact/resource 当作本地 workspace path 交给 `Read/Bash/Edit/Write` 的 action。SFT 和 ToolRL 会忠实学习这些历史动作，即使它们在新的 DSH path contract 下不可访问。v7 后来将 17,080 个 legacy literal 转为 resource URI 或 workspace path，并 mask 掉 115 个涉及 116 次已知错误 local access 的 assistant message，正是对 v6 这一残留问题的修补。

同样，v6 的“轨迹成功”是历史记录语义，并未在 release 构建时把每个 tool call 放回当前 MCP server 全量重放。重复失败、stale resource、环境版本漂移和 teacher 最终 evidence 是否仍可复算，没有一个完整的 executable replay gate。故 v6 数据应称为经过协议审计的历史轨迹 corpus，而不是当前环境下逐条重放验证的 executable corpus。

## 这条样本进入 SFT 后，loss 究竟落在哪里

SFT 不把样本拆成两道题，而是一次渲染完整五-message 轨迹。Qwen3.5 native thinking 被关闭，数据中的显式 `<thought>` 作为普通 assistant target 学习；`qwen3_5` loss mask 只覆盖 assistant token。当前样本因此提供两个监督条件：

```text
A: P(teacher 的 thought + 三个属性工具调用 | system + 原始问题)

B: P(teacher 的筛选 thought + structured final |
     system + 原始问题 + logged tool call + 与它配对的真实 observation)
```

这里的 A、B **不是两次独立调用 loss，也不是两个 optimizer step**。它们只是为了理解因果条件而把同一条自回归序列中的两段 assistant target 写成两个条件概率。实际实现先用 Qwen3.5 chat template 把五个 message 一次性渲染、tokenize 成一条 token 序列，再产生同长度的二值 loss mask：system、user 和 observation token 为 0，两段 assistant message 中可监督 token 为 1。因果 LM 对所有 `mask=1` 的位置做 next-token cross entropy；所以 A 中的 thought/tool-call token 与 B 中的 thought/final token，在这条轨迹参与的同一次 forward/backward 中共同形成一个 loss。

若一个 optimizer step 全局拿到两条完整轨迹（`s=1,2`），第 `s` 条的第 `t` 个 token 为 `x[s,t]`，mask 为 `m[s,t] ∈ {0,1}`，那么因为实际命令启用了 `--calculate-per-token-loss`，这一步的核心监督量可以写成：

```text
L_step =
  - sum_s sum_t m[s,t] * log P_theta(x[s,t] | x[s,<t])
    ----------------------------------------------------
                 sum_s sum_t m[s,t]
```

因此应当形成四个具体认识。第一，optimizer 看到的是一个合并了该步两条轨迹中**全部 assistant target token** 的标量 loss，不是“A 的 loss”和“B 的 loss”各更新一次。第二，A 与 B 对总梯度的权重按各自 assistant token 数自然累加；长 thought/final 会比短 action 贡献更多 token 项，而不是每个 decision 等权。第三，teacher tool call 和真实 observation 虽然在 B 的预测 prefix 中，却都不作为 B 的 target 重算一遍；前者已经在 A 位置受监督，后者始终 `mask=0`、只提供条件。第四，B 的每个 token 仍是普通 causal next-token prediction：较后的 final token 还会条件于 teacher final 中较早的 token。

以当前五-message 样本为例，概念上可以把它想成下列一条 mask 序列，而不是两个训练 example：

```text
system(0) + question(0)
+ teacher thought/tool-call A(1)
+ 与该 logged call 配对的 observation(0)
+ teacher thought/final B(1)
                ↓
         一个 trajectory loss
```

这确实教了两项与 MS-1 直接相关的能力：从题目识别需要哪些属性工具，以及在拿到正确 descriptor 后应用阈值并交付答案。模型并不会在 token 中看到 teacher/student 身份标签，所以它完全可以把前一 assistant call 和 observation 理解成“自己的对话历史”；这里使用 `logged` 只是标记数据来源。关键限制是，B 的 prefix 固定来自数据中已有的 action/observation pair，而不是模型在 A 位置自由采样的行为及其实际执行结果。因此 SFT 没有经历“模型漏调某个属性”“参数错误”“工具报错”“observation 超长”之后如何恢复。这是标准 teacher forcing 的 exposure gap，不是 Slime 实现异常。

365 条 unique trajectory 为了满足 GBS=2，被物化为 366 行；追加的 batch-alignment 行恰好是这条最短样本，渲染后 4,462 token。因此它在一个 SFT epoch 内出现两次，其余 364 条各一次。汇报时应同时说“365 条 unique、366 条 materialized”，否则专家会无法对上 183 个 optimizer step。

实际 SFT 参数为：Qwen3.5-9B 全参数，8×H200，TP=4、PP=1、DP=2，GBS=2；最大序列 131,072，每 GPU 动态 token budget 16,384；一个 epoch、183 次 optimizer update。学习率 `5e-6 → 5e-7`，5% warmup 后 cosine decay，Adam β=0.9/0.95，weight decay=0.1，full recompute、per-token loss。训练 loss 183 步均有限，均值约 0.1713、首步约 0.1690、末步约 0.0707；grad norm 均值约 1.667，最大约 10.55。

这组并行和 batch 参数的物理含义如下：

| 参数 | 含义 | v6 SFT 中实际发生的事 |
|---|---|---|
| TP=4 | tensor parallel | 一份 9B 模型的矩阵/张量计算横向切在 4 张 GPU 上；它不是 4 份独立样本或 4 次更新 |
| PP=1 | pipeline parallel | 不按层切成多个 pipeline stage；每个 TP group 共同承载整套层的 tensor shard |
| DP=2 | data parallel | 有 2 个数据并行 replica，每个 replica 各由 4 张 TP GPU 构成；两边处理不同轨迹，再同步/归并梯度 |
| GBS=2 | global batch size | 每次 optimizer update 在所有 DP replica 合计消费 2 条完整 trajectory，而不是 2 个 assistant decision、也不是 2 token |

所以 8 张卡的拓扑是 `TP 4 × PP 1 × DP 2 = 8`。在一个典型 step 中，DP replica 0 拿一条完整轨迹，DP replica 1 拿另一条；每条轨迹内部又由相应的 4 张 TP 卡协同完成 forward/backward；两份梯度在 DP 维归并后，Adam 才更新一次参数。366 个 materialized trajectory / GBS 2 正好等于 183 个 optimizer step。日志的 dynamic-batch audit 也逐步记录为 `samples=2, microbatches=2, per_rank=1`。

### SFT 的长度合同与超长处理

SFT 确实有硬长度限制：`--sft-max-sequence-len 131072`。这限制的是一整条轨迹经过 Qwen chat template 后的 token 总数，包括 system、问题、全部 assistant action 和全部 observation。实际 366 行的长度为：最短 4,462、均值约 23,702、p50 14,076、最大 93,970 token；超过 131,072 的记录为 0。日志也没有任何 `Truncated oversized SFT sample`，所以 v6 正式 SFT **没有一条样本真的被截断**。

`--max-tokens-per-gpu 16384` 容易被误读。它是 dynamic batching 的装箱/负载预算，不是样本硬上限；183 个 step 中有 122 个 step 至少含一条超过 16,384 token 的轨迹，这些长样本作为 oversized singleton 仍完整训练。真正的截断门槛是 131,072。

若未来某条渲染序列超过 131,072，当前 `sft_rollout.py` 会机械保留开头 8,192 token 与结尾 122,880 token，删掉中间，再对 token ids 与 loss mask 同步切片：

```text
original: [ head 8192 ][-------------- middle --------------][ tail 122880 ]
kept:     [ head 8192 ]                                      [ tail 122880 ]
```

这种 head+tail 策略有利于保留 system/原题和末尾 final，却不是 message-aware 或 tool-boundary-aware；它可能删掉中间某次 tool call/observation，使保留下来的后续 action 失去因果前提。v6 没触发它，所以这是当前实现的潜在风险而非本次 run 已发生的数据损坏。截断后 response span 从 loss mask 中第一个 1 一直延伸到末尾，其中 observation/user 位置仍保持内部的 0 mask。

这些日志证明优化链路确实运行、数据确实进入 loss，也证明 SFT 能拟合训练 teacher；它们不能证明泛化或 agent 成功率提高。v6 没有在同一评测 harness 下保存并汇报 base→SFT-only 的独立 control，因此后面的 58% 无法拆成“预训练模型贡献、SFT 贡献、ToolRL 贡献”。另外，SFT checkpoint 目录名是 `iter_0000000`，不代表 SFT 做了 0 步；这是该串行脚本的 stage-specific 保存/加载语义，实际 183 步由日志和数据计数确定。

## 从完整 episode 到 5,329 道静态 next-decision 题

SFT 结束后，同一条五-message 样本被转换为两个独立的 ToolRL row：

```text
row A / initial tool step
prompt = [system, user]
gold   = teacher thought + 三个属性 tool calls

row B / final
prompt = [system, user,
          teacher 的三个 tool calls,
          teacher 执行后的三个 observations]
gold   = teacher thought + structured final
```

实际 JSONL 每行包含 `prompt`、`label` 和 `metadata`，并冗余保存 reward 直接读取的结构化 target。其形状是：

```json
{
  "prompt": ["当前 decision 之前的全部 teacher messages"],
  "label": {
    "decision_type": "tool_call | final_answer",
    "assistant_index": 2,
    "assistant_content": "teacher 的下一段 assistant 文本",
    "target_tool_calls": ["结构化 teacher calls"],
    "target_final_answer": "结构化 teacher final 或 null"
  },
  "metadata": {
    "source_id": "react_pf_303eea476077228e",
    "task_type": "pf",
    "decision_role": "tool_step | final",
    "is_initial_step": true,
    "prompt_tokens_final": "tokenizer 实测值",
    "canonical_target_tokens": "tokenizer 实测值"
  },
  "target_assistant": {
    "role": "assistant",
    "content": "与该 row 的 teacher assistant message 完全相同的原文",
    "parsed": "对原文做确定性 XML/JSON parse 后的结构",
    "tool_call_count": 3
  },
  "target_tool_calls": ["供 reward 使用"],
  "target_final_answer": "供 reward 使用"
}
```

这里的字符串是 schema 说明，不是假装成当前样本的具体 token 数；selector 实际使用 tokenizer 计算的整数。全体 raw rows 为 5,329：4,964 个 tool step、365 个 final。

### `target_assistant` 到底是什么，是否完全没用

原文把它简写成“冗余 teacher target”容易造成两种误解。它实际是一个 dict，不是字符串；而“冗余”表示 schema 中重复保存同一事实，不表示它在整个数据工具链中绝对无人读取。转换脚本 `convert_react_to_toolrl_steps.py` 对每个 teacher assistant message 做确定性 Python 解析：

```python
content = message["content"]
parsed = parse_assistant_decision(content)
target_assistant = {
    "role": "assistant",
    "content": content,
    "parsed": parsed,
    "tool_call_count": len(parsed["tool_calls"]),
}
```

这里没有调用 LLM，也没有重新生成或改写 teacher target。相同对象被复制到 top-level、`label.target_assistant` 和 `metadata.target_assistant` 三处；其中原始文本还与 `label.assistant_content` 重复，结构化 calls/final 又分别与 `target_tool_calls`、`target_final_answer` 重复。它本质是为了审计、兼容不同 consumer 和避免下游重复 parse 所做的 schema denormalization。

在本文追踪样本的 initial row（assistant index 2）里，`target_assistant.content` 就是上一节展示的完整第一段 teacher action：开头为：

```text
<thought>This is a cheminformatics filtering task over a provided set of SMILES...
Let me calculate the required molecular properties.</thought>
<tool_call>
{"arguments":{"smiles_list":[10 条原始 SMILES]},"tool_name":"calculate_mol_basic_info"}
{"arguments":{"smiles_list":[同一组 10 条 SMILES]},"tool_name":"calculate_mol_hbond"}
{"arguments":{"smiles_list":[同一组 10 条 SMILES]},"tool_name":"calculate_mol_hydrophobicity"}
</tool_call>
```

这里方括号中的中文仅为文档避免把同一 10 项数组重复排版三遍；JSONL 字段中三组数组都逐项完整保存。它的 `parsed.decision_type="tool_call"`、`parsed.tool_calls` 长度为 3、`final_answer=null`、`tool_call_count=3`。同一样本 final row（assistant index 4）的 `target_assistant.content` 则是完整的逐分子筛选表和 `<final_answer>`，`parsed.decision_type="final_answer"`、calls 为空、final 为两条 `selected_smiles` 加三条 evidence、`tool_call_count=0`。

Slime 这次正式 ToolRL dataset loader 的直接入口只有 `prompt`、`label`、`metadata`，reward 首选读取 `label.target_tool_calls/target_final_answer`；因此 top-level `target_assistant` 不是额外 prompt，不参与一次额外 loss，也不是 reward 的主输入。但是它并非整个代码库“完全没用”：selector 在 `assistant_content` 缺失时会回退到 `label/metadata.target_assistant.content`；context compaction、training-view materialization 和部分统计脚本也读取 top-level 对象。所以准确结论是：**它不增加一份监督信号；对本次标准 row 的主训练路径是重复信息，但对审计与 fallback consumer 仍有用途。**

这里是 v6 最关键的因果切口。对 row A，模型采出的四组 tool calls 只被 parser/reward 检查，不会发送给 MolClaw；对 row B，模型得到的是数据中固定记录的历史 action 及其配对 observation。这里不应把核心区别说成“teacher observation 对 student observation”：chat transcript 没有这种 provenance role，模型完全可以把它当作自己的既往交互来理解。真正的区别只是它在训练计算图中并不是 row A 本轮 sampled action 的执行结果。row A 的 sampled branch 和 row B 没有连接：

```text
v6 实际训练：
logged H_t → sample a_t^sample → 与 logged target 做局部 reward → 更新结束
H_(t+1)^log = H_t + a_t^log + o_t^log → 另一个 sample → 另一个 reward

v6 没有训练：
a_t^sample → Env(a_t^sample) → o_t^sample → 继续生成 → final → benchmark reward
```

所以真正的区别不是 observation 的 teacher/student 归属，而是它有没有和本轮 sampled action 因果配对。v6 ToolRL 可以局部提高“在 logged state 下像不像 target action”，却没有把“如果不调用属性工具，最终 MS-1 就错”这一后果直接回传给 initial decision。即使 terminal reward 比 v7 严格，它也只更新 final row 的 completion，无法 credit 给 row A 的 sampled 工具选择。

## 2,252 个 production decision 是如何选出的

实际 v6 run 使用的不是全部 5,329 个 raw decision，而是 `drug_pipe_production` 静态 view。selector 优先保留初始 action 与 final，再按 task type、工具/参数形状、深度、长度和 observation 状态等 deterministic strata 覆盖中间 decision。最终保留：

| role | selected group |
|---|---:|
| tool step | 1,888 |
| final | 364 |
| 合计 | 2,252 |

排除的 3,077 个 row 中，4 个是 no-progress repeat，2 个 teacher target 超过 16K response limit，3,071 个属于 diversity-equivalent middle decision。一个 final 因 target 长达 34,926 token，未进入 ToolRL，但仍存在于 SFT；另有三个确定性 coverage filler 用于让总数被 RBS=4 整除。最大选中 prompt 为 89,518 token，最大 target 为 12,687 token；运行合同是 prompt 245,760、response 16,384、context 262,144，因此没有样本需要 context compaction。

“diversity-equivalent middle decision”不是说两条 action 的文本、分子、observation 或语义完全等价。selector 先无条件保留每条轨迹的 initial decision 与 final；只对其余 middle tool row 计算一个粗粒度 stratum key：

```text
(
  task_type,
  sorted[(tool_name, sorted(argument key names)) for each target call],
  depth_decile,                  # decision ordinal / 轨迹总 decision 数，分 0..9 桶
  approximate_prompt_length_bin, # 按 prompt 字符数：short/medium/long/very_long
  previous_observation_status,   # none/unknown/success/failure
  single_or_multi_call
)
```

长度桶的字符边界依次为 32,768、131,072 和 524,288；previous observation status 由 prompt 中最近一个 `<observation>` 的 `ok/status` 字符串保守判断。只要两个 middle row 的上述六项相同，就进入同一 diversity stratum。注意 argument **值**不进入 key，真实 observation 内容不进入 key，trajectory/source ID、化学输入、teacher thought 和 target action 全文也不进入 key。因此“同一工具名 + 同一参数名集合 + 相似深度/长度 + 同类前态”就可能被当作 diversity-equivalent，即便它们处理的是不同蛋白或分子、得到不同 observation、要求不同后续判断。

每个 stratum 只保留一个 deterministic representative：对 `decision_key + stratum` 计算 SHA-256，hash 最小者胜出；其余 row 标成 `diversity_equivalent_middle_decision`。v6 共形成 1,521 个 middle strata，并因此排除 3,071 个 middle decision；最后再从未选 row 中按确定性 rank 加回 3 个 alignment filler，使 2,252 能被 4 整除。这是一个**静态覆盖/去重复启发式**，不是根据模型当前会不会、reward 方差或科学信息价值做的 learnability selection。它可能减少机械重复，也可能误删“结构形状相同但 observation→下一步语义不同”的训练状态；专家诊断时应把这视作需要消融验证的数据选择假设。

### ToolRL decision 的长度合同与超长处理

ToolRL 与 SFT 的长度单位不同：一行只含 `teacher prefix prompt + 当前一次生成 target`，不是整条 episode 的新自主 rollout。静态 view 的合同为 prompt 最多 245,760 token、生成 response 最多 16,384 token、两者合计 context 最多 262,144 token。

物化前，selector 先用 Qwen tokenizer 计算 canonical teacher target；target 超过 16,384 的 row 直接排除，而不是截断 target。v6 实际排除了两条：一个 34,926-token final 和一个 32,059-token tool action。随后对 prompt 做最多三次 budget compaction 尝试；若仍超过 245,760，则以 `context_compaction_failed` 排除，若 prompt+target 超过 262,144，则以 `context_exceeds_limit` 排除。v6 配置的 semantic summarizer 是 `none`；本次被选中的最大 prompt 仅 89,518、最大 target 12,687，所以 2,252 条 selected row 的实际 compaction 数为 0。

训练采样时，模型自己的 completion 仍受 16,384-token runtime cap。若采样撞到 cap，sample 标记为 truncated；reward guard 会把原本为正的 reward 压到 0，原本为负的 reward 不抬高。按恢复后有效的 9,008 个 completion 重建，14 个触发 truncated，约 0.155%。这只说明“静态单次 next-decision completion 很少超过 16K”，绝不等于完整 agent episode 不会在多轮累计后到达 length/max steps；正式评测的 MS-2 正好展示了后者。

selected rows 覆盖 53 个工具名。raw decision 的 10,984 次 teacher target call 中，8,526 次是 MolClaw、2,458 次是 Bash/Read/Write/Edit/Glob/Grep 等 local tool；2,177 个 tool decision 含 local call，35 个同时含 local 与 MolClaw。这说明 corpus 中确有 skill discovery、文件读写和远程计算行为，但它们依然只来自 teacher history。

实际 production prompt 不注入完整 official tool catalog：catalog 用于 reward/runtime，模型需要从 system contract、历史上下文以及 SFT 已学到的工具名中恢复行动先验。manifest 另行物化了 5,316-row 的 `official_baseline` view，它向 prompt 注入 official catalog，并定义 base checkpoint→ToolRL；**实际 v6 run 并没有执行这条 arm**。实际是 SFT warm start→2,252-row production view。专家汇报时必须把“release 中存在 official-baseline 文件”与“实际 checkpoint 跑了 official baseline”分开。

这个 prompt 设计需要更精确地评价。**“prompt 不注入完整 catalog，而部署 harness 允许模型用 Glob/Read 发现 skill”本身是逻辑通顺的 agent 协议，并不与训练天然冲突。** 全轨迹 SFT 确实分别学到了 `P(发现动作 | 原题)`，也学到了 `P(后续工具选择 | logged 发现动作 + 与之配对的 observation/skill 内容)`；因此不能笼统说 v6 完全没有学过“发现结果影响后续选择”。这里的 history 对模型而言可以被理解为自己的历史；称为 `logged` 是为了说明它来自固定数据，而非强行赋予 teacher 身份。

真正缺失的是 ToolRL 中 sampled branch 的闭环和跨步 credit。offline GRPO 可以在一条 initial row 上采样 Glob/Read，并对它与 logged target action 的局部相似度打分；也可以在另一条 later row 上看到历史中已经读回的 skill 内容，再优化下一次 tool call。但 initial row 中模型本轮生成的 Glob/Read 不会执行，它的真实返回不会构成 later row；later row 是否出现、看到什么 observation，与前一 row 的 sampled action 无关。因此 GRPO 不能学习“这次 sampled 发现动作实际找到了什么，进而是否令下一步和最终答案更好”，不能把最终成功 credit 回早期发现，也没有暴露于自己写错 path、读错 skill 后的恢复状态。

所以部署协议是成立的，SFT 也提供了 teacher-forced 条件模仿；问题是 RL objective 没有优化这条**由自身动作造成的因果链**。在 MS-1 中，已记住的直接属性工具模式就足以工作，实际 50 题没有一次 skill read；在 MS-2 中，复杂工作流更依赖发现、path 和多步纠错，离线 gap 才被放大。

## v6 的四采样 GRPO 与 reward 信号

ToolRL 从 SFT checkpoint warm-start，Qwen3.5-9B 全参数，8×H200，TP=4、PP=2、DP=1。每个 rollout batch 取 4 个 decision prompt（RBS=4），每个采样 4 个 completion（n=4），得到 16 个 response 对应 GBS=16 的一次 GRPO update；2,252 个 group 理论上组成 563 个 update、9,008 个 sampled response。temperature=1.0，prompt/response/context limit 分别为 245,760/16,384/262,144。

actor 相对冻结的 SFT reference 使用 `low_var_kl` loss，系数 0.001；reward KL 为 0，entropy coefficient 为 0，PPO clip low/high 均为 0.2。ToolRL 学习率恒定 `2e-7`，weight decay 0.1，Adam β=0.9/0.95；actor 常驻，rollout engine 按阶段 offload。真正的 rollout 入口是普通 `slime.rollout.sglang_rollout.generate_rollout`，不是会调用 MolClaw 的自定义 environment rollout。

工具 row 使用 hierarchical teacher-local reward，原始标量严格落在 `[-0.5, 1.0]`。它不是几个 component 的简单加权和，而是按门逐级判定；前一门不通过，就不进入后一门：

```text
1. 非法 ReAct，或不是“有 tool_call 且无 final_answer”
   reward = -0.5

2. envelope 合法，但预测与 teacher 之间没有同名工具可以配对
   reward = -0.4

3. 至少有同名配对，但工具 multiset/数量并不完全匹配
   reward = -0.05 + 0.35 * tool_F1

4. 工具集合与数量完全匹配后，才检查每一对调用的参数；
   多调用采用所有调用中最差的一项作为该门指标：

   required_coverage < 1:
       reward = 0.30 + 0.20 * required_coverage

   critical_exact < 1:
       reward = 0.55 + 0.15 * critical_exact

   configurable_validity < 1:
       reward = 0.72 + 0.13 * configurable_validity

   所有参数 key/value 与 teacher 等价:
       reward = 1.0

   否则（关键参数正确、其余是 schema 合法的替代配置）:
       reward = 0.90
```

预测和 teacher calls 先经过 canonical tool/argument normalization，再做 order-insensitive、保 multiplicity 的一对一配对；不要求 multi-call 的输出顺序逐字相同。8 个调用以内穷举最高总配对质量，超过 8 个时用 greedy matching。`tool_F1` 只把 canonical tool name 相同的 pair 当 matched，precision/recall 分母分别是预测和 teacher call 数。

required 参数优先来自当时 frozen tool catalog 的 JSON schema；没有 catalog schema 的 local tool 退回把 teacher argument keys 视作 required。critical 参数优先看 schema 的 `x-toolrl-importance=identity|critical`，其余按参数名规则识别 `id/name/target/gene/protein/ligand/receptor/sequence/smiles/mutation/chain/artifact/file/path/input/structure/complex/pdb/cif/sdf/mol2` 等身份/资源字段，并要求与 teacher 归一化值 exact。configurable 参数检查一个确定性的 JSON-schema 子集，包括 type、enum、数值上下界、字符串长度、array 长度及 items 类型。

这比字符串 exact match 更细，也允许某些非 teacher 参数；但它仍没有执行工具，无法知道一个 schema 合法调用是否真的得到有用 observation。`<thought>` 是否存在、长度多少会被记入 diagnostics，却不直接改变上述 reward gate。

final row 与 v7 有决定性差异。v6 日志显示 `structured_final_exact_enabled=True`：合法且结构化内容与 teacher target 精确一致时得 +1，否则通常得 -0.5；比较前会忽略 duplicated human-readable `summary` 字段。`terminal_correctness` 只有 exact 时为 1。它确实对 selected_smiles/answer 等内容施加了信号，因此不能把 v6 与 v7 的 format-only final 混为一谈。

更精确地说，final 必须同时满足 parser 成功、含一个 `final_answer`、不含 tool call；然后 scorer 递归删除预测与 teacher object 中名为 `summary` 的字段，再比较剩余整个结构。只有完全相等时 raw reward=+1；只要 envelope 无效或任一内容字段不等，raw reward=-0.5。另行记录的 format component 是合法时 +1、否则 -0.3；`terminal_correctness` 是 exact 时 1、否则 0，但正式用于 GRPO 的 `score` 仍是前述 +1/-0.5，而不是这两个 component 相加。

但也不能把它称为官方 benchmark correctness。v6 exact scorer 比较整份 teacher structured object；日志中存在这样的实际情形：预测的 `answer_smiles` 正确，但 evidence 与 teacher 不完全一致，仍得 -0.5。于是它可能奖励 teacher-identical 答案，也可能错罚 benchmark scorer 会接受的语义等价答案。更准确的表述是：

> v6 terminal reward 有内容监督，但目标是 **teacher structured-final exactness**，不是 environment-grounded MolBench correctness。

### raw reward 怎样成为一次 GRPO 参数更新

每次 rollout batch 固定取 4 个不同 decision prompt，每个 prompt 在 temperature=1.0 下各采 4 个 completion，因此形成 4 个互不混合的 GRPO group、共 16 条 response。对一个 group 的四个 raw reward `r1,...,r4`，实际启用了 reward normalization 与 `grpo_std_normalization=True`：

```text
mean = (r1 + r2 + r3 + r4) / 4
std  = 四个 reward 的 sample standard deviation（unbiased=True）
A_i  = (r_i - mean) / (std + 1e-6)
```

例如一个工具 group 得分 `[1.0, 0.9, -0.4, -0.5]`，均值为 0.25、sample std 约 0.810；四个 advantage 约为 `[+0.926,+0.802,-0.802,-0.926]`。这表示 GRPO 学的是同一 prompt 下的相对偏好：第一条比第二条更受鼓励，后三条受抑制；它不把 +1 当作跨 prompt 可直接比较的绝对价值。

每条 completion 的标量 advantage 会广播到它的全部生成 token，policy objective 使用 rollout old policy 与当前 actor 的 token probability ratio，并在 `[1-0.2, 1+0.2]` 范围做 PPO-style clipping。16 条 completion 共同构成 GBS=16 的一次 optimizer update。与此同时，另有系数 0.001 的 `low_var_kl` loss 把 actor 约束在 frozen SFT reference 附近；`reward KL=0` 表示 KL 没有先从 raw reward 中扣除，`entropy coefficient=0` 表示没有额外 entropy bonus。

因此，同一 prompt 四个 sample 同分时，不论全是 +1 还是全是 -0.5，四个 reward advantage 都为零，task-reward policy gradient 没有方向；若 actor 已偏离 reference，KL regularization 仍可能产生梯度。由于本次 run 在 iter 224 保存失败并恢复，原始 traversal 文件包含重复采样；按恢复后最终生效分支取重复 key 的最后一次、其余取唯一一次，可重建 2,252 个 effective unique group：

| role | effective group | 四样本同分 | 可提供组内偏好 | group raw reward 均值 |
|---|---:|---:|---:|---:|
| tool step | 1,888 | 609（32.26%） | 1,279 | 约 +0.03039 |
| final | 364 | 272（74.73%） | 92 | 约 -0.22802 |
| 合计 | 2,252 | 881（39.12%） | 1,371 | — |

272 个零方差 final group 中，255 个是四个 sample 全为 -0.5，17 个全为 +1。全部 1,456 个 effective final response 中，264 个得 +1、1,192 个得 -0.5，teacher-structured exact positive rate 为 18.13%。这说明 v6 final reward 虽然比 v7 有内容约束，但 74.73% 的 final prompt 没有提供 GRPO 组内偏好；尤其大量全错组不会产生 policy-gradient 方向。“负均值”不能理解成模型被强烈教会改正，因为 GRPO 关心的不是绝对负分，而是组内能否区分。

这组数也解释了为什么不能将 v6 `-0.228` 与 v7 `+0.586` 直接当作能力进步。两版 reward 定义不同：v6 要 teacher structured exact，v7 只要合法 envelope；v7 大量内容错误的 final 会从 -0.5 变成 +1。标尺变宽本身足以抬高均值。

## 中途失败、恢复与 563 个 update 应怎样解释

v6 并非一次不间断完成。SFT 和长度/production-shaped/multi-update gates 通过后，ToolRL 于 08-17 13:49 左右启动；运行到 rollout/checkpoint 224 时，`torch.distributed.checkpoint.api.CheckpointException` 导致保存失败。不完整目录后来被移为：

```text
toolrl/iter_0000224.failed_incomplete_20260818
```

08-18 03:27 左右从上一有效 checkpoint 恢复，最终于 13:02 左右完成 iter 562。这个故障属于 checkpoint serialization/保存，不是 OOM、NCCL、NaN 或 reward 函数崩溃；训练恢复后确实完成。

但是恢复影响了 exact-epoch 证据。`toolrl_fixed_traversal.jsonl` 有 2,352 条记录，却只有 2,252 个 unique decision key：group index 800–899 对应的 100 个 decision 被失败分支和恢复分支各记录一次，即 25 个 rollout batch×4 decision。79/100 个重复 decision 的 sampled reward 在两次采样间发生变化，符合 temperature=1.0 下恢复后重采样。最终所有 2,252 个预期 key 都出现、没有 unexpected key；但 v6 没有像 v7 那样产出一份明确认证“exactly once”的 post-run traversal audit。

因此本文的 reward 统计采用一个需要显式标注的恢复推断：对重复 key 取最后一次记录，视为恢复后最终生效分支；对单例取唯一记录。它能还原 selected view 的 1,888 tool + 364 final，也与 rollback/resume 语义一致，但不应把原始 2,352 行误报为 2,352 个训练 group，更不应声称 v6 原始 traversal 天然无重复。v7 后来补齐 clean exact-epoch audit，正是对这个可复核性缺口的升级。

## 到在线 MS 评测时，因果链才第一次真的闭合

v6 iter 562 的评测使用 50 道 MS-1 与 33 道 MS-2，共 83 题，并根据 overlap audit 排除 4 个与训练重合的候选任务。评测不是训练时的静态 next-decision generator，而是自定义 `drug_agent.rollout.generate_with_drug_agent.generate` 在线 agent loop：模型 action 被真实解析、工具被执行、observation 回填、模型继续，直到 final、非法格式、长度或 max steps。

评测配置包括 temperature=0、max steps=128、单题 timeout=3,600 秒、MCP timeout=14,400 秒、max new tokens=16,384、max prompt/context=49,152/65,536；允许 81 个 MCP 工具和 6 个 local tool，并冻结了当时的 skill snapshot。与此同时，它启用了 `v6_mol_eval_compat`、compatibility parser/decision stop 和 parse recovery。19 个 MS-1 continuation action 实际由 compatibility path 接住；这不是一个可以忽略的标签，而是会改变 episode 是否继续执行的评测脚手架。

这里所谓“针对 v6-mol 已观察问题做兼容和提示增强”，不是一个模糊名称，而是五项很具体的 eval-only 行为：

| compatibility 行为 | 它替 v6 堵住的已观察问题 | 它明确没有替模型做什么 |
|---|---|---|
| 在 live tool catalog 与普通 ReAct reminder 后，追加 `skills/L1_tools/`、Glob pattern、SKILL.md path，并规定首次 specialized MolClaw call 前必须 Read relevant skill | base prompt 对 mounted skill 的发现路径提示不足 | 不替模型选择相关 skill，不替它修不存在的 path |
| 生成参数在首个完整 `</tool_call>` 或 `</final_answer>` 处 stop，并用 `no_stop_trim` 保留 closing tag | v6 会在一个完整 decision 后继续生成额外内容，导致本来可解析的 action 被污染 | 不截断/修复尚未闭合的 JSON 或 XML；单次生成仍可能打满 16K |
| strict parser 失败后，仅兼容“一个 `<tool_call>` 容器内，两个及以上完整 top-level JSON object 用逗号分隔” | v6 将 canonical 的空白分隔 multi-call 生成为逗号分隔 | 不接受 JSON array、trailing comma 或 malformed JSON，不改 payload 字段 |
| continuation context 中去掉 Qwen transport marker `<|im_end|>` | transport marker 被当成下一轮 assistant 内容继续喂回时，可能诱发边界污染 | 不改变模型本轮 decision 的语义或工具参数 |
| continuation 中只对一次已观察到的多余 leading `</tool_call>` 做窄恢复；若一整轮只输出这个 close，也最多允许一次 no-op | v6 在 observation 后偶发重复输出上一轮 closing tag | 不无限容忍重复 close；其他格式错误仍立即失败 |

这套 profile 始终先跑 strict parser，并且不会掩盖 unknown tool、缺 required argument 或工具执行错误。它的目的是真实执行“模型已经表达清楚、只差一个已知 Qwen 表面变体”的 action，而不是替模型做 agent reasoning。最终 trace 中 compat-assisted action 共 19 次，全部是 MS-1 observation 后单独多吐出的 `</tool_call>` 被当作一次 no-op；comma-separated-object recovery 与 transport-marker stripping 的实际计数都是 0，MS-2 也没有任何 compat recovery。换言之，compat 的五项能力在 profile 中都启用，但不能误报为五项都在本次 83 题上实际触发。该分布说明不能把 v6 的 MS-2 失败归因于“compat parser 把动作改坏了”；19 次 no-op 也发生在此前工具已经调用之后，所以它不能解释 MS-1 为何 50/50 都开始调用工具，但它确实避免了 19 个 episode 在工具返回后立刻因多余 closing tag 失败。至于显式 skill prompt、live catalog 暴露和 decision stop 对 policy 行为的影响没有独立 ablation，MS-1 仍不能被当成完全无 scaffold 的裸 checkpoint 结果。

83 题的终态是：36 题到达 `final_answer`，34 题以 `invalid_react_format` 结束，8 题 length，5 题 max steps；也就是 47/83 没有交付 final。分 task 看：

| suite | 题数 | final | invalid format | length | max steps |
|---|---:|---:|---:|---:|---:|
| MS-1 / PF | 50 | 33 | 17 | 0 | 0 |
| MS-2 / AC | 33 | 3 | 17 | 8 | 5 |

MS-1 的 50 题全部至少发起一次真实工具调用，合计 210 次，其中 208 次 execution success、2 次 semantic error；最终官方 exact 为 29/50（58%）。其中 33 题交付合法 final，29 题正确，另外 4 题虽到达 final 但答案错误；17 题因非法 ReAct 格式没有交付可评分答案。17 个 failure 中，2 个是 supported ReAct tag 外仍有额外内容，15 个是 `final_answer.selected_smiles` 没有给 list。也就是说，MS-1 的主要剩余损失已经不是 RDKit 属性工具算不出来，而是 **算完后未按严格 final schema 交付**，其次才是 4 个已交付但筛选内容错误的答案。

MS-2 则只有 1/33（3.03%）正确：仅 3 题到达 final，其中 2 题答案仍错；17 题在第一个 decision 就因 `arguments` 不是 object 或 JSON object sequence 无法解析而结束；另外 8 题单次生成打满 16K response cap，5 题走满 128 step。故原文的 “invalid、length/max steps 和反复 local/skill”需要拆开说：

- `length` 和 `max_steps` 的 13 题全部在 MS-2，反复操作也主要集中在这些复杂 MS-2 rollout；
- `invalid` 不是 MS-2 独有，MS-1 与 MS-2 各 17 题，只是 MS-1 多为 final schema 交付错误，MS-2 多为首轮 tool-call JSON/arguments 错误；
- skill loop 不是均匀发生在所有 MS-2：384 次 `skills/L1_tools/...` Read 集中在 4 题，其中 3 题各重复 127 次；它是少数 catastrophic loop 拉高总量，而非 33 题都在重复读 skill。

MS-2 的 33 题总计产生 1,056 个 agent step 和 1,042 个工具调用，只有 142 次 semantic success、900 次 semantic error；其中 17 次还在 schema validation 阶段失败。错误高度结构化：381 次 Read 报 `file does not exist`，295 次 Bash 全因 `unsupported shell redirection`，101 次 `pred_pocket_prank` 返回失败状态，82 次 `server_file_to_base64` 失败，另有 fpocket/path/required-argument 问题。五个 max-step 任务本身就贡献 640 step。模型面对同一种明确失败 observation 时，常常原样或近似重试，缺少“识别不可恢复错误→换路径/降级/停止”的控制策略。

从一条 MS-1 trace 看，模型通常在首步成组调用 `calculate_mol_basic_info`、`calculate_mol_drug_chemistry`、`calculate_mol_hbond`、`calculate_mol_hydrophobicity`，拿到真实属性后再筛选。这与本文追踪的 PF teacher 样本模式高度一致。因而 v6 的 58% 至少说明：在该 compatibility harness 下，SFT+ToolRL checkpoint 能把已学到的“属性题先算属性”行为部署为真实调用；它不是靠 50 题都不调用工具而猜出 29 题。

MS-1 明显好于 MS-2，不只是因为“前者容易”。MS-1 与 145 条 PF teacher 轨迹中的稳定模板高度同构：题目给候选 SMILES 和确定性阈值，模型通常只需一次成组调用几个直接 descriptor tool，再按 observation 做机械过滤；它不依赖蛋白结构检索、server/local artifact 转换、skill path、shell 文件处理、pocket/docking 的长链依赖。实际 MS-1 skill-read 次数为 0、平均仅 2.96 step，且工具执行成功率为 208/210。MS-2 则要求从 target 语义到结构/口袋/结合证据的多步选择，每一步都可能产生 resource/path 或 server failure；一旦模型进入错误状态，离线训练又没有覆盖自己的失败分支和跨步 credit，错误便累积成 loop 或超长生成。

因此，从本次测评能够直接归纳出的 v6 模型突出问题是：ReAct/JSON 与 task-specific final schema 仍脆弱；复杂任务首步 action 就可能坏掉；local skill/path 和 server artifact contract 没有真正掌握；对失败 observation 缺乏状态更新，容易机械重复；复杂 workflow 的工具选择、required argument 和停止策略不可靠；即使到达 final，仍有内容错误。compat profile 已经窄幅堵住 comma-separated multi-call、decision 后续写、transport marker、单次 redundant close 和 skill 路径“完全没提示”这些接缝问题，所以上述残余 failure 不能再归咎于这些已知表面变体；它们更接近模型 policy、训练 state distribution 和 credit assignment 本身的问题。

但因果归属仍不闭合。训练时没有执行这些 action，评测时才执行；SFT 和 ToolRL 都可能影响调用，base model 先验也可能影响，而 compatibility parser、tool exposure 和 deterministic decoding 进一步改变行为。没有同一 harness 下的 base、SFT-only、ToolRL 对照，就不能说 58% 是“v6 ToolRL 带来的提升”。它是整套 checkpoint+harness 的结果。

## v6 与部署目标之间真正剩下的差距

沿当前样本看，训练与评测之间的界面差异可以压缩为：

```text
SFT：完整 teacher episode，一次 teacher forcing；每个局部条件来自 teacher

ToolRL：teacher state → 一次 sampled decision → teacher/schema reward → 结束

评测：原始问题 → 自己的 decision → 真工具 observation → 自己的新 state
      → 错误恢复/继续调用 → 自主停止 → benchmark correctness
```

v6 的 SFT 覆盖了完整成功轨迹，所以能建立较强的 tool-use prior；offline GRPO 又能优化局部 XML/tool/final exactness。但它从未在训练中让 policy 承担自己的 action 后果。模型漏调工具时看不到最终失败；错误参数没有形成真实 error observation；一次调用成功后是否继续、何时停止、episode token/tool cost 都没有统一 reward；final exact 的负 credit 也回不到此前 tool decision。

这就是为什么 v6 可以同时出现两种看似矛盾的事实：旧 harness 中 MS-1 50/50 都调用工具且 58% 正确，说明 pipeline 有实际能力；而训练定义仍不够成为面向 DSH 的 full-episode ToolRL baseline。前者是部署结果，不能反向改变后者的训练计算图。

## v6 是否具备“正规正确、baseline”属性

| 命题 | 结论 | 证据与限定 |
|---|---|---|
| 是符合 Slime/Qwen 机械合同的真实 SFT+GRPO 吗？ | 是 | 全参数 SFT/ToolRL 完成，模板、assistant mask、batch、KL、checkpoint 和有限梯度均有运行证据 |
| 数据协议和 SFT/ToolRL target 一致吗？ | 是 | 5,329 个 target byte-exact；multi-call/runtime/reward parser 审计通过 |
| corpus 是当前环境逐条重放验证的 executable trajectories 吗？ | 不是 | 来源是历史 teacher 轨迹；legacy artifact/path 行为未做全量当前环境 replay |
| ToolRL 是在线完整 episode RL 吗？ | 不是 | 普通单 completion rollout，不执行工具，不延续 sampled branch |
| final reward 有答案内容信号吗？ | 有，但目标偏窄 | teacher structured-final exact；不是官方 benchmark correctness，可能错罚语义等价/evidence 不同答案 |
| 2,252 个 decision 是否 clean exactly-once？ | 最终 key 覆盖完整，但证据不如 v7 干净 | 保存失败后 100 个 key 重采样；需按恢复分支重建 effective traversal |
| 58% 可作为 checkpoint-only 的科学基线吗？ | 目前不足 | compatibility harness 与后续 DSH 不同，且缺 base/SFT-only/ToolRL 同环境对照 |

因此 v6 的“baseline 属性”应分层表达。作为可运行的历史轨迹 imitation + offline local-action GRPO 基线，它成立，而且比此前松散版本正规；作为证明 ToolRL 提升 MolBench-MS 的科学 baseline，它缺少同 harness control、不可变 code provenance 和与 benchmark 对齐的 episodic reward；作为标准 DSH agent baseline，它还受 compatibility scaffold 与环境合同差异限制。

## v6 留给 v7 的升级起点，以及不能被误解的变化

v7 不是从无到有建立 XML/multi-call/SFT→ToolRL。v6 已经完成 canonical XML、同容器 multi-call、serializer parity、静态 coverage selection、SFT warm start、冻结 reference、hierarchical tool reward 和长上下文 gates。v7 真正继承并加强的是可审计性与数据合同：清理重复 evidence/失败重试，将 98 条 VS final 的 `selected_smiles` 从 scalar string 包成 singleton list，解开 artifact wrapper，建立 resource URI/workspace path contract，mask 已知错误 local path action，并生成 clean exactly-once traversal audit。这个 VS 字段修复不是“把所有问题和答案统一成 list”：user 问题文本没有被重写；PF 的 `selected_smiles` 和 VS 的 `ranked_smiles` 在 v6 已经是 array；AC 的 `answer_smiles` 到 v7 仍是 string，E2E 的 `result` 是 array。

与此同时，v7 在最重要的 terminal learning signal 上发生了退化：v6 虽是过严的 teacher structured exact，至少区分答案内容；v7 将 `terminal_correctness` 置零，只按 final envelope 合法性给 +1/-0.5。于是 v7 final reward 均值从约 -0.228 升到约 +0.586，主要反映评分尺变化，不是答案能力提高。v6 的 74.7% final 零方差多数来自“四个都不 exact”，v7 的 46.7% final 零方差中则有大量“四个都格式合法”；两者都缺 GRPO 偏好，却以不同方式缺信号。

评测层面也不能直接说“v6 58%→v7 7% 完全由权重退化”。v6 是旧 compatibility harness；v7 是标准 DSH，两者的 prompt scaffolding、tool/skill 暴露、parser 和 stop behavior 不同。v6 告诉我们的最强事实是：当模型稳定进入真实属性计算链时，MS-1 能达到明显更高的正确率；v7 同一 DSH 内部又显示未调用 MCP 的 76 次全部错误。二者结合指向 tool-use initiation/continuation 是首要诊断轴，但严格版本比较仍必须在同一 DSH 条件重跑 base、v6、v7。

## 给专家诊断时最值得追问的几个因果问题

汇报 v6 时，问题不应停在“学习率是否太小”或“context 是否够长”。应请专家沿同一条样本逐层判断：365 条历史轨迹是否足以代表部署 state distribution；teacher/provider 与当前环境是否一致；SFT-only 是否已经形成调用 prior；initial prompt 没有完整 catalog 时，模型凭什么可靠发现工具；offline action reward 是否会强化 teacher quirks；structured exact 是否过度惩罚 evidence 表达差异；39.1% 零方差 group 是否让大量 update 无任务 advantage；恢复后的 100 个重采样 decision 是否影响可比性；compatibility parser 对 100% MS-1 tool-call rate 有多大贡献；以及相同 checkpoint 在标准 DSH 下是否仍会稳定调用 MCP。

最小而有解释力的重评不是再比较两份旧汇总，而是在冻结的同一 DSH commit、skill snapshot、tool server、prompt、decoding 和题集上运行 base、v6 SFT-only、v6 iter562、v7 SFT-only、v7 final，多 seed 汇报 MCP invocation、合法 action、tool execution、合法 final、官方 correctness、`correct | MCP-called`、总步骤/token 和错误终态。这样才能把数据清理、SFT、offline GRPO、reward 变更和 harness 影响分别识别出来。

## 汇报时可以使用的一段总述

我们从 605 条历史 teacher agent 轨迹中按 MolBench membership 固定出 365 条 PF/AC/VS/E2E 完整轨迹，v6 将它们正规化为统一 XML ReAct，并对 5,329 个 assistant decision 做了 SFT/ToolRL serializer byte-exact、multi-call 和 runtime/reward parser 审计。SFT 把整条 teacher episode 送给 Qwen3.5-9B，以 assistant-only teacher forcing 学全部 5,329 个局部决策；365 条 unique 因 GBS=2 物化为 366 行，实际训练 183 步。随后转换器把同一批轨迹拆成 5,329 个 `teacher-prefix→next-action` row，静态选择 2,252 个，每个采四次做 GRPO。工具 reward 是 teacher/schema-local 分层评分；final reward 要求与 teacher structured final 精确一致而非官方 benchmark correctness。rollout 不执行 sampled call、不追加自己的 observation，也不继续完整 episode，所以 terminal correctness 无法给此前工具动作分配 credit。ToolRL 在 iter224 保存失败后从有效 checkpoint 恢复，最终 iter562 完成；恢复造成 100 个 decision 被重采样，按最终分支可重建 2,252 个 effective group，其中约 39.1% 零方差，final 零方差约 74.7%。在旧 compatibility 在线评测中，50 道 MS-1 全部真正调用工具，共 210 次，29/50 正确；这说明 checkpoint+harness 能稳定进入属性计算链，却不能单独证明 offline ToolRL 的增益。v6 因而是工程成立的离线局部决策 baseline，而不是已与标准 DSH、官方 episodic correctness 闭环的 ToolRL baseline。

## 关键证据索引

v6 frozen data 位于 `outputs/slime_drug_agent_data/live_tool_catalog_v6-final-mol-sftnrl`。数据谱系、hash、正规化和实际/official 两套 view 见 `dataset_manifest.json`；decision 转换统计见 `toolrl/toolrl_steps.report.json`；静态选择见 `toolrl/context_manifest.json`；协议审计见 `audit/reasoning_action_segmentation.json`、`audit/sft_toolrl_serializer_parity.json`、`audit/runtime_parser_compatibility.json` 和 `audit/validation.production.json`。追踪样本 ID 为 `react_pf_303eea476077228e`，可在 `react_trajectories.jsonl`、`canonical_trajectories.jsonl` 与 `toolrl/toolrl_steps.jsonl` 中对应复核。

实际 run 位于 `outputs/slime_drug_agent_runs/Qwen3.5-9B_v6_final_mol_production_20260817`。实际 profile/参数见 `serial_config.env` 与 `resolved_config.env`；SFT/ToolRL 命令、loss、reward component 和故障见 `logs/sft.log`、`logs/toolrl.log`、`logs/toolrl.log.attempt_20260818_032720` 与 `status.log`；固定遍历记录为 run 根目录的 `toolrl_fixed_traversal.jsonl`；失败 checkpoint 被保存在 `toolrl/iter_0000224.failed_incomplete_20260818`，最终 checkpoint 为 iter 562。

评测位于 `outputs/slime_drug_agent_evals/molbench_ms1_ms2_v6mol_iter562_compat_skills_final_20260819_043926`。评测合同见 `run_manifest.json`、`eval_config.yaml` 与 `benchmark_manifest.json`；逐题 action/工具/终态见 `traces.jsonl`；训练重合排除见 `overlap_audit.jsonl`；MS-1/MS-2 官方结果分别见 `preds/rdkit_bench/all.json` 与 `preds/acnet_curated/all.json`。任何 v6→v7 分数比较都应同时注明 v6 使用 `v6_mol_eval_compat`、v7 使用标准 DSH，除非先完成同 harness 重评。
