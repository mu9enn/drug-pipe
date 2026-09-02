# v7-release 训练的样本因果生命线：从 365 条历史轨迹到 DSH 失配

> 本文用于向具备 SFT、GRPO/ToolRL 和 agent 后训练经验的专家汇报，也用于我们自己重新完整掌握这次训练。它不把“v7 相比此前做了什么改进”和“v7 自身究竟在训练什么”拆成两个话题，而是沿一条真实样本的生命线，把每次数据变换、模型看到的条件、loss/reward 的来源、梯度能归因到哪里，以及最终评测为什么会退化连起来讲。

配套的 v6 同定位文档见 [`v6_final_sample_causal_lifecycle_zh.md`](v6_final_sample_causal_lifecycle_zh.md)。两份文档追踪同一个 PF 样本，因此既可以分别独立汇报，也可以逐阶段对照版本变化；严格比较时仍应服从各自 run 产物和不同 evaluation harness 的边界。

## 先给出准确结论

v7-release 是一套**工程上完整跑通、数据和 batch 遍历经过较强审计的 Qwen3.5-9B 全参数 SFT + 离线 decision-level GRPO**。它符合 Slime 的训练接口，Qwen chat template、loss mask、上下文长度、SFT warm start、冻结 reference model、GRPO 分组和 checkpoint 都真实工作了。它继承了 v6 已经建立的 XML ReAct、同容器 multi-call、SFT/ToolRL serializer parity、静态 coverage selection、长样本 gate 和串行 SFT→ToolRL；相较 v6 真正新增或加强的重点，是 evidence/失败重试清理、artifact/resource/path contract、错误 local path action masking，以及无重复无遗漏的 exact-epoch traversal 审计。

但是，如果“baseline”指的是“能够回答这 365 条 MolBench-style 训练轨迹所代表的任务，并在 DSH 中从零自主调用 MolClaw、读取 observation、处理错误、停止并交付正确答案”的 SFT+ToolRL 基线，那么 v7 **还不具备这个 baseline 属性**。更准确的名称应是：

> **full-trajectory teacher-forced SFT + teacher-prefix offline next-decision GRPO**

而不是未经限定的“端到端 ToolRL”。其根本原因不是 Slime 没有训练、Qwen 没有更新，也不是 rollout 普遍被 16K 截断，而是 ToolRL 的“rollout”只生成一个离线前缀之后的下一段 assistant 文本：生成的工具调用不被执行，生成的 observation 不会进入后续上下文，episode 不会从该分支继续，最终 benchmark correctness 也不会向此前工具选择分配 credit。更严重的是，当前 hierarchical reward 对终局语义正确性明确置零：一个内容错误甚至语义空洞、但 envelope 合法的 `<final_answer>` 得 +1。

两次 v7 DSH 评测把这个目标错位暴露得非常直接：两次各 50 道 MS-1，各自只有 12 道真正调用了 MolClaw MCP；合计 100 个 rollout 中，调用 MCP 的 24 个答对 7 个，未调用 MCP 的 76 个答对 0 个。也就是说，当前 MS-1 的首要失败是模型没有稳定进入计算链；在已经进入计算链的条件下仍只有 7/24（29.17%）答对，计算后的筛选、格式和答案交付才是第二层问题。

## v7 相较 v6 的 upgrade 总览

为了汇报时能直接定位版本变化，这里把 v6→v7 单独列成一栏。但“发生了变化”不自动等于“训练目标得到升级”：v7 的主要进步在数据合同和运行可审计性，核心训练范式基本沿用 v6，terminal reward 反而退化。

| 维度 | v6 状态 | v7 的变化 | 应怎样评价 |
|---|---|---|---|
| XML ReAct 与 multi-call | 已统一 XML、同一 `<tool_call>` 容器内多 JSON object，SFT/ToolRL serializer 与 parser 已审计 | 原样继承；仍有 1,527 个 multi-call turn | 是必要基础，但不是 v7 新增 upgrade |
| 历史轨迹内容清理 | 365 条、5,329 个 assistant decision；仍有重复 evidence、一次完全相同的失败重试；98 条 VS final 的 `selected_smiles` 是 scalar string | 删除 841 个重复 evidence item、1 对相同失败重试；仅将这 98 个 VS `selected_smiles` 规范成 singleton list；decision 变为 5,328 | 减少重复/自相矛盾监督，是真实的数据质量升级；不是把所有题目和所有答案统一成 list |
| artifact/path 合同 | 保留大量 `<artifact:...>`，没有严格区分 server resource 与 local workspace path | 解开 1,843 个 local artifact wrapper；将 17,080 次 legacy artifact literal 出现全部正规化为 `resource://...` 或 `workspace/...`；catalog 参数说明同步区分两类输入 | 真正的接口合同升级；这些数字是 literal occurrence，不是独立文件数 |
| 已知错误 path action | v6 会把这类历史 action 当正常 teacher target 学习 | 识别 116 个“server resource 交给本地 filesystem tool”的违规 call，mask 对应 115 个 assistant message，并保留 observation | 避免直接模仿已知错误，是真升级；但没有补采同状态下的正确替代 action |
| SFT 监督规模 | 5,329 个 assistant decision 全部可监督 | canonical 仍有 5,328 个 decision，其中 115 个 path action `step_loss_mask=0`，所以实际有 loss 的 assistant message 为 5,213 个 | 监督数量减少是质量过滤的代价，不代表增加了能力覆盖 |
| ToolRL 数据视图 | 5,329 raw decision→2,252 production decision，1,888 tool + 364 final | 先因 mask 得到 5,213 个 eligible decision，再选 2,208 个，1,844 tool + 364 final | selector 思路沿用 v6；upgrade 是输入更干净，不是从 offline selection 升成 online rollout |
| traversal 与恢复证据 | 最终覆盖 2,252 个 key，但保存失败/恢复使 100 个 key 在原始 traversal 中重复采样，需重建有效分支 | 2,208 个 group、552 个 batch、8,832 个 completion；post-run audit 证明每个 expected key 恰好消费一次，无重复、遗漏或越界 | 是运行可复核性的明确升级 |
| tool-step reward | hierarchical teacher/schema-local reward | 核心定义基本沿用 | 不是任务级 upgrade；仍未执行 sampled call，也没有 environment outcome |
| final reward | 合法且与 teacher structured final exact 才得 +1，虽过严但检查内容 | 改成只检查合法 final envelope；`terminal_correctness=0` | 这是关键退化，不是 upgrade；`-0.228→+0.586` 主要来自换了宽松标尺 |
| ToolRL 因果单位 | teacher prefix→单个 sampled next decision | 未变；仍不执行工具、不续写 sampled episode、不回传官方 correctness | 没有升级成 full-episode ToolRL，是 v7 最核心的科学缺口 |
| 评测与 provenance | v6 主要在 compatibility harness 评测；run code commit 也是 `unknown` | v7 用标准 DSH 暴露了真实部署失配，并增加 dataset hash/path/traversal audit；但本次 run 的 `CODE_COMMIT` 仍是 `unknown` | 诊断与数据可追溯性改善；不能把 harness 变化算作 checkpoint 能力升级 |

因此，最简洁而不失真的汇报口径是：**v7 把 v6 的历史轨迹离线 SFT+GRPO 管线做得更干净、更符合资源/文件接口、更可审计；它没有升级 ToolRL 的因果单位，也没有对 MolBench correctness 建立端到端 credit，且把 final 内容 reward 降级成了格式 reward。**

## 这份判断依据的时间边界

`docs/1 on 1 2607&08.pdf` 的最后一次汇报是 2026-08-19；v7-release 数据在 2026-08-21 固化，实际 SFT→ToolRL 运行发生在 08-21 至 08-22，两次关键 DSH 评测发生在 09-01。因此 PDF 覆盖了问题演化的背景——例如 local tool decision 没有被充分学习、长 observation、重复 thought/loop、单/多 call 协议、reward 和 context 的讨论——但**没有覆盖本文分析的最终 v7-release run**。本文对 v7 的事实以固化 dataset manifest、run-local resolved config、训练日志、traversal audit、checkpoint 和 DSH 评测产物为准；PDF 只用于解释为什么后来会做这些改造。

另一个需要主动向专家披露的 provenance 限制是：该 run 的 `serial_config.env` 虽记录了数据 hash，却把 `CODE_COMMIT` 记成了 `unknown`。所以数据版本可以精确复现，实际运行代码只能由 run 产物与当前代码相互印证，不能声称已经由 commit hash 完整冻结。这本身也是它尚未成为严格科学 baseline 的一处缺口。

## 一条真实样本最初是什么

下面固定追踪 v7-release 中的真实样本 `react_pf_303eea476077228e`。它是一道典型的 property-filtering（PF，和 MS-1 的确定性属性过滤机制相同）任务：用户给出 10 个 SMILES，要求同时满足 Lipinski 条件，并额外满足 `MolLogP <= 3.55`、`MolWt <= 285.37`，输出所有合格分子。

SFT/canonical 文件是“一行一条完整轨迹”的 JSONL，而不是一行一个 prompt-response pair。删去长文本后，这条记录的外形如下；注意工具 observation 在 chat 层被保存为 `role="user"` 的 XML observation message，而不是另设一个非 Qwen 标准 role：

```json
{
  "id": "react_pf_303eea476077228e",
  "messages": [
    {"role": "system",    "content": "...", "step_loss_mask": 0},
    {"role": "user",      "content": "...", "step_loss_mask": 0},
    {"role": "assistant", "content": "<thought>...</thought><tool_call>...</tool_call>", "step_loss_mask": 1},
    {"role": "user",      "content": "<observation ...>...</observation>", "step_loss_mask": 0},
    {"role": "assistant", "content": "<thought>...</thought><final_answer>...</final_answer>", "step_loss_mask": 1}
  ],
  "metadata": {"path_reference_contract": "workspace-path-or-resource-uri-v1"}
}
```

历史 teacher 轨迹只有五条 message，却已经构成一条完整、可验证的因果链：

```text
system: 统一 XML ReAct 约定
user:   10 个候选 SMILES + 确定性属性阈值
assistant:
  <thought>需要计算 MW/HBD/HBA/LogP……</thought>
  <tool_call>
    {calculate_mol_basic_info, 10 个 smiles}
    {calculate_mol_hbond, 10 个 smiles}
    {calculate_mol_hydrophobicity, 10 个 smiles}
  </tool_call>
user:
  <observation tool_name="calculate_mol_basic_info">真实计算结果</observation>
  <observation tool_name="calculate_mol_hbond">真实计算结果</observation>
  <observation tool_name="calculate_mol_hydrophobicity">真实计算结果</observation>
assistant:
  <thought>逐个对照阈值……</thought>
  <final_answer>{
    "task_type": "pf",
    "selected_smiles": [
      "Clc1cccc(Cl)c1N=C1NCCN1",
      "CCc1nc(N)nc(N)c1-c1ccc(Cl)cc1"
    ],
    "evidence": [...]
  }</final_answer>
```

这里最重要的不是 XML 长什么样，而是 teacher 轨迹中存在真实的因果依赖：最终两个 SMILES 之所以能被选出，是因为前一步实际计算出了每个分子的 MW、HBD、HBA 和 LogP，后一步读取这些 observation 再应用阈值。MS-1 本质上也正是这种可确定执行的 RDKit 属性过滤。模型如果不计算，就不是“可能凭推理答对但精度稍差”，而是在缺少决定答案所需变量的情况下猜测。

这条轨迹属于 365 条 MolBench-style 完整历史轨迹之一。其更上游的谱系与 v6 相同：历史 v5 release 有 605 条由 `deepseek-v4-flash-0731` teacher 生成并清理的轨迹，v5-mol 依据审计过的 source-ID membership 精确保留 365 条、排除 240 条；v7 没有另行采集新 episode，而是在这 365 条内容上继续清理和修复。需要避免一个容易混淆的表述：这 365 条不是“365 道独立的 DSH MS-1”；它们覆盖 `pf/ac/vs/e2e` 四种 task type，共有 5,328 个 assistant decision 位置，其中 4,963 个是工具 decision，365 个是 final decision。后续 path-contract pass 又把其中 115 个已知错误的 assistant action 置为不参与 loss，因此真正可进入 ToolRL 的 decision 是 5,213 个。轨迹的长短差异很大；这里选中 PF 样本，是因为它短、因果关系清楚，而且恰好也是为 SFT batch 对齐而重复一次的样本。

## v7 先怎样修复这条历史轨迹

v7 没有重新采集环境交互，而是在 v6 的 365 条历史轨迹上做了更严格的清理和协议固化。所有 system/user/observation message 的 `step_loss_mask=0`；正常、可监督的 assistant message 的 mask 为 1，115 个 path-contract violation assistant message 则被显式置为 0。observation 作为条件保留，但不让模型拟合工具返回文本。它继续使用 v6 已经正规化并审计过的 assistant decision 序列化形式：

```xml
<thought>...</thought>
<tool_call>
{"tool_name":"...","arguments":{...}}
{"tool_name":"...","arguments":{...}}
</tool_call>
```

或终局：

```xml
<thought>...</thought>
<final_answer>{...}</final_answer>
```

multi-call 不被拆成多个 assistant turn，也不使用 JSON array，而是在**同一个** `<tool_call>` 容器内放置多行 JSON object。追踪样本中三个属性计算正是一个 multi-call decision；SFT serializer、ToolRL parser 与 runtime parser 因而对“这是一个决策还是三个决策”达成一致。365 条轨迹中共有 1,527 个这种 multi-call turn。这是 v6 已经完成、v7 继续保持的实质性协议基础，而不是 v7 新增加的排版变化。

相较 v6，清理还删除了 841 个重复 evidence item 和 1 对完全相同的失败重试，解开了 1,843 个 local artifact wrapper，规范化了 98 个 VS `selected_smiles`，并把 4,963 条 observation 全部确认成不参与 loss。随后 path-contract pass 把 17,080 次 legacy artifact literal 出现全部转成 `resource://...` 或 `workspace/...`。这里的 17,080 是整个 JSONL 中文本引用的出现次数，不是 17,080 个独立文件，也不是 17,080 个违规调用。

### v7 是否把所有问题和答案都统一成 list

没有。对 v6 与 v7 的 365 条 canonical trajectory 逐条检查后，实际变化严格限定在 VS structured final 的一个字段：v6 的 98 条 VS 都是 `"selected_smiles":"某一个 SMILES"`，v7 将其变成 `"selected_smiles":["某一个 SMILES"]`。这是 Python 清理脚本的确定性类型转换，不是 LLM 重写。

其他任务类型没有被统一成同一种 list schema：145 条 PF 的 `selected_smiles` 在 v6 已经全部是 array；98 条 VS 的完整 `ranked_smiles` 在 v6 也已经全部是 array；104 条 AC 到 v7 仍使用 scalar string `answer_smiles`；18 条 E2E 的 `result` 是 array。清理脚本没有重写原始非-observation user message；对 v6 和 v7-final 的这些问题文本逐条比较完全相同。因此不能汇报成“v7 把所有问题要求和答案改成 list”，只能说：

> **v7 修复了 VS structured final 中 `selected_smiles` 字段与 PF/集合式消费约定不一致的问题：98 个 scalar 被包装成 singleton list；问题文本、其他 task schema 和已经为 list 的 `ranked_smiles` 没有做全局统一。**

### “把 server resource 错当本地文件路径”的 116 个调用究竟是什么

这里原来的简写确实容易误导。116 个调用并不是“凡是工具参数里出现 server 文件就判错”，也不是说远端工具不应接收 server 文件。它们在原始历史轨迹里通常写成 `<artifact:structure/P36544_8V8A.pdb>` 这类 legacy artifact reference；它所指向的实体位于 MolClaw/server 一侧，但它本身不是可跨环境复用的裸 `/server/...` 绝对路径。v7 将它正规化成模型可见、稳定的逻辑句柄 `resource://structure/P36544_8V8A.pdb`。运行时的 per-task `ArtifactRegistry` 保存该句柄到真实 server path 的映射，并只在执行兼容的远端工具前解析回 server path。

v7 实际存在三个文件/资源域：

| 模型可见形式 | 所属域 | 谁可以使用 |
|---|---|---|
| `resource://structure/a.pdb` | MolClaw/server 侧资源的逻辑句柄 | 接受 server resource 的远端 MolClaw/MCP 工具；runtime 会在执行前解析成真实 server path |
| `workspace/a.pdb` | 当前 task 的本地隔离 workspace | 本地 `Read/Write/Edit/Bash/Grep/Glob` |
| `skills/.../SKILL.md` | 本地只读 skill mount | 本地 discovery/读取工具；不是 server artifact，也不是一般输出目录 |

所以你的理解对了一半：`calculate_pdb_basic_info`、`fpocket_toolkit`、`fix_pdb` 等大量远端工具，当然应该继续使用 server resource；在模型接口上应传 `resource://...`，由 runtime 解析，而不应让模型记住某台机器上的裸绝对路径。真正违规的是把这个**远端资源句柄交给仅能访问本 task 本地 workspace 的 filesystem tool**。116 个违规 call 的工具分布恰好是：`Read` 81、`Bash` 32、`Edit` 2、`Write` 1；没有一项是 `calculate_pdb_basic_info`、`fpocket_toolkit` 等远端 resource consumer。

真实轨迹中有一个很直观的 multi-call：

```xml
<tool_call>
{"tool_name":"Read","arguments":{"file_path":"resource://structure/P36544_8V8A.pdb","limit":100}}
{"tool_name":"calculate_pdb_basic_info","arguments":{"pdb_file_path":"resource://structure/P36544_8V8A.pdb"}}
</tool_call>
```

紧接着的历史 observation 显示：本地 `Read` 返回 `EACCES ... resource://structure/P36544_8V8A.pdb`，而远端 `calculate_pdb_basic_info` 对同一资源成功计算出 PDB 指标。这正好证明问题不在“路径参数中出现 server resource”，而在**消费该参数的工具属于哪个执行域**。

若模型确实需要用本地 `Read/Bash` 检查 server 文件，正确链路不是直接读取 `resource://...`，而是先调用远端 `server_file_to_base64(file_path=resource://...)`；harness 会在模型不可见地解码并原子写入 task workspace，再返回如 `workspace/artifacts/a.pdb` 的本地路径，之后本地工具才能使用。反方向若要让远端工具消费本地生成文件，则需要先走对应的上传/注册链路，得到新的 server resource handle。路径字符串看起来都像“文件”，但其可达性和生命周期不同，不能互换。

path audit 没有改写这 116 个 action、假装 teacher 当时做对了，而是将其所在的 115 个 assistant message 标记为 `step_loss_mask=0` 和 `path_contract_supervision_masked=true`。116 是违规 call 数，115 是 assistant message 数；二者不同是因为一个 assistant message 可以是 multi-call turn。原 action 文本仍作为历史上下文保留，紧随其后的 115 条 observation 也保留且不参与 loss；ToolRL 转换器不把这 115 个 action 建成训练 target，但更晚的 decision prompt 仍可能包含这段失败 action/observation 历史。这样做可以让后续 recovery decision 保持原有因果条件，同时避免直接最大化错误 action 的似然。

这个处理仍有明确局限：它只是“不给错误动作正监督”，并没有在相同 prefix 下提供 `server_file_to_base64→workspace/...→Read` 或“直接改用远端 consumer”的正确 teacher action。错误 action 也没有从上下文中完全消失。因此它提高了监督合同的正确性，却不等价于模型已经学会了 server/local resource transfer。

这个选择改善了监督的正确性，却也带来一个应明确报告的取舍：SFT canonical trajectory 仍统计到 5,328 个 assistant decision，但 ToolRL 转换只允许 5,213 个未被 path-contract mask 的 decision 进入 raw pool，少掉的正是这些不应强化的 action。也就是说，v7 的升级主要是“不要学习已知错误的本地/资源路径行为”，并没有补采一条在新 path contract 下真实可执行的替代轨迹。对依赖 Read/Bash/Edit/Write 的任务，这能减少错误模仿，却不能凭空提供正确动作的正监督。

### v7 是否把四类任务的答案都统一成了 list

没有。逐条比较 v6 与 v7 的 365 条 canonical trajectory 后，`system + 第一条原始 user question` 完全相同；两版投影内容的 SHA-256 都是 `65f6853cbdffc572a912b4b63551f486e6e1154357e49d9ff6bea29b20131b6a`。因此 v7 没有重写问题要求，也没有告诉四类题“一律以 list 作答”。v7 cleanup 只执行了一条窄规则：当 `task_type` 是 PF 或 VS 且 `selected_smiles` 仍为 string 时，将它包成 singleton list。PF 原本已经全是 list，所以实际改变的恰好是 98 条 VS。

| task type | v6 final target | v7 final target | v7 是否改变 |
|---|---|---|---|
| 145 条 PF | `selected_smiles: list`，实际长度 1–8 | 相同 | 否 |
| 98 条 VS | `ranked_smiles: list`；`selected_smiles: string` | `ranked_smiles: list`；`selected_smiles: singleton list` | 是，仅改后者 |
| 104 条 AC | `answer_smiles: string` | 相同 | 否 |
| 18 条 E2E | `result: list`，本数据中均为长度 1 | 相同 | 否 |

这里还暴露出一处独立的合同不一致：当前 online final-contract formatter 给 E2E 展示的是 `"result":"task result"`，而 18 条历史 E2E target 的 `result` 都是 singleton list；parser 对 E2E 又只要求 `result` 字段存在而不约束类型。这不是 v6→v7 upgrade，而是 prompt、training target 与 validator 尚未完全统一，应该在下一版单独修复。

“全部 list 化是否更利于学习”应分两层回答。**统一模型面对的外层 answer schema，原则上有小幅潜在收益**：减少字段名和 string/list 类型分支，让格式错误更少，也让 reward/parser 更容易共享同一合同。但是，**把所有现有字段机械地套成 list 并不自动改善任务能力**，因为这些任务的答案语义和基准接口不同：PF 是零到多个分子的无序集合；AC 是二选一、恰好一个 SMILES，官方 MS-2 scorer 接收 string；VS 需要有序 ranking list；E2E 是异构任务结果，不一定是分子集合。当前 adapter 也明确要求 MS-1 projection 是 list、MS-2 是 string、MS-3 是 ranking list。若只改训练 target 而不同时改 prompt contract、parser、reward、projection adapter 和 evaluator，会直接制造新的 train/eval mismatch。

如果下一版要降低 schema entropy，更稳妥的设计不是“所有旧字段都变 list”，而是让 **PF/AC/VS 三类分子答案共享一个模型内部字段**，例如 `answer_smiles: [...]`，再按 `task_type` 施加 cardinality/ordering 约束：PF 为 0..N 个且按集合评分，AC 必须恰好 1 个，VS 为有序列表；adapter 在送官方 scorer 前将 AC 的 singleton list 确定性解包为 string。E2E 保留单独的 `result`，但必须选定一种类型并让训练与部署完全一致。这个改动可能提高结构稳定性，应该作为 v8 的独立 ablation，而不能预期它解决 v7 的主要失败：当前 76 个未调用 MCP 的 MS-1 全错，首先仍是 tool-use/credit-assignment 问题；而 format-only final reward 即使得到更整齐的 list，也没有教模型 list 里应该放哪些正确分子。

数据层面因此可以这样评价：v7 比此前更正规、更可解析、更少自相矛盾；但它仍是对旧成功历史的离线整理，而不是用当前 DSH/MCP 环境重新生成、重新验证的 executable trajectory corpus。

## 同一条样本进入 SFT 时，模型实际学了什么

SFT 不拆这条样本，而是把完整五-message 轨迹作为一条训练记录送入 Slime 的 SFT rollout。Qwen3.5 native thinking 被关闭，训练显式学习数据里的 `<thought>`；`qwen3_5` loss mask 只在 assistant token 上计算交叉熵。于是这条样本对模型提供了两个监督位置：

```text
位置 A：P(teacher 的三个属性工具调用 | system + 用户问题)

位置 B：P(teacher 的筛选推理和正确 final |
         system + 用户问题 + logged 工具调用 + 与它配对的真实 observation)
```

这里已经能看到 SFT 的能力与边界。它确实学习了“看到这种属性过滤问题，应调用哪些工具”，也学习了“拿到正确 descriptor observation 后，应怎样筛选并输出”。模型看到的只是正常 assistant/action/observation history，没有 teacher/student 身份标签；`logged` 只说明这段 prefix 来自固定历史数据。位置 B 的条件始终是数据中已有的 action/observation pair，而不是模型在位置 A 自由生成的 action 及其执行后果。标准 teacher forcing 使每一个局部条件都干净，却没有训练模型在自己的动作偏离历史 action 后恢复，也没有直接优化从零开始整个 episode 的成功概率。

365 条轨迹为了满足 global batch size 2 被物化为 366 条：追加一条最短记录而不是丢尾，重复的恰好就是当前样本，渲染后 4,462 token。因此该样本在一个 SFT epoch 中出现两次，其余 364 条各一次。这不是数据增强，而只是 batch alignment；报告样本量时应同时说清“365 个 unique trajectory、366 个 materialized row”。

实际 SFT 是 Qwen3.5-9B 全参数训练，8×H200，TP=4、PP=1、DP=2，GBS=2，一个 epoch 共 183 个 optimizer step；最大序列 131,072，每 GPU 动态 token budget 16,384。学习率从 `5e-6` 经 5% warmup 后按 cosine 降到 `5e-7`，Adam β 为 0.9/0.95、weight decay 0.1，使用 full recompute 和 per-token SFT loss。训练日志中 183 个 step 均有有限 loss/grad norm：loss 均值约 0.171、从首步约 0.168 到末步约 0.071，grad norm 均值约 1.66。它证明 SFT 数值上确实训练了，不证明 held-out agent 能力变好，因为这一阶段没有对应的 held-out SFT/DSH 评测。这里说“学习 teacher 的 assistant decisions”必须带上刚才的限定：115 个已知 path-contract 错误 action 被保留为上下文但不贡献监督 loss。

从“是否符合 Slime 和 Qwen 的 SFT 范式”看，这一段是 v7 最接近合格 baseline 的部分：模型、模板、mask、batch、优化器和上下文合同彼此一致。它的主要科学限制是只有一个小型历史训练集、无 held-out control、teacher forcing 与部署状态分布不一致，而不是实现本身不成立。

## 因果链在 ToolRL 转换处怎样被切断

SFT 结束后，ToolRL 不是让模型拿着原问题在 MolClaw 环境里重新完成这条五-message episode。转换器把所有轨迹的每个可训练 assistant decision 都变成一条 `teacher-prefix → next assistant segment` 样本。每行 ToolRL JSONL 都有 `prompt`、`label`、`metadata` 三个核心域，并冗余保存便于 reward 读取的 `target_assistant/target_tool_calls/target_final_answer`：

```json
{
  "prompt": ["截至当前 decision 之前的完整 teacher messages"],
  "label": {
    "decision_type": "tool_call | final_answer",
    "assistant_index": 2,
    "assistant_content": "teacher 的下一段 assistant 文本",
    "target_tool_calls": ["结构化 teacher calls"],
    "target_final_answer": "结构化 teacher final 或 null"
  },
  "metadata": {
    "source_id": "...",
    "task_type": "pf|ac|vs|e2e",
    "decision_role": "tool_step|final",
    "is_initial_step": true,
    "prompt_tokens_final": 1234,
    "canonical_target_tokens": 567
  }
}
```

其中上面的 token 数只是 schema 示意，不是当前样本的实值；实际 selector 使用 tokenizer 计算并把实值写入每一行。当前 PF 样本因此变成两道彼此独立的题：

```text
ToolRL row A（initial tool step）
prompt = [system, user]
gold   = teacher 的 <thought> + 三个属性工具调用

ToolRL row B（final）
prompt = [system, user,
          数据集中记录的三个工具调用,
          与这些历史调用配对的真实 observation]
gold   = teacher 的 <thought> + 正确 <final_answer>
```

对 row A，SGLang 从同一个 prompt 以 temperature 1.0 采四个 completion，reward 比较这四个 completion 与 logged target action，然后 GRPO 更新模型。**这四个 completion 中的工具调用没有任何一个被执行。**它们不会产生 observation，也不会成为 row B 的 prefix。到了 row B，模型看到的是静态数据中已经记录的历史调用及其配对 observation，再单独采四个 final completion。

这里不应把 observation 在模型主观上严格归属于 teacher 或 student。chat history 里没有 `teacher_observation` 这种角色标记；从 token 条件的角度，当前模型完全可以把此前 assistant action 与随后 observation 理解为“我刚才做了这个动作并收到了这个结果”。因此，把它简称为“历史 observation”或“logged observation”更准确；“teacher observation”最多只能描述数据来源，不能作为这里的核心问题。

但在 RL 的因果图里必须区分两件事：**模型可以把一段历史当作自己的历史来理解，不等于这段历史是由本轮当前 policy 的 sampled action 实际产生。** row B 中固定的是数据采集时的 `a_t^log` 与环境返回的 `o_t^log`；row A 本轮采样的是另一个 `a_t^sample`。训练没有计算 `Env(a_t^sample)`。即使某次 `a_t^sample` 恰好与 `a_t^log` 相同，系统也没有执行和验证这种对应；一旦二者不同，row B 的 observation 仍不会随之改变。

因此 v7 ToolRL 的真实计算图是：

```text
logged prefix H_t ──> 当前 policy 采样 a_t^sample(1..4)
                  ──> 与 logged target 做局部 reward ──> GRPO update

另一条固定数据 row：
H_(t+1)^log = H_t + a_t^log + o_t^log
             ──> 当前 policy 采样下一 decision ──> 另一个局部 reward

不存在：a_t^sample ──> Env(a_t^sample) ──> o_t^sample
                   ──> 基于 o_t^sample 继续生成 ──> episode reward
```

所以真正重要的不是给 observation 贴 teacher/student 标签，而是它是否与本轮 sampled action **因果配对**。row A 的模型如果完全没调用属性工具，只能在 row A 得到 action-level 负奖励；它不会在同一 sampled branch 中经历“因为没算属性，所以最终 MS-1 答错”。row B 的 final 如果答错，也无法把负 credit 分给 row A 的 sampled tool selection，因为 row B 使用的是固定 logged history，不是同一 sampled trajectory 的延续。

全数据从 5,213 个 raw decision 进一步静态选择成 2,208 个 production decision：1,844 个 tool step、364 个 final。初始 action 和 final 优先保留；中间 action 按 task type、调用形状、tool/argument names、轨迹深度 decile、近似长度 bin、前一 observation 状态等形成 deterministic stratum，每个 stratum 留一个代表。4 个 no-progress repeat、2 个超过 16K target limit 的 decision 被排除，另外 2,999 个中间 decision 被视为 diversity-equivalent 而不进入本轮；再加 2 个 deterministic filler，使总数能被 rollout batch size 4 整除。365 条轨迹只有 364 个 final 进入 ToolRL，是因为其中一个 terminal target 超过 16K；它仍在 SFT 中，但没有 terminal GRPO 监督。

这个选择比早期运行更可控：没有 runtime dynamic filter，不会随着当前 policy 改变训练集合；2,208 个 decision 被 552 个 rollout batch 恰好消费一遍，traversal audit 证明无重复、无遗漏、无越界。v6 的 production view 是 2,252 个 decision/563 batch，但中途恢复使 traversal 留下 100 个重复 key，需要按恢复分支重建；v7 因 path-contract mask 等变化缩为 2,208/552，并补上了 clean exact-epoch 验证。它是一项真正的工程升级，但它解决的是“离线 decision 数据是否被稳定遍历”，不是“agent 是否完成完整 episode”。

还应把模型在 initial row 上能获得的工具信息说清。实际 v7 production prompt 不注入完整 official tool catalog；catalog 只供 reward/runtime 使用。也就是说，对追踪样本的 row A，模型主要依赖 system contract、预训练/SFT 参数中记住的工具名和已有历史上下文来生成 `calculate_mol_*`，而不是现场拿到一份完整 MCP schema。历史长轨迹中虽然包含 Bash/Read 等 skill-discovery action，但本轮离线 rollout 不执行这些 action，也就不会让“发现到的 skill 内容”进入 sampled branch 的下一轮。这一 prompt/环境差异很可能影响 DSH 中从零启动工具链的稳定性，至少必须作为专家诊断变量，而不能只归咎于参数学习。

## 四次采样怎样得到 reward，以及为什么大量 GRPO 没有任务学习信号

ToolRL 从 SFT checkpoint warm-start，仍是 Qwen3.5-9B 全参数；8×H200 上 TP=4、PP=2、DP=1。每次拿 4 个 decision prompt（RBS=4），每个 prompt 采 4 次（n=4），正好产生 16 个 response，对应 GBS=16 的一次更新；共 552 次更新、8,832 个 sampled response。rollout 使用普通 `slime.rollout.sglang_rollout.generate_rollout`，不是执行 MolClaw episode 的 custom agent rollout。最大 prompt/response/context 分别为 245,760/16,384/262,144，temperature=1.0。

优化采用 GRPO，本次实际命令的 PPO clip low/high 都是 0.2，entropy coefficient=0；ToolRL 学习率恒定 `2e-7`。当前 actor 相对冻结 SFT reference 使用 `low_var_kl` loss，系数 0.001；reward-level KL 系数为 0。actor 常驻、SGLang rollout 权重按阶段 offload，属于内存/稳定性实现，不改变“一个 prompt 只生成一个 completion”的语义。

对工具 row，hierarchical reward 先检查 XML envelope，再把预测 calls 与 teacher calls 做顺序不敏感的匹配，并逐级看 tool set、required arguments、critical values 和 configurable validity。其主要台阶是：非法 envelope `-0.5`，完全错 tool `-0.4`，部分 tool set 为 `-0.05 + 0.35×F1`，缺 required argument 约 `0.30–0.50`，critical argument 不同约 `0.55–0.70`，config 不合法约 `0.72–0.85`，与 teacher 等价为 `1.0`，满足 schema 的可接受替代配置为 `0.9`。所以把它概括成“teacher action 相似度”是正确方向，但要补一句：它不是单纯字符串 exact match，而是**以 teacher action 和静态 tool schema 为参照的分层局部评分**；仍然不是工具执行结果或任务成功评分。

对 final row，当前代码先调用结构化 exact scorer，随后主动覆盖了其结论：

```python
valid = parsed.ok and parsed.has_final_answer and not parsed.has_tool_call
score = 1.0 if valid else -0.5
terminal_correctness = 0.0
structured_final_exact_enabled = False
```

所以以下两个输出在训练目标中没有区别：

```xml
<final_answer>{"task_type":"pf","selected_smiles":[正确的两个分子],"evidence":[...]}</final_answer>
```

```xml
<final_answer>{"task_type":"pf","selected_smiles":[],"evidence":[]}</final_answer>
```

只要第二个能被 parser 接受、没有同时夹带 tool call，也得到 +1。这里“格式合法”不只是成对 XML tag，还包括 parser 支持的结构化 final schema；但它仍然不验证 selected SMILES 是否正确，也不验证 evidence 是否由真实 observation 支持。

GRPO 对同一 prompt 的四个 reward 做组内比较。简化地说，优势取决于 `r_i - group_mean`；若四个 response 同分，不论全是 +1 还是全是 -0.5，policy-gradient advantage 都为零。v7 实际 traversal 中：

| decision role | group 数 | 四样本同分、零方差 | 可提供组内偏好 |
|---|---:|---:|---:|
| tool step | 1,844 | 601（32.59%） | 1,243 |
| final | 364 | 170（46.70%） | 194 |
| 合计 | 2,208 | 771（34.92%） | 1,437 |

170 个零方差 final group 中，144 个是四个 sample 全部 +1，26 个是全部 -0.5。全部 1,456 个 final response 里有 1,054 个得 +1、402 个得 -0.5，`terminal_correctness` 对全部样本恒为 0。也就是说，很多 final group 的确只在确认“模型已经稳定地产生某种合法 envelope”，完全没有告诉模型四个答案中哪个更正确。

这解释了 v7 final-group raw reward 均值约 `+0.58585`，而按 v6 恢复后有效 traversal 重建的均值约为 `-0.22802` 时为什么不能说“答案能力大幅提升”。v6 的 terminal reward 要求与 teacher structured final 精确一致（比较时忽略重复的 summary 字段），确实检查内容，却也不是官方 MolBench correctness：即使核心 answer/selected SMILES 正确，只要 evidence 等结构化字段与 teacher 不同，也可能被判 -0.5。v7 又从这种“有内容但可能过严”的 teacher-exact 标尺退化成 format-only，reward 标尺本身发生根本变化。跨 definition 比 raw reward，等同于用两套评分规则的考试分数比较。v7 的工具 row raw reward 均值约 `0.03560`，也提示模型在 teacher-local action objective 上远没有“普遍掌握”。

训练日志中的 grad norm 全程有限，说明更新链路不是断的；但 policy-gradient loss 均值接近零、clip fraction 为零，加上 34.92% 的零方差 group，说明“系统能反传”与“任务目标提供了强而正确的学习信号”是两件不同的事。KL 仍会约束模型靠近 SFT reference，却不能补出缺失的 terminal correctness 信号。

同理，约 0.2% 的 rollout truncation 只能回答“8,832 个离线 next-decision completion 中，有多少单段生成撞到 16K response cap”。它完全不能推出“真实 agent episode 很少跑到 16K/总 context 上限”，因为训练根本没有把连续 action、observation、错误重试和最终回答串成一个在线 episode。v7 的 length gates 证明了短、P50、P95、near-limit 单步 batch 可以运行和保存；它们不是 agent completion/termination gate。

## 为什么同一模型到 DSH 后面对的是另一个学习问题

DSH 评测不提供任意一个 teacher prefix。模型从原始问题开始，需要自行发现/读取 skill，选择 MolClaw MCP tool，构造参数，等待真实 observation，根据 observation 决定下一步，处理失败或长结果，最后在预算内停止并交付可被官方 MolBench scorer 解析的答案。每一步都会改变后续状态，前一步错误会让模型进入训练数据从未出现过的 prefix。

于是，训练和评测的接口虽然都叫“tool-using assistant”，实际存在一整条连续失配：

```text
训练：teacher state → 一次局部 action → teacher-local reward → 分支结束
评测：initial state → 自己的 action → 真环境 observation → 自己的新 state
      → 多次 action/error/recovery → 自主停止 → benchmark correctness
```

v7 训练了“在成功历史的某个截面上，下一步像不像 teacher”；DSH 考的是“从初始状态起，能否把自己产生的状态分布维持到成功终局”。前者可以改善 XML 合法性和局部 tool/argument 选择，却不会自动得到后者需要的探索、持续调用、错误恢复、终止和长程 credit assignment。

MS-1 的实测尤其能排除“只是最后筛选小错”的解释。v6 的旧 compatibility harness 上，50/50 道 MS-1 都至少调用一次工具，共 210 次调用，最终 29/50 正确（58%）。v7 的两次标准 DSH run 各 50 道 MS-1：第一次 12 道调用 MCP、其中 4 道正确；第二次也是 12 道调用、其中 3 道正确。合并后：

| v7 MS-1 rollout 状态 | 正确 | 错误 | 条件正确率 |
|---|---:|---:|---:|
| 真正调用 MCP | 7 | 17 | 29.17% |
| 未调用 MCP | 0 | 76 | 0% |
| 总计 | 7 | 93 | 7% |

因此，证据支持如下因果排序：第一瓶颈是 tool-use initiation/continuation——没有调用就没有计算所需属性，76 次全部失败；第二瓶颈才是调用之后能否选对工具、读对 observation、应用所有阈值并合法 final，因为调用后仍有 17/24 失败。不能反过来说“只修 final answer 就能解决 MS-1”。

同时必须保留比较边界：v6 的 58% 来自旧 compatibility harness，v7 的 7% 来自标准 DSH，两者 prompt scaffolding、skill/tool 暴露和执行 harness 不完全相同，所以 `58% → 7%` 不能被当成纯 checkpoint 因果效应。最强、最干净的证据是 v7 同一 DSH 条件内部的 `7/24 vs 0/76`；它直接说明 tool invocation 是当前必要条件。要严格测量 v6→v7 权重退化，必须把 base、v6、v7 在同一 DSH commit、同一 skill snapshot、同一 decoding 配置和多 seed 下重评。

## v7 到底算不算“正规正确、具备 baseline 属性”

若把问题拆成可证伪的几句话，答案会更准确：

| 要判断的命题 | 结论 | 原因 |
|---|---|---|
| v7 是符合 Slime/Qwen 机械合同的一次真实后训练吗？ | 是 | 全参数 SFT 与 GRPO 均完成；模板、mask、并行、batch、KL、checkpoint、traversal 和有限梯度有日志证据 |
| v7 的 SFT 是合理的历史轨迹 imitation baseline 吗？ | 基本是，但证据不完整 | 全轨迹、assistant-only loss 和协议统一合理；缺 held-out、同 harness 的 base/SFT-only 对照，且仍有 teacher-forcing exposure gap |
| v7 的 ToolRL 是 full-episode、environment-interactive ToolRL 吗？ | 不是 | 普通单 completion rollout，不执行工具，不延续 sampled branch，无 episodic correctness/termination/token credit |
| v7 的 reward 与“MolBench-MS 答对”一致吗？ | 不一致 | 工具 reward 是 teacher-local action shaping；final reward 把语义 correctness 置零 |
| v7 可作为当前 MolBench-MS 的科学性能 baseline 吗？ | 目前不足 | 训练目标错位、关键 control arm 缺失、v6/v7 harness 不一致、run 未冻结 code commit |

还有一个命名上容易误导专家的事实：v7-release manifest 同时物化了一条所谓 `official_baseline` 数据视图，共 5,200 个 eligible decision，采用 official tool catalog 注入，并定义为 base checkpoint→ToolRL；但**实际这次 run 不是那条 arm**。实际运行的是 `drug_pipe_production`：SFT warm start→2,208 个静态 decision 的 hierarchical GRPO，prompt 不注入完整 tool catalog，catalog 只用于 reward/runtime。不能因为 manifest 里存在 official-baseline 文件，就称实际 checkpoint 已经跑过 official baseline。

所以最公允的总体判断是：v7 把一个此前较脆弱、协议和运行层面容易混乱的 pipeline，升级成了一个可审计的**离线局部决策训练基线**；但它尚未把目标升级成 DSH 所考的端到端 agent 行为。其工程正规性显著提升，科学目标有效性没有随之闭环。final reward 变宽还让一项核心训练指标产生了虚假的“进步感”。

## 下一版应怎样成为可交给专家诊断的真正 baseline

真正的下一步不是继续调 `2e-7`、KL 系数或 16K cap，而是先让每个训练 sample 的因果单位与评测单位一致。对于 MS-1，一条 RL sample 应从原题开始；policy 生成 tool call 后由同一 DSH/MolClaw contract 真正执行，observation 追加回同一上下文，policy 继续行动，直到合法 final、失败终止或预算耗尽。最终用官方 MS scorer 判断 selected SMILES，episode correctness 作为主 reward；XML/tool schema、有效工具调用、参数合法、及时停止和 token/tool cost 只能作为 shaping，不能替代语义正确性。这个 episode reward 必须能归因到此前的工具选择，否则模型仍然不会学到“稳定计算是答对 MS-1 的必要步骤”。

在实现 full-episode RL 之前，也可以把当前 v7 保留为一个明确标注的 offline decision-level control，而不是丢弃。最有解释力的一组同环境 control 应是：base Qwen、SFT-only、SFT+当前 offline next-decision GRPO、SFT+online episodic ToolRL；四者使用同一 DSH harness、skill snapshot、tool server、decoding 参数、题集与多 seed。这样才能分别回答 SFT 是否提高了 tool-call prior、offline GRPO 是否只是提高 XML/teacher-local action、online credit 是否提高 MCP call rate 和最终 correctness。

训练与评测都应至少统一记录：episode 级 MCP invocation rate、首次有效工具调用率、每题工具步数、tool error/recovery、合法 final rate、官方 correctness、`correct | MCP-called` 与 `correct | no-MCP`、完成前总 token、context/truncation、按 task type 的曲线；GRPO 侧继续记录 reward 方差，但要把“零方差 +1 格式组”与“真正语义同分组”分开。对 MS-1 来说，一个非常直接的早期 gate 应是：从原题开始的完整 rollout 是否稳定调用 RDKit/MolClaw 属性工具，并且工具输出改变最终 selected SMILES；不是只检查一个静态 action completion 是否没被截断。

最后，v7 已有的数据 hash、manifest、path audit、固定 traversal 和 checkpoint gate 应保留；同时补齐 code commit/container/tool-server/DSH commit 的不可变记录，并冻结 train/validation/test 题目边界。这样，下一次向专家说“reward 提高、工具调用率提高、MS-1 提高”时，三个指标才来自同一任务定义和可追溯实现，而不是分别证明格式、局部 imitation 和最终任务的三套不同事情。

## 汇报时可以用的一段总述

我们从 365 条历史完整 MolBench-style 轨迹出发，继承 v6 已统一的 Qwen XML ReAct 与 multi-call turn，再清理重复 evidence/失败重试并修复 artifact/path contract。SFT 保留整条轨迹，对正常 teacher assistant decision 做 assistant-only teacher forcing，同时 mask 掉 115 个已知错误的 path action；这一步让模型学习“在正确历史状态下，teacher 下一步怎么做”。之后同一批轨迹形成 5,213 个可训练的 `teacher-prefix→next-action` decision，再静态选择 2,208 个，每个采样四次做 GRPO。工具 row 的 reward 是以 teacher action 和 tool schema 为参照的局部分层评分；final row 只奖励合法 `<final_answer>`，语义正确性被置零。普通 SGLang rollout 不执行 sampled tool call，也不沿 sampled branch 继续完整 episode，所以最终 benchmark correctness、能否成功停止和 episode 总 token 都无法向此前动作分配 credit。DSH 却要求模型从原题开始发现 skill、连续调用真实 MCP、处理自己的 observation/error 并交付正确 final。两次 v7 MS-1 中，24 个调用 MCP 的 rollout 答对 7 个，76 个没调用的一个也没答对，说明当前首要失败是 tool-use 与长程 credit assignment，调用后的答案质量是第二层失败。故 v7 是一个工程上正规的 offline next-decision SFT+GRPO baseline，不是已经与 DSH 目标闭环的 full-episode ToolRL baseline。

## 关键证据索引

数据事实以 `outputs/slime_drug_agent_data/live_tool_catalog_v7-release-mol-sftnrl/dataset_manifest.json`、同目录 `manifest.json`、`toolrl/context_manifest.json`、`toolrl/toolrl_steps.report.json` 和 `...-source/path_contract_audit.json` 为准。当前样本位于 `react_trajectories.jsonl`，ID 为 `react_pf_303eea476077228e`；其两个 selected ToolRL row 位于 `toolrl/toolrl_steps.jsonl`。

实际 run 为 `outputs/slime_drug_agent_runs/Qwen3.5-9B_v7_release_mol_sft_toolrl_drug_pipe_production_20260821_163914`。SFT 物化事实见 `training_data/sft.manifest.json`，实际参数见 `serial_config.env` 与 `resolved_config.env`，训练事实见 `logs/sft.log`、`logs/toolrl.log`，完整遍历见 `toolrl_fixed_traversal.jsonl` 与 `toolrl_fixed_traversal.audit.json`。

实现语义可从以下位置复核：path/resource 正规化、违规判定与 loss mask 在 `slime/drug_agent/scripts/normalize_v7_path_contract.py`；模型可见句柄到真实 server path 的执行期映射在 `slime/drug_agent/tools/artifact_registry.py`；`server_file_to_base64` 到本地 workspace 的物化在 `slime/drug_agent/tools/server_file_materializer.py`；完整轨迹转 decision 的逻辑在 `slime/drug_agent/toolrl/convert_react_to_toolrl_steps.py`；静态 selector 在 `slime/drug_agent/scripts/select_toolrl_decisions.py`；普通单步 rollout 入口在 `slime/drug_agent/toolrl/scripts/run_toolrl_grpo.sh:68`；SFT assistant-only loss 配置在 `slime/drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh:163`；hierarchical tool reward 在 `slime/drug_agent/toolrl/molclaw_reward.py:814`；format-only final override 在同文件 `:367`；hierarchical dispatch 在同文件 `:990`；production batch 和 fixed traversal 在 `slime/drug_agent/scripts/run_qwen3_5_9b_v4_plan_sft_toolrl_v2.sh:222` 与 `:317`。这些链接用于解释当前实现和 run 产物的一致部分；由于实际 run 的 `CODE_COMMIT=unknown`，不能把当前 dirty worktree 当作 v7 运行时的不可变源码快照，发生歧义时仍以 run-local config/log/traversal 为准。

两次 v7 评测分别位于 `outputs/dsh_molbench_evals/dsh_qwen35_9b_v7_release_toolrl_iter0551_ms1_ms2_worker_rollout_2g_20260901_7875340` 和 `...worker_rollout_retry_20260901_8096559`。v6 对照位于 `outputs/slime_drug_agent_evals/molbench_ms1_ms2_v6mol_iter562_compat_skills_final_20260819_043926`；由于 harness 不同，只应用作历史对照，不应用作严格的 checkpoint-only ablation。
