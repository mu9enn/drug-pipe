# v6 与 v7 模型失败模式及根因：从 MolBench-MS 评测反推训练计算图

> 本文回答一个比“哪个版本分数高”更严格的问题：v6、v7 在真实 MolBench-MS agent 评测中分别失败在什么位置，这些失败由模型、训练目标、数据、运行协议还是评测环境中的什么机制产生。它与 [`v6_final_sample_causal_lifecycle_zh.md`](v6_final_sample_causal_lifecycle_zh.md) 和 [`v7_release_sample_causal_lifecycle_zh.md`](v7_release_sample_causal_lifecycle_zh.md) 配套：两篇生命周期文档从一条训练 sample 向前解释数据与优化；本文从评测 rollout 向后追责，直到训练计算图。

## 先给出诊断结论

v6 和 v7 不是“同一种失败的程度不同”。两者在评测中的主要断点不同：

```text
v6 MS-1：原题 → 几乎总能调用属性工具 → 大多拿到有效 observation
         → 仍可能因 final schema 或少量筛选错误丢分

v6 MS-2：原题 → 首步 tool-call/JSON 可能已经非法
         → 或进入长工具链后遭遇 path / shell / server error
         → 不会有效恢复，重复、超长或耗尽 step → 极少正确 final

v7 MS-1：原题 → 多数 rollout 没有调用 MCP
         → 没有题目所需的 RDKit 属性 → 空答案、错误答案或未完成

v7 MS-2：原题 → 大多数 rollout 也没有形成可评分的完整工具工作流
         → 即使局部调用 MCP，仍很少交付唯一候选
```

共同根因不是 Slime 没有真的更新参数，也不首先是学习率、TP/DP 或单步 16K 截断，而是**训练的因果单位与评测的因果单位不同**。SFT 对历史成功轨迹做 teacher forcing；ToolRL 把历史轨迹拆成相互独立的 `logged prefix → sampled next decision`，对局部动作评分后立即结束，不执行本轮 sampled tool call，也不沿它产生的 observation 继续。正式 DSH 则要求模型从原题开始，对自己每一步 action 的真实后果负责，持续到正确终局。训练优化的是历史状态上的局部模仿，评测要求的是自己状态分布上的完整闭环控制。

v7 还引入了一个比 v6 更直接的目标错位：v6 final reward 虽然过严，但至少检查与历史 structured final 的内容一致性；v7 把 terminal correctness 明确置为 0，只要 `<final_answer>` envelope 合法就给 `+1`。因此 v7 的 final reward 变高不能说明答案能力提高，反而会让许多“四个答案都格式合法但内容不同”的 GRPO group 没有语义偏好。

不过，现有证据也不允许把 v6 的 `58%` 与 v7 的 `7%` 直接解释为“v7 权重导致 51 个百分点退化”。v6 使用旧的 `v6_mol_eval_compat` harness；v7 使用标准 DSH，prompt、tool/skill 暴露、parser compatibility 与停止行为不同。我们现在能确定的是两件事：

1. v6 checkpoint 加旧 harness 能稳定进入 MS-1 属性计算链，并取得 29/50；
2. 在 v7 自己的同条件 DSH 评测内部，是否调用 MCP 几乎决定了 MS-1 是否还有答对的可能。

严格的 v6→v7 权重归因仍需同一 DSH、同一 tool server、同一 skill snapshot、同一 decoding 和多 seed 对照。

## 证据口径：哪些数字可以直接比较，哪些不能

### v6：旧 compatibility harness 下的一次 83 题评测

v6 iter 562 的记录位于：

```text
outputs/slime_drug_agent_evals/
  molbench_ms1_ms2_v6mol_iter562_compat_skills_final_20260819_043926
```

题集为 50 道 MS-1 与 33 道 MS-2。在线 loop 会真实执行模型 action、追加 observation，再让模型继续，直到 final、非法格式、length 或 max steps。temperature 为 0，max steps 为 128，单次模型 response cap 为 16,384 token。该 harness 还启用了 live tool catalog、显式 skill discovery 提示、decision stop，以及针对 v6 已知 Qwen 表面变体的窄 compatibility recovery。

| suite | 题数 | 到达 final | invalid format | length | max steps | 官方正确 |
|---|---:|---:|---:|---:|---:|---:|
| MS-1 | 50 | 33 | 17 | 0 | 0 | 29/50 = 58% |
| MS-2 | 33 | 3 | 17 | 8 | 5 | 1/33 = 3.03% |
| 合计 | 83 | 36 | 34 | 8 | 5 | 30/83 |

compatibility path 实际协助了 19 个 action，全部是 MS-1 中 observation 后多生成了一个 `</tool_call>`，被作为一次窄 no-op 接住。comma-separated multi-call recovery 和 transport-marker stripping 在这 83 题上都没有触发；MS-2 没有 compat recovery。因而不能把全部 58% 当成裸 checkpoint 能力，也不能反过来把 MS-2 的灾难性失败归咎于 parser 替模型改坏了动作。

### v7：标准 DSH 下两次各 83 题评测

v7 iter 551 的两次记录位于：

```text
outputs/dsh_molbench_evals/
  dsh_qwen35_9b_v7_release_toolrl_iter0551_ms1_ms2_worker_rollout_2g_20260901_7875340
  dsh_qwen35_9b_v7_release_toolrl_iter0551_ms1_ms2_worker_rollout_retry_20260901_8096559
```

第一次 83 题中 69 个 DSH turn completed、14 个 failed；第二次 76 个 completed、7 个 failed。这里的 `completed` 只表示 DSH turn 结束，不等于产生了可评分答案，更不等于答对。

| run | MS-1 accuracy | MS-1 调过 MCP 的题 | MS-2 accuracy | DSH failed |
|---|---:|---:|---:|---:|
| 第一次 | 4/50 = 8% | 12/50 | 2/33 = 6.06% | 14/83 |
| 第二次 | 3/50 = 6% | 12/50 | 2/33 = 6.06% | 7/83 |

两次 v7 MS-1 合并后共有 100 个 rollout：

| 是否真正调用 MCP | 正确 | 错误 | 条件正确率 |
|---|---:|---:|---:|
| 调用过 MCP | 7 | 17 | 29.17% |
| 没有调用 MCP | 0 | 76 | 0% |
| 总计 | 7 | 93 | 7% |

这是当前最强的内部证据：**MCP invocation 对 MS-1 是实测必要条件，但还不是充分条件。**没有调用的 76 次全部错，说明首要断点是没有进入计算链；调用后的 17/24 仍错，又说明工具选择、结果使用、阈值筛选和答案交付构成第二层断点。

DSH wrapper 的 `valid_output`/`final_output_conforming` 与 MolBench evaluator 自己的 `valid_rate` 不是同一指标，不应混用。例如第一次 MS-2 record 层只有 2 个 `valid_output`，而官方 AC evaluator 的 valid rate 是 4/33；后者可以从非标准文本恢复候选。本文凡称“官方 accuracy/valid rate”，均以 `metrics.json`/`evaluation_summary.json` 为准；凡称 DSH completion、projection 或 conforming，则以逐题 `record.json` 为准。

### 尚未成功的同 harness 对照

2026-09-02 已尝试把 v6 iter 562 与 v7 iter 551 放到同一 canonical bridge/DSH 下比较。模型 view 转换、非权重 asset 一致性审计和 bridge unit test 已完成，但 v6 的 DSH 在 readiness 前因 MCP reverse tunnel 返回 `Proxy response (502) !== 200` 退出，尚未产生任何可比较的模型 rollout。该记录位于：

```text
outputs/dsh_v6v7_bridge_compare_20260902_133157_6d4402
```

所以目前必须把“v7 的绝对 DSH 表现很差”与“v7 权重严格劣于 v6”分开：前者有直接证据，后者仍待实验。

## v6 的问题：简单确定性任务能计算，复杂任务无法稳定控制完整 episode

### MS-1 暴露的是答案交付脆弱，而不是主要的工具不可用

v6 的 50 道 MS-1 全部至少发起一次真实工具调用，共 210 次，其中 208 次执行成功。典型策略是首步成组调用 `calculate_mol_basic_info`、`calculate_mol_drug_chemistry`、`calculate_mol_hbond`、`calculate_mol_hydrophobicity`，再根据 observation 做阈值过滤。这与训练集中 145 条 PF 轨迹的高频结构高度同构。

它最后只得到 29/50，主要损失却发生在计算之后：33 题交付 final，其中 29 题正确、4 题内容错误；17 题以 invalid ReAct 结束，其中 15 题是 `final_answer.selected_smiles` 没有给 list，2 题是在支持的 ReAct tag 外带了额外文本。也就是说：

```text
50 道都开始计算
  ├─ 33 道合法交付
  │    ├─ 29 道正确
  │    └─ 4 道筛选/内容错误
  └─ 17 道格式失效，无法交给 scorer
```

这说明 v6 已形成有用的“属性题先计算”行为先验，却没有同样稳固地掌握 task-specific final contract。最终答案字段的 string/list 类型不是纯 UI 问题：严格 evaluator 无法消费时，前面的正确计算全部归零。

为什么训练后仍会这样？v6 SFT 的确见过完整 final，ToolRL 的 final reward 也检查 structured exact，但两者都没有直接优化正式 evaluator 的投影成功率。SFT 是 teacher forcing；final ToolRL row 只在已经给定的 logged history 后单独生成。它不会让“初始工具选择→真实 observation→最终 list schema→官方正确”成为同一条 rollout 和同一个可归因回报。v6 final GRPO 的 364 个 group 中有 272 个四采样同分，零方差率 74.73%；其中 255 个 group 四个 sample 全为 `-0.5`。GRPO 对组内相对优势学习，四个都错并不会因绝对负分而自动得到纠错方向。

### MS-2 暴露的是多步 agent policy 的系统性失效

v6 MS-2 只有 1/33 正确。17 题在第一个 decision 就因为 `arguments` 不是 object 或 JSON object sequence 不能解析；8 题单次生成打满 16K；5 题走满 128 step。length/max-step 的 13 题全部来自 MS-2。

进入工具链也不意味着有效推进。33 题共发生 1,056 个 agent step、1,042 个工具调用，只有 142 次 semantic success，900 次 semantic error，另有 17 次 schema validation failure。主要错误为：

| 错误 | 次数 | 说明的 policy 问题 |
|---|---:|---|
| `Read: file does not exist` | 381 | 没有稳定区分 local workspace path、skill path 与 server resource |
| `Bash: unsupported shell redirection` | 295 | 没有根据 harness 能力约束命令形式，收到确定性报错后仍重复 |
| `pred_pocket_prank` failed | 101 | workflow/前置产物不满足仍继续走同一路径 |
| `server_file_to_base64` failed | 82 | server artifact 到 local workspace 的物化合同未掌握 |

384 次 skill file Read 集中在 4 道题，3 道各重复 127 次；5 个 max-step 任务贡献了 640 step。因此“v6 总在读 skill”并不准确，更准确的是少数 rollout 进入 catastrophic loop，遇到相同 observation 后没有更新内部策略。

这里同时存在五类能力缺口：

- 首步协议不稳：复杂题尚未执行工具就因 XML/JSON/arguments 失败；
- 世界状态表示不稳：不知道某个 handle 是 server resource、artifact URI 还是本地可读文件；
- 工作流前置依赖不稳：没有可靠建立结构、口袋、文件物化等先后关系；
- 错误恢复不稳：面对确定性失败仍近似重试，缺少换路径、降级或终止策略；
- 停止策略不稳：不能判断任务已不可恢复或证据已足够，最终 length/max steps。

### 为什么 v6 的 MS-1 好得多

MS-1 不是在证明 v6 已经会一般化的 agent reasoning，而是在证明一个高度匹配训练模板的短链条可以工作。PF 题只要求对给定 SMILES 做确定性 descriptor 计算和阈值过滤，通常一轮成组调用即可；实际无需 skill Read，平均仅 2.96 step，工具执行成功率为 208/210。MS-2 则要求跨多个资源域维护长依赖，每一步都可能把模型带入训练中没有的错误状态。

因此，v6 的强项是**短、直接、高频、与历史 PF 模板同构的工具调用**；它的弱项是**状态会由自身 action 改变、并且需要根据失败 observation 重规划的长程控制**。这恰好对应 offline teacher-prefix 训练能覆盖与不能覆盖的边界。

## v7 的问题：格式工程更正规，但模型更少进入真正的计算链

### MS-1 首要失败是没有调用 MCP

v7 两次评测各只有 12/50 道 MS-1 调过 MCP。其余 rollout 即使 DSH 状态是 `completed`、甚至产生了一个形式上可解析的空 list，也没有获取完成属性过滤所需的 RDKit 结果。逐题记录中可以看到大量 `prediction: []`、只进行本地 Read/Write、或直接结束的情形。

这不是“工具算错”能解释的：未调用 MCP 的 76 次没有一道正确。因而当前 v7 MS-1 诊断的优先级必须是：

```text
第一层：为什么从原题没有稳定启动并持续执行 MolClaw/RDKit 调用？
第二层：调用之后，为什么仍有 17/24 没得到正确答案？
第三层：正确内容能否稳定投影为 evaluator 要求的 schema？
```

如果只统一 final list、放宽 parser 或润色 reasoning，最多影响第二、三层，不会让那 76 个没有任何 MCP observation 的 rollout凭空获得属性。

### v7 的 final reward 教会“有信封”，没有教会“信封里答案正确”

当前 v7 reward 的 terminal 分支是：

```python
out["score"] = 1.0 if valid else -0.5
out["components"]["terminal_correctness"] = 0.0
out["diagnostics"]["structured_final_exact_enabled"] = False
```

因此，只要 completion 是一个合法 `<final_answer>`，其中 `selected_smiles` 为空、候选错误或证据语义空洞，都与正确答案同得 `+1`。v7 的 364 个 final group 中，170 个四采样同分；其中 144 个是四个 sample 全部 `+1`。1,456 个 final completion 有 1,054 个得 `+1`，但 `terminal_correctness` 全部为 0。

这会产生两种后果。第一，reward 日志会显得明显变好：v7 final-group raw mean 约 `+0.586`，v6 约 `-0.228`；但两者标尺不同，前者主要是评分放宽。第二，GRPO 只使用组内相对差异；四个内容不同的合法答案全部 `+1` 时 advantage 为零，模型没有任何信号去选择语义更正确的那个。

v6 的 teacher-structured exact 也不是理想答案：它可能因为 evidence 表达不同而错罚语义等价答案，并导致大量四个全错的零方差 group。但从 v6 到 v7 的正确升级应是把 terminal 目标改成官方 benchmark correctness，并适当配合格式 shaping，而不是删掉内容判断。

### v7 的工具 reward 仍只是 logged state 上的局部 action 评分

v7 tool row 的 hierarchical reward 比字符串 exact match 更合理：它依次检查 XML envelope、tool set、required arguments、critical values 与 configurable validity，允许某些 schema 合法的非 teacher 配置。但它仍然回答的是：

```text
“这个 sampled action 在当前 logged prefix 下，是否像一个合法、接近 logged target 的动作？”
```

它没有回答：

```text
“这个 action 在当前真实 MolClaw/DSH 环境执行后，是否推进了本题并最终答对？”
```

实际 v7 ToolRL 由 5,213 个可训练 decision 静态选出 2,208 个，每个 prompt 采四次。对某个 row：

```text
H_t^log → sample a_t → local reward → 该 sampled branch 结束
```

数据中的下一 row 虽然包含前一历史 action 与其配对 observation，模型从 token 上完全可以把这段 history 理解为“自己的过去”；问题不在 teacher/student 命名。问题是它在训练计算图中并不是本轮 `a_t` 的执行结果：

```text
H_(t+1)^log = H_t^log + a_t^log + o_t^log
             → 另一次独立 sample → 另一个局部 reward
```

所以模型本轮不调用工具，不会在同一 sampled branch 中经历“没有属性→最终答错”；它本轮给错路径，也不会看到实际 file-not-found 后学会恢复；最终答案的 reward 更不可能 credit 给此前那次 sampled tool decision。

这正是 v7 训练日志里单步 rollout truncation 约 0.2%，而完整 DSH 仍会 max-token 的原因：前者只测一个离线 next-decision completion，后者累计完整 episode。两者不是同一长度随机变量。

### v7 的 path 清理修了错误监督，却没有自动补出正确 policy

v7 将 legacy artifact literal 正规化为 resource URI 或 workspace path，并识别出 116 个错误 local access call，mask 掉其所在的 115 个 assistant message。这里的错误不是“server 上的文件不能作为 server tool 参数”；恰恰相反，server-side tool 需要 server resource。错误是把仅在 server/tool 侧存在的 handle 当成当前 DSH 本地 workspace 中可由 `Read/Bash/Edit/Write` 打开的路径。

mask 是必要的数据卫生升级：不应继续要求模型复现确定会失败的 local action。但 mask 只删除错误梯度，没有为这些状态补入“先调用物化工具、取得 local path、再 Read”之类的正确替代 action。后续 logged history 还可能保留这段 action/observation 作为上下文，以维持轨迹因果顺序。因此它提高了数据合同的正规性，却不能单独提高新环境中的闭环完成率。

### v7 的答案 schema 清理也不是主要瓶颈

v7 没有把全部任务答案统一为 list：PF 的 `selected_smiles` 本来就是 list；98 条 VS 只把 scalar `selected_smiles` 包为 singleton list，而 `ranked_smiles` 本来就是 list；AC 的 `answer_smiles` 仍是 string；E2E 的 `result` 仍是 list。这是按任务合同做的局部修复，不是统一 schema。

统一模型内部的分子答案表示可能减少格式熵，但必须同步修改 prompt、parser、reward、projection adapter 与 evaluator，并保留任务语义：PF 是 0..N 的集合，AC 是恰好一个候选，VS 是有序 ranking。仅机械地把字段套 list 不能解决 v7 的首要问题——模型多数时候根本没有调用 MCP；format-only final reward 也不会因为 list 更整齐而知道 list 中应放什么。

## 共同根因：从评测 failure 向训练系统逐层追责

### 根因一：训练优化“局部 next action”，评测优化“完整自主 episode”

这是最高层、影响最大的根因。两版都采用相同的核心范式：

```text
SFT：在完整 logged trajectory 上做 assistant-only teacher forcing

ToolRL：logged prefix → sampled one decision
        → teacher/schema-local reward → branch ends

DSH：raw question → sampled action → real execution → own observation
     → sampled next action/error recovery → ... → final
     → official benchmark correctness
```

SFT 可以让模型记住成功轨迹中的行为模式，v6 MS-1 的稳定属性调用说明这种先验确实可能部署出来。但 teacher forcing 不覆盖模型自己遗漏工具、给错参数或收到新错误后的状态。offline ToolRL 又没有把 sampled branch 接上环境，因此不能修复这个 exposure gap，也不能完成长程 credit assignment。

### 根因二：reward 与真实任务成功没有对齐

两版的 tool reward 都以 logged action/tool schema 为参照，不以执行是否成功、observation 是否有用、最终是否答对为主目标。v6 terminal reward 是 teacher structured-final exact，内容敏感但过严；v7 terminal reward 是纯 envelope validity，连答案内容都不再区分。

因此两版都会出现 reward 与真实能力脱钩：

- 一个 schema 合法但在环境中会失败的 tool call 可以拿正 reward；
- 一个不同于 teacher、但同样能解决任务的 action 可能被低估；
- v6 一个核心答案正确但 evidence 不同的 final 可能得负分；
- v7 一个内容错误或空洞的合法 final 可以得满分；
- terminal reward 只作用在独立 final row，无法归因给更早的工具选择。

### 根因三：GRPO 的大量 group 没有组内可学习偏好

GRPO 不是看到负 reward 就一定“向正确方向更新”。同一 prompt 的四个 reward 经组内中心化；若四个同分，其 task advantage 为零。v6 的 2,252 个有效 group 中约 39.1% 零方差，final group 零方差 74.73%；v7 的 2,208 个 group 中 771 个零方差，占 34.92%，final group零方差 46.70%。

两者的零方差语义不同：v6 final 多为四个都不满足过严 exact；v7 final 大量是四个都满足过宽的格式条件。前者告诉我们 reward 太稀疏/苛刻，后者告诉我们 reward 太容易/不含任务信息。共同结果都是大量 optimizer update 没有来自当前任务 reward 的区分方向，只剩 KL 等约束项可能产生梯度。

### 根因四：历史成功轨迹不能代表模型在部署中制造的状态分布

365 条 v6/v7 轨迹是历史 MolBench-style logged trajectories，不是用当前 DSH、当前 MolClaw server 和当前模型逐条重放得到的 executable corpus。v6 甚至保留了后来确认不符合新 path contract 的动作；v7 mask 了已知错误，但没有系统补齐正确替代轨迹。

训练主要看见成功历史上的 state；部署却会进入：工具未调用、arguments malformed、server handle 被当 local path、shell 能力不支持、pocket prediction 失败、observation 超长等状态。v6 MS-2 的 900 次 semantic error 与机械重试，就是 state-distribution gap 的实证。

### 根因五：静态数据选择按结构覆盖，不按当前 policy 的可学习性选择

v6 从 5,329 个 decision 选 2,252 个，v7 从 5,213 个选 2,208 个。中间 decision 使用静态 stratum/确定性 hash 去重，以减少重复并保证覆盖；它不知道当前 policy 在哪个状态会失败，也不知道两个表面结构相同的 observation 是否要求语义不同的下一步。

这种选择可以作为可复现的数据压缩 baseline，却可能删掉对错误恢复或状态条件非常关键的变体。它也不会随着模型能力变化采集 hard negative。因此不能把“覆盖了每种 decision shape”理解为“覆盖了部署中的每种因果状态”。

### 根因六：tool discovery 与运行合同只有部署时真正闭合

实际 production ToolRL prompt 不注入完整 official tool catalog；模型依赖 system contract、SFT 参数中记住的工具名以及历史 context。通过 Glob/Read 发现 skill 在部署逻辑上是通顺的，问题不是“不提供 catalog 就必然错误”。问题仍是：offline ToolRL 中 sampled Glob/Read 不执行，发现到的内容不会进入同一 sampled branch 的下一轮。

v6 compatibility harness 额外提供 live catalog 和显式 discovery 路径；v7 标准 DSH 的启动条件不同。这可能解释两者 tool-use initiation 的一部分差异，但没有同 harness ablation 前不能量化。

### 根因七：科学对照和 provenance 还不足以做版本因果归因

v6、v7 实际 run 的 `CODE_COMMIT=unknown`，运行数据虽有 hash，代码状态没有被不可变 commit 完整冻结。没有在统一 harness 下系统保存并评测 base、SFT-only、ToolRL checkpoint；旧 v6 与 v7 的 harness 又不同。v6 还发生过 iter 224 checkpoint 保存失败与恢复，100 个 decision 被重采样，虽可按最终生效分支还原完整 key，但审计洁净度低于 v7。

这不会推翻已经观测到的模型失败，却限制了“是哪一个训练 stage 导致”的归因。当前可以说 v7 的 reward/objective 没有提供所需信号；不能仅凭两个最终分数说 ToolRL 一定使 v7 比 v6 更差，也不能说 v6 的 58% 全由 ToolRL 带来。

## 把“模型问题、harness 问题、基础设施问题”分开

| 层级 | 已有证据 | 不应误判为 |
|---|---|---|
| 模型/policy | v6 final schema 错、MS-2 首步 malformed、报错后循环；v7 多数 MS-1 不调用 MCP | 单纯 tool server 算错 |
| 训练目标 | sampled action 不执行、无 episodic credit；v7 final correctness 关闭 | Slime 没有反传或训练没有跑 |
| 数据 | 历史成功 prefix、legacy path、mask 后缺正确替代、静态去重 | 所有 365 条都是当前环境 replay-verified |
| harness | v6 有 compat/catalog/scaffolding，v7 是标准 DSH | 纯 checkpoint apples-to-apples 比较 |
| 基础设施 | 同 harness 重评被 MCP tunnel 502 阻断 | 模型在该次重评中失败；模型尚未开始 rollout |

尤其要避免把 v7 标准 DSH 的 `completed` 当成任务成功。一个 turn 可以正常结束、输出空 prediction，并被记为 completed；正式正确性仍为 0。同样，工具调用总数也不是能力指标：v6 MS-2 有 1,042 次调用却只有 1 题正确，因为大量调用是重复失败。

## 面向专家应验证的最小诊断矩阵

下一轮不应只跑一个“v8 final”。为了让每个结论可归因，至少需要在冻结的同一 DSH commit、tool server、skill snapshot、prompt、decoding、题集与 seed 集合上评测：

| arm | 回答的问题 |
|---|---|
| base Qwen | 预训练模型原始 tool-use/格式能力是多少 |
| v6 SFT-only | 历史全轨迹 teacher forcing 是否建立 MS-1 调用先验 |
| v6 iter 562 | v6 offline GRPO 在同 harness 下增加还是损害该先验 |
| v7 SFT-only | path mask/数据清理本身的影响是什么 |
| v7 iter 551 | format-only terminal + offline tool reward 的净影响是什么 |
| SFT + online episodic ToolRL | 执行 sampled call、回填 observation、官方终局 reward 后是否真正改善 |

每个 arm 至少统一记录：

- 从原题开始的首次有效 MCP invocation rate；
- `correct | MCP-called` 与 `correct | no-MCP`；
- 合法 tool-call、schema validation、execution success、semantic success；
- 按错误类型统计 recovery、重复同错次数和不可恢复后的停止；
- DSH final conforming、evaluator projection、官方 correctness；
- 完整 episode 的总 step、tool call、token、max-token/max-step；
- GRPO group 的 reward 方差，并区分“全都正确”“全都错误”“仅格式全合法”；
- 多 seed 均值与方差，而不是单次 deterministic/非 deterministic 差异。

对 MS-1，最先建立的训练 gate 不应是单步 completion truncation 很低，而应是：完整 rollout 从原题出发，是否稳定调用正确 RDKit/MolClaw 属性工具；真实 observation 是否改变 selected set；最终是否通过官方 scorer。对 MS-2，则还必须加入失败注入与恢复评测，验证模型看到 `file does not exist`、unsupported command、server conversion failure 后是否换策略，而不是重试到 128 step。

## 最终判断

v6 证明了这条路线并非完全没有能力：在带 compatibility scaffolding 的旧在线 harness 中，它对与 PF 历史模板高度同构的 MS-1 能做到 50/50 启动真实工具、208/210 调用执行成功、29/50 答对。但同一个模型在 MS-2 暴露出协议、路径、工作流、错误恢复与停止策略的系统性缺陷；训练没有覆盖自身失败状态，是这些问题反复出现的核心原因。

v7 的升级主要发生在数据合同、path/resource 正规化、错误 action mask、exact-epoch 审计和训练工程可复核性，而不是端到端 agent objective。它仍然是 offline next-decision GRPO，并进一步把 final 语义 correctness 关闭。两次标准 DSH 中，大多数 MS-1 没有调用 MCP，未调用的 76 次全部错误；所以 v7 当前首先是 tool-use initiation/continuation 与长程 credit-assignment failure，其次才是调用后的计算、筛选和答案交付问题。

最根本的修复方向不是继续美化静态 trajectory、单独统一 list 或只调优化器，而是把训练 sample 的因果边界扩展到真实 episode：模型 action 真正执行，自己的 observation 进入自己的下一状态，错误恢复和及时停止被纳入轨迹，最终由官方 MolBench correctness 给主 reward，并把这一终局信号归因到此前的工具选择。XML 合法、schema、工具成功、token/step 成本可以作为 shaping；它们不能再替代“本题最终答对”。

## 证据索引

- v6 数据、SFT、ToolRL、reward、恢复和评测全链：[v6 lifecycle](v6_final_sample_causal_lifecycle_zh.md)
- v7 path 修复、SFT mask、离线 decision、format-only reward 与两次 DSH：[v7 lifecycle](v7_release_sample_causal_lifecycle_zh.md)
- v6 逐题评测：`outputs/slime_drug_agent_evals/molbench_ms1_ms2_v6mol_iter562_compat_skills_final_20260819_043926/traces.jsonl`
- v7 两次逐题 DSH：各 run 的 `workspaces/*/record.json`；官方汇总为 `metrics.json` 与 `evaluation_summary.json`
- v7 terminal reward：`slime/drug_agent/toolrl/molclaw_reward.py` 的 format-only final override
- 同 harness 未完成对照：`outputs/dsh_v6v7_bridge_compare_20260902_133157_6d4402/status.log` 与 `v6_iter562/dsh.log`
