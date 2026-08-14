# ToolRL 五项局部改进：最终实现方案与训练状态

更新日期：2026-08-13

适用范围：Qwen3.5-9B、v4 冻结轨迹、Slime ToolRL/GRPO 训练链路

> 说明：本文后半部分的训练状态是 2026-08-13 已启动旧 view 的历史记录。新版 LLM planning clean 与 LLM context summary 代码不会自动改写该运行中的数据；必须重新物化版本化数据 view 后由后续 run 显式加载。

## 1. 结论先行

当前主线保持 `SFT → ToolRL` 两阶段。第 4 项不再使用 first-thought heuristic，也不增加 Plan-SFT；新的数据生产代码改为在 LLM clean 时根据完整 teacher trajectory 生成首位高层 planning thought。planning 当前不作为独立 RL role 或 reward。已经放弃早期 `decision_aware` 方案中的三项启发式设计：

- 不把“第一条非空 thought”自动标为 planning，也不构建或加载 Plan-SFT；
- 不再因为轨迹出现重复调用就额外复制 final；
- 不再固定挑选 2,500 个 middle decision，也不再使用 `max 8 steps/trajectory` 之类的人工上限。

当前采用的总体结构是：

```text
                    冻结的 605 条成功 teacher trajectory
                                      │
                                      │
                                      ▼
                              已完成的 ReAct SFT
                                      │
                                      ▼
                              ToolRL step view
                         state → tool_step / final
                                      │
                           删除 no-progress repeat
                           Microcompact 超长上下文
                           diversity candidate pool
                                      │
                                      ▼
                                           当前 policy 对每题 rollout 4 次
                                                        │
                                           只训练 reward 有方差的 group
                                                        │
                                                        ▼
                                             hierarchical ToolRL/GRPO
```

此前的三阶段实验已经停止且不再作为当前训练路线。它曾完成 Plan-SFT 和若干 gate，但其 checkpoint 不会被新的两阶段训练加载。当前新 launcher 直接从已有 ReAct SFT checkpoint 启动 ToolRL。

- 当前生产链路：`ReAct SFT checkpoint → ToolRL`；
- planning 数据生成已并入新版 LLM clean；Plan-SFT 仍不进入主线；
- 第 1、2、3、5 项代码、数据视图、reward 和动态筛选保留不变；
- 新两阶段 run 必须重新完成长度 gate 和 multi-update gate，不能沿用旧 Plan-SFT run 的完成 marker。

因此，旧实验只能说明基础机制和 262k 长度路径可以工作；新的两阶段 run 仍需独立 gate 后才能进入正式训练。

## 2. 五项要求与最终方案总览

| 原始要求 | 早期方案 | 最终方案 |
| --- | --- | --- |
| 1. 工具名比 teacher 参数精确匹配更重要 | 固定 `55/10/10/15` 加权和 | 层级门控 reward：格式 → 工具集合 → required args → critical args → configurable validity |
| 2. 重复调用时提高 final 相对权重 | 重复调用轨迹额外复制 final | 删除“成功后、无状态变化的相同调用”；final 每轨迹只保留一次，其相对占比自然上升 |
| 3. 超长上下文压缩 | generic head/suffix compactor | Microcompact 风格 typed observation compaction；仍超限时调用独立、grounded 的 LLM step summarizer |
| 4. planning 与 tool execution 分开 | `planning = first non-empty thought`，再复制 planning | LLM clean 根据完整 teacher trajectory rewrite/prepend 首位 planning thought；不增加 Plan-SFT，planning 暂不作为独立 RL 目标 |
| 5. 不训练所有 decision | 固定挑 2,500 middle decisions | 静态 diversity candidate pool + 当前 policy 的 n=4 reward-variance 动态筛选 |

## 3. 改进一：从固定加权 reward 改为层级门控 reward

### 3.1 要解决的核心问题

原始要求不是简单地说“tool name 占 55%，parameter value 占 15%”，而是说工具选择和参数正确性处于不同层级：

```text
工具错
  < 工具对但缺 required argument
  < 工具对、required argument 齐全但 identity-critical 值错
  < 工具和关键状态一致，配置参数合法
  < 与 teacher 完全等价
```

如果仍然使用所有 feature 的加权求和，模型可能通过“错误工具 + 很像的参数”拿到不合理的分数，也可能因为一个允许多解的配置参数与 teacher 不同而受到过重惩罚。

### 3.2 最终 reward 阶梯

新 launcher 默认使用：

```bash
TOOLRL_REWARD_MODE=hierarchical
```

工具 decision 的分数区间为 `[-0.5, 1.0]`，逐级门控如下：

| 阶段 | 条件 | reward |
| --- | --- | ---: |
| ReAct envelope 非法 | 不是合法且唯一的 tool-call decision | `-0.5` |
| 工具完全错误 | 没有匹配到任何正确工具名 | `-0.4` |
| 工具集合不完整 | 多调、漏调或只匹配部分工具 | `-0.05 + 0.35 × tool_name_F1` |
| required args 缺失 | 工具集合完全正确，但必需参数不齐 | `0.30 + 0.20 × required_coverage` |
| critical args 错误 | required args 齐全，但身份/状态参数不一致 | `0.55 + 0.15 × critical_exact` |
| configurable args 非法 | 关键参数正确，但配置不符合 schema/range | `0.72 + 0.13 × configurable_validity` |
| 合法替代配置 | 工具、必需参数、关键状态均正确，配置合法但不与 teacher 逐值相同 | `0.90` |
| teacher-equivalent | 工具和全部参数与 teacher 规范化后完全一致 | `1.00` |

这保证了“选对工具但参数不精确”仍能拿到明显正分，同时 teacher exact 仍然是最高分，但不再是获得高 reward 的唯一方式。

### 3.3 critical 与 configurable 参数

参数分为两类：

1. `critical/identity` 参数

   这些参数决定任务对象、状态或 artifact 身份，例如：

   - `pdb_id`、`uniprot_id`、`gene_name`；
   - protein/ligand/sequence/SMILES；
   - mutation、chain；
   - input/output file、path、artifact、complex/structure。

   这类值错误通常意味着执行了另一个任务，因此要求与 teacher 规范化值严格一致。

2. `configurable` 参数

   例如 `top_k`、采样数、box size、可选 preset、某些 flag。它们不统一要求 teacher exact，而是检查：

   - 参数是否在工具 schema 中；
   - JSON 类型是否正确；
   - enum、minimum/maximum、长度和数组大小是否有效。

工具 schema 可以通过 `x-toolrl-importance=critical|identity|configurable|configuration` 显式覆盖分类；没有扩展标记时，代码使用参数名模式做保守识别。没有 catalog schema 的本地工具退化为“teacher keys 作为 required keys”，但不会假装所有 teacher value 都必须完全相同。

### 3.4 多工具调用与顺序

工具调用先做规范化配对，工具集合用 precision/recall/F1 和调用完整性约束。多工具调用的重复检测和 tool-set 判断均不依赖调用顺序，因此同一集合仅顺序不同不会被错误判为另一种 decision。

### 3.5 final reward

final 保持简单：

- 合法、唯一的 `<final_answer>` 且结构化内容精确匹配：`1.0`；
- 格式错误或结构化结果错误：`-0.5`；
- 忽略重复的人类可读 `summary` 字段；
- response 被 Slime 标记为 truncated 时，不允许获得正 reward。

### 3.6 可回退与消融

`official`、`molclaw`、早期 `decision_aware` 模式仍保留，没有覆盖官方公式。新训练分支只把默认模式改为 `hierarchical`，因此可以直接做 ablation。

主要代码：

- `drug_agent/toolrl/molclaw_reward.py`
- `drug_agent/toolrl/metrics.py`
- `drug_agent/toolrl/tests/test_reward.py`
- `drug_agent/toolrl/tests/test_metrics.py`

## 4. 改进二：不复制 final，删除 no-progress repeat

### 4.1 为什么删除“重复轨迹额外复制 final”

复制 final 只改变同一 final prompt 在数据集中出现的次数，并没有把负 credit 分给导致浪费的重复 tool step。它还会把“合理 retry”和“无意义重复”混在一起。

最终实现采用 `no_progress_repeat`，只删除满足全部条件的 decision：

1. 当前工具名和参数集合与同一轨迹中之前某次调用完全相同；
2. 之前那次调用有显式、可用的成功 observation；
3. 两次调用之间没有其他成功 observation 表明 relevant state 已经变化。

调用集合使用规范 JSON 序列化，并按调用排序后比较，所以多工具集合顺序无关。

### 4.2 合理 retry 不会被删除

以下情况继续保留：

- 上一次 timeout；
- 上一次 error/failed；
- observation 状态未知；
- 两次相同调用之间已有其他成功 step 改变了状态。

因此它不是简单的 `same tool name + same params` 去重。

### 4.3 v4 实际结果

- 11,909 个派生 decision 中识别出 18 个 `no_progress_repeat`；
- 分布在 15 条轨迹中；
- 这 18 个 decision 从候选 RL view 中删除；
- 605 个 final 每条轨迹只保留一次，不做额外复制。

早期统计中的“138 条含 literal repeat 的轨迹”使用的是更宽松的字面重复定义，不能作为新语义定义的预期值。新定义更保守，目的是不误删合理 retry。

删除重复和中间 diversity 去重后，final 在静态候选池中的比例从原始 `605 / 11,909 = 5.08%` 上升到 `605 / 4,823 = 12.54%`。这是自然的相对占比变化，而不是特殊复制规则。最终进入梯度的 final 比例还会受到当前 policy reward 方差筛选影响，不是预先写死的常数。

主要代码：

- `drug_agent/toolrl/convert_react_to_toolrl_steps.py`
- `drug_agent/scripts/select_toolrl_decisions.py`
- `drug_agent/toolrl/tests/test_converter.py`
- `drug_agent/tests/test_toolrl_v2.py`

## 5. 改进三：Microcompact 风格的确定性长上下文压缩

### 5.1 长度契约

训练链路固定使用模型文件声明的上限：

```text
max context  = 262,144
max prompt   = 245,760
max response = 16,384
```

launcher 启动前同时检查：

- HF `text_config.max_position_embeddings >= 262,144`；
- tokenizer `model_max_length >= 262,144`；
- 每条物化 prompt 使用 Qwen3.5 的真实 chat template 后不超过 245,760；
- 当前 gold decision 的最小规范序列化不超过 16,384。

gold label 不允许被截断。若 gold decision 本身超限，直接排除并写入 manifest。

### 5.2 三层压缩

#### Level 1：typed observation Microcompact

只处理较早的、体积超过 32,768 字符的 observation payload，最近 4 个 observation 保持原样。工具调用、消息拓扑和 call/result 配对不变。

大对象被替换为可审计描述：

- 大字符串：类型、字符数、SHA256、有限 head/tail；
- 大数组：元素数、SHA256、前 4 项、后 4 项；
- 大字典：优先保留 status、error、tool name、path、file、artifact、ID、metadata 等关键字段；
- 文件正文、base64、PDB/CIF、表格和日志因此不再原样复制进上下文。

每个替换块记录原始大小、输出大小和 SHA256。

#### Level 2：保留最近完整历史

若 Level 1 后仍超限：

- 永久保留 system message；
- 永久保留原始 task；
- 从最近的完整 assistant step 边界向前填充最大连续后缀；
- 不切断 assistant/observation 的逻辑边界。

#### Level 3：grounded LLM 结构化摘要

被移除的中间历史交给独立 `summarize-react-step-context` skill，转为按时间顺序的 JSON 摘要，保留：

- 简短 rationale；
- tool name；
- 压缩后的 arguments；
- observation 的 status/error/path/artifact/ID 等状态。

摘要器只看到历史 prefix 和 source inventory，当前 gold response/label 不会写入其 workdir。超大历史按完整 assistant/observation 单元 map/reduce，摘要预算最多 32,768 token；最后使用真实 Qwen chat template 精确复核 token 数。schema、grounding 或长度校验最多尝试三次，仍失败只排除该 decision。

ToolRL/GAD 使用相同的 state hash、skill/prompt/schema version cache，因此同一 decision prefix 只总结一次。没有引入 LLMLingua，也没有实现 Claude Code 的在线 cache-edit/memory 状态机。

### 5.3 v4 物化结果

- 静态候选池：4,823 条；
- 发生上下文压缩：70 条；
- 最大最终 prompt：245,654 token；
- 最大保留 gold target：14,355 token；
- 因 gold target 超过 16,384 排除：1 条；
- 未修改任何 gold call/final；
- manifest 保存原始/最终 token 数、移除消息数、保留后缀边界、摘要 hash、版本和 Microcompact entries。

主要代码：

- `drug_agent/scripts/compact_rl_context.py`
- `drug_agent/scripts/select_toolrl_decisions.py`
- `drug_agent/tests/test_structured_rl_context.py`
- `drug_agent/tests/test_toolrl_length_probes.py`

### 5.4 当前局限

LLM 自由文本仍可能产生语义偏差，因此 tool name、arguments 标量、status、path、artifact、ID 和 error 都要通过 source message index 做 grounding 校验。无法校验通过的数据不进入 view。LLMLingua 或 ACON/SUPO 类训练式 compressor 仍留作后续 ablation。

## 6. 改进四：planning 进入 LLM clean，但不增加 Plan-SFT

### 6.1 当前决定

当前实现不使用：

```text
first valid decision + non-empty thought = planning
```

因为“先读取文件”和“先生成候选、再做 ADMET、再预测结合、最后分析复合物”虽然都可能出现在第一段 thought 中，但只有后者是真正的高层规划。仅凭位置和非空 thought 无法生成可信 planning label。

LLM clean 读取完整 teacher trajectory，并输出 `planning_action`：已有 task-level plan 时 rewrite；只有局部 rationale 或没有 thought 时 prepend。每条 accepted trajectory 同时生成带前后 hash 的 planning annotation sidecar。

ToolRL 中仍然只有：

- `decision_role=tool_step`
- `decision_role=final`
- 额外布尔字段 `is_initial_step`

ToolRL 中不存在 `planning` 或 `initial_tool_step` role。`is_initial_step` 只用于审计和覆盖，不声称它是 plan。

### 6.2 为什么不使用规则 Plan view

代码库中曾实现一个从完整 tool sequence 映射到 tool-family subgoal 的规则 Plan view，并完成过一次实验。但它不是 LLM 对任务语义和完整轨迹的总结，也没有经过逐条人工审核，因此不能作为当前主线的 planning supervision。

当前仍明确禁止以下行为：

- 不调用 `build_plan_view.py`；
- 不物化 `plan_view.jsonl`；
- 不运行 Plan-SFT；
- ToolRL 不加载 `plan_sft/` checkpoint；
- 不给第一条 thought 或 initial step 额外采样权重。

训练链路保持：

```text
含 LLM planning thought 的 ReAct SFT → hierarchical/policy-boundary ToolRL
```

### 6.3 当前 planning 数据契约

planning label 满足：

- 根据原始任务和完整成功轨迹总结，而非复制原 thought；
- 表达 scientific subgoals、依赖顺序和终止条件；
- 不泄漏具体 artifact path、运行时 ID 和冗余参数；
- 经过 schema 校验、自动一致性检查和人工抽查；
- 单独版本化，并与原始轨迹一一可追溯。

当前首个 tool decision 仍参加普通 ToolRL，planning 仅随工具 outcome 接受间接梯度；没有 planning 专用 reward、复制或采样。未来若引入独立 planning RL，直接消费 sidecar，不再从 first thought 猜测 label。

### 6.4 保留的离线代码

`build_plan_view.py` 暂时保留作为历史原型和未来对照，但不被当前 launcher 引用。保留文件不代表当前训练启用了 Plan-SFT。

## 7. 改进五：learnability × coverage 的两阶段 decision 选择

### 7.1 静态阶段：构建多样性候选池

静态阶段不预测“难度”，只负责合法性、去重、上下文预算和覆盖：

- 保留全部 initial step 候选；
- 保留全部 final 候选；
- 删除 `no_progress_repeat`；
- middle decision 按以下 stratum 保留一个确定性代表：
  - task type；
  - tool-call shape（工具名与参数名集合）；
  - trajectory depth decile；
  - prompt 长度桶；
  - 上一 observation 的 success/failure/unknown/none；
  - single-tool / multi-tool；
- stratum 内用 decision key 的 SHA256 排序选代表，结果可复现；
- 不设固定 2,500 条预算，也不设每轨迹最多 8 条。

v4 结果：

| 指标 | 数值 |
| --- | ---: |
| 原始派生 decision | 11,909 |
| 删除 no-progress repeat | 18 |
| diversity-equivalent middle 排除 | 7,067 |
| 超长 gold target 排除 | 1 |
| 最终静态候选 | 4,823 |
| 其中 tool_step | 4,218 |
| 其中 final | 605 |
| middle diversity strata | 3,614 |
| task 覆盖 | 5/5 |
| observed tool 覆盖 | 83/83 |
| 人工复制行 | 0 |

### 7.2 在线阶段：当前 policy 的 n=4 方差筛选

每个 candidate decision 使用当前 checkpoint rollout 4 次：

```text
reward standard deviation > 1e-6  → policy-boundary，进入 GRPO
4 次全对                           → mastered，丢弃
4 次全错                           → too hard，丢弃
其他零方差                         → zero variance，丢弃
```

Slime 动态补采样，直到凑齐本次 rollout batch 所需的有效 group。所有尝试过的 group——包括被过滤的 group——都写入 `learnability.jsonl`，记录 source、decision role、工具、4 个 reward、均值、标准差和 policy-boundary 标志。

设置 `dynamic_sampling_max_dropped_groups=128` 和 strict 模式：如果连续补采样仍无法凑齐有效 group，任务直接失败，不允许为了继续训练而把零方差 group 静默放回 batch。

这使 diversity 成为候选池约束，而 learnability 成为真正的训练 selector。

主要代码：

- `drug_agent/scripts/select_toolrl_decisions.py`
- `drug_agent/toolrl/policy_boundary.py`
- `slime/utils/arguments.py`
- `slime/rollout/sglang_rollout.py`

### 7.3 当前局限

online selector 会增加 rollout 成本。当前 gate 已观察到全错 group 被丢弃，并通过补采样形成有效 batch；也观察到部分 final group 因 exact-final 全错而成为零方差。这符合 selector 定义，但意味着：

- 当前模型若几乎无法 exact-match final，final 可能在实际梯度中被系统性低采样；
- 长响应在被判定为 truncated/全错前仍会消耗完整生成时间；
- 需要统计正式运行中的 drop reason、有效 final 比例和每个 optimizer update 的补采样倍数。

这是当前最需要用多 update gate 继续验证的问题，而不是通过重新复制 final 来掩盖。

## 8. 数据与测试验收

### 8.1 数据审计

- v4 源文件 SHA256：`be4ed789b45b280b338a3344558736cc43847b19478df7d71d53853a2de91e1e`
- 源轨迹：605；
- 派生 decision：11,909；
- `tool_step=11,304`，`final=605`；
- `planning=0`，`initial_tool_step=0`；
- 转换 skipped rows：0；
- 静态候选池：4,823；
- 5/5 task、83/83 tool 覆盖；
- 最大 prompt 245,654；
- 最大 gold target 14,355；
- 源文件未修改，所有排除和压缩均有 manifest 可追踪。

### 8.2 测试

最后一次相关测试命令：

```bash
python -m pytest -q \
  drug_agent/toolrl/tests \
  drug_agent/tests/test_toolrl_v2.py \
  drug_agent/tests/test_structured_rl_context.py \
  drug_agent/tests/test_validate_sft_messages.py \
  drug_agent/tests/test_toolrl_length_probes.py \
  drug_agent/tests/test_qwen35_9b_v4_official_launcher.py
```

修改后的两阶段 launcher 相关测试结果：`46 passed`。命令覆盖 ToolRL 单元测试、v2 launcher、结构化压缩、长度 probes 和 official launcher 回归。

覆盖范围包括 hierarchical reward、final exact、truncation guard、顺序无关多工具匹配、no-progress repeat、合理 retry、deterministic selection、超长 observation、artifact/hash 保留、精确 token 上限和两阶段 launcher 契约。当前主线测试不再把 Plan schema 作为训练链路验收项。

## 9. 当前两阶段训练状态与历史问题

### 9.1 当前主线

```text
ReAct SFT checkpoint → ToolRL v2
```

新 run 使用全新的运行目录，不复用旧三阶段实验的 marker、gate 日志或 Plan-SFT checkpoint。ToolRL 的 `LOAD` 明确指向已经完成的 ReAct SFT checkpoint。启动顺序为：

```text
物化 ToolRL candidate view
  → shortest / p50 / p95 / near-limit gates
  → 10-update multi gate
  → 正式 ToolRL
```

这不是增加了一个新的训练阶段；这些 gate 只是正式 ToolRL 前的运行健康检查。实际模型训练路线仍是 `SFT → ToolRL`。

当前 run：

```text
RUN_ID=Qwen3.5-9B_v4_sft_toolrl_v2_20260813_121417
TMUX=toolrl_v2_sft_0813_121417
RUN_ROOT=/root/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/outputs/
         slime_drug_agent_runs/Qwen3.5-9B_v4_sft_toolrl_v2_20260813_121417
```

启动前确认 8 张 H200 均为 `0 MiB / 0%`、主机内存和磁盘充足、无遗留训练/tmux 进程。基础 SFT 日志确认完成 step 0–302 共 303 个 optimizer update。当前 launcher 从该 ReAct SFT 权重开始，未加载旧 Plan-SFT。

截至 2026-08-13 12:22（Asia/Hong_Kong），新 run 已完成 11,909 条 ToolRL decision 转换、4,823 条 candidate view 物化及长度 probes，正在执行 `toolrl_gate_shortest`。物化结果与上文验收值一致：605 个 final、83/83 tools、5/5 tasks、70 条压缩样本、最大 prompt 245,654、最大 target 14,355、0 复制行。Ray job、8 个 SGLang engine、Megatron actor 和 ReAct SFT checkpoint 均已加载，权重同步完成。shortest 第一个 group 的 reward mean/std 为 `-0.45/0.05`，nonzero-std group ratio=1.0，truncation=0；step 0 已完成，grad norm=6.964。第二个 rollout/update 正在继续，因此整个 shortest gate 尚未标记为 PASS。后续 stage 和健康结果以 `status.log` 及各 stage 独立日志为准。

### 9.2 已废弃三阶段实验的可复用结论

旧运行：

```text
Qwen3.5-9B_v4_plan_sft_toolrl_v2_20260812_201500
```

该实验不再继续，也不能作为当前主线 checkpoint 来源。它仅提供以下工程事实：

| 阶段 | 状态 | 关键结果 |
| --- | --- | --- |
| 数据转换/旧规则 Plan view/候选池 | 历史完成 | 当前不复用 Plan view |
| 旧 Plan-SFT | 历史完成 | checkpoint 明确弃用 |
| shortest gate | 完成 | 2 updates，grad norm 6.272 / 5.892 |
| p50 gate | 完成 | 2 updates，grad norm 5.815 / 3.959 |
| p95 gate | 完成 | 2 updates，grad norm 5.456 / 2.487 |
| near-limit gate | 完成 | 2 updates，grad norm 2.225 / 8.029 |
| multi-update gate | 历史运行被停止 | 只完成 step 0，不可恢复 |
| 正式 ToolRL | 未启动 | 无可复用 ToolRL checkpoint |

near-limit 成功重试时：

- rollout raw reward：0.55、0.625；
- reward std：0.606、0.650；
- nonzero-std group ratio：1.0；
- truncation ratio：0；
- 最长 response：5,680 token；
- 平均 total length 约 247,567 token；
- 未降低 245,760 prompt/262,144 context 契约。

multi-update 第一个 batch：

- 使用正式 `RBS=4、n=4、GBS=16`；
- 动态筛选丢弃 1 个全错 group 后补齐有效 batch；
- 16 个训练样本，reward std=0.313；
- nonzero-std group ratio=1.0；
- truncation ratio=0；
- 9 个动态 microbatch，最大 packing bin=14,926 token；
- step 0 grad norm=3.186。

### 9.3 历史问题一：旧 Plan-SFT 第一次恢复时没有真正训练

现象：launcher 加载已完成的 ReAct SFT checkpoint 后，继承了旧数据进度，命令很快结束，看似完成但没有新的 optimizer update。

根因：finetune 入口没有显式把 rollout/data cursor 归零。

修复：SFT launcher 增加 `--start-rollout-id 0`，并重新执行。随后实际完成 303 个更新。

状态：旧实验中已解决；当前主线完全不运行 Plan-SFT，因此不再适用。

### 9.4 历史问题二：第一次 near-limit backward OOM

第一次 near-limit 尝试中，rollout 成功，但后续 backward 发生：

```text
Tried to allocate 1.88 GiB
GPU free: 1.12 GiB
PyTorch reserved but unallocated: 5.39 GiB
```

这更符合 allocator fragmentation，而不是 prompt 超过契约。没有改 TP/PP/CP、batch 或上下文长度，只给 fully-resident actor 设置：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

相同 near-limit gate 重试后两次 backward 均成功，峰值阶段仍有约 3–6 GiB 显存余量。

状态：已解决，但正式长跑仍需持续观察碎片累积。

### 9.5 历史问题三：成功重试被旧日志中的 OOM 误判失败

现象：near-limit 重试已经成功退出，但 gate checker 扫描同一个追加日志时读到了上一次尝试留下的 traceback/OOM，因此误报失败。

根因：一个 stage 的多次 attempt 共用 append-only log，健康检查没有 attempt 边界。

修复：`run_logged` 在重试未完成 stage 前先归档旧日志，新 attempt 使用独立日志；本次成功段隔离后重新检查为 PASS。

状态：已解决。历史失败日志仍保留用于审计。

### 9.6 仍需关注：policy-boundary 补采样可能较慢

multi gate 的第一个有效 batch 用时较长，主要来自：

- 某些 response 接近 16,384 上限才结束；
- 全对/全错 group 必须丢弃并补采样；
- exact-final 对当前 policy 较难，容易产生全错零方差 group。

这不是 hang：SGLang 解码吞吐稳定、请求持续完成，最终形成了有效 batch 并完成非零梯度更新。但它会显著影响正式训练 wall time，也可能改变 final/tool-step 的实际训练比例。

状态：机制正常，效率和 final 覆盖尚未完成长期验证。

### 9.7 为什么旧训练不能直接恢复

旧 multi-update gate 使用 `DISABLE_CHECKPOINT_SAVE=1`，暂停前的第 1 个 update 没有 checkpoint。更重要的是，旧 run 从 Plan-SFT checkpoint 加载，与当前 `SFT → ToolRL` 实验定义不同，因此即使存在 checkpoint 也不应恢复。新 run 必须从 ReAct SFT checkpoint 独立重跑全部 gate。

当前没有：

- `toolrl_multi_update.complete` marker；
- 正式 `toolrl/` checkpoint；
- 正式训练的长期 reward/gradient 曲线；
- 与 official baseline 的最终对比结果。

## 10. 当前主线的推广门槛

正式 ToolRL 必须依次满足：

1. fresh run root 和空闲 8 卡 preflight；
2. 候选池、源 hash、模型/tokenizer 长度契约验收；
3. shortest、p50、p95、near-limit 均产生有限 reward 和非零梯度；
4. 完整 10-update gate 连续健康，无 OOM、NaN 和 truncation 激增；
5. policy-boundary 补采样能够形成有效 group，并记录各 role 的 drop reason；
6. 只有全部 gate 通过后才自动进入正式 1,259-rollout ToolRL。

新 run 不沿用旧 `RUN_ID`，也不设置 `RESUME_V2_RUN=1`。若新 run 在非语义性故障后需要恢复，只能复用该新 run 自己已经通过的 marker；不能跨实验复用旧三阶段 marker。

## 11. 关键文件索引

| 文件 | 职责 |
| --- | --- |
| `drug_agent/toolrl/convert_react_to_toolrl_steps.py` | 提取 `tool_step/final`、标记 initial、识别 no-progress repeat |
| `drug_agent/scripts/build_plan_view.py` | 历史离线原型；当前 launcher 不引用，未来 LLM plan 标注完成后再评估 |
| `drug_agent/scripts/compact_rl_context.py` | Microcompact + recent suffix + deterministic structured summary |
| `drug_agent/scripts/select_toolrl_decisions.py` | 构建 diversity/coverage candidate pool、长度审计 |
| `drug_agent/toolrl/policy_boundary.py` | n=4 当前策略 reward 方差筛选及 learnability audit |
| `drug_agent/toolrl/molclaw_reward.py` | hierarchical tool reward 与 exact final reward |
| `drug_agent/toolrl/metrics.py` | 按 decision role 记录 reward/格式/工具/参数/truncation 指标 |
| `drug_agent/toolrl/scripts/run_toolrl_grpo.sh` | ToolRL 启动参数和动态筛选接线 |
| `drug_agent/scripts/run_qwen3_5_9b_v4_sft_toolrl_v2.sh` | 当前 canonical 两阶段入口：ReAct SFT → gates → ToolRL |
| `drug_agent/scripts/run_qwen3_5_9b_v4_plan_sft_toolrl_v2.sh` | 兼容旧文件名的底层入口；不构建或训练 Plan-SFT |
| `slime/utils/arguments.py` | strict dynamic-sampling 参数 |
| `slime/rollout/sglang_rollout.py` | strict max-drop 行为，不允许零方差 fallback |
