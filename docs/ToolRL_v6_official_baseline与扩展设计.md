# ToolRL v6：official baseline 与 Drug-Pipe extensions

本设计以 [`原始轨迹数据审计.md`](./原始轨迹数据审计.md) 中的 ordered
reasoning/action block 为数据依据，以 `qiancheng0/ToolRL@8cee13e` 为算法依据。

## 结论

canonical reference 固定为 ToolRL commit
`8cee13ec0ca72f0461da372a93a6fd8140dbb840`。v6 不再把项目 reward、KL、selector
和官方方法统称为 “official”；通过 `V6_PROFILE` 解析为两条独立路径：

- `official_baseline`：ToolRL 官方 reward、GRPO、reference-KL 顺序和全量 fixed traversal。
- `drug_pipe_production`：SFT warm start、hierarchical/structured-final reward、skill discovery、curated view 和 direct frozen-SFT KL。

## 数据语义与 serializer

一个 assistant message 按原始 block 顺序扫描，可以展开为 1..N 个 RL decision。每个有效
unit 是“零个或多个 thought + 一类 action（一个或多个 tool call，或 final）”。
当新 thought 出现时，若当前 unit 已有 action，先 flush 该 unit；后续 decision 的
state 是原 history 加上同一 assistant response 中已生成的精确 prefix。不伪造
observation，不移动 thought，不跨 observation 合并。

同一 unit 的多个 invocation 进入唯一 `<tool_call>` 容器，JSON object 逐行排列，
不加逗号、不使用 array。observation 只进入下一 decision 的 state。
full v5 中有 8 个 `tool_call → thought` interleaved assistant turn；mol v5 为 0。
新切分不再 quarantine 它们：其中 4 个展开为两个有效 action decision，另 4 个保留
第一个 action decision，并将最后无 action 的 thought 记为 incomplete tail。full 另有
6 个独立 thought-only assistant message，因此 incomplete thought 总数为 10；它们保留作为
后续 state，但 `step_loss_mask=0`，不作为 SFT/RL action target。

对 multi-action assistant message，canonical 整轨迹仅用作 history，该 message 在 base SFT record
中 loss-masked；每个 action segment 生成一条 prefix-conditioned SFT record，用
`loss_char_start` 仅监督当前 action。这使 SFT target 与 ToolRL gold action 字节级一致，同时
保留后续 segment 的真实 causal prefix。

### 五个真实 serializer 样例

省略超长参数值，但调用顺序和 multiplicity 不省略。

1. `react_ac_7d6e3eb5bdf4c686:2` 的 skill 目录 single call：

```text
before: <thought>plan ALOX5 comparison...</thought><tool_call>{Bash ls skills}</tool_call>
after:  <thought>plan ALOX5 comparison...</thought>
        <tool_call>
        {"tool_name":"Bash","arguments":{"command":"ls ..."}}
        </tool_call>
```

2. `react_ac_7d6e3eb5bdf4c686:6` 的 `Read boltz2-affinity/SKILL.md` single call：
保持一个 decision 和一个 invocation，只把 JSON 规范化到单一 container。

3. `react_ac_7d6e3eb5bdf4c686:8` 的
`retrieve_protein_sequence(identifier=ALOX5, organism=Homo sapiens)` single call：
tool name、arguments 和 thought 文本均不改变。

4. `react_ac_7d6e3eb5bdf4c686:4` 的 `Read + Bash` multi-call：

```text
before: <thought>...</thought><tool_call>{A}</tool_call><tool_call>{B}</tool_call>
after:  <thought>...</thought><tool_call>\n{A}\n{B}\n</tool_call>
```

5. `react_ac_99a15051f0181668:4` 的两个 `Bash ls` multi-call：两个 invocation
保持原顺序和各自 arguments，不会因 tool name 相同而折叠。

五组未省略的 before/after 原文保存在 release 的
`audit/serializer_examples.json`；此处只做可读摘要。

## official ToolRL baseline objective/config

repo 实际 objective 按下列顺序实现：工具 action 获得 format `0/1` 与
fine-grained correctness `-3..3`，response/final action 只获得 envelope format `0/1`。
对每个 sampled response，先将 frozen-reference token KL 以 `0.001` 从 token reward
扣除，再对整段 token reward 求和得到 sequence score；同 prompt 的 4 个 score
使用 sample standard deviation 做 group normalization。该 scalar advantage 广播给完整
response policy mask，然后应用对称 `0.2` PPO clip 与 `0.001` entropy bonus。
因此 thought 和 action tokens 接受同一 sequence-level GRPO credit，但 thought 文本
本身不做 teacher semantic matching。

| 项目 | `8cee13e` 行为 | v6 baseline |
|---|---|---|
| action | `<think>` + tool/response | `<thought>` + tool/final；仅 tag/schema rename |
| multi-call | 单 container，多行 JSON | 一致；调用 multiplicity 保留 |
| tool reward | format `[0,1]` + correctness `[-3,3]` | 一致 |
| final/response reward | 只检查 response envelope；correctness=0 | 一致；不比较 structured final 内容 |
| n / temperature | 4 / 1.0 | 4 / 1.0 |
| advantage | group mean + unbiased std normalization | 一致 |
| PPO clip | 对称 0.2 | 一致 |
| policy-loss reduction | response token masked mean | `calculate_per_token_loss=1` 后由 Slime 全局 token normalizer 归约 |
| entropy coefficient | 0.001 | 一致 |
| reference KL | `use_kl_loss=False`，但 fixed KL=0.001 在 group normalization 前扣入 reward | custom adapter 精确复现顺序 |
| reference | rollout 初始模型的 frozen reference | 一致 |
| filtering | 无 policy-boundary dynamic filter | 无 |
| traversal | shuffled DataLoader，drop_last，15 epochs | fixed auditable view；decision/group/cursor accounting；训练轮数为运行参数 |
| prompt | 每题 system 中列 available tools | `official_catalog` view 注入离线 catalog |

仍有两项明确的运行系统映射，不能伪装成逐字相同：官方示例的
`train_batch_size=512`、`ppo_mini_batch_size=128`、`total_epochs=15`；当前 8×H200 Slime
profile 使用 RBS=4、GBS=16 和一次 fixed traversal。它保持每个 n=4 group 的 GRPO 数学
目标，但 update 的批量混合与优化噪声不同。另一个差异是官方每题提供相关 tools，Drug-Pipe
baseline 当前注入完整离线 catalog；这是数据域适配，不是 reward/objective extension。两项都
写入 resolved config，后续可单独做 batch 与 per-task catalog 对齐，不与 production extension
混写。

paper 与 repo 对 KL 的表述不一致时按 repo：`actor.use_kl_loss=False` 只表示 KL 不作为独立
actor loss；官方仍创建 RefPolicy，并用 `algorithm.kl_ctrl.kl_coef=0.001` 做 reward shaping。

baseline resolved config：

```text
V6_PROFILE=official_baseline
TOOLRL_REWARD_MODE=toolrl_official_8cee13e
TOOLRL_STRUCTURED_FINAL_EXACT=0
TOOLRL_PROMPT_STRATEGY=official_catalog
TOOLRL_VIEW=all_static
n=4, temperature=1.0
clip_low=0.2, clip_high=0.2
entropy_coef=0.001
use_kl_loss=0, kl_coef=0.001, kl_type=k1
custom_advantage=compute_official_8cee13e_advantages
calculate_per_token_loss=1
dynamic_filter=off
SFT_warm_start=off
```

## Drug-Pipe extensions

| extension | 开关/选择 | production 默认 |
|---|---|---|
| SFT warm start | `V6_PROFILE=drug_pipe_production` | on |
| frozen-SFT direct KL | `TOOLRL_USE_KL_LOSS=1` | on，0.001 low-var |
| hierarchical reward | `TOOLRL_REWARD_MODE=hierarchical` | on |
| structured final exact | `TOOLRL_STRUCTURED_FINAL_EXACT=0/1` | on |
| skill-based discovery/no catalog prompt | `drug_pipe_skill_discovery` view | on |
| static curated selector | production `toolrl_steps.jsonl` | on |
| long-context microcompact/summary | materializer | on |
| dynamic policy-boundary filter | explicit ablation only | off |

production resolved config：

```text
V6_PROFILE=drug_pipe_production
pipeline=SFT -> ToolRL
TOOLRL_REWARD_MODE=hierarchical
TOOLRL_STRUCTURED_FINAL_EXACT=1
TOOLRL_PROMPT_STRATEGY=drug_pipe_skill_discovery
TOOLRL_VIEW=curated_static
n=4, temperature=1.0
clip_low=0.2, clip_high=0.2
use_kl_loss=1, kl_loss_coef=0.001, kl_type=low_var_kl
reference=frozen SFT checkpoint
dynamic_filter=off
```

## Thought 与 loss

teacher thought 不参与 reward semantic matching。rollout 的完整生成（thought 与 action）均在
response policy mask 内，sequence reward 的 GRPO advantage 广播到所有这些 token。官方 4K
RL target 的 4,000/4,000 都含非空 think，其中 3,518 个 tool target 使用同一个 placeholder，
482 个 response target 使用另一个 placeholder。v6 不改写 Drug-Pipe teacher 的无-thought
turn；baseline runtime format reward 与官方一样要求 thought tag（官方数据本身均为非空）。先前“materializer 删除 thought”
并不成立：v5 `target_assistant` 始终保留 thought；发生过重建的只是旧 length-estimation target。

## Grammar parity 与 production parser

`drug_pipe_production` 的 SFT action target 和 ToolRL gold action 共用
`toolrl_turn.serialize_decision`。full 中多段 turn 通过 prefix-conditioned supplemental SFT
record 实现，不会让 SFT 学“多个 `<tool_call>` block”、RL 学“单一 container”。

production runtime 和 reward 共用 strict ToolRL-turn grammar。end-to-end contract 为：

| container body | runtime | reward parser |
|---|---|---|
| `A\nB\nC` 三个 JSON object | valid，3 invocations | valid，3 invocations |
| `A,B,C` | invalid | invalid |
| `[A,B,C]` | invalid | invalid |
| 多个 `<tool_call>` container | invalid | invalid |

因此 production 不再用普通 `json.loads` 解析整个 container body，而是用连续 JSON decoder
解析空白分隔的 object，并显式拒绝逗号、array、尾随文本与多 container。

## Invocation multiplicity

官方 matcher 使用 `Counter` 计算 name multiset，并用 `used_pred_indices` 做一对一最佳匹配。
v6 baseline 保持这一点：`dock(arg1)` 和 `dock(arg2)` 是两个 invocation；一个 predicted dock
不能匹配两条 gold。测试覆盖 two-gold/one-pred 得分低于满分，以及 two-gold/two-pred
顺序颠倒仍得到 4.0。

真实数据中的极值样例是 `react_kg_c24a3e7991f6e7b4:40`：同一 turn 含
12 个 `pred_binding_affinity_boltz2` invocation，每个有不同 ligand arguments。v6
serializer 和 matcher 均保留这 12 重 multiplicity。

## 分段与 action-space 审计

| release | assistant messages | expanded decisions | multi-action turns | interleaved turns | incomplete thought tails | tool/final |
|---|---:|---:|---:|---:|---:|---:|
| full | 11,915 | 11,913 | 4 | 8 | 10 | 11,308 / 605 |
| mol | 5,329 | 5,329 | 0 | 0 | 0 | 4,964 / 365 |

full 有 1,934 个 thoughtless action，mol 有 428 个；不伪造 placeholder thought。full 的
multi-tool decision 为 2,481，mol 为 1,527。

`max_response_length=16,384` 是 runtime action-space 硬上限，因此 baseline 也不再保留永远
无法完整生成的 gold action。full 原始 11,913 decisions 中有 46 个超限，最大
445,297 token；mol 原始 5,329 decisions 中有 13 个超限，最大 45,571
token。排除原因统一记为 `target_action_exceeds_runtime_response_limit`，禁止截断 label。

计数术语固定为：一个 decision 对应一个 GRPO group；每个 group 采样 n=4 条
response；RBS=4 表示一个 rollout batch 消费 4 个 decisions/groups。例如 11,900
decisions 是 11,900 groups、47,600 sampled responses 和 2,975 rollout batches，不是
2,975 groups。物化后的精确 final accounting 由各 release `dataset_manifest.json` 记录。

## 可审计产物

每个 release 同时生成：

- `toolrl/toolrl_steps.official_baseline.jsonl`：catalog prompt + all eligible fixed view。
- `toolrl/toolrl_steps.jsonl`：Drug-Pipe skill-discovery + curated production view。
- 两份独立 context manifest；记录 selection、长度排除、prefix-conditioned
  segmentation 和 RBS 对齐。
- launcher 的 resolved config 记录 profile、reward、KL 路径、reference、decision count 和
  `num_rollout=N/4`。gate、multi-update 和正式 run 使用独立 traversal 日志；正式
  run 逐 group 记录 decision key、group index、dataset epoch/cursor、是否进入 update
  和消费次数。训练结束前 `validate_fixed_toolrl_traversal` 必须验证无重复、无遗漏，
  否则不写入 `toolrl.complete`。
