# 首次 9B 模型训练与在线评测报告

## 1. 报告范围

本文记录 Drug-Pipe 第一次正式训练得到的 Qwen3.5-9B Drug Agent 在真实工具环境中的测试情况，重点回答：

- 首次 SFT、ToolRL 训练得到了什么模型；
- 模型已经具备哪些基础能力；
- 截至 2026-08-06，MolBench 测试取得了什么结果；
- 测试暴露了哪些模型问题；
- 哪些失败来自评测 runtime 或基础设施，不能直接归因于模型。

本文是阶段性工程报告，不是最终 benchmark 报告。MS-1 和 MS-2 已完整落盘并可计算指标；MO 只落盘了部分题目，随后按人工指令停止了评测任务。

## 2. 被测模型

本轮测试使用首次 9B SFT→ToolRL 训练结果：

```text
Base model: Qwen3.5-9B
Training data: 首版 373 条 canonical ReAct 训练数据
Checkpoint stage: ToolRL
Checkpoint iteration: iter_0000756
Checkpoint run:
slime-wd/outputs/slime_drug_agent_runs/
  Qwen3.5-9B_current373_full_20260727_110712/
  toolrl_procfsfix_20260728_091212/iter_0000756
```

当前评测针对该 ToolRL checkpoint，不是 base model，也不是尚未完成正式验收的 GAD checkpoint。

在线推理使用项目统一的 XML ReAct 协议：

```xml
<thought>...</thought>
<tool_call>{"tool_name":"...","arguments":{...}}</tool_call>
```

或：

```xml
<thought>...</thought>
<final_answer>{...}</final_answer>
```

评测启动时加载了：

- 81 个由 molclaw-scp 实时 `list_tools` 返回的 MCP 工具；
- 6 个在线评测允许的本地工具；
- 62 份 L1 tool-level skill 文档快照；
- 每题独立 workspace 和 artifact registry。

MolBench 全量运行的主要参数是：

```text
MAX_WORKERS=2
NUM_GPUS=2
TENSOR_MODEL_PARALLEL_SIZE=2
TEMPERATURE=0
MAX_STEPS=0
TASK_TIMEOUT_SEC=10800
MAX_NEW_TOKENS=16384
EVAL_MAX_CONTEXT_LEN=65536
```

其中 `MAX_STEPS=0` 表示没有 assistant decision 次数上限。后续结果证明，在当前模型和 runtime 尚无可靠 no-progress guard 的情况下，这一配置并不安全。

## 3. 评测环境打通情况

在正式 MolBench 批量测试前，曾使用短任务验证模型权重、XML parser、工具 registry、MCP relay、本地工具和 artifact 映射是否真正连通。

以下五个短任务均成功完成：

1. EGFR protein retrieval → `fix_pdb`；
2. `is_valid_smiles("CCO")`；
3. `visualize_molecule("CCO")`；
4. EGFR protein retrieval → `visualize_protein`；
5. Write `smoke ok` → Read。

这些成功说明：

- 指定的 ToolRL checkpoint 能被 Slime eval-only 路径正确加载；
- 模型能生成可解析的 XML ReAct decision；
- MolClaw MCP 工具能够被真实调用；
- Read/Write 等本地工具能够在隔离 workspace 中执行；
- MCP 返回的服务器路径能够经过 artifact registry 映射并被后续工具继续使用；
- 模型具备完成一至三步、目标明确且较熟悉的工具链的能力。

这些 smoke tests 只能证明基础链路和短任务能力，不能证明模型已经具备稳定的长程规划能力。

## 4. MolBench 批量测试范围与停止状态

本轮计划运行：

| Suite | 任务类型 | 计划题数 |
|---|---|---:|
| MS-1 | property filtering | 50 |
| MS-2 | activity comparison | 33 |
| MO edit | molecule editing | 78 |
| 合计 |  | 161 |

评测在 2026-08-06 按人工指令停止。停止后：

- Ray job 已终止；
- SGLang engine、rollout actor 和评测进程均已退出；
- 两张 H200 的显存均已释放；
- 已完成或失败的逐题结果均被保留；
- 共享 MCP relay 未停止。

结果目录中共有 115 条唯一任务记录：

| Suite | 已落盘 | completed/final | failed | truncated |
|---|---:|---:|---:|---:|
| MS-1 | 50 | 40 | 4 | 6 |
| MS-2 | 33 | 11 | 7 | 15 |
| MO edit | 32 | 0 | 18 | 14 |
| 合计 | 115 | 51 | 29 | 35 |

还有 46 题没有任务结果。停止前当前 runner 的进度条约为 `107/161`；结果目录中的 115 条记录还包含 resume/retry 前已经持久化的任务，因此该数字与当前进度条位置并不等价。

按结束原因汇总：

| done reason | 数量 |
|---|---:|
| `final_answer` | 51 |
| `length` | 35 |
| `fatal_error` | 23 |
| `task_timeout` | 6 |

## 5. MS-1 与 MS-2 指标

MS-1 和 MS-2 的 83 题均已落盘，因此可以按全题分母计算阶段性指标。没有生成合法 final 的题目使用空 prediction，并按 0 分进入端到端指标。

### 5.1 MS-1

| Metric | Value |
|---|---:|
| accuracy | 0.72 |
| precision | 0.774 |
| recall | 0.77 |
| F1 | 0.7633 |
| sensitivity | 0.77 |
| specificity | 0.96 |
| validity | 0.79 |
| n_samples | 50 |

MS-1 表明当前模型已经具备一定的 property filtering 能力。其端到端 accuracy 达到 0.72，但仍有 10/50 题没有产出 final，因此该结果同时受到模型决策和评测 runtime 成功率影响。

### 5.2 MS-2

| Metric | Value |
|---|---:|
| accuracy | 0.1515 |
| valid rate | 0.2727 |
| n_samples | 33 |

MS-2 只有 11/33 题产生 final，端到端 accuracy 为 5/33。相较 MS-1，activity comparison 的稳定性明显不足；长工具交互、工具失败和 runtime 故障都对该结果产生了影响。

当前指标文件位于：

```text
slime-wd/outputs/slime_drug_agent_evals/
  molbench_ms1_ms2_mo_toolrl9b_2gpu_persistfix_20260805_103459/
  interim_metrics/ms1_ms2_stopped_20260806/evaluation_summary.json
```

## 6. 模型自身暴露出的主要问题

### 6.1 长任务不能稳定收敛到 final

这是本轮最严重的模型问题。模型经常已经获得足够信息，却继续调用相同工具或重复验证，无法稳定完成：

```text
信息充分
→ 汇总证据
→ 输出 final_answer
```

典型轨迹包括：

- `molbench_mo_edit_delete_008`：541 个 decision，重复调用 `is_valid_smiles` 541 次；
- `molbench_mo_edit_delete_009`：800 个 decision；
- `molbench_mo_edit_sub_005`：794 个 decision；
- `molbench_ms2_014`：1,036 个 decision，其中 1,027 次为工具错误；
- `molbench_ms2_025`：583 个 decision，其中 578 次为工具错误。

模型也没有稳定掌握失败恢复策略：工具连续失败后，它经常不改变参数、不切换工具、不缩小任务，而是继续重复基本相同的调用。

### 6.2 MO final schema 遵循失败

在部分 molecule editing 轨迹中，模型已经推导出正确修改结果并成功验证 SMILES，却使用了错误的 final 字段：

```json
{
  "modified_smiles": "...",
  "original_smiles": "...",
  "task_type": "mol_edit"
}
```

评测协议要求的是：

```json
{
  "output_smiles": "..."
}
```

runtime 随后明确返回：

```text
mol_edit `final_answer.output_smiles` must be a string
```

但模型仍反复提交相同错误结构。例如：

- `molbench_mo_edit_delete_001`：428 个 decision，其中 426 个非法；
- `molbench_mo_edit_sub_001`：417 个 decision，其中 415 个非法；
- `molbench_mo_edit_sub_003`：325 个 decision，其中 322 个非法；
- `molbench_mo_edit_add_008`：456 个 decision，其中 453 个非法。

MO 已落盘的 32 题累计出现 1,762 个非法 decision，且没有一题成功提交合规 final。这说明：

- 首次训练数据对 MO terminal schema 的覆盖不足；
- 模型在长上下文中容易失去协议约束；
- 模型不能可靠地利用 runtime schema-error observation 做自我修正。

### 6.3 Reasoning 重复且冗长

模型经常在相邻 thought 中重复同一个计划。例如同一条轨迹开头连续输出：

```text
I need to analyze the given SMILES...
I'll analyze the given SMILES...
Let me validate the SMILES...
```

完成科学判断后，又连续输出多个意思相同的结论：

```text
The modified SMILES is valid...
The task is complete...
Task completed successfully...
Summary: ...
```

这种行为会：

- 浪费生成 token；
- 延长单题工具交互时间；
- 增大 tool-call/final JSON 被截断的概率；
- 快速挤占上下文窗口；
- 使长任务最终进入 context overflow。

训练数据中发现的相邻重复 thought 此前已进行过一次确定性清理，但在线测试说明，单纯删除训练数据中的重复文本还不足以建立可靠的终止策略。

### 6.4 复杂任务中的 XML/JSON 协议稳定性下降

在短任务中，模型通常可以生成严格合法的 XML ReAct。但当出现以下条件时，协议稳定性明显恶化：

- 上下文变长；
- 同一工具多次失败；
- 需要生成任务类型特定的 final schema；
- 分子编辑需要先提出结构、再验证、再提交最终答案。

表现包括：

- final 字段不符合 task schema；
- 收到 schema error 后原样重复；
- tool-call JSON 过长或被截断；
- 连续产生大量非法 decision；
- 正确科学结果没有转换成合规 terminal answer。

### 6.5 工具失败后的 replanning 能力不足

单次工具失败本身不是模型错误；真实药物任务中，模型应当能够分析失败原因并重规划。当前模型在短链 retrieval→repair 测试中曾成功做到失败后改道，但在批量长任务中不稳定。

主要问题不是“轨迹里出现工具 error”，而是：

```text
同一错误反复发生
+ 参数与路径没有实质变化
+ 模型仍持续调用
+ 最终没有形成可用结论
```

这说明其局部 replanning 能力存在，但尚未泛化为稳定策略。

### 6.6 不同任务类型之间的能力差距明显

截至本轮停止时，模型表现呈现以下分层：

```text
短、熟悉、目标明确的 1–3 步工具链：较稳定
MS-1 property filtering：已有一定能力
MS-2 activity comparison：端到端成功率较低
MO molecule editing：尚不能稳定完成
长链、重复失败和复杂 terminal schema：明显薄弱
```

因此，首次 9B ToolRL 模型已经是一个能够执行真实短程工具任务的 agent，但还不是能够无人值守执行任意长程药物任务的成熟 agent。

## 7. 不能直接归因于模型的评测系统问题

本轮结果同时暴露出在线评测 runtime 的缺陷。在修复这些缺陷前，当前 MolBench 分数只能视为端到端系统结果或模型能力下界，不能视为模型能力的纯净估计。

### 7.1 配置的 context cap 没有被实际执行

虽然 manifest 中记录了：

```text
EVAL_MAX_CONTEXT_LEN=65536
```

custom rollout 仍不断把完整历史追加到下一轮输入。实际观察到：

```text
input tokens:       245,797
requested output:    16,384
total:              262,181
Qwen native limit:  262,144
```

最终请求必然触发 HTTP 400。该问题来自 runtime 未在生成前执行上下文预算，而不是模型主动选择超过模型上限。

### 7.2 确定性 HTTP 400 被无意义重试

context overflow 属于同一输入下不可恢复的客户端错误，但 generic HTTP retry 会继续重复请求约 60 次。这样既不能得到新结果，也显著增加失败任务耗时。

### 7.3 MCP 连接和生命周期管理仍不完全稳定

日志中仍有：

```text
MCP connect timeout after 60s
Attempted to exit cancel scope in a different task
asynchronous generator is already running
MCP close timeout
```

23 条 fatal error 的直接错误类型为：

| Error | 数量 |
|---|---:|
| HTTP 503 | 10 |
| HTTP 400 | 7 |
| MCP connect timeout | 6 |

HTTP 503 和 MCP connect timeout 不能作为模型科学能力失败直接计分；HTTP 400 中则包含被模型无界循环放大、最终由 runtime context overflow 终止的混合问题。

### 7.4 缺少 no-progress/repeated-action guard

`MAX_STEPS=0` 允许模型无限决策，而 runtime 没有对下列行为实施受控终止：

- 相同工具和参数连续重复；
- 相同 schema-invalid final 连续重复；
- 连续工具错误且状态没有变化；
- token budget 已接近模型上下文上限。

因此，本可在几十步内报告失败或触发明确保护的任务，最终扩张为数百至上千个 decision。

### 7.5 Artifact 路径识别会误伤带斜线键方向的 SMILES

重新物化 MS-1/MS-2 指标时发现，通用 POSIX 路径识别可能把 SMILES 中的 `/N=`、`/C=` 误认为绝对路径。例如：

```text
NS(=O)(=O)c1ccc(/N=C/c2ccccc2)cc1
```

被错误转换成包含：

```text
<artifact:local/N>=
```

的字符串，导致 RDKit 无法解析。这是 artifact sanitizer 的类型识别缺陷，而不是模型的化学错误，也可能使少量指标偏低。

## 8. 对首次模型的阶段性判断

首次 9B SFT→ToolRL 训练已经取得以下实质成果：

- 模型能够使用项目统一的 XML ReAct 协议；
- 能够真实调用 MolClaw MCP 工具；
- 能够调用隔离 workspace 中的本地文件工具；
- 能够处理 canonical artifact reference；
- 能够稳定完成一部分短程任务；
- 在完整 MS-1 上取得 0.72 的端到端 accuracy。

当前最核心的模型问题可以归纳为：

```text
长程规划不收敛
+ reasoning 重复
+ 工具调用重复
+ 连续失败后缺少稳定改道
+ task-specific final schema 遵循不稳定
+ MO 等低覆盖任务泛化不足
```

当前最核心的 runtime 问题可以归纳为：

```text
context cap 未生效
+ 不可恢复 HTTP 错误仍重复重试
+ MCP 生命周期不稳定
+ 缺少 no-progress guard
+ artifact sanitizer 误伤 SMILES
```

因此，不能简单地把所有未完成题归因于模型，也不能仅凭短任务成功就认为模型已具备完整药物智能体能力。更准确的结论是：

> 首次训练已经打通了从 checkpoint、XML ReAct 到真实工具执行的基本闭环，并在短任务和 MS-1 上显示出有效能力；但模型的长程终止、失败恢复、协议稳定性和 MO 泛化仍明显不足，同时评测 runtime 尚有会放大失败的工程缺陷。

## 9. 后续评测前的优先事项

为了获得能够指导下一轮训练的可信结果，建议按以下顺序处理：

1. 修复 runtime context budget，确保配置的上下文上限真实生效；
2. 对确定性 4xx 禁止重复重试；
3. 修复 MCP session teardown 和连接恢复；
4. 修复 SMILES 与 POSIX artifact path 的类型区分；
5. 增加可审计的 repeated-action/no-progress 保护；
6. 在同一 checkpoint 上重跑失败题，先分离 runtime failure 与 model failure；
7. 基于干净失败集补强 MO final schema、终止决策和失败 replanning 数据；
8. 再决定下一轮 SFT、ToolRL 或 GAD 的训练调整。

不应使用 runtime alias 或静默修改模型输出的方式掩盖模型协议错误。runtime 可以防止无限循环和资源失控，但模型是否能够输出正确 decision，仍应在 trace 和指标中如实保留。

## 10. 结果与证据位置

本报告对应的评测目录：

```text
slime-wd/outputs/slime_drug_agent_evals/
molbench_ms1_ms2_mo_toolrl9b_2gpu_persistfix_20260805_103459/
```

关键文件：

```text
run_manifest.json
benchmark_manifest.json
task_results/*.json
interim_metrics/ms1_ms2_stopped_20260806/evaluation_summary.json
interim_metrics/ms1_ms2_stopped_20260806/metrics.json
interim_metrics/ms1_ms2_stopped_20260806/predictions.jsonl
interim_metrics/ms1_ms2_stopped_20260806/failures.jsonl
```

报告中的数量和指标均以停止后保存的这些文件为准。
