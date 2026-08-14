# ReAct planning clean 与超长 step 压缩

## 数据流

正式数据路径保持两阶段训练：

```text
ReAct production
  -> LLM clean + initial high-level planning thought
  -> SFT
  -> raw ToolRL/GAD decisions
  -> context-budget materialization
  -> ToolRL/GAD
```

没有独立 Plan-SFT。planning 不作为 ToolRL 的独立 role、reward 或复制样本；首个 tool decision 仍正常参与 RL。

## Planning clean

`llm_clean_patch_v2` 要求每条 trajectory 提供 `planning_action`。LLM 根据完整 teacher trajectory 决定：

- 首 thought 已经是任务级 plan：`rewrite_first_thought`；
- 首 thought 只是当前一步 rationale：在前面 `prepend_planning_thought`；
- 没有 thought：在首个 tool call 前 prepend。

Python gate 校验目标必须是第一条有效 assistant decision，并继续冻结 tool calls、arguments、observations、final facts 和消息顺序。成功输出除 `react_trajectories.jsonl` 外还包含 `planning_annotations.jsonl`，供未来 planning RL 使用。

## ToolRL/GAD context materialization

超长 step 使用统一的三级路径：

1. 真实 Qwen chat template 精确计数；未超 245,760 token 时不改动。
2. 对旧 observation 的 blob、base64、大数组和文件正文做 typed microcompact；保留 hash、长度、有限 head/tail、path/artifact/ID/status/error。
3. 仍超限时调用 `summarize-react-step-context` skill。它只接收历史 prefix，不接收当前 teacher response；按完整 assistant/observation 单元做 map/reduce，并保留最大可用的近期完整后缀。

硬约束为：

```text
max context  = 262144
max prompt   = 245760
max response = 16384
summary      <= 32768
```

LLM summary 最多尝试三次。schema、grounding 或长度校验仍失败时，只排除对应 decision；不截断 label。ToolRL/GAD 共享由 `source state hash + skill/prompt/schema version` 定位的 summary cache。

在可联网 CPU worker 上物化两个 view：

```bash
INPUT=/shared/react_trajectories.jsonl \
OUTPUT_ROOT=/shared/react_rl_views_v1 \
HF_CHECKPOINT=/shared/Qwen3.5-9B \
bash drug_agent/scripts/materialize_react_rl_views.sh
```

GPU worker只读取物化结果，并在启动训练前复核 hash、schema 和 token gate。
CPU worker 的 Slime 环境必须能真实执行 Qwen chat template；当前登录环境的 `jinja2 3.0.3` 不满足现有 Transformers 所需的 `jinja2>=3.1`，因此正式物化脚本会在 preflight 阶段拒绝该环境，而不是使用近似 token 计数。

9B ToolRL launcher 通过 `PREMATERIALIZED_RL_VIEW_ROOT` 加载该 view；设置后只校验 source hash/长度契约并构建 gate probes，不会在 GPU worker 调用 LLM：

```bash
PREMATERIALIZED_RL_VIEW_ROOT=/shared/react_rl_views_v1 \
CANONICAL_DATA=/shared/react_trajectories.jsonl \
EXPECTED_CANONICAL_SHA256=<cleaned-source-sha256> \
bash drug_agent/scripts/run_qwen3_5_9b_v4_sft_toolrl_v2.sh
```

## SFT 超长边界

ToolRL/GAD 的 step compactor 不处理整条 SFT trajectory。当前 v4 审计为：

- 29/606 行超过当前 SFT cap 131,072；
- 14/606 行超过 245,760；
- 13/606 行超过模型上限 262,144；
- 最大 1,710,101 token。

606 行包含为 batch 对齐复制的一行，canonical source 为 605 条。当前 SFT loader 仍保留 8,192-token 头部和最近尾部、删除中间；本轮只扩展审计，不改变该策略。planning clean 后必须重新运行 SFT length probes。
