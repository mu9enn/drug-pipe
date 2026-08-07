# Canonical v3（510 条）SFT、ToolRL 与 GAD 训练交接

> 更新时间：2026-08-07  
> 适用对象：接手当前 Drug-Pipe 模型训练的研究人员或智能体

## 1. 必须使用的数据版本

当前训练事实源是：

```text
slime-wd/outputs/slime_drug_agent_data/live_tool_catalog_v3/
```

它由两组互不重叠的 canonical ReAct 合并而成：

```text
365 条 live_tool_catalog_v2
+ 145 条外部 canonical 迁移数据
= 510 条 canonical v3
```

合并审计确认 ID、规范化问题和完整记录的交集均为 0。canonical SFT 文件的 SHA256 为：

```text
8c0bb6b53cc5e58d297a17d17245186b1c597fb5dc58132b03cbee53ddff1e57
```

不要再用 v1/v2 派生的 ToolRL 或 GAD steps。尤其不能把 510 条 SFT 与旧的、只把 MolClaw call 当 decision 的 RL 数据混用。

## 2. 本轮修正的 decision contract

ToolRL 和 GAD 现在对下列 87 个可执行工具一视同仁：

```text
81 个实时 MolClaw 工具
+ Read / Write / Edit / Bash / Grep / Glob
= 87 个训练可见工具
```

每个 assistant decision 的完整可执行 tool calls 按原顺序保留。包含本地工具和 MolClaw 工具的 mixed decision 不再被拆开或丢弃。`Skill` 不在本合同中：当前 canonical v3 不把它当作可执行训练工具。

所有 state 都是 history-only fixed expert state；ToolRL/GAD converter 不执行 MCP，本地工具也不在线执行。Formal SFT、ToolRL 和 GAD 继续保持 offline。

## 3. 正式数据路径、数量与 hash

| 用途 | 相对 `live_tool_catalog_v3` 的路径 | 条数 | SHA256 |
|---|---|---:|---|
| SFT | `react_trajectories.jsonl` | 510 | `8c0bb6b53cc5e58d297a17d17245186b1c597fb5dc58132b03cbee53ddff1e57` |
| ToolRL | `toolrl/toolrl_steps.jsonl` | 9003 | `26c45bbebc60e36517c794d22bd666fd267a1478a4bf38c889862f48383e2275` |
| GAD | `gad/gad_steps.jsonl` | 9003 | `db12b328b14cf88c73fe81cb5e84d1d34763f7233e1926691702b9662f482020` |
| 工具目录 | `tool_catalog.json` | 87 tools | `f279c02cba16e5a6beba65aa83bf183064e3cab39d8f689dfd8545870f9c7c7a` |

ToolRL 与 GAD 都包含：

- 8493 个 tool-call decisions；
- 510 个 final-answer decisions；
- 15821 个目标 tool calls；
- 12186 个 MolClaw calls；
- 3635 个本地 calls；
- 3352 个含本地工具的 decisions；
- 47 个同一 decision 内混合本地与 MolClaw calls 的 decisions；
- 0 个跳过的 decisions。

本地调用分布：

| 工具 | 调用数 |
|---|---:|
| Read | 1228 |
| Write | 958 |
| Edit | 542 |
| Bash | 831 |
| Grep | 1 |
| Glob | 75 |

详细事实源：`merge_manifest.json`、`derived_data_manifest.json`、`toolrl/toolrl_steps.report.json`、`gad/gad_steps.report.json`。ToolRL 已通过 9003/9003 的 offline schema/catalog validation。

## 4. 训练阶段的正确继承关系

```text
canonical v3 SFT
├── SFT checkpoint → ToolRL（v3 ToolRL steps）
└── SFT checkpoint → GAD generator warmup（v3 GAD steps）
```

不要把 ToolRL checkpoint 当作 GAD 的 generator warmup，除非实验设计明确改变并单独记录。GAD 仍需使用配对且通过 manifest 校验的 generator/discriminator warmup checkpoint。

如果 SFT 已经用上述 canonical SHA256 启动或完成，不需要为了 converter 修复重跑 SFT；从 SFT 之后改用本文件列出的新 ToolRL/GAD 数据即可。如果 SFT 的输入 hash 不是上述值，则它不是这套 510 条 v3 实验。

## 5. Worker 上的显式环境设置

登录 worker 后先固定路径，避免继承旧 shell 中的 v1/v2 环境变量：

```bash
ROOT=/root/slime_sxy/group-space/sunxiangyu/drug-pipe
export LIVE_DATA_ROOT="$ROOT/slime-wd/outputs/slime_drug_agent_data/live_tool_catalog_v3"
export CANONICAL_DATA="$LIVE_DATA_ROOT/react_trajectories.jsonl"
export TOOLRL_DATA="$LIVE_DATA_ROOT/toolrl/toolrl_steps.jsonl"
export GAD_DATA="$LIVE_DATA_ROOT/gad/gad_steps.jsonl"

wc -l "$CANONICAL_DATA" "$TOOLRL_DATA" "$GAD_DATA"
sha256sum "$CANONICAL_DATA" "$TOOLRL_DATA" "$GAD_DATA"
```

预期行数依次为 `510 / 9003 / 9003`，hash 必须与第 3 节一致。通用 9B 串行 launcher 和 `qwen3_large_profile.sh` 的新 shell 默认已指向 v3，但显式设置仍是跨 worker 交接最稳妥的方式。

已经启动的 shell、已写出的 `serial_config.env` 或带显式 `PROMPT_DATA` 的命令不会自动继承代码中的新默认值；必须核对实际 resolved config。

## 6. 122B compact 数据的特别约束

`run_qwen35_122b_lora_rl_serial.sh` 目前仍显式引用旧 v1 的 `*_ctx10240.jsonl`。不要把旧 compact steps 与 v3 SFT 混用，也不要直接把未压缩 v3 steps 填进去冒充 10240-token 数据。

122B 训练必须先从本轮 v3 的 9003 条 steps 生成新的、经过 Qwen3.5 tokenizer 验证的 compact/bounded-context 派生集，记录保留/截断策略、条数和 hash，再显式覆盖：

```bash
QWEN122_TOOLRL_DATA=/path/to/v3/toolrl_steps_ctx10240.jsonl
QWEN122_GAD_DATA=/path/to/v3/gad_steps_ctx10240.jsonl
```

在这两个 v3 compact 文件尚未生成并验证前，122B 串行 RL launcher 应视为 data preflight 未通过。

## 7. 可复现重建

在 Slime 环境中执行：

```bash
ROOT=/root/slime_sxy/group-space/sunxiangyu/drug-pipe
SLIME="$ROOT/slime-wd/slime"
V3="$ROOT/slime-wd/outputs/slime_drug_agent_data/live_tool_catalog_v3"
export PYTHONPATH="$SLIME${PYTHONPATH:+:$PYTHONPATH}"
export DRUG_AGENT_TOOL_CATALOG="$V3/tool_catalog.json"

python -m drug_agent.toolrl.convert_react_to_toolrl_steps \
  --input "$V3/react_trajectories.jsonl" \
  --output "$V3/toolrl/toolrl_steps.jsonl" \
  --skipped-report "$V3/toolrl/toolrl_steps.skipped.jsonl" \
  --report "$V3/toolrl/toolrl_steps.report.json"

python -m drug_agent.toolrl.validate_toolrl_offline_data \
  --input "$V3/toolrl/toolrl_steps.jsonl" \
  --report "$V3/toolrl/validation_report.json" \
  --errors "$V3/toolrl/validation_errors.jsonl"

python -m drug_agent.gad.data \
  --input "$V3/react_trajectories.jsonl" \
  --output "$V3/gad/gad_steps.jsonl" \
  --skipped-report "$V3/gad/gad_steps.skipped.jsonl" \
  --report "$V3/gad/gad_steps.report.json"
```

重建前应写到临时目录并验证，再原子替换正式文件；不要让正在运行的训练读取半写入 JSONL。
