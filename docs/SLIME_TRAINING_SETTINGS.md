# Slime 训练参数决策

本文记录 Drug-Pipe 当前 canonical 数据在 Slime 上进行 SFT、ToolRL 和 GAD 训练时的参数依据。
它保留已经在 4-GPU worker 上验证的 Qwen3.5-4B 经验、早期 Qwen3.5-27B 规划，以及最终采用的
Qwen3.5-9B 训练配置。4B/27B 段落是历史决策记录，不表示对应模型资产仍需驻留工作目录；当前
主目录只保留 9B 基座、转换权重和最终训练 checkpoint。参数不是脱离数据的固定模板；数据或模型改变后必须重新统计长度并做 smoke。

## 1. 已验证的 4B SFT 基线

验证运行使用 373 条 canonical ReAct，完整跑完 1 epoch、373 个 optimizer steps，并成功保存
checkpoint：

```text
model                         Qwen3.5-4B
GPU                           4（该 run 日志未固化 GPU 型号）
TP / PP / CP / DP             4 / 1 / 1 / 1
rollout batch size            373
global batch size             1
epochs                        1
max tokens per GPU            8192
learning rate                 1e-5
minimum learning rate         1e-6
warmup fraction               0.1
schedule                      cosine
recompute                     full, uniform, 1 layer
train/rollout offload          disabled
optimizer CPU offload         disabled
```

结果目录：

```text
slime-wd/outputs/slime_drug_agent_runs/
  Qwen3.5-4B_sft_current373_optimized_20260727_051224
```

这次运行形成了几条可复用经验：

- Offline SFT 不实例化 SGLang；`--debug-train-only` 时应保持 actor 常驻 GPU，不能为了不存在的
  rollout model 做 sleep/wake 或 CPU/GPU offload。
- `RBS=373` 一次装入完整 epoch，把 rollout/train 边界开销压到一次；`GBS=1` 仍然产生每条记录
  一次 optimizer update。
- 当前记录很长，`GBS` 表示记录数而不是 token 数；单条记录本身已经提供大量 supervised tokens。
- `max-tokens-per-gpu` 是 dynamic microbatch packing 目标，不是截断长度。单个超限样本会独占
  microbatch，因此必须单独验证最长样本的显存。
- 完整轨迹不能为了提高表面 GPU 利用率随意截断。应先使用 full recompute 和合理并行，再考虑
  optimizer CPU offload。
- checkpoint 在 epoch/rollout 边界保存；不能只根据 `save-interval` 推断实际产生多少 checkpoint。

## 2. 当前数据长度

使用本地 Qwen3.5 tokenizer 对最新数据统计：

| 数据 | 数量 | P50 | P95 | 最大 |
|---|---:|---:|---:|---:|
| SFT 完整轨迹 | 373 | 14,130 | 62,238 | 94,059 |
| ToolRL prompt | 3,028 | 9,266 | 32,778 | 89,599 |
| ToolRL target | 3,028 | 520 | 2,363 | 34,948 |
| GAD prompt | 3,234 | 9,460 | 33,293 | 89,599 |
| GAD target | 3,234 | 458 | 2,363 | 34,948 |

旧 RL 默认 `prompt=6144, response=512, context=6656` 不适合这些数据。ToolRL 有 2,188 条、
GAD 有 2,364 条 prompt 超过 6,144 tokens。正式 RL 使用：

```text
ROLLOUT_MAX_PROMPT_LEN=98304
ROLLOUT_MAX_RESPONSE_LEN=4096
ROLLOUT_MAX_CONTEXT_LEN=102400
```

4096-token response 覆盖超过 99% 的 teacher target；个别约 35K 的长尾不应迫使所有 rollout
预留同样大的生成空间。

## 3. 4-GPU Qwen3.5-4B 串行正式配置

`run_qwen3_5_4b_sft_toolrl_gad_serial.sh` 面向当前 373 条 canonical 数据。运行顺序虽然是
SFT、ToolRL、GAD，但算法分支保持为：

```text
4B base -> SFT -> ToolRL
                -> GAD
```

ToolRL 不作为 GAD 的初始化，以便比较两种算法。

| 阶段 | GPU/并行 | RBS / N / GBS | rollout 数 | LR | 其他关键设置 |
|---|---|---|---:|---:|---|
| SFT | 4 GPU, TP4 | 373 / - / 1 | 1 epoch | 1e-5 | max tokens/GPU 8192, full recompute |
| ToolRL | 4 GPU, TP4 | 4 / 2 / 8 | 757 | 5e-7 | official reward, temp 1.0 |
| GAD negative | 4 GPU, TP4 | 2 / 1 / 2 | 1617 | 0 | temp 0.8 |
| discriminator warmup | GPU 3 | batch 2 | 1 epoch | 1e-6 | max length 4096, grad clip 0.2 |
| GAD generator | GPU 0-2, TP1/DP3 | 2 / 3 / 6 | 1617 | 2e-7 | pure reward, KL 0.001 |
| online discriminator | GPU 3 | GRPO group request | continuous | 1e-6 | grad clip 0.2 |

ToolRL/GAD 使用 `prompt/response/context = 98304/4096/102400`，SGLang static memory fraction
为 0.75。GAD 必须采用 3+1 布局，因为 pure reward 的 4B discriminator 需要在 generator
训练期间常驻一张 GPU。所有 formal stage 都不执行 MCP。

## 4. 历史方案：8 x H200 的 27B 选择与准备

本节仅保留当时的容量规划依据。27B 模型资产已经归档，不是当前可直接启动的主线配置。

当前本地可用、明显大于 4B 且已有 Slime 模型定义的是 Qwen3.5-27B：

```text
HF checkpoint:  slime-wd/data/Qwen3.5-27B
model args:     slime-wd/slime/scripts/models/qwen3.5-27B.sh
```

训练前还必须生成：

```text
slime-wd/data/Qwen3.5-27B_torch_dist
```

Qwen3.5-27B 有 4 个 query groups，优先使用已经由 Slime 27B 配置验证的 TP4，而不是未经验证的
TP8。单机 8 卡 SFT 使用 `TP4 x PP2`；RL 可使用 `TP4 x DP2` 提高吞吐。

worker Ready 后先确认：

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
nvidia-smi topo -m
df -h /root/slime_sxy/group-space/sunxiangyu
```

单节点设置 `NCCL_IB_DISABLE=1` 不会关闭 NVLink/NVSwitch，但必须用拓扑输出确认卡间连接。

## 5. Qwen3.5-27B SFT

正式基线：

```text
GPU                           8
TP / PP / CP / DP             4 / 2 / 1 / 1
decoder last pipeline layers  30
epochs                        1
RBS / GBS                     373 / 1
max tokens per GPU            32768
LR / min LR                   2e-6 / 2e-7
warmup fraction               0.05
schedule                      cosine
recompute                     full
train/rollout offload          disabled
optimizer CPU offload         disabled initially
```

先用 2--4 条覆盖短、中、最长长度的样本 smoke。若最长样本 OOM，处理顺序是：

1. 确认 TP4/PP2、full recompute 和 pipeline layer balance 生效；
2. 启用 optimizer CPU offload；
3. 最后才考虑经过审查的长尾处理，不能静默截断。

## 6. ToolRL

ToolRL 从 SFT checkpoint 分支，不从 base model 重新开始：

```text
training TP / PP / DP         4 / 1 / 2
rollout TP / engines          4 / 2
RBS / samples per prompt      4 / 4
GBS                           16 completions
NUM_ROLLOUT                   757
prompt / response / context   98304 / 4096 / 102400
max tokens per GPU            32768
learning rate                 2e-7
reward mode                   official
temperature                   1.0
SGLang static memory fraction 0.75
recompute                     full
save interval                 250
```

`3028 x 4 = 12112` 次 completion，属于正式 RL 运行。`N=4` 保留有意义的 group-relative
variance；不能降到 `N=1`。Formal ToolRL 继续完全 offline，不执行 MCP。

## 7. GAD

实验分支应为：

```text
Qwen3.5-27B SFT checkpoint
├── ToolRL
└── GAD
```

不要默认把 ToolRL checkpoint 再交给 GAD，否则无法区分两种算法的贡献。

### 7.1 Stage 2 negative generation

```text
GPU / TP / DP                 8 / 4 / 2
rollout TP / engines          4 / 2
RBS / N / GBS                 6 / 1 / 6
NUM_ROLLOUT                   539
prompt / response / context   98304 / 4096 / 102400
temperature                   0.8
student                       27B SFT checkpoint
learning rate                 0
```

`3234 / 6 = 539`，可以完整覆盖而不复制尾部。

### 7.2 Discriminator warmup

当前 discriminator 是单卡 Hugging Face 全参数 Adam 实现。27B 的参数、梯度、master weights 和
optimizer states 不能放进单张 H200。因此当前可执行的正式 variant 是：

```text
model                         Qwen3.5-4B
GPU                           独立 1 x H200
epochs                        1
batch size                    2，smoke 后可升到 4
max length                    4096
learning rate                 1e-6
gradient clip                 0.2
save interval                 200
```

这必须在 manifest 中标记为 `27B generator + 4B discriminator efficiency variant`，不能声称是
同规模 discriminator。要训练 27B discriminator，需要把当前单卡实现改造成 FSDP/Megatron 并提供
额外 GPU，不是调整一个 batch 参数即可解决。

### 7.3 Stage 3 GAD

推荐给 discriminator 单独申请 1 张 H200，让当前 8 张卡全部服务 generator：

```text
generator TP / DP             4 / 2
rollout TP                    4
RBS / N / GBS                 2 / 4 / 8
NUM_ROLLOUT                   1617
prompt / response / context   98304 / 4096 / 102400
student learning rate         2e-7
KL coefficient                0.001
temperature                   0.8
reward mode                   pure
```

若只能使用同一个 8 卡 worker，保守布局是 GPU 0--3 运行 TP4 generator、GPU 7 运行 4B
discriminator，剩余卡空闲；此时 generator 需要 optimizer CPU offload，吞吐明显更低。

## 8. 启动顺序

```text
27B HF -> torch_dist conversion
-> 27B SFT length-stratified smoke
-> 27B full SFT
-> ToolRL smoke
-> ToolRL full
-> GAD negative generation
-> 4B discriminator warmup
-> GAD smoke
-> GAD full
```

每个 full run 前记录代码 commit、数据 SHA256、模型来源、并行拓扑、长度上限、实际 GPU 拓扑和
上游 checkpoint。SFT、ToolRL、GAD formal training 均保持 offline；只有在线 debug/evaluation 执行工具。

## 9. 8 x H200 的 Qwen3.5-9B 串行配置

当前 8 x H200、508 GiB host RAM worker 使用：

```text
drug_agent/scripts/run_qwen3_5_9b_sft_toolrl_gad_serial.sh
```

9B 有 4 个 query groups，因此不使用不整除 query groups 的 TP8。SFT 和 ToolRL 使用
`TP4/PP2/DP1`；这既使用全部 8 卡，也让 373 条奇数数据在 `GBS=1` 下不需要复制或丢弃。
GAD pure reward 需要一张卡常驻同源 9B discriminator，因此 Stage 3 使用 6 卡
`TP2/DP3` generator、GPU 7 discriminator，并故意留下 GPU 6 作为服务隔离余量。

| 阶段 | GPU/并行 | RBS / N / GBS | rollout 数 | LR | 关键设置 |
|---|---|---|---:|---:|---|
| SFT | 8, TP4/PP2 | 373 / - / 1 | 1 epoch | 5e-6 | max tokens 16384, cosine, full recompute |
| ToolRL | 8, TP4/PP2 | 4 / 4 / 16 | 757 | 2e-7 | official reward, temperature 1.0, max tokens 8192 |
| GAD negative | 8, TP4/DP2 | 2 / 1 / 2 | 1617 | 0 | SFT branch, temperature 0.8, max tokens 8192 |
| discriminator warmup | GPU 7 | batch 1 | 1 epoch | 5e-7 | same-source 9B, max length 2048 |
| GAD generator | GPU 0--5, TP2/DP3 | 2 / 3 / 6 | 1617 | 1e-7 | pure reward, KL 0.001, max tokens 8192 |
| online discriminator | GPU 7 | one GRPO group/request | continuous | 5e-7 | max length 2048, clip 0.2 |

所有 RL 阶段沿用 `prompt/response/context=98304/4096/102400`。9B HF checkpoint 首次运行时
会在 GPU 0 上生成单-rank `Qwen3.5-9B_torch_dist` release checkpoint，随后由 MCore 在正式
拓扑中确定性 reshard。

该 profile **完整保存 optimizer 和 RNG state**，不使用 `--no-save-optim`。GPU-resident Adam
采用当前 MCore distributed checkpoint 表示；只有显式启用 optimizer CPU offload 时才回到
旧的 pre-MCore-0.14 表示。这样保留精确 resume 能力，同时避免 4B 旧格式保存时观察到的
host-memory 峰值。

长 decision state 的 policy loss 使用 `log-probs-chunk-size=2048` 和 loss recompute。若
`entropy-coef=0`，不会再构建无效的全词表 entropy 反向图。SGLang static memory fraction 为
0.70。这些设置不截断输入，也不改变 log-prob/reward 定义，只降低训练阶段临时显存峰值。
ToolRL/GAD 显式向 Qwen3.5 chat template 传入 `enable_thinking=false`，避免模型原生 `<think>`
envelope 包住项目选定的 canonical `<thought>` 协议并导致所有 completion 获得相同格式罚分。

串行任务因后续阶段失败时，可对同一个 `RUN_ROOT` 设置 `RESUME_SERIAL_RUN=1`。脚本会校验
canonical 数据 SHA256，并只跳过带有 `.complete` marker 且 checkpoint 仍存在的阶段；不会把
仅生成 rollout、但没有 checkpoint 的失败阶段当作已完成。

## 10. Checkpoint retention

正式 SFT、ToolRL 和 GAD generator 默认使用：

```bash
CHECKPOINT_KEEP_LAST=2       # 训练中保留 latest + previous
CHECKPOINT_FINAL_KEEP=1      # 阶段成功后只保留 final
DISCRIMINATOR_KEEP_LAST=2   # GAD discriminator 周期权重
```

Slime 的公开参数为 `--save-retain-last`。启用后，每次 checkpoint 会等待完整落盘，并校验
`latest_checkpointed_iteration.txt`、metadata 和分布式 shard，然后才删除更旧的
`iter_NNNNNNN`。阶段失败时保留最新两份；串行 launcher 只有在阶段成功后才收敛为最终一份。
最终 checkpoint 继续包含 optimizer、scheduler 和 RNG state，可以精确恢复训练。
