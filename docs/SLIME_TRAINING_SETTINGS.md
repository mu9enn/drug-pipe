# Drug-Pipe 大模型训练实战指南：27B、35B 与 122B-FP8 在 4/8×H200 上的 SFT、ToolRL 和 GAD

> 更新时间：2026-08-04
>
> 项目：`drug-pipe/slime-wd/slime`
>
> 目标读者：第一次系统接触大语言模型训练、希望理解“为什么这样配参数”的项目成员

本文不是一份“抄过去就一定能跑”的静态命令表，而是本项目长时间实机实验后的决策记录和学习教程。它回答四类问题：

1. 每个模型在 4 卡或 8 卡 H200 上到底能做什么；
2. SFT、ToolRL、GAD 分别使用什么参数，以及每个参数是什么意思；
3. 我们遇到过哪些显存、内存、并行、数值、奖励和工程问题；
4. 下一次换 worker、换模型或换数据时，应该怎样重新验证，而不是盲目照搬。

本文以当前 launcher、profile、测试、真实日志和 `resolved_config.env` 为事实来源。历史报告中一些较早结论已经被后续实验修正。例如：

- “122B 全参数 colocated ToolRL 跑不通”仍然成立；
- “122B 在单机 8 卡上完全不能做 ToolRL/GAD”已经被后来的 **官方 FP8 基座 + LoRA** 方案推翻；
- 两句话描述的是不同训练制度，并不矛盾。

> 当前 510 条 canonical v3 及其新 ToolRL/GAD decision 数据的训练交接，以
> [CANONICAL_V3_510_TRAINING_HANDOFF.md](CANONICAL_V3_510_TRAINING_HANDOFF.md) 为准。
> 本文后续出现的 v1/v2 条数和路径只代表对应历史实验，不能覆盖该数据合同。

---

## 1. 先看结论：哪些组合已经验证到什么程度

### 1.1 状态定义

本文用四个状态避免把建议值误写成既成事实：

| 标记 | 含义 |
|---|---|
| **A：正式/多步实机通过** | 已完成正式阶段或连续多个真实 optimizer update，关键生命周期已观察 |
| **B：短 gate 通过** | 已完成一个或少量真实闭环，可证明接口和容量，但不能自动外推到完整 epoch |
| **C：推荐起始配置** | 由理论、相邻模型和实测容量推导，尚未完成该方法的实机闭环 |
| **D：当前资源不可行** | 已实测失败或理论预算明确超出；必须换训练制度或资源 |

### 1.2 模型 × 卡数 × 方法总表

| 模型 | 资源 | SFT | ToolRL | GAD | 结论摘要 |
|---|---|---|---|---|---|
| Qwen3.5-27B | 4×H200，约 530 GB host | **B** | **B** | **D/B** | SFT 和 ToolRL 单组闭环通过；在线 GAD generator 在约 507/518 GiB 被杀，negative/discriminator 子阶段可单独做 |
| Qwen3.5-27B | 8×H200，约 1 TiB host | **A** | **A（多步，完整 pass 当时仍在推进）** | **C** | SFT 已完成；ToolRL 最终稳定到大量更新；GAD 参数已确定但必须等 ToolRL 后串行启动验证 |
| Qwen3.6-35B-A3B | 4×H200，约 530 GB host | **B** | **C** | **C/D** | 47.5K-token SFT step 通过；RL 不能从 SFT 容量直接外推，GAD 需要更大 host |
| Qwen3.6-35B-A3B | 8×H200，约 1 TiB host | **B（多更新吞吐 gate）** | **C** | **C** | SFT 多更新通过；ToolRL/GAD 尚未完整实机闭环 |
| Qwen3.5-122B-A10B-FP8 | 4×H200 | **D** | **D** | **D** | 不是合理目标；仅权重、训练状态和 rollout 生命周期就远超单机 4 卡容量 |
| Qwen3.5-122B-A10B-FP8 | 8×H200，约 1 TiB host | **A：全参数 SFT** | **A：LoRA 多步运行** | **B/C：关键 gate 通过、正式阶段串行排队** | 全参数 SFT 完成 91 steps；全参数 RL 不可行；LoRA ToolRL 正常多步运行，LoRA GAD 的 ref/discriminator/adapter 关键 gate 已通过 |

最重要的阅读原则是：**SFT 能跑不代表 ToolRL/GAD 能跑；一个短 batch 能跑也不代表最大长度、保存 checkpoint 和 train↔rollout 切换都能跑。**

---

## 2. 初学者需要先建立的训练心智模型

### 2.1 SFT、ToolRL、GAD 分别在学什么

#### SFT：监督微调

SFT（Supervised Fine-Tuning）把一条正确轨迹当成标准答案。模型看到 prompt，逐 token 学习 teacher 的正确回答、思考格式和工具调用。

本项目的 SFT 主要学习：

- canonical ReAct 标签，例如 `<thought>`、`<tool_call>`、`<observation>`、`<final_answer>`；
- MolClaw 工具名；
- 参数名、参数值和调用顺序；
- 最终回答必须由已记录 observation 支撑。

SFT 的优点是稳定、容易检查；缺点是它只模仿已有答案，不会直接探索“不同回答哪个更好”。

#### ToolRL：用工具调用奖励做强化学习

ToolRL 让模型对同一个或一批 prompt 生成回答，然后根据格式、工具名、参数名、参数值和匹配程度计算 reward。训练目标不是逐 token 模仿 teacher，而是提高高 reward 行为的概率。

本项目的 dense MolClaw reward 不是只有“对/错”两个值，而是尽量把奖励拆细，使部分正确的调用也有可学习信号。

#### GAD：用在线判别器帮助 generator

GAD 可以理解成“两名学生互相促进”：

- generator 负责生成轨迹；
- discriminator 学习区分更好的真实/正例轨迹和 generator 的较差负例；
- discriminator 的分数再作为 generator 的一部分 reward。

本项目的 GAD 有三个阶段：

```text
SFT 权重
  -> 生成 aligned negatives
  -> discriminator warmup
  -> online GAD：generator 与 discriminator 交替更新
```

ToolRL 和 GAD 都从 SFT 分叉。默认实验设计不是 `SFT -> ToolRL -> GAD`，而是：

```text
base -> SFT -> ToolRL
             -> GAD
```

这样才能判断 ToolRL 和 GAD 各自带来什么变化。

### 2.2 一次训练 step 里发生什么

以在线 RL 为例：

1. rollout engine（SGLang）加载当前 policy；
2. 对 prompt 生成 response；
3. reward function 或 discriminator 评分；
4. 计算 advantage：哪些样本比基线好、哪些差；
5. Megatron actor 做 forward/backward；
6. optimizer 更新参数或 LoRA adapter；
7. 把新权重/adapter 同步回 SGLang；
8. 下一轮重复。

因此在线 RL 同时需要：训练权重、训练激活、梯度、optimizer state、rollout 权重、KV cache、reward/discriminator 和权重同步缓冲。它通常比 SFT 更难塞进同一台机器。

### 2.3 GPU 显存和 host 内存不是一回事

- **HBM/GPU 显存**：H200 每卡约 140 GiB 可用，保存当前 rank 的权重、激活、梯度、optimizer shard、KV cache 和临时 workspace。
- **host memory**：worker 的 CPU 内存/cgroup 限额，4 卡 worker 约 518 GiB，8 卡 worker约 1 TiB。CPU optimizer offload、actor sleep backup、checkpoint 序列化都会使用它。
- **磁盘/GPFS**：保存 HF checkpoint、torch_dist、训练 checkpoint 和日志。

常见误区是：看到 GPU 还有空间，就以为训练能继续。实际上本项目最难的失败之一是 **8 张 H200 的计算已完成，但 actor pause 时把 CUDA allocation 备份到 host，1 TiB host 被打爆**。

### 2.4 Dense 与 MoE：active parameters 不等于要保存的参数

- 27B 是 dense 模型；每个 token 基本经过同一套参数。
- 35B-A3B 和 122B-A10B 是 MoE；每个 token 只激活一部分 experts，所以计算量较低。

但训练时所有 experts 的参数仍需存放、更新和保存。因此：

- “122B-A10B 每 token 只激活约 10B”不代表它按 10B 模型占内存；
- 内存预算必须按总训练参数量，而不是 active parameters。

本项目读取 safetensors header 得到的训练语言塔规模约为：

| 模型 | 当前训练语言塔 + LM head BF16 权重规模 |
|---|---:|
| Qwen3.5-27B | 约 53.8 GB，约 26.9B 参数 |
| Qwen3.6-35B-A3B | 约 69.3 GB，约 34.7B 参数 |
| Qwen3.5-122B-A10B | 约 244.2 GB，约 122.1B 参数 |

视觉塔和 MTP 层没有在当前 model args 中启用，因此不属于这轮训练对象。不能拿完整 HF 目录大小直接当作训练参数量。

---

## 3. 并行参数：TP、PP、DP、CP、EP、ETP 到底是什么

### 3.1 TP：Tensor Parallel

TP 把一个大矩阵运算拆到多张卡。例如 TP2 表示一层内部由两张卡共同计算。

TP 受模型结构约束。本项目这些 Qwen3.5/3.6 大模型只有 2 个 KV query groups，Megatron 要求：

```text
num_query_groups % TP == 0
```

因此 actor 使用 TP2，不能随意设成 TP4/TP8。多出来的 GPU 要用于 PP、EP 或 DP。

### 3.2 PP：Pipeline Parallel

PP 把模型的不同层放在不同 stage。例如 48 层模型 PP4，可以每 stage 12 层。

PP 的主要作用：

- 切分权重；
- 切分 optimizer state；
- 降低单卡激活压力。

但不同 stage 不一定一样重：

- 第一个 stage 可能有 embedding；
- 最后一个 stage 有 LM head、全词表 logits 和 loss；
- 所以最后 stage 往往需要少放几层。

27B 的 8 卡长序列最终采用 20/16/16/12 层，而不是平均 16/16/16/16，就是为了给最后的 LM-head/loss 留空间。

### 3.3 DP：Data Parallel

DP 让多个副本处理不同样本，再同步梯度。DP 能提高吞吐，也能通过 distributed optimizer 分摊某些训练状态。

长短样本差异大时，DP 可能出现负载失衡：一个 rank 处理 70K tokens，另一个只处理 5K，短的 rank 会等待长的 rank，看起来像 NCCL 卡死。解决方法是：

- `GBS` 足够大；
- `--balance-data` 按 token 长度分配；
- 或建立长度 bucket。

### 3.4 CP：Context Parallel

CP 沿序列维度切长上下文，看起来很适合 94K 长轨迹。但在当前 Megatron optimizer 实现中，CP rank 会复制一部分 optimizer state。

27B/4 卡的 CP2 在 70% 和 80% optimizer offload 下都把约 518-GiB host 推到约 492 GiB并被 Ray 杀死。最终我们优先使用 PP，而不是 CP。

### 3.5 EP 与 ETP：Expert Parallel

- EP 把不同 MoE experts 分到不同 rank；
- ETP 再切 expert 内部张量。

35B/122B 使用 EP2、ETP1。必须同时满足 dense grid 与 expert grid 的整除关系：

```text
world_size % (TP × PP × CP) == 0
world_size % (ETP × EP × PP) == 0
```

### 3.6 本项目的并行总览

| 模型/资源 | 主要训练拓扑 | 解释 |
|---|---|---|
| 27B/4卡 | TP2/PP2/CP1 | dense，无 EP；PP 切权重和 optimizer |
| 27B/8卡吞吐 gate | TP2/PP2/DP2 | 适合中短样本，高吞吐 |
| 27B/8卡长序列正式 | TP2/PP4/DP1 | 用更多 PP 换极长序列和 loss stage 容量 |
| 35B/4卡 | TP2/PP2/EP2 | 4 rank 同时满足 dense/expert grid |
| 35B/8卡吞吐 gate | TP2/PP2/EP2/DP2 | DP2 提吞吐，需 balance-data |
| 122B/8卡 | TP2/PP4/EP2 | 8 rank 全部用于模型/experts，无 DP |

---

## 4. Batch、长度和 rollout 参数怎么理解

### 4.1 RBS、N、GBS

- `RBS` / `rollout_batch_size`：每轮取多少个 prompt。
- `N` / `n_samples_per_prompt`：每个 prompt 生成多少个 response。
- `GBS` / `global_batch_size`：一次 optimizer update 使用多少个生成样本。

应满足基本关系：

```text
每轮生成样本数 = RBS × N
```

生成样本数要能被 GBS 合理消费。比如 122B ToolRL 的 RBS8/N1/GBS8，每轮正好产生一次 update 所需的 8 个样本。

### 4.2 为什么 N=1 也能训练

传统 GRPO 常在同一个 prompt 的 N 个回答内部做中心化。如果 N=1，就没有组内相对差异。

122B/27B 最终 ToolRL 使用 REINFORCE++ normalization 跨一批不同 prompt 归一化 reward，因此可以使用 RBS8/N1/GBS8。这样解决了“同一个 prompt 的 4 个回答 reward 完全相同，导致 advantage=0”的问题。

GAD 仍保留 grouped 结构，例如 122B 使用 RBS4/N2/GBS8，并用动态 filter 丢弃 reward 没有方差的组。

### 4.3 `max_tokens_per_gpu` 不是截断长度

它是动态 microbatch 的 token packing 目标：尽量把若干短样本放在同一个 microbatch 中。

如果单条样本长于这个值，它通常会单独占一个 microbatch，而不是自动被切成几段。因此必须另设：

- prompt 最大长度；
- response 最大长度；
- context 最大长度；
- 或明确的 head/tail truncation 策略。

### 4.4 本项目的数据长度

canonical v1、Qwen3.5 tokenizer、`enable_thinking=false` 的统计为：

| 数据 | 条数 | P50 | P90 | P95 | P99 | 最大 | 总 tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT 完整 chat | 364 | 14,209 | 54,732 | 65,634 | 83,954 | 94,063 | 8,708,386 |
| ToolRL prompt+target | 3,182 | 10,078 | 27,795 | 33,958 | 55,690 | 94,061 | 44,368,735 |
| GAD prompt+target | 3,147 | 10,194 | 27,864 | 34,123 | 55,917 | 94,061 | 44,180,700 |

所以“只有 364 条 SFT”不等于训练很小；它有约 870 万 tokens。SFT 首轮只做 1 epoch，避免对少量轨迹过拟合。

122B LoRA 为单机容量和吞吐使用了 compacted 10,240-token context 数据；这是一项明确的数据变换，必须保存来源和审计，不能和原始 94K 数据混为一谈。

---

## 5. 优化参数是什么意思

### 5.1 Learning rate（LR）

LR 控制每次更新走多大一步：

- 太大：loss、KL、输出格式可能迅速崩坏，甚至 NaN；
- 太小：训练非常慢，reward/held-out 指标不动。

大模型、RL 和全参数训练一般使用更小 LR。LoRA 只更新少量 adapter，通常能承受比同尺寸全参数训练更高的 LR。

### 5.2 Warmup

训练开始时 optimizer state、reward 分布和模型输出都还没稳定，直接使用峰值 LR 容易冲坏模型。warmup 让 LR 从较小值逐步升到峰值。

例如 398 个 ToolRL rollout、warmup fraction 0.05，大约前 20 步处于 warmup。

### 5.3 Cosine decay 与 min LR

cosine schedule 让 LR 从峰值平滑下降到 `min_lr`。它适合已知训练步数的 SFT 或 LoRA RL。constant LR 保持不变，调试更直观，但更依赖选对峰值。

### 5.4 Adam β、weight decay 和 grad clip

- `beta1=0.9`：一阶动量，平滑梯度方向；
- `beta2=0.95`：二阶动量，估计梯度尺度；
- `weight_decay`：抑制权重无限增大；
- `grad_clip=1`：当梯度过大时按范数裁剪，防止单个异常 batch 破坏训练。

日志里的 raw `grad_norm` 大于 1 不代表裁剪没有生效。通常先记录原始 norm，再执行 clipping。

### 5.5 Recompute

activation recompute 不保存所有中间激活，backward 时重新计算。代价是更多计算，收益是显著降低 HBM。

本项目大模型默认 full recompute；这是“用 FLOPs 换显存”的核心手段。

### 5.6 Optimizer offload

把 optimizer state 从 GPU 移到 CPU 可节省 HBM，但会：

- 占用 host memory；
- 引入 PCIe/CPUAdam 开销；
- 在异步 D2H/H2D 下产生额外 staging buffer 和同步风险。

因此 offload 不是越高越好。27B/4 卡 ToolRL 的甜点是 70%，而 60% 会 HBM 不足、80% 会 host OOM。

### 5.7 SGLang static memory fraction

它控制 rollout engine 为模型、KV cache 和运行时预留多少 GPU 内存。太高会挤压 Megatron actor，太低会没有 KV pool。

这必须和 actor 是否常驻、rollout 是否 offload、KV dtype、context 长度一起调。

### 5.8 BF16、FP8 和 LoRA

- BF16：16-bit 浮点，训练稳定性较好；
- FP8：8-bit 浮点，可显著降低权重/通信/部分 state 内存，但不同 tensor 的量化方式与兼容性不同；
- LoRA：冻结大部分 base weights，只训练小型低秩 adapter。

必须分别说清：checkpoint 权重、forward recipe、param gather、master params、grad、Adam moments、rollout 权重、KV cache各是什么 dtype。

“使用官方 FP8 模型”不意味着 KV cache 也必须 FP8。122B 最终 LoRA 使用官方 FP8 权重，但保留 BF16/default KV cache，以减少未校准 FP8 KV 对 reward fidelity 的影响。

---

## 6. Qwen3.5-27B：4 卡配置

### 6.1 共同设置

| 参数 | 值 | 解释 |
|---|---|---|
| actor topology | TP2/PP2/CP1/EP1 | dense 模型；PP2 比 CP2 更节省 host optimizer 内存 |
| full recompute | 开启 | 控制长序列激活显存 |
| FP32 grad accumulation | 开启 | 优先保证数值稳定 |
| host 要求 | SFT/ToolRL 建议 ≥480 GiB；在线 GAD ≥600 GiB | 530-GB worker 已证明不够在线 GAD |

### 6.2 SFT：已通过真实长 step

| 参数 | 最终值 |
|---|---:|
| 数据 | 364 条 canonical，首轮 1 epoch |
| RBS / GBS | 364 / 4 |
| updates | 91 |
| LR / min LR | `1e-6 / 1e-7` |
| warmup / schedule | 3% / cosine |
| max tokens/GPU | 8192（packing target） |
| log-prob chunk | 1024 |
| optimizer CPU offload | 40% |
| recompute | full, uniform, 1 layer |

实测 47,499-token step：

- loss `0.14236`；
- grad norm `3.1299`；
- actor train 约 48.1 秒；
- 约 988 tokens/s；
- HBM 峰值约 108.7 GiB/GPU；
- host used 约 169.8 GiB。

### 6.3 ToolRL：单组闭环已通过，完整训练仍需正式 gate

容量 gate 的关键设置：

| 参数 | 实测甜点 |
|---|---:|
| actor | `--no-offload-train`，保持常驻 |
| rollout | `--offload-rollout` |
| optimizer offload | 70% |
| SGLang static fraction | 0.18 |
| capacity gate batch | RBS1/N4/GBS4 |
| 完整训练起始 batch | RBS2/N4/GBS8 |
| LR | `1e-7` |
| temperature | 1.0 |
| clip | 0.20 / 0.28 |
| KL | ToolRL 基线不加 reference KL |

单组 gate 完成 rollout、reward、actor update、权重同步和 rollout restore。它证明方法闭环可用，但完整 3,182 条数据仍需多步与 checkpoint gate。

### 6.4 GAD：530-GB host 上不可做在线 generator

推荐起点仅在换到更大 host 后使用：

| 参数 | 建议值 |
|---|---:|
| generator RBS/N/GBS | 2/4/8 |
| generator LR | `5e-8` |
| temperature | 0.8 |
| KL coefficient | 0.001 |
| optimizer offload | 约 70%，重新 gate |
| SGLang fraction | 0.18 |
| discriminator | 小模型独立 warmup，context 先 8192 |
| host | 至少 600 GiB，最好留 96–128 GiB 余量 |

已通过的子阶段：

- 1 条 negative cache 生成；
- 0.8B discriminator 单 pair warmup/manifest；
- online service、trajectory JSONL、reference log-prob、discriminator 多次更新和 generator backward 的接口。

未通过的是资源生命周期：65%/70% offload 下 host 达到约 502–507/518 GiB 后被杀。降低 offload 没有线性降低 CPUAdam 瞬时 RSS，不能靠关闭 Ray 保护硬挤。

---

## 7. Qwen3.5-27B：8 卡配置

### 7.1 SFT：最终长序列正式配置

早期高吞吐 gate 使用 TP2/PP2/DP2、34/30 层，约 177K tokens 达到 2,369 tokens/s。但极长样本使最后 loss rank 接近 140 GiB。正式完整数据最终改为：

| 参数 | 最终值 | 原因 |
|---|---:|---|
| topology | TP2/PP4/DP1 | PP4 为 94K 长尾和 loss stage 留空间 |
| PP layers | 20/16/16/12 | 最后 stage 含 LM head/loss，少放层 |
| RBS / GBS / epoch | 364 / 4 / 1 | 91 updates |
| LR / min LR | `1e-6 / 1e-7` | 27B SFT 实测尺度 |
| warmup / decay | 3% / cosine | 平稳进入训练 |
| max tokens/GPU | 6144 | 动态 packing，保留 GDN 激活余量 |
| vocab log-prob | recompute，chunk 64 | 限制大词表 FP32 workspace |
| optimizer offload | 关闭 | 8 卡有足够 HBM；避免 CPU 瓶颈/同步问题 |
| optimizer moments | BF16 | FP32 moments 在后续 update OOM |
| overlap | 关闭 | overlap buffer 在长桶触发 OOM |
| balance data | 开启 | 减少长短样本不均 |

正式 SFT 已完成并保存可供 RL 分叉的 checkpoint。

### 7.2 ToolRL：最终稳定生产参数

| 参数 | 最终值 |
|---|---:|
| 从哪里启动 | 27B SFT checkpoint，不加载 SFT optimizer/RNG |
| topology | TP2/PP4/DP1，20/16/16/12 |
| RBS / N / GBS | 8 / 1 / 8 |
| rollout 数 | `ceil(3182/8)=398` |
| advantage | REINFORCE++，跨 batch normalize |
| reward | dense MolClaw |
| LR | `1e-8` constant |
| warmup | 5%，约 20 steps，从 0 开始 |
| temperature | 0.7 |
| clip | 0.20 / 0.28 |
| reference KL | 默认关闭 |
| prompt/response/context | 65536 / 4096 / 69632 |
| SGLang fraction | 0.12 |
| CUDA graph | 关闭 |
| CPU optimizer offload | 关闭 |
| behavior log-prob | 使用 rollout log-probs |
| save interval | 200 |

为什么 LR 最终只有 `1e-8`：一次 `5e-8` 尝试在第一步后迅速破坏 ReAct validity，后续 batch 大量变成 `-0.3/-0.5` 无效格式。降低到 `1e-8`、加入约 20 步 warmup、temperature 降为 0.7 后，训练稳定推进到大量 step，loss/grad/KL/clip 均保持有限。

为什么 N=1：同 prompt 的严格四样本组多次得到完全相同 reward，GRPO 组内标准化产生零 advantage。改用八个不同 prompt 和 REINFORCE++ normalization 后恢复学习信号。

### 7.3 GAD：最终 launcher 参数，尚需正式串行验证

GAD 必须从同一 SFT checkpoint 分叉，不能读取 ToolRL policy。

#### Stage 2 negatives

| 参数 | 值 |
|---|---:|
| RBS | 2 |
| rollout 数 | `ceil(3147/2)=1574` |
| prompt/response/context | 65536 / 4096 / 69632 |
| generator | SFT checkpoint |

#### Discriminator warmup/service

| 参数 | 值 |
|---|---:|
| model | Qwen3.5-0.8B（27B 资源折中版） |
| warmup GPU | GPU 7，独立阶段 |
| epochs / batch | 1 / 1 |
| LR | `1e-6` |
| max length | 8192 |
| grad clip | 1.0 |
| online service | actor 使用 8 卡时转 CPU service |

#### Stage 3 generator

| 参数 | 值 |
|---|---:|
| RBS / N / GBS | 2 / 4 / 8 |
| rollout 数 | 1574 |
| advantage | GSPO/项目 GAD launcher |
| reward | pure discriminator reward |
| dynamic filter | 非零 reward std |
| LR | `5e-8` |
| temperature | 0.8 |
| KL | 0.001，low-var KL |
| SGLang fraction | 0.12 |
| CUDA graph / CPU optimizer offload | 关闭 |

这些参数是经过相邻 stage、内存和接口约束确定的生产起点，但在该窗口结束时 ToolRL 尚未完成，GAD 正式阶段还没有获得“A”级实证。因此不能写成“已经完整跑完”。

---

## 8. Qwen3.6-35B-A3B：4 卡配置

### 8.1 SFT：已通过真实长 step

| 参数 | 最终值 |
|---|---:|
| topology | TP2/PP2/CP1/EP2/ETP1 |
| dispatcher | DeepEP/flex（4卡 SFT gate） |
| RBS / GBS / epoch | 364 / 4 / 1 |
| LR / min LR | `2e-6 / 2e-7` |
| warmup / schedule | 3% / cosine |
| max tokens/GPU | 8192 |
| log-prob chunk | 1024 |
| optimizer offload | 40% |
| recompute | full |

47,499-token step 实测：

- loss `0.15024`；
- grad norm `2.8944`；
- actor train 约 212.6 秒；
- 约 223.5 tokens/s；
- HBM 峰值约 131.2 GiB/GPU，最紧卡只剩约 8.9 GiB。

### 8.2 ToolRL：推荐起点，未完成实机闭环

| 参数 | 起始值 |
|---|---:|
| RBS / N / GBS | 2 / 4 / 8 |
| LR | `2e-7` constant |
| temperature | 1.0 |
| KL | 0 |
| optimizer offload | 60% 起 |
| SGLang fraction | 0.35 起 |
| actor | 常驻，rollout offload |

这里不能直接照抄 27B 的 70%/0.18，因为 MoE dispatcher buffer、expert 权重和 SGLang 恢复占用不同。正式训练前要做完整单组 gate。

### 8.3 GAD：推荐起点，530-GB host 高风险

| 参数 | 起始值 |
|---|---:|
| generator RBS/N/GBS | 2/4/8 |
| LR | `1e-7` |
| temperature | 0.8 |
| KL | 0.001 |
| optimizer offload | 70% 起，必须重测 |
| SGLang fraction | 0.20 起 |
| host 建议 | ≥700 GiB |

27B 在线 GAD 已在 530-GB host 失败，35B 参数更多，因此 4 卡/530 GB 不应直接启动完整在线 GAD。

---

## 9. Qwen3.6-35B-A3B：8 卡配置

### 9.1 SFT：多更新吞吐 gate 已通过

| 参数 | 最终 gate 值 |
|---|---:|
| topology | TP2/PP2/EP2/DP2 |
| PP layers | 22/18 |
| dispatcher | ordinary all-to-all |
| GBS | 4，开启 balance-data |
| LR / min LR | `2e-6 / 2e-7` |
| max tokens/GPU | 8192 |
| log-prob chunk | 256 |
| optimizer offload | 关闭 |
| Adam moments | BF16 |
| overlap | 关闭 |

约 177.2K tokens 实测：

- actor train 约 124.0 秒；
- 约 1,429 tokens/s；
- loss `0.1154`；
- grad norm `1.0329`；
- 各卡峰值约 117–136 GiB。

GBS2 且未 balance-data 的对照发生 DP 长短样本 collective 失配：一张卡持续跑 NCCL，其余卡等待。最终使用 GBS4 + balance-data。

对于 P95–max 长桶，建议从 TP2/PP4/EP2/DP1、12/10/10/8 层开始 gate，而不是直接假设 PP2 能覆盖 94K。

### 9.2 ToolRL：推荐起点

| 参数 | 起始值 |
|---|---:|
| topology | TP2/PP2/EP2/DP2；长桶另用 PP4 |
| RBS / N / GBS | 2 / 4 / 8 |
| LR | `2e-7` |
| temperature | 1.0 |
| KL | 0 |
| CPU optimizer offload | 40% 起 |
| SGLang fraction | 0.22 起 |
| custom all-reduce | 建议关闭，使用 NCCL |

这是容量合理的起点，不是已完成的 ToolRL 结论。

### 9.3 GAD：推荐起点

| 参数 | 起始值 |
|---|---:|
| RBS / N / GBS | 2 / 4 / 8 |
| LR | `1e-7` |
| temperature | 0.8 |
| KL | 0.001 |
| optimizer offload | 50% 起 |
| SGLang fraction | 0.18 起 |
| host 建议 | ≥900 GiB |
| discriminator | 独立阶段 warmup；在线阶段需单独核算 GPU/CPU service |

同样必须完成 negatives、warmup、service、generator update、weight restore 和 checkpoint gate 后才能升为已验证配置。

---

## 10. Qwen3.5-122B-A10B-FP8：8 卡全参数 SFT

### 10.1 模型来源

最终使用官方 FP8 版本：

```text
Qwen/Qwen3.5-122B-A10B-FP8
```

actor 的 torch_dist 和 rollout HF view 必须来自同一官方 FP8 lineage，不能混用早期 BF16-derived actor checkpoint。

### 10.2 最终 SFT 配置

| 参数 | 最终值 | 解释 |
|---|---:|---|
| topology | TP2/PP4/EP2/ETP1 | 8 卡完整分片 |
| PP layout | 12/12/12/12 | 13-layer stage 在 FP8 Adam 临时展开时 OOM |
| RBS / GBS / epoch | 364 / 4 / 1 | 91 updates |
| LR / min LR | `1e-7 / 1e-8` | 早期更高 LR 在后续 backward 出 NaN |
| warmup / schedule | 3% / cosine | 保守全参数 FP8 schedule |
| max tokens/GPU | 6144 | 动态 packing |
| SFT max sequence | 12288 | 对长轨迹做明确 head/tail 保留 |
| head tokens | 4096 | 保留 system/task 前缀 |
| recompute | full + vocab log-prob recompute |
| log-prob chunk | 512 |
| FP8 recipe | delayed | pinned full-param optimizer 与 blockwise 不兼容 |
| main params / grads | FP16 / BF16 |
| Adam m1/m2 | FP8 / FP8 |
| optimizer groups | 最终 profile 为 32M elements；早期 train-only gate 曾用 64M |
| optimizer state offload | pageable moments-only |
| master-weight offload | 关闭 |
| checkpoint | 最终 weights-only；不保存 optimizer/RNG |

正式 SFT 最终完成 91 steps，step 90 的 loss `0.4495`、grad norm `1.7332`，随后生成 SFT-aligned FP8 HF rollout view。

### 10.3 为什么不是所有 tensor 都用 FP8

完整训练至少涉及：

- FP8 primary/model weights；
- FP16 main parameters；
- BF16 main gradients；
- FP8 Adam moments；
- 某些 FP32 临时展开和 normalization workspace。

这是混合精度系统，不是“按一下 FP8 开关”。任何一个临时 FP32 expansion 都可能在只剩几十 MiB 的 rank 上触发 OOM。

---

## 11. 122B 全参数 ToolRL/GAD 为什么失败

我们先解决了 BF16 rollout 的 GPU 容量：

- BF16 rollout 权重约 250 GB，TP8 后每卡裸权重约 31 GB，几乎没有 KV 空间；
- 官方/正确 block-FP8 rollout 约 126–127 GB，每卡约 17.1 GB；
- static fraction 0.25 后成功建立约 181 万 token 的 KV pool。

短 ToolRL gate 进一步完成了：

- 4 条 rollout；
- 4 次 actor forward/backward；
- TE FusedAdam optimizer step。

但 train→rollout transition 仍失败：actor pause 需要把约 120 GiB/rank 的 CUDA allocation 备份到 host，叠加 optimizer state 后 cgroup 超过 1 TiB；即使把 Ray threshold 提到 99% 也会真实越界。

所以失败根因不是“再少生成几十个 tokens 就能解决”的 KV 小问题，而是 **全参数 actor 生命周期的 host 峰值**。正确解决方向是：

1. external rollout/discriminator 节点；或
2. 改成 LoRA，冻结 122B base，仅同步小 adapter。

本项目最终选择第二条，以满足“只用单机 8 卡”的约束。

---

## 12. 122B-FP8 单机 8 卡 LoRA ToolRL：最终配置

### 12.1 LoRA 参数

| 参数 | 最终值 | 意义 |
|---|---:|---|
| rank | 32 | adapter 的低秩维度；越大表达力和显存越高 |
| alpha | 64 | LoRA 输出缩放；这里 alpha/rank=2 |
| dropout | 0 | 已有小数据/RL 噪声，避免额外随机性 |
| train dtype | adapter FP32 | adapter 很小，优先稳定性 |
| target modules | `linear_qkv`、`linear_proj`、shared experts FC1/FC2 | 覆盖注意力投影和共享 expert |
| excluded | GDN/linear-attention | SGLang online LoRA 不支持该部分热加载 |
| adapter export gate | 384 keys，0 个 GDN keys | 防止漏导出或导出不支持模块 |

### 12.2 模型和显存设置

| 参数 | 最终值 |
|---|---:|
| actor topology | TP2/PP4/EP2，12/12/12/12 |
| rollout topology | SGLang TP8 |
| actor FP8 recipe | blockwise |
| FP8 param gather | 开启 |
| optimizer state | adapter FP32 Adam，无 CPU offload |
| actor/rollout offload | 都关闭，actor 常驻 |
| SGLang fraction | 0.25 |
| KV cache | BF16/default，不用未校准 FP8 KV |
| max tokens/GPU | 10240 |
| prompt/response/context | 10240 / 2048 / 12288 |
| CUDA graph | 关闭 |
| custom all-reduce | 关闭，使用 NCCL |
| overlap schedule | 关闭 |

为什么 LoRA 可以用 blockwise，而全参数 SFT 用 delayed：全参数 optimizer 需要让 FP8 tensor 支撑 FP16 optimizer shard，当前 pinned MCore 的 blockwise 不兼容；LoRA optimizer 只管理小型 FP32 adapter，base 的 blockwise FP8 只用于 forward/gather，因此可以工作，而且 gate 中更快。

### 12.3 ToolRL 算法参数

| 参数 | 最终值 |
|---|---:|
| 数据 | compacted 10,240-context ToolRL，3,182 条 |
| RBS / N / GBS | 8 / 1 / 8 |
| rollout 数 | 398 |
| advantage | REINFORCE++ + normalize |
| reward | dense MolClaw |
| LR / min LR | `2e-7 / 2e-8` |
| schedule / warmup | cosine / 3% |
| Adam betas | 0.9 / 0.95 |
| weight decay | 0.01 |
| temperature | 0.8 |
| clip | 0.20 / 0.28 |
| KL loss | 关闭 |
| rollout behavior log-prob | 不直接使用，`USE_ROLLOUT_LOGPROBS=0` |

### 12.4 为什么不用 SGLang rollout log-probs

我们做了 FP8、BF16 rollout、temperature 1.0、blockwise 等多组 A/B gate，SGLang 和 Megatron 对同一 token 的 log-prob 仍相差约 14–15 nats/token。

如果把 SGLang log-prob 当 PPO old policy，第一步大量 token 会被错误 clip。最终方案是在 Megatron 中冻结旧策略并重算 log-prob，使初始 ratio=1。

LoRA 下 reference policy 不加载第二份 122B checkpoint，而是临时 `disable_adapter`，用冻结 base 做 reference forward。这同时保证数学含义和 host 容量。

正式 ToolRL 已连续完成多个真实 update，loss/grad 有限、reward 有差异、adapter 热加载成功、无 OOM/NCCL/NaN。

---

## 13. 122B-FP8 单机 8 卡 LoRA GAD：最终配置

### 13.1 Stage 2 negatives

| 参数 | 值 |
|---|---:|
| generator | 同一个 122B SFT policy |
| RBS | 8 |
| rollout 数 | `ceil(3147/8)=394` |
| context | 10240 prompt + 2048 response |

### 13.2 Discriminator warmup

| 参数 | 最终值 |
|---|---:|
| model | Qwen3.5-4B |
| epochs | 1 |
| batch size | 2 |
| LR | `1e-5` |
| max length | 4096 |
| save interval | 100 |

这里必须显式覆盖 generic profile 的旧 0.8B fallback。4B 是最终 gate 的质量/容量折中点。

### 13.3 Online service

| 参数 | 值 |
|---|---:|
| device | `cuda:0`，请求后 offload |
| LR | `1e-5` |
| update steps/request | 1 |
| max length | 4096 |
| save interval | 50 |

### 13.4 Stage 3 generator

| 参数 | 最终值 |
|---|---:|
| RBS / N / GBS | 4 / 2 / 8 |
| rollout 数 | `ceil(3147/4)=787` |
| advantage | GSPO |
| dynamic filter | reward nonzero std，最多丢 32 组 |
| reward mode | hybrid |
| discriminator/format/tool 权重 | 0.8 / 0.1 / 0.1 |
| LR / min LR | `1e-7 / 1e-8` |
| schedule / warmup | cosine / 3% |
| weight decay | 0.01 |
| temperature | 0.8 |
| KL | 0.001 |
| reference | 同一 base，临时关闭 LoRA adapter |

GAD reference-KL gate 已真实完成 reference forward、有限 `kl_loss`、LoRA adapter export/reload，并观察到约 89.5 GiB/GPU 的峰值。正式 GAD 按串行设计在 ToolRL 完成后启动，所以“参数和关键 gate 已确定”不等于“GAD 完整 epoch 已完成”。

---

## 14. 本窗口遇到的问题与解决方法

### 14.1 集群、SSH 和挂载

#### 问题：旧 worker hostname 失效

`rjob` pod 名带时间戳，worker 重建后旧 SSH 地址不能继续使用。

**解决：**每次以用户最新提供或 `rjob` 输出的 pod 为准；技能和文档不硬编码旧 worker。

#### 问题：登录节点与 worker 路径不同

登录节点通常是 `/home/sunxiangyu/...`，worker mount 是 `/root/slime_sxy/...`。

**解决：**统一 source `slime_env.sh`，通过 `SLIME/WD/DATA/DRUG_AGENT_RUNS_ROOT` 解析；preflight 同时检查模型、数据和 GPFS mount。

#### 问题：外部 launcher 重启 Ray，破坏正在运行的任务

旧脚本可能直接 `ray stop --force`。

**解决：**增加 `guard_ray_restart.sh`，发现 RUNNING/PENDING submission 时 fail closed；任何手动重启前先列出 Ray jobs 和 GPU processes。

### 14.2 模型下载与 torch_dist 转换

#### 问题：只看到 `config.json`，实际 shard 没下全

大模型下载中断时，index 存在但部分 safetensors 缺失。

**解决：**按 `model.safetensors.index.json.weight_map` 检查每个 shard；转换目录必须为空、source/output 分离、加原子 lock。

#### 问题：转换完成 marker 不等于训练可加载

`latest_checkpointed_iteration.txt=release` 只能说明转换程序结束。

**解决：**在目标 TP/PP/EP topology 做 load-only，再做真实 forward/backward/optimizer step。

#### 问题：自制 FP8 converter 漏掉 Qwen3.5 fused experts

原工具会跳过 `experts.gate_up_proj/down_proj`，输出看起来是 FP8，实际大量 expert 仍是 BF16。

**解决：**补丁把 fused experts 按 expert 拆成 gate/up/down、生成 `weight_scale_inv`、正确统计 tensor bytes；与官方 128×128 block FP8 shard 做 bit-exact 对照。最终 122B 生产优先使用官方 FP8 checkpoint。

### 14.3 并行与长序列显存

#### 问题：试图把 TP 设成 4/8

大模型只有 2 个 KV query groups，违反 Megatron 整除约束。

**解决：**固定 TP2，其余 GPU 给 PP/EP/DP。

#### 问题：CP2 看起来能切长序列，却造成 host OOM

CP 复制 optimizer state，27B/4 卡在约 492/518 GiB 被杀。

**解决：**优先 PP；只有在明确预算 optimizer 复制后才考虑 CP。

#### 问题：最后 pipeline rank 比其他 rank 更容易 OOM

LM head、全词表 logits、cross entropy 和 log-prob workspace 都在最后 stage。

**解决：**不均匀 PP 层分配；27B/8 卡从 18/16/16/14 进一步调到 20/16/16/12。

#### 问题：减小 log-prob chunk 仍然 OOM

旧实现会在 backward 前保留每个 chunk 的 softmax clone，减小 chunk 只是延后同一总量。

**解决：**实现 vocab log-prob recompute，在 backward 中重新计算 FP32 normalization；再使用 512/256/64 等合适 chunk。

#### 问题：开启 overlap 后反而 OOM

grad reduce/param gather overlap 需要额外通信 buffer/staging。

**解决：**先关闭 overlap 建立稳定基线，只有在最紧 rank 有充足余量后再逐项打开。

#### 问题：GBS2 的 35B/8卡像 NCCL hang

实际是一份 DP 处理极长样本、另一份早已完成并等待 collective。

**解决：**GBS4 + `--balance-data`；必要时按长度 bucket，而不是先修改 NCCL timeout。

### 14.4 Optimizer 和 host memory

#### 问题：以为 BF16 moments 能让 CPUAdam 省一半 host

当前 pinned `HybridDeviceOptimizer` 的 CPU path 强制创建 master、gradient、m1、m2 四个 FP32 tensor，约 16 bytes/parameter。

**解决：**按实际实现预算，不按命令行 dtype 名称想象；validator 明确拒绝虚假的低内存配置。

#### 问题：offload 越高不一定越安全

27B/4卡中：

- 60% 让 GPU restore 太紧；
- 80% 让 CPUAdam host OOM；
- 70% 才是 HBM/host 平衡点。

**解决：**同时监控每卡 HBM 与 host cgroup；逐档 gate，不把 offload 当单调旋钮。

#### 问题：异步 H2D optimizer copy 后权重异常

当前 stream patch 原先等待了错误方向的 event，optimizer master/live params 存在竞态。

**解决：**在 current stream 等待 H2D stream 正确完成；添加回归测试。27B ToolRL 的多次重试中这是关键 correctness fix 之一。

### 14.5 122B 全参数 FP8 optimizer

#### 问题：blockwise FP8 在 full-param optimizer 报 `BlockwiseQTensor.view`

当前 MCore/TE 组合无法让 blockwise tensor 支撑 FP16 optimizer shard。

**解决：**全参数路径使用 delayed FP8；LoRA 路径因 optimizer 只管理 FP32 adapter，仍可使用 blockwise。

#### 问题：Adam 第一次 lazy 初始化只差 20–24 MiB 也会 OOM

122B 各 rank 已非常接近 HBM 上限，FP32 临时展开或过大的 optimizer group 会触发边界 OOM。

**解决：**缩小参数组到 32M/64M elements、逐组更新、立即释放 FP32 expansion，把 FP8 moments 逐组放到 pageable CPU，并调用适当的 cache 清理。

#### 问题：把 FP16 master weights 也 offload，host 达到 975–1016 GiB

即使提高 Ray threshold，1-TiB cgroup 仍真实越界。

**解决：**只 offload moments，master weights 留 HBM；Ray 保护恢复到安全值。

#### 问题：较高 122B SFT LR 在第二个 warmup update 后 NaN

容量已通过不代表数值稳定。

**解决：**正式 SFT 降到 `1e-7 -> 1e-8`，重新做多更新 gate，最终完成 91 steps。

### 14.6 Rollout 与权重同步

#### 问题：122B BF16 rollout 没有 KV pool

每卡裸权重约 31 GB，static budget 不够权重 + runtime + KV。

**解决：**官方 block-FP8 rollout，每卡权重约 17.1 GB，static 0.25 后建立大 KV pool。

#### 问题：TP8 custom all-reduce 报 CUDA invalid argument

与 CUDA graph input sharing 的兼容问题。

**解决：**`--sglang-disable-custom-all-reduce`，改用 NCCL。

#### 问题：更新权重后生成结果损坏

Qwen3.5 GDN/Mamba CUDA graph 保留旧 state，in-place weight update 后没有正确 recapture。

**解决：**在线 RL 关闭 CUDA graph；每次权重同步后做真实生成 correctness gate。

#### 问题：SGLang 与 Megatron log-prob 差约 15 nats/token

尝试 BF16 rollout、temperature 对齐、FP8 recipe 对齐都没有消除。

**解决：**122B LoRA 不使用 rollout behavior log-prob，改在 Megatron 中重算冻结 old policy；第一步 `ppo_kl=0`、`pg_clipfrac=0` 是正确现象。

#### 问题：LoRA target 包含 GDN，SGLang 无法热加载

训练端能创建 adapter 不等于 rollout backend 支持该模块。

**解决：**只 target QKV、attention projection 和 shared experts FC1/FC2；gate 检查 384 keys 且 GDN keys=0。

#### 问题：GAD reference 需要第二份 122B checkpoint

这会破坏 LoRA 的内存优势。

**解决：**reference forward 临时 disable adapter，使用同一冻结 base；不做全模型备份。

### 14.7 Reward、训练信号和算法稳定性

#### 问题：GRPO loss/grad=0，但程序没有报错

同一组所有 response reward 完全相同，中心化后 advantage 全为 0。

**解决：**先检查 parser、reward components、采样多样性和 discriminator；ToolRL 可使用 RBS8/N1 + REINFORCE++，GAD 使用 dynamic nonzero-std filter。不要盲目加 LR。

#### 问题：27B ToolRL 第一更新后 ReAct 格式崩坏

LR `5e-8` 对当前全参数 RL 仍太大，且 reference switching path 也产生不稳定。

**解决：**LR 降 `1e-8`、约 20 步 warmup、temperature 0.7、默认关闭 ref switching；观察到大量稳定更新。

#### 问题：GAD discriminator 单 pair accuracy=1.0 看起来很好

一个 pair 的 accuracy 没有统计意义，只能证明接口能跑。

**解决：**完整 negative cache、1 epoch warmup、记录 margin/loss/截断率，再做 online generator gate。

#### 问题：122B GAD 意外继承 0.8B discriminator

generic profile 的 fallback 覆盖了最终质量配置。

**解决：**122B serial launcher 显式 pin Qwen3.5-4B，并写入 `resolved_config.env`。

### 14.8 Checkpoint、恢复和监控

#### 问题：一次 optimizer step 成功，但保存 checkpoint 时 host OOM

checkpoint serialization 有独立瞬时内存峰值。

**解决：**把 save/reload 作为单独 gate；122B 全参数 SFT 最终只保存 weights，不保存 optimizer/RNG。

#### 问题：Ray CLI 返回 0，serial script 错误标记 COMPLETE

operator stop 或无 checkpoint 的任务不应算完成。

**解决：**stage 完成后检查语义 artifact：tracker、negative cache、manifest、adapter 或 PASS marker。

#### 问题：监控脚本在 rollout sample 中搜到 `Traceback` 就报警

训练数据本身可能包含失败日志；SGLang server args 也含 `nan_detection=False`。

**解决：**只扫描当前 Ray submission segment；排除完整 rollout sample 行；NaN regex 只匹配 `loss/grad_norm = nan/inf` 形式。

#### 问题：tmux pane 还活着就认为训练健康

pane 可能只是一个空 bash，真正 child 已退出。

**解决：**同时检查 Ray job、actor/rollout processes、日志 freshness、step advancing、GPU 使用、有限 metrics 和 stage markers。

---

## 15. 一套可复用的训练放行流程

不要从“模型加载成功”直接跳到完整 epoch。按下面顺序：

1. **worker preflight**：GPU 数量/型号/HBM、host cgroup、磁盘、mount、CUDA、Torch、Megatron、SGLang、mbridge、FLA；
2. **HF source**：config/index/shards 全部存在；
3. **HF→torch_dist**：新目录、原子 lock、tracker=`release`；
4. **目标 topology load**：真正 reshard/load；
5. **最短样本 step**：forward/backward/optimizer；
6. **P50 step**；
7. **P95/max step**；
8. **checkpoint save/reload**；
9. **RL 单组**：rollout→reward→非零 advantage→update→同步→restore；
10. **第二个 update**：观察 lazy state 与内存复用；
11. **多步数值/奖励 gate**；
12. **完整阶段**。

任何一步失败都创建新 retry tag，保留旧日志，找“当前 submission 的第一个真实错误”，一次只修改一个原因。

---

## 16. 怎样判断训练真的健康

### 16.1 进程层

- Ray job 为 RUNNING；
- actor、RolloutManager、SGLangEngine、GAD service 按阶段存在；
- 不是只有 tmux shell。

### 16.2 资源层

- 预期 GPU 都有合理显存占用；
- utilization 会随 rollout/train 阶段变化；
- 最紧 rank 仍保留可解释余量；
- host available 不逼近 Ray threshold；
- GPFS 有 checkpoint 空间。

### 16.3 数值层

- loss、grad norm 有限；
- 没有 NaN/Inf；
- clip fraction 不长期接近 1；
- KL 没有无控制上升；
- LR 符合 warmup/decay 预期。

### 16.4 RL 信号层

- reward 有多个取值；
- parse success 不持续下降；
- 不连续 6 步零 grad；
- response truncation 可解释；
- held-out tool/schema/value accuracy 没有退化。

### 16.5 生命周期层

- 权重/adapter 同步后 rollout 生成正确；
- checkpoint marker 与实际文件一致；
- stage 分支来源正确；
- GAD discriminator manifest 与 model path 正确。

---

## 17. 初学者下一步应该学习什么

建议按以下顺序深入：

1. **先学数据**：tokenization、chat template、loss mask、长度分布、train/held-out split；
2. **再学单卡训练**：forward、loss、backward、optimizer、gradient clipping；
3. **再学并行**：先 DP，再 TP/PP，最后 MoE 的 EP；
4. **学习内存公式**：权重、梯度、master weights、Adam moments、激活、KV cache；
5. **学习 RL batch 语义**：prompt、response、reward、advantage、old policy、KL、clip；
6. **学习工程 gate**：为什么短 step、长 step、save、restore、第二 step 都要独立验证；
7. **最后再追求 MFU/吞吐**：先正确、再稳定、再快。

值得亲手做的练习：

- 用 4B/小数据复现一次 SFT，观察 LR、grad norm 和 loss；
- 构造一个全同 reward group，看 GRPO advantage 为什么为 0；
- 比较 TP2/PP2 与 TP2/PP4 的每 rank 显存；
- 观察开启/关闭 full recompute 的显存和速度；
- 读取一个 checkpoint 中 parameter、optimizer state 的 dtype 和字节数；
- 对同一 response 比较 SGLang 和 Megatron log-prob，理解 backend 数值一致性为何重要。

---

## 18. 项目中的事实来源和入口

主要 profile 与 launcher：

```text
slime-wd/slime/drug_agent/scripts/qwen3_large_profile.sh
slime-wd/slime/drug_agent/scripts/run_qwen3_large_probe.sh
slime-wd/slime/drug_agent/scripts/run_qwen3_large_training_serial.sh
slime-wd/slime/drug_agent/scripts/run_qwen35_122b_lora_aligned_gate.sh
slime-wd/slime/drug_agent/scripts/run_qwen35_122b_lora_rl_serial.sh
slime-wd/slime/drug_agent/scripts/preflight_large_model_worker.sh
slime-wd/slime/drug_agent/scripts/monitor_qwen3_large_serial.sh
```

回归测试：

```text
slime-wd/slime/drug_agent/tests/test_large_model_profiles.py
slime-wd/slime/drug_agent/tests/test_recomputed_log_probs.py
```

更偏探针记录的历史材料：

```text
slime-wd/slime/drug_agent/docs/qwen3_large_training_plan_zh.md
slime-wd/slime/drug_agent/docs/qwen3_122b_8xh200_probe_20260803_zh.md
```

Codex 可复用技能：

```text
.codex/skills/slime-h-cluster-training/
```

外部一手资料：

- Slime quick start：<https://thudm.github.io/slime/get_started/quick_start.html>
- Slime low precision：<https://thudm.github.io/slime/advanced/low-precision.html>
- Megatron-Core distributed optimizer：<https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html>
- Megatron-Core optimizer CPU offload：<https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/optimizer_cpu_offload.html>
- Megatron-Core MoE：<https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html>
- Qwen3.5-122B-A10B-FP8：<https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8>
- ToolRL：<https://arxiv.org/html/2504.13958>
- GAD：<https://arxiv.org/html/2511.10643>

---

## 19. 最后的方法论总结

这轮实验真正有价值的不是某一组神奇参数，而是以下方法：

1. **把结论限定在证据范围内**：SFT gate 不外推 RL，短上下文不外推 94K，训练 step 不外推 checkpoint save；
2. **区分训练制度**：全参数和 LoRA、官方 FP8 权重和 BF16 KV、colocated 和 external engine 是不同问题；
3. **同时看 GPU 与 host**：offload 只是把压力搬家；
4. **先保证训练信号，再优化吞吐**：GPU 100% 但 reward 全同、grad=0，没有学习价值；
5. **一次只修一个因果问题**：用最小 gate 验证，再升级规模；
6. **把每次踩坑变成 fail-closed 检查和回归测试**：这样下一次 Codex 窗口或新成员不必从头踩一遍。

如果未来更换模型版本、镜像、Megatron/TE/SGLang 版本、数据长度或 worker 内存，本文的数值应作为经过验证的起点，而不是永恒常量。重新执行放行流程，才是真正掌握训练大语言模型的标志。
