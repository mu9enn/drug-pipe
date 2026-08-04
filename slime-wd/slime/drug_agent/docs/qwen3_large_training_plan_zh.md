# Drug-agent 大模型训练计划（Qwen3.5/3.6）

更新时间：2026-08-03。本文记录当前可复现的模型、数据和硬件假设；122B 最新探针细节见 `qwen3_122b_8xh200_probe_20260803_zh.md`。

## 当前结论

- Qwen3.5-27B 已完成 HF→torch_dist（16 shards，53,793,990,659 bytes）。4×H200 上已通过 47,499-token SFT 与 ToolRL 单组闭环；8×H200 的 `TP2/PP2/DP2`、34/30 层不均衡 PP、BF16 Adam moments 在约 177.2k tokens/两次 update 上通过，`74.81 s`、`2368.7 token/s`，最紧卡 138,816 MiB。4 卡 ToolRL 的实测甜点是 70% optimizer offload + SGLang 0.18。4 卡在线 GAD generator CPUAdam 在 530 GB worker 上达到 `507.31/517.58 GiB` 被杀；至少换 600 GiB host。
- Qwen3.6-35B-A3B 已完成 HF→torch_dist（16 shards，69,410,999,507 bytes）。4×H200 的 47,499-token SFT 使用 `TP2/PP2/EP2`、40% optimizer offload，`212.55 s`、`223.5 token/s`。8×H200 使用 `TP2/PP2/EP2/DP2`、22/18 层、BF16 moments、GBS4 + `--balance-data`，约 177.2k tokens 的真实 update 为 `124.02 s`、`1428.9 token/s`，最紧卡 135,694 MiB。GBS2 无 balance 的对照发生 DP/NCCL 长短样本失配，不能使用。ToolRL/GAD 尚不能由 SFT 成功外推。
- Qwen3.5-122B-A10B 已在 8×H200/1 TiB host 上完成 HF→torch_dist、`TP1/PP8`→`TP2/PP4/EP2` 重分片和真实 4501-token SFT optimizer step。旧失败只排除了 **BF16 SGLang rollout**（权重约 30.5 GiB/GPU 后没有 KV 余量）。按 Slime 官方稳定路径保留 BF16/torch_dist actor、改用 128×128 block-FP8 rollout（39 shards，126.71 GB）和 FP8 KV 后，每卡 rollout 权重约 17.1 GiB，并成功建立 1,815,954-token KV pool。6,008-token ToolRL 单组已经完成 rollout、4 次 forward/backward 和 optimizer step；但随后 train→rollout pause 的 host backup 超过 1 TiB，Ray 99% gate 仍会杀 actor，所以单机 colocated 长跑未通过。GAD 还增加在线 discriminator，不能由 ToolRL 外推。
- 三个 HF checkpoint 都包含视觉塔和一层 MTP；当前 model args 没有启用 `--mtp-num-layers/--enable-mtp-training`，所以 torch_dist/训练对象是文本语言塔与 LM head，排除视觉塔和 MTP。不能把完整目录大小直接当成训练参数量。

## 实测模型规模

直接读取 safetensors header 得到下表（十进制 GB）：

| 模型 | 全 checkpoint | 语言塔 | LM head | MTP | 视觉塔 | 当前训练 BF16 权重（排除视觉塔、MTP） |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-27B | 55.563 | 51.249 | 2.543 | 0.849 | 0.921 | 53.792 GB（约 26.896B 参数） |
| Qwen3.6-35B-A3B | 71.904 | 68.304 | 1.017 | 1.689 | 0.893 | 69.321 GB（约 34.661B 参数） |
| Qwen3.5-122B-A10B | 250.173 | 242.697 | 1.526 | 5.047 | 0.903 | 244.223 GB（约 122.112B 参数） |

MoE 只减少每 token 的计算量，不减少必须保存/更新的全部 expert 参数和 optimizer state。

## 数据长度（Qwen3.5 tokenizer，`enable_thinking=false`）

| 数据 | 条数 | p50 | p90 | p95 | p99 | 最大 | 总 tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT 完整 chat | 364 | 14,209 | 54,732 | 65,634 | 83,954 | 94,063 | 8,708,386 |
| ToolRL prompt+target | 3,182 | 10,078 | 27,795 | 33,958 | 55,690 | 94,061 | 44,368,735 |
| GAD prompt+target | 3,147 | 10,194 | 27,864 | 34,123 | 55,917 | 94,061 | 44,180,700 |

因此：

- `rollout_max_prompt_len=98304`、`rollout_max_response_len=4096`、`rollout_max_context_len=102400` 能覆盖当前数据；不能继续沿用 6144/512 的旧 smoke 默认值。
- `max_tokens_per_gpu=8192` 是动态 micro-batch 的打包目标，不会截断超过它的单条样本。当前低卡 profile 用 PP、full recompute、chunked log-prob 和实际显存探针覆盖超长单条；不要误以为该参数会把一条样本自动切片。
- SFT 只有 364 条但约 8.7M tokens；先做 1 epoch。ToolRL 论文也指出约 400 条 SFT 足以学习工具格式，过度 SFT 可能过拟合。

## 初始 profile

| profile | 训练拓扑 | rollout TP | SFT LR | ToolRL LR | GAD generator LR | optimizer |
|---|---|---:|---:|---:|---:|---|
| `qwen35-27b-4xh200` | TP2/PP2/CP1/EP1 | 4 | 1e-6→1e-7 cosine | 1e-7 constant | 5e-8 | offload=40/70/70%；SFT/ToolRL host≥480 GiB、在线 GAD host≥600 GiB；ToolRL/GAD SGLang=0.18 |
| `qwen35-27b-8xh200` | TP2/PP2/CP1/EP1/DP2；34/30 层 | 2×TP4 | 1e-6→1e-7 cosine | 1e-7 constant | 5e-8 | SFT 无 CPU offload、BF16 moments；ToolRL/GAD CPUAdam 初值=35/40%；SGLang=0.22/0.18；GBS4 + balance-data |
| `qwen36-35b-4xh200` | TP2/PP2/CP1/EP2 | 4 | 2e-6→2e-7 cosine | 2e-7 constant | 1e-7 | SFT 实测 offload=40%；ToolRL/GAD 初值=60/70%；DeepEP；一般 host≥480 GiB、在线 GAD 未实测保守设≥700 GiB |
| `qwen36-35b-8xh200` | TP2/PP2/CP1/EP2/DP2；22/18 层 | 2×TP4 | 2e-6→2e-7 cosine | 2e-7 constant | 1e-7 | SFT 无 CPU offload、BF16 moments；ToolRL/GAD CPUAdam 初值=40/50%；ordinary all-to-all；GBS4 + balance-data |
| `qwen35-122b-8xh200` | TP2/PP4/CP1/EP2 | 外置 TP8 | 5e-7→5e-8 cosine | 5e-8 constant | 2e-8 | BF16 torch_dist actor + FP8 param gather/FP8 moments；128×128 FP8 rollout + FP8 KV；单机 colocated 默认 fail-closed |

这些 LR 是第一轮探针值。最终值由下列信号决定：训练 loss、grad norm、clip fraction、ToolRL group reward 标准差/全同率、KL（只监控，ToolRL 基线不加 KL）、GAD discriminator margin/accuracy，以及 HBM/RSS 峰值。

## 为什么 TP/CP/EP 这样选

- 35B/122B 只有 2 个 KV query groups；Megatron 要求 `num_query_groups % TP == 0`，所以 TP 不能设成 4/8，初始用 TP2。
- Megatron 分别建立 dense grid 与 expert grid：dense 约束是 `world % (TP×PP×CP)==0`，expert 约束是 `world % (ETP×EP×PP)==0`。CP 能切长序列激活，但会复制参数/optimizer；PP 同时切模型层和 optimizer。27B 的两次 CP2 实测均在约 492/517.58 GiB 被 Ray 杀死，PP2 则一步成功，因此三个低卡数 profile 都优先 PP、CP1。
- 当前样本最长约 94K；用 full recompute、PP 和 `log_probs_chunk_size=1024` 控制峰值。`max_tokens_per_gpu=8192` 只是动态打包目标，超长单条不会被它切成多个 microbatch。
- 27B 是 dense 模型，不设 EP；rollout 用 TP4 使完整 27B 推理权重均匀分布到 4 卡。

## 27B 实测记录（4×H200，host 517.58 GiB）

- 转换成功：16 个 `.distcp` shard，BF16 训练权重 53,793,639,224 bytes，可在加载时重分片。
- 47,499-token SFT 一步：`TP2/PP2/CP1`、40% optimizer offload、full recompute、log-prob chunk 1024；loss `0.1423622690`、grad norm `3.1299273`、48.06 秒、约 988.4 token/s；峰值约 108.72 GiB/GPU，host used 169.78 GiB。权重 checkpoint 保存并被 ToolRL 成功加载。
- 对照失败：同一数据用 `TP2/CP2` 时，70% 和 80% offload 都在约 492 GiB host usage 被 Ray 杀死；原因是 CP rank 复制 optimizer，而不是 GPU 不够。
- ToolRL 调参：actor sleep 会创建整份 CUDA allocation 的 host backup，和 optimizer offload 叠加后 OOM，因此 colocated RL 固定 `--no-offload-train --offload-rollout`。40% offload + SGLang 0.35 在 cache 恢复时 HBM OOM；60% + 0.25 在最终恢复时仍超 HBM；80% + 0.18 则在 CPUAdam update 达到 `512.82/517.58 GiB` host usage，即使 Ray 阈值 99% 仍被杀。最终 70% + 0.18 完整闭环成功：4 条 rollout 平均 response 1,271.5 tokens，rollout 21.58 秒；actor train 58.94 秒、约 575.8 token/s，grad norm `18.0457`；最终同步后最紧卡约 111.42 GiB used，host used 294.54 GiB。该组平均 advantage 接近 0 是 GRPO 组内中心化的定义，不代表无梯度。
- 上述成功探针按设计不保存 step-end checkpoint。此前 60% 配置的权重序列化把 host 推到约 492.21 GiB，因此“稳态训练”和“可恢复 checkpoint 保存”仍是两个 gate；正式长跑需更高 host 余量或经过单独验证的保存时序。

### 27B GAD 实测

- Stage 2 negative cache 成功生成 1 条、29,505 bytes，response 391 tokens，rollout 4.63 秒；当前 Slime 入口仍以 `LR=0` 走了一次无效 generator update（grad norm 0），因此它证明数据接口但不是推荐的批量 negative 生成效率。
- Qwen3.5-0.8B discriminator 用 1 个 aligned pair、batch 1、`max_length=8192` 完成 warmup并保存 checkpoint/manifest：loss `0.5556863`、accuracy `1.0`、positive/negative score `-2.03125/-2.328125`、margin `0.296875`、grad norm `496`（实际按 1.0 clipping）。单 pair 的 accuracy 没有质量含义；正式 GAD 必须用全量 pair 完成至少 1 epoch warmup。
- 在线链路已验证 discriminator checkpoint contract、4 条 rollout、trajectory JSONL、判别器持续更新（最终 version 6）、reference log-prob 与两次 actor backward。最终判别器指标为 loss `4.01e-7`、margin `14.7285`，checkpoint 位于 `gad_discriminator_online_probe_final`。
- 本次 warmup 使用 8K 窗口，但已启动的在线 service 仍是旧的 4K 参数；它足以验证协议，不足以验证长轨迹判别质量。profile/launcher 已统一把后续默认改为 8K，正式实验必须把 service 的启动日志与 profile 一起归档，避免这种静默不一致。
- 该 probe 的 4 条 response 完全相同，raw GAD reward 都为 `0.7042451`，所以 GRPO 报告 `zero_std/count=1`、advantage/return/actor grad 信号为 0。这只是管线 gate，不是有效策略学习；正式运行必须先让 group reward 有方差。
- 资源 gate 未通过：70% offload 在 Ray 97% 阈值达到 `502.47/517.58 GiB`，65% offload 在 97% 达到 `502.17 GiB`，最后 65%/98% 达到 `507.31 GiB` 后仍被杀。降低 offload 没有线性降低 CPUAdam 的瞬时 RSS，继续关闭/提高 Ray 保护不安全。profile 因此为在线 GAD 单独设置 host≥600 GiB；530 GB 只支持已验证的 SFT、ToolRL、negative 与 discriminator-only 阶段。

### 27B 实测记录（8×H200，host 1025 GiB）

- 转换 checkpoint 在 `TP2/PP2/DP2` 下实际重分片加载成功。FP32 moments 在第二次 update 的 vocab log-prob tile 处连续以 478/242/122 MiB 级别差额 OOM；10% HybridDeviceOptimizer CPU offload与 gradient/param overlap 又因额外 staging buffer 在第一步 OOM，均不采用。
- 最终稳定点使用 34/30 层不均衡 PP、log-prob chunk 256、FP32 master/grad + BF16 m1/m2、无 CPU offload/无 overlap。4 条长样本约 177.2k tokens、两次 update 完整成功：`actor_train=74.812 s`、`2368.66 token/s`、loss `0.10203/0.12439`、grad norm `1.593/1.646`；最紧卡 138,816 MiB。
- 正式 SFT 用 GBS4 + `--balance-data`，让 DP2 每次共同处理四条样本并按 token 长度平衡。p95–max（65–94K）单条另设 `TP2/PP4/DP1` 长桶：18/16/16/14 层、GBS4；PP4 切参数和 optimizer，优先于复制 optimizer 的 CP。

## 35B 实测记录（4×H200，host 517.58 GiB）

- 转换成功：16 个 `.distcp` shard，BF16 训练权重约 69.321 GB（约 34.661B 参数），可在训练时按目标并行拓扑加载。
- 47,499-token SFT 一步完整成功：`TP2/PP2/CP1/EP2/ETP1`、DeepEP/flex dispatcher、40% optimizer offload、full recompute、log-prob chunk 1024；loss `0.1502420579`、grad norm `2.8944246630`、actor train `212.55` 秒、约 `223.5 token/s`。
- 峰值约 131.21 GiB/GPU（最紧卡仅余约 8.85 GiB），host used 184.81 GiB；随后权重 checkpoint 保存成功。结论仅覆盖 SFT，colocated ToolRL/GAD 还会同时引入 SGLang 权重和 cache，必须另做单组 gate。

## 35B 实测记录（8×H200，host 1025 GiB）

- torch_dist 在 `TP2/PP2/EP2/DP2` 下实际重分片加载成功。最终配置为 22/18 层不均衡 PP、ordinary NVLink all-to-all、log-prob chunk 256、FP32 master/grad + BF16 m1/m2、无 CPU offload/无 overlap。
- GBS2、没有 `--balance-data` 的对照中，一份 DP 的首 microbatch 为 70.57 秒，随后出现单卡 NCCL kernel 100%、其余 7 卡等待的长短样本 collective 失配，主动停止；这不是显存失败。
- GBS4 + `--balance-data` 完整成功：约 177.2k tokens，`actor_train=124.017 s`、`1428.87 token/s`、loss `0.115396`、grad norm `1.03293`。首 microbatch 降至 18.08 秒；各卡峰值为 117,192–135,694 MiB，最紧卡仍余约 7.9 GiB，所以 overlap 默认关闭。
- 与 4 卡实测相比吞吐约 6.4×，主要来自去掉 40% CPUAdam offload、DP2 和单机 ordinary all-to-all，不应解释为单纯卡数线性加速。p95–max 长桶建议 `TP2/PP4/EP2/DP1`、12/10/10/8 层、GBS4，仍需最大样本 gate。

## 122B 实测预算（8×H200，host 1025 GiB）

- 训练对象约 122.112B 参数。`TP2/PP4/CP1/EP2` 已成功从 `TP1/PP8` release checkpoint 重分片加载；TP2 满足 2 个 KV query groups，CP1 避免 optimizer 复制。
- 传统 CPUAdam 的 FP32 state 仍需 >1 TiB，已排除。当前 profile 改用 precision-aware TE FusedAdam：FP8 gathered params、FP16 main params、BF16 main grads、两个 FP8 moments；SFT 只把 moments 放入 pageable CPU buffers，master weights 留在 HBM。
- 4501-token shortest SFT 完整一步已成功：loss `0.2070251374`、grad norm `3.4294734`、forward/backward microbatch `28.08 s`、actor train `85.68 s`、`52.53 token/s`。最紧卡使用 `131.78/140.06 GiB`，host used/available `489.14/536.25 GiB`；Ray job 正常退出。完整 master+moment CPU offload则分别在 Ray 95%/99% gate 达到 975.59/1016.10 GiB，不能使用。
- BF16 rollout 在 static fraction 0.10 下无法建立 KV pool；换成 block-FP8 rollout、`fp8_e4m3` KV、static fraction 0.25 后建立了 1,815,954-token pool。`prompt/context/response=16384/16448/64` 的单组实际抽到 5,944+64=6,008 tokens；4 条 rollout、全部 actor forward/backward 和第一步 TE FusedAdam 均完成。关键容量参数是 32M-element optimizer groups、512 MiB TorchMemorySaver margin、moments-only pageable offload和关闭 SGLang custom all-reduce。短探针 HBM 峰值达到 141,869/143,771 MiB；下一次 actor pause 又把 cgroup 推过 99% host threshold。因此 FP8 解除了 rollout/KV 显存限制，却没有解除单机 colocated 的训练后 host backup 限制。生产默认继续 fail-closed；正式 ToolRL/GAD 拆外部 rollout/discriminator 节点。

## 首轮完整训练日程（通过全部短跑 gate 后）

| 方法 | 27B/35B 首轮 | 122B 首轮 | epoch / 更新数 | 关键采样参数 |
|---|---|---|---|---|
| SFT | RBS 364、GBS 4 | RBS 364、GBS 4 | 1 epoch，91 updates | 3% warmup，cosine 到表中 min LR；按 assistant loss mask |
| ToolRL | RBS 2、n 4、GBS 8 | 外置 rollout 后 RBS 1、n 4、GBS 4；colocated 只作短计算 gate | 1 epoch，约 1,591 / 3,182 updates | temperature 1.0，GRPO，KL=0，clip 0.2/0.28 |
| GAD | RBS 2、n 4、GBS 8 | 外置 discriminator/rollout 后 RBS 1、n 4、GBS 4 | discriminator warmup 1 epoch；online GAD 先 1 epoch，稳定后最多再 1 epoch | temperature 0.8，KL 0.001；原论文 n=8，本项目 n=4 是资源折中 |

这里的 batch 是样本数，不是 token 数；单条长度从约 10K 到 94K，必须继续用 dynamic batching，并按每次 update 的有效 target tokens 同时报告 loss/throughput。若 SFT grad norm 连续触发 1.0 clipping，就把 LR 减半；若 ToolRL/GAD 全同 reward 组比例高于 20%，先提高采样多样性或修复 reward/discriminator，不加 LR；若 KL 持续上升且 held-out 工具准确率无改善，则 LR 减半或提前停止。checkpoint 保存必须作为独立内存 gate，不能因为无保存的训练 step 成功就开始长跑。

## 当前镜像的 optimizer 语义

精确镜像内的 Megatron `HybridDeviceOptimizer` 在 CPU offload 开启时强制 `param_update_in_fp32=True`，CPUAdam 路径不使用 precision-aware 的 `main_params/exp_avg/exp_avg_sq` dtype 来创建低精度 CPU state。host 上必须按每参数四个 FP32 张量（master、gradient、m1、m2，共 16 bytes）估算。profile validator 已显式执行这一规则，防止用 BF16 配置得到虚假的低内存结论。

## torch_dist 转换

先 source 环境和 profile，再运行通用转换脚本。转换目录必须为空；脚本不会覆盖 partial output。

```bash
source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
cd "$SLIME"

export MODEL_PROFILE=qwen36-35b-4xh200  # 换成另外两个 profile 即可
source drug_agent/scripts/qwen3_large_profile.sh

EXPECTED_GPUS="$NUM_GPUS" HF_CHECKPOINT="$HF_CHECKPOINT" \
  bash drug_agent/scripts/preflight_large_model_worker.sh

NUM_GPUS="$NUM_GPUS" MODEL_ARGS_FILE="$MODEL_ARGS_FILE" \
HF_CHECKPOINT="$HF_CHECKPOINT" SAVE_DIR="$REF_LOAD" \
  bash drug_agent/scripts/prepare_qwen3_torch_dist.sh
```

转换完成条件：`latest_checkpointed_iteration.txt` 内容为 `release`，且目录中至少有一个 `.distcp` 文件。正式训练前还要做一次 Megatron load-only 或最短 SFT forward/backward，不能只看 tracker。

转换脚本还会核对 index 中的所有 safetensor shard、输出磁盘余量、source/output 路径隔离，并用原子 lock 防止两个转换进程同时写入同一目录。

## 可复现短跑入口

统一入口只提供转换和短程 gate，没有完整训练 action：

```bash
export MODEL_PROFILE=qwen36-35b-4xh200

bash drug_agent/scripts/run_qwen3_large_probe.sh validate
bash drug_agent/scripts/run_qwen3_large_probe.sh preflight
bash drug_agent/scripts/run_qwen3_large_probe.sh convert
bash drug_agent/scripts/run_qwen3_large_probe.sh sft-one-step

# 以下步骤从已完成的 SFT checkpoint 分叉。27B/35B profile 可直接做
# 对应资源 gate；122B 的短上下文 ToolRL 需显式设置
# ALLOW_UNSUPPORTED_COLOCATED_RL=1，避免把 6K 单组误当成 94K 全数据通过。
SFT_LOAD=/path/to/sft/checkpoint \
  bash drug_agent/scripts/run_qwen3_large_probe.sh toolrl-one-group

SFT_LOAD=/path/to/sft/checkpoint \
  bash drug_agent/scripts/run_qwen3_large_probe.sh gad-negatives-one
```

GAD 后续按 `gad-discriminator-one` → 单独终端 `gad-serve` → `gad-one-group` 执行。用新的 `RUN_TAG` 创建新输出；入口拒绝覆盖非空 probe 目录。SFT probe checkpoint 使用 `--no-save-optim --no-save-rng`；ToolRL 单组探针默认完全禁用 step-end checkpoint，以便把稳态计算与高 host 峰值的序列化分别设 gate。正式恢复能力必须另设 gate 验证。

## SFT / ToolRL / GAD 顺序

1. 基础 HF → torch_dist。
2. SFT 从 release torch_dist 开始；1 epoch，推荐 `rollout_batch_size=364`、`global_batch_size=4`（91 updates）、3% warmup + cosine，使用 Qwen3.5 多轮 loss mask，只训练 `step_loss_mask=1` 的 assistant 区域。
3. ToolRL 做两个独立臂：论文忠实的 cold start（直接从 release torch_dist）为主臂，SFT warm start 为对照臂且不恢复 SFT optimizer/RNG。27B/35B 推荐 `RBS=2, n=4, GBS=8`；122B 用外置 rollout、`RBS=1, n=4, GBS=4`，再按 ≤8K、8–16K、>16K 逐桶 gate。已通过的 6K colocated 单组只用于容量诊断，不能作为训练臂。temperature 1.0，无 reference KL，reward 使用 format + tool-name/schema/value 的细粒度分解。
4. GAD 与 ToolRL 是两个从 SFT 分叉的实验，不让 GAD 继续加载 ToolRL。先生成 aligned negatives，再 warm up discriminator，最后训练 generator。
5. GAD discriminator 不使用 27B/35B/122B 主模型，而使用同族 Qwen3.5-0.8B 全参数模型。27B/35B 可先验证 CPU 或分时服务；122B 的最紧 rank 仅余约 1.9 GiB、host pause 又超过 1 TiB，因此 discriminator 与 rollout 都必须外置，不能与 actor 共用第 8 张 H200。

ToolRL 论文的可复现基线是 4K step-level 样本、每 prompt 4 responses、temperature 1.0、无 KL、LR `1e-6`、batch 512、15 epochs；论文也报告 GRPO cold start 的泛化优于 SFT 初始化。本项目的序列 p50 已约 10K、模型为 27B–122B，不能照搬 batch/LR：第一轮把 LR 降到表中数值、GBS 降到 8/4，并先跑 1 epoch；只有 held-out tool/schema 指标继续改善且 KL/clip/reward 方差健康时才加到 2–3 epochs，不预先跑 15 epochs。

GAD 原论文采用 Bradley–Terry 判别器、GRPO `n=8`、temperature `0.8`、KL coefficient `0.001`，总计 3 epochs（generator/discriminator 各 1 epoch warmup，再做 2 epochs online GAD），并在消融中发现同尺寸 discriminator 最好。本项目首轮受显存限制采用 `n=4` 和 0.8B discriminator，因此是资源约束版而非论文等价复现。正式运行仍须完成全量 aligned-negative 与 discriminator warmup；单条 pair 只验证接口。判别器 tokenizer 采用 left truncation 以保留候选回答和最近状态，但 4K 默认会丢失长轨迹开头的系统约束：先实测 `max_length=8192`，如共卡峰值允许再升到 16K/32K，并单独统计截断率。若组内 reward 全同，优先检查 discriminator warmup、上下文覆盖和 response 多样性，而不是盲目提高 generator LR。

## 分阶段实测门槛

每个模型/方法均按以下顺序推进，任何一步失败都不进入下一步：

1. worker preflight：GPU 数量/型号、CUDA、Megatron、SGLang、mbridge、FLA、两个 GPFS mount。
2. HF→torch_dist；核对 tracker、distcp 数量、磁盘增量。
3. load-only/单 batch forward。
4. 最短样本 1 次 optimizer step；记录每卡 HBM 和 host RSS。
5. p50 长度样本 1 step。
6. 最大约 94K 样本 1 step。
7. ToolRL 单 rollout group（4 samples），确认 reward 不全同、解析成功且有非零 advantage。
8. GAD 单组：negative cache、discriminator update、generator update、checkpoint contract 全部通过；27B 在当前 530 GB worker 停在 generator CPUAdam，需换 ≥600 GiB host worker 后重做最后一关。

本项目不在参数未收敛前启动完整 epoch/完整 RL 数据集。

## 当前 worker 与并发约束

8 卡 worker 的模型、torch_dist、代码、数据和 1 TiB host 均已通过 preflight；最终 v20 在干净窗口成功，早期 v13 被外部 9B 评测重启的结果不再用于容量判断。

本项目的 SFT/ToolRL/GAD launcher 已增加 `guard_ray_restart.sh`：发现 RUNNING/PENDING Ray submission 时先退出；dashboard 不可达时最多等待 5 秒。这个防护能避免本项目打断已有任务，但无法阻止未更新的外部 launcher 反向重启 Ray；正式长跑仍要求独占窗口或统一入口。

4 卡 worker 的 `--memory=530000` 满足已验证的 27B SFT/ToolRL 与 35B SFT，但不满足 27B 在线 GAD（需要至少 600 GiB）。8 卡 122B 不再建议走 >1.8 TiB 的 FP32 CPUAdam；当前 1 TiB worker 支持实验性的 train-only SFT，以及 FP8 rollout 的短上下文 colocated ToolRL 单组计算。后者最紧卡只余约 1.9 GiB，尚未通过 16K/94K、checkpoint-save 或多步长跑，因此 profile 仍默认 fail-closed；colocated GAD 未通过。

## 一手资料

- [slime 官方 quick start：HF→torch_dist 与多 GPU 转换](https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md)
- [slime 官方 usage：GRPO、per-token loss 与训练参数](https://github.com/THUDM/slime/blob/main/docs/en/get_started/usage.md)
- [slime PR #1662：35B 的 KV groups=2，因此 TP=2；官方 TP2/EP8](https://github.com/THUDM/slime/pull/1662)
- [Qwen3.6-35B-A3B 官方模型卡](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.5-122B-A10B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)
- [Megatron-Core distributed optimizer 内存公式](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html)
- [Megatron-Core optimizer CPU offload](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/optimizer_cpu_offload.html)
- [Megatron-Core MoE 并行与内存优化](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [ToolRL 论文：细粒度 reward、GRPO、4 samples、temperature=1、无 KL](https://arxiv.org/html/2504.13958)
- [GAD 论文：online Bradley–Terry discriminator、warmup 与 GRPO 配置](https://arxiv.org/html/2511.10643)
