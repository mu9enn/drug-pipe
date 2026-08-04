# Measured 4/8-GPU profiles

## Contents

- Capacity principles
- 27B profiles
- 35B profiles
- 122B full-parameter and LoRA profiles
- Tuning priorities

## Capacity principles

- Size total trainable parameters, not MoE active parameters. MoE reduces FLOPs per token but all experts still occupy checkpoint, optimizer, and synchronization memory.
- Separate HBM steady state, HBM transient allocations, host optimizer state, train↔rollout backup, and checkpoint serialization.
- Qwen3.5/3.6 large models here have two KV query groups. Use TP2; TP4/TP8 violates `num_query_groups % TP == 0` on the actor.
- Check dense `world % (TP×PP×CP) == 0` and expert `world % (ETP×EP×PP) == 0` grids independently.
- Prefer PP to CP under host-memory pressure. Two 27B CP2 attempts replicated optimizer state and were killed near the 4-card worker's 518-GiB cgroup.
- Disable overlap until headroom is demonstrated. Communication/staging buffers can turn a near-fit into an OOM.

## 27B

### 4×H200, about 530 GB host

Measured SFT baseline:

- TP2 / PP2 / CP1 / EP1.
- Full recompute, log-prob chunk 1024.
- CPUAdam optimizer offload 40%.
- A 47,499-token step completed in about 48 s at about 988 tokens/s; peak roughly 109 GiB/GPU.

Measured colocated ToolRL capacity point:

- Keep actor resident: `--no-offload-train --offload-rollout`.
- Optimizer offload 70%, SGLang static fraction 0.18.
- 40%/0.35 and 60%/0.25 ran out of HBM during restore; 80%/0.18 exhausted host memory during CPUAdam.
- Online GAD with a colocated discriminator exceeded 507/518 GiB and was killed. Require at least 600 GiB host and rerun the complete GAD gate.

### 8×H200, about 1 TiB host

Throughput SFT gate:

- TP2 / PP2 / DP2, first/last pipeline stages 34/30 layers.
- No CPU offload; FP32 master/grad and BF16 Adam moments.
- Log-prob chunk 256; GBS4 plus balanced data.
- About 177K tokens over two updates reached about 2,369 tokens/s; tightest GPU about 138.8 GiB.

Long-corpus production path in `run_qwen3_large_training_serial.sh`:

- TP2 / PP4 / DP1, layer layout 20/16/16/12.
- `MAX_TOKENS_PER_GPU=6144`, recomputed vocab log-probs, chunk 64.
- Disable CPU optimizer offload/overlap; use BF16 moments.
- The PP4 path avoids the final loss rank OOM observed with the faster PP2 topology on extreme sequences.

Use the current launcher as authoritative: later production stability fixes supersede the initial profile table.

## Qwen3.6-35B-A3B

### 4×H200

- TP2 / PP2 / CP1 / EP2 / ETP1.
- DeepEP/flex dispatcher was used for the measured 4-card SFT gate.
- Full recompute, log-prob chunk 1024, CPUAdam offload 40%.
- A 47,499-token SFT step completed in about 213 s at about 224 tokens/s; peak roughly 131 GiB/GPU.
- ToolRL/GAD do not inherit this proof. Gate rollout, restore, optimizer, and checkpoint lifecycle separately.

### 8×H200

- TP2 / PP2 / EP2 / DP2; 22/18 pipeline layer split.
- Single-node ordinary all-to-all; no CPU offload; BF16 Adam moments; log-prob chunk 256.
- GBS4 with balanced data completed about 177K tokens at about 1,429 tokens/s; tightest GPU about 135.7 GiB.
- GBS2 without balancing produced long/short DP collective skew: one rank ran NCCL while the others waited. This was not an HBM failure.
- Use TP2/PP4/EP2/DP1 as a long-bucket candidate, then gate it; do not infer RL capacity from SFT.

## Qwen3.5-122B-A10B-FP8 on 8×H200

### Full-parameter SFT only

- Source actor conversion from the official FP8 checkpoint, not the older BF16-derived checkpoint.
- TP2 / PP4 / CP1 / EP2 / ETP1, uniform 12/12/12/12 pipeline layout.
- Full recompute and recomputed vocab log-probs.
- The full-parameter pinned stack needs delayed FP8 recipe, FP16 main params, BF16 main grads, FP8 moments, bounded optimizer groups, and pageable moments-only offload. Blockwise tensors cannot back the pinned FP16 optimizer shard.
- A short real optimizer step passed, but optimizer checkpoint serialization and extreme lengths remain separate gates.
- Full-parameter colocated ToolRL completed short compute only, then actor pause exceeded the 1-TiB host cgroup. Do not call this production-capable.

### Single-node ToolRL/GAD production: LoRA

Use the actual measured launcher `drug_agent/scripts/run_qwen35_122b_lora_rl_serial.sh`:

- Base: completed SFT torch_dist plus its SFT-aligned official-FP8 HF rollout view.
- TP2 / PP4 / EP2 / ETP1, uniform 12 layers per stage; SGLang rollout TP8.
- LoRA rank 32, alpha 64, dropout 0.
- Target `linear_qkv`, `linear_proj`, `*.shared_experts.linear_fc1`, and `*.shared_experts.linear_fc2`; omit GDN/linear-attention modules because online SGLang LoRA does not support them.
- Train adapters in FP32; no optimizer/actor/rollout offload. Keep the actor resident.
- Actor FP8 recipe `blockwise` plus FP8 parameter gather is valid because the optimizer owns only small FP32 adapters; this does not contradict delayed FP8 being required for the full-parameter optimizer path.
- SGLang static fraction 0.25; disable CUDA graph, custom all-reduce, and overlap scheduling.
- Keep BF16/default KV cache for reward fidelity unless calibrated FP8 KV equivalence is proven.
- Prompt/response/context limits 10,240 / 2,048 / 12,288; dynamic packing target 10,240 tokens/GPU.
- Use standard Megatron old-policy PPO/GSPO with `USE_ROLLOUT_LOGPROBS=0` because measured SGLang-versus-Megatron log-prob differences were about 14–15 nats/token.

This LoRA path is the later solution to the earlier full-parameter host-memory failure. Do not collapse the two regimes into one conclusion.

## Tuning priorities

Optimize in this order:

1. Correctness and finite updates.
2. Ability to cover the required sequence-length distribution.
3. HBM and host transient headroom, including save/restore.
4. Reward diversity and held-out task quality.
5. Throughput.

For throughput, first remove unnecessary CPU offload, balance DP tokens, rebalance PP stages, and increase dynamic token packing while preserving headroom. Do not enable overlap, CUDA graphs, FP8 KV, or larger batches without a matching correctness gate.
