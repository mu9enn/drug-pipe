# Measured 4/8-GPU profiles

## Contents

- Capacity principles
- Qwen3.5-9B profiles
- 27B profiles
- 35B profiles
- 122B full-parameter and LoRA profiles
- Tuning priorities

## Capacity principles

- Size total trainable parameters, not MoE active parameters. MoE reduces FLOPs per token but all experts still occupy checkpoint, optimizer, and synchronization memory.
- Separate HBM steady state, HBM transient allocations, host optimizer state, train↔rollout backup, and checkpoint serialization.
- Always read `num_key_value_heads`/`num_query_groups` from the exact model. The
  27B/35B/122B profiles here have two KV query groups and therefore use TP2;
  Qwen3.5-9B has four and admits TP4.
- Check dense `world % (TP×PP×CP) == 0` and expert `world % (ETP×EP×PP) == 0` grids independently.
- Prefer PP to CP under host-memory pressure. Two 27B CP2 attempts replicated optimizer state and were killed near the 4-card worker's 518-GiB cgroup.
- Disable overlap until headroom is demonstrated. Communication/staging buffers can turn a near-fit into an OOM.

## Qwen3.5-9B on 8×H200

Measured 2026-08-06 on an 8×143,771-MiB H200 worker with about 1.37 TiB host
memory, Slime plus Megatron, and the 365-record canonical v2 corpus:

- Dense 32-layer model, hidden size 4096, FFN 12288, 16 attention heads and
  four KV query groups. TP4 is valid.
- TP1/PP1/DP8 failed at the 248,320-way LM head even after recomputing vocab
  log-probs and loss with chunk size 64. The contiguous logits allocation
  needed about 6.5 GiB when only 2–3 GiB remained. Do not retry TP1 by merely
  lowering `log-probs-chunk-size`.
- TP2/PP1/DP4 passed an eight-record p50 batch (112,962 tokens) in 109.3 s,
  about 1,033 tokens/s, with about 123.6 GiB used on the measured rank.
- TP4/PP1/DP2 passed the identical p50 batch in 61.4 s, about 1,840 tokens/s,
  with about 119.0 GiB used. Losses agreed (0.49358 vs 0.49352), so TP4 is the
  measured throughput winner rather than a numerically different path.
- With full recompute, recomputed loss/vocab log-probs, chunk size 64,
  balanced dynamic packing and `MAX_TOKENS_PER_GPU=16384`, a p95 pair reached
  about 2,523 tokens/s and the maximum pair (individual lengths up to 94,016)
  reached about 3,176 tokens/s. Both had finite loss/gradients and no HBM or
  host-memory failure. Higher long-batch throughput reflects fuller dynamic
  token packing, not cheaper tokens.
- Complete model/Adam/RNG/data-cursor checkpoint serialization took about
  21 s. A real interruption at iteration 0 resumed at rollout/step 1 and
  successfully saved iteration 1.

Recommended full-parameter SFT profile:

- TP4 / PP1 / CP1 / DP2, BF16 training, distributed Adam, no CPU offload.
- GBS2, dynamic batch, balanced data, full recompute, recomputed loss and
  vocab log-probs, chunk 64, 16,384 tokens/GPU.
- Adam betas 0.9/0.95, weight decay 0.1, LR 5e-6 to 5e-7 cosine and about 5%
  warmup. Derive total updates and decay steps from the final token budget and
  held-out convergence rather than freezing them to the current record count.
  Retain optimizer/RNG for resumability.
- Never drop a non-divisible tail. Use
  `drug_agent/scripts/materialize_batch_aligned_sft.py` to append the minimum
  number of unchanged shortest records required by GBS2, with source/output
  hashes in a manifest. This remains valid as the corpus grows.

Measured colocated ToolRL profile:

- Keep both sides resident: TP4/PP1/DP2 actor plus eight one-GPU SGLang
  engines, `--no-offload-train --no-offload-rollout`. This avoids host copies
  and the Qwen3.5 GDN release/resume correctness path.
- Use `SGLANG_MEM_FRACTION_STATIC=0.25`, CUDA graphs/custom all-reduce/overlap
  scheduling disabled, BF16/default KV, and `MAX_TOKENS_PER_GPU=16384`.
  The one-step stress gate peaked near 92.3/140.1 GiB, leaving about 47.8 GiB.
- Use RBS8, n1, GBS8, REINFORCE++ with normalized advantages across prompts,
  dense MolClaw reward, temperature 0.8 and no reference KL. Strict grouped
  GRPO was rejected for this corpus because the discrete reward frequently
  gives every sample in a group the same value.
- A real three-update gate completed rollout, backward, weight sync and the
  next post-update generations. Actor throughput was about 957, 843 and 592
  tokens/s as length mix changed; all losses/gradients were finite, rollout
  versus train log-prob absolute differences were 0.0068–0.0125, and the first
  three batches had zero repetition and truncation.
- Relative to the 8192-token/0.35 baseline, a comparable first shuffled prompt
  batch at 16,384/0.25 reached 1,068 actor tokens/s (about 11.6% higher) while
  handling a 4096-token stress response. Do not compare end-to-end wall time
  blindly: that stochastic response was correctly flagged as both repeated and
  truncated and dominated rollout latency.
- The validated optimizer point is LR `2e-7`, betas 0.9/0.95 and weight decay
  0.1. For a production pass, use about 3% warmup and cosine decay to `2e-8`,
  checkpoint at a coarse interval, and select by held-out behavior rather than
  training reward alone.
- Use an 8,192-token ordinary rollout cap, a 16,384-token cap only for declared
  long terminal `vs`/`pf` tasks, and a 131,072-token total context. The project
  hook `drug_agent.rollout.length_aware_generate.generate` selects the tier
  from task/decision metadata and never reads gold response length. These new
  caps are code/test validated but still require a fresh long-output GPU gate;
  the worker used for the original 4K gates has expired.

Do not enable CUDA graphs merely for speed: a post-update generation gate is
mandatory for hybrid GDN/Mamba weights. Keep `USE_ROLLOUT_LOGPROBS=0` at the
measured small log-prob disagreement. Monitor repetition, truncation, response
length, per-component reward and held-out schema/recovery tasks; LR changes do
not repair missing stopping/recovery reward coverage.

Measured GAD profile and recommended quality-first production setting:

- Preserve the three branches: Stage 2 negatives from the SFT actor, one full
  discriminator warmup epoch, and online GAD from the same SFT checkpoint.
- Stage 2 generator: the same resident TP4/PP1/DP2 actor and eight TP1 rollout
  engines, RBS8/n1, temperature 0.8, 98,304/4,096/102,400 prompt/response/
  context limits, static fraction 0.25. The historical eight-row gate wrote
  eight unique, complete teacher/student pairs and exited successfully. New
  launches use 114,688 prompt, 8,192 ordinary response, 16,384 declared-long
  response and 131,072 context; promote those caps only after a fresh GPU gate.
- Discriminator: Qwen3.5-0.8B full-backbone BF16 AdamW on physical GPU 7,
  batch 2, max length 8,192, LR `1e-6`, weight decay 0.01, clip grad 1.0,
  one epoch. Four gate updates reduced Bradley–Terry loss from 0.288 to 0.0967
  and raised the teacher/student margin from 1.41 to 2.37; peak standalone HBM
  was about 12.3 GiB. Its checkpoint includes roughly 3.0 GB of optimizer state.
- Online actor: TP4/PP1/DP2, eight TP1 engines, RBS1/n8/GBS8, GSPO, temperature
  0.8, hybrid reward 0.8 discriminator + 0.1 format + 0.1 decision schema,
  low-variance KL 0.001, dynamic nonzero-reward-std filter with a finite drop
  cap, 16,384 tokens/GPU, full recompute and chunk 64.
- Set `ROUTER_POLICY=round_robin` for n8 grouped sampling. On an almost
  token-identical first batch (28,275 versus 28,279 tokens), round-robin used
  all eight engines and cut rollout 61.77→34.24 s and full step
  134.90→86.91 s versus cache-aware routing. It intentionally gives up the
  measured 82.5% shared-prefix hit rate because parallel decode wins here.
- Keep both actor and rollout resident, CUDA graphs/custom all-reduce/overlap
  disabled, and pin the online discriminator with `CUDA_VISIBLE_DEVICES=7`
  plus logical device `cuda`. The combined gate peaked around 114 GiB on the
  shared discriminator rank and retained useful HBM margin.
- The two-update correctness gate advanced discriminator version 4→6, had
  finite actor grad norms 0.98 and 8.33, no repetition/truncation, and survived
  post-update generation. Round-robin's one-step confirmation advanced it to
  version 7 and produced actor throughput about 6,059 tokens/s.
- Use generator LR `1e-7`→`1e-8` cosine, 3% warmup, betas 0.9/0.95 and weight
  decay 0.1 for a production pass. The two-step gate's cosine scheduler reached
  its minimum immediately; that gate validates mechanics, not the full
  schedule. Keep discriminator LR `1e-6`, one online update per group.

RBS1/n8 is the quality-first, paper-aligned setting and consumes eight
generations per v2 state. If rollout budget is the limiting resource, explicitly
label an RBS2/n4/GBS8 efficiency arm; do not silently call it equivalent. A
near-zero logged mean `rollout/rewards` is expected after within-group
centering—require nonconstant raw/hybrid scores and nonzero generator gradients
before declaring the signal dead.

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
