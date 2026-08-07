# SFT, ToolRL, GAD, FP8, and LoRA

## Contents

- Method branching
- SFT guidance
- ToolRL guidance
- GAD guidance
- 122B LoRA production settings
- FP8 interpretation and compatibility

## Method branching

Use this experiment graph:

```text
HF/official FP8 -> torch_dist -> SFT
                                  |-> ToolRL
                                  `-> GAD negatives -> discriminator warmup -> online GAD
```

Load ToolRL and GAD from the SFT weights with `--finetune --no-load-optim --no-load-rng`. Do not feed ToolRL weights into GAD unless explicitly designing a different experiment.

## SFT

- Use canonical v2 for current work; retain the historical v1 statement only
  when reproducing an old run. Choose total updates from effective assistant
  tokens and held-out convergence after the corpus is frozen. Do not infer the
  required epochs from a small-looking row count.
- Use assistant/step loss masks and `enable_thinking=false` consistently with data preparation.
- Start with GBS4, 3% warmup, cosine decay, Adam beta1/beta2 0.9/0.95, weight decay selected by the measured launcher, and grad clip 1.
- Evaluate held-out tool, schema, parameter, and task-type strata. Preserve migration audits; exclude rejected rows.
- Gate shortest, median, upper-tail, maximum sequence, and save separately. `max_tokens_per_gpu` does not split a 94K sample.

Measured first-pass LR ranges were 1e-6→1e-7 for 27B, 2e-6→2e-7 for 35B, and 5e-7→5e-8 for early 122B full-parameter probes. The later official-FP8 full SFT required a more conservative schedule after a NaN; consult its actual run configuration rather than blindly restoring the probe LR.

## ToolRL

Maintain batch arithmetic: every rollout creates `RBS × n` samples and the result must be compatible with GBS.

Generic initial resource-aware baseline:

- 27B/35B: RBS2, n4, GBS8, then adjust based on reward variance and memory.
- Temperature 1.0 for the paper-like grouped baseline, clip 0.20/0.28, entropy 0, and no reference KL for the original ToolRL arm.
- Treat cold start and SFT warm start as separate arms if reproducing the paper claim; the production request in this project explicitly used SFT warm start.

Project-specific lessons:

- Identical rewards within every GRPO group yield zero centered advantages by definition. This is not fixed by raising LR.
- The measured 9B v2 profile uses RBS8/n1/GBS8, REINFORCE++ normalized across
  prompts, dense MolClaw reward, temperature 0.8, LR 2e-7→2e-8, and no KL.
  Eight TP1 engines plus a TP4/DP2 resident actor passed three updates and
  post-update generation. Use 16,384 tokens/GPU and SGLang static 0.25.
- For new 9B rollouts, use 8K for ordinary decisions and 16K only for declared
  long `vs`/`pf` terminal decisions, with total context 131K. Treat these as
  generation caps, not SFT truncation or dynamic-packing limits. Record the
  resolved tier and truncated ratio; reject positive credit for incomplete
  length-truncated decisions.
- The 27B production serial path switched ToolRL to RBS8/n1/GBS8 with REINFORCE++ normalization across prompts because strict four-sample groups repeatedly had identical rewards. Its later stability settings are LR 1e-8, 5% warmup, temperature 0.7, and no reference switching by default after an earlier collapse. Use the launcher as source of truth.
- Inspect parse success, reward component distribution, nonzero gradients, clip fraction, KL/log-prob diagnostics, response truncation, and held-out tool accuracy together.
- Hyperparameters cannot manufacture missing stopping, schema-correction or
  failure-recovery supervision. The old 9B model's long loops require v2 data,
  hard-state sampling and explicit low-weight repetition/length/protocol
  constraints; do not respond by increasing LR or context alone.

## GAD

Use three explicit stages:

1. Generate aligned negatives from the SFT generator.
2. Warm the Bradley–Terry discriminator for a complete pass and save both checkpoint and manifest.
3. Start online GAD from the same SFT policy, with a running discriminator service.

The GAD paper uses one epoch of generator and discriminator warmup, group size
8, temperature 0.8, and KL coefficient 0.001. Its discriminator uses
Bradley–Terry preference loss and stays online; the paper reports reward hacking
from a frozen/off-policy discriminator. The project's hybrid reward adds small
format and tool-schema guards to the discriminator score. Resource-constrained
runs can use fewer samples, but must report the deviation.

Do not interpret one successful pair, accuracy 1.0 on one pair, or an all-equal reward group as learning evidence. Track discriminator loss, margin, accuracy, context truncation, response diversity, and generator gradients.

For Qwen3.5-9B on one 8×H200 worker:

- Use Qwen3.5-0.8B as a resident discriminator on one physical GPU. Batch 2,
  max length 8192, LR 1e-6, AdamW weight decay 0.01 and clip 1.0 passed the
  measured warmup gate with about 12.3 GiB peak HBM.
- Use RBS1/n8/GBS8, GSPO, hybrid 0.8/0.1/0.1, KL 0.001, generator LR
  1e-7→1e-8 and a finite nonzero-std resampling cap. This is the quality-first
  arm; RBS2/n4/GBS8 is an explicitly cheaper ablation.
- Apply the same 8K/16K metadata-selected response tiers and 131K context to
  Stage 2 and online GAD. Keep discriminator `max_length` separate: its 8K
  gate left-truncates state while preserving the candidate, and 16K/32K or a
  structured state representation needs its own memory/quality gate.
- Set the Slime router to round-robin for same-prompt n8 sampling. The measured
  token-matched A/B reduced rollout time by 44.6% and full-step time by 35.6%
  versus cache-aware routing. This result is workload-specific; remeasure if
  engine count, prompt lengths or group size changes.
- A centered group can log mean reward approximately zero while carrying useful
  advantages. Inspect the per-sample trajectory rewards, raw discriminator
  scores, service version and actor grad norm before diagnosing zero signal.
- Pinning with `CUDA_VISIBLE_DEVICES=7` renumbers that GPU to logical `cuda:0`;
  pass `cuda` to the service. Do not pass `cuda:7` in that process.

For the final 122B LoRA design, use Qwen3.5-4B as the discriminator, not the generic profile's older 0.8B fallback. The service offloads after requests. Verify the actual resolved model path before starting GAD.

## Final 122B LoRA settings

The measured serial launcher uses:

| Stage | Batch/sampling | Optimizer schedule | Objective |
|---|---|---|---|
| ToolRL | RBS8, n1, GBS8; 398 rollouts | 2e-7→2e-8 cosine, 3% warmup, wd 0.01 | REINFORCE++ normalized; dense MolClaw reward; temp 0.8; no KL |
| GAD | RBS4, n2, GBS8; 787 rollouts | 1e-7→1e-8 cosine, 3% warmup, wd 0.01 | GSPO; dynamic nonzero-std filter; hybrid 0.8/0.1/0.1; KL 0.001; temp 0.8 |

Both stages branch from SFT. GAD first produces negatives and warms the 4B discriminator. The serial script uses completion markers so it can resume stages.

Critical LoRA mechanics:

- Export and hot-reload only adapter weights; the gate expects 384 adapter tensors and no GDN adapter tensors.
- Evaluate the GAD reference policy by temporarily disabling adapters on the live base, not by loading/backing up a second 122B checkpoint.
- Set `USE_ROLLOUT_LOGPROBS=0`. SGLang and Megatron differed by about 15 nats/token even after FP8/BF16/temperature alignment, so direct rollout behavior log-probs caused incorrect first-step clipping. Recompute the frozen old policy in Megatron so the initial ratio is one.
- Keep CUDA graphs disabled because Qwen3.5 GDN/Mamba state became stale after in-place weight updates in earlier runs.

## FP8 interpretation and compatibility

Name every precision domain separately:

- checkpoint weights
- Megatron forward/backward recipe
- parameter gather/synchronization
- master parameters and gradients
- Adam moments
- SGLang rollout weights
- KV cache

The official `Qwen3.5-122B-A10B-FP8` checkpoint is the model source. Mentioning BF16 later does not mean silently replacing it with a BF16 model: the final LoRA run kept BF16/default KV cache for fidelity, while actor/rollout weights remained the official FP8 lineage.

Known compatibility boundary:

- Full-parameter pinned optimizer: use delayed FP8; blockwise cannot back its FP16 optimizer shard.
- LoRA optimizer: use blockwise FP8 actor plus FP8 param gather; only FP32 adapters are optimized, and this path was faster in the real gate.

The local FP8 converter was patched to expand Qwen3.5 fused expert tensors and write `weight_scale_inv` correctly. Revalidate converter/bridge tests after dependency upgrades.
