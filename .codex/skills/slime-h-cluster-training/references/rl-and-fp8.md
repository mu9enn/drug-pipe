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

- Train the 364 canonical trajectories for one initial epoch. They comprise about 8.7M tokens; do not multiply epochs because the row count looks small.
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
- The 27B production serial path switched ToolRL to RBS8/n1/GBS8 with REINFORCE++ normalization across prompts because strict four-sample groups repeatedly had identical rewards. Its later stability settings are LR 1e-8, 5% warmup, temperature 0.7, and no reference switching by default after an earlier collapse. Use the launcher as source of truth.
- Inspect parse success, reward component distribution, nonzero gradients, clip fraction, KL/log-prob diagnostics, response truncation, and held-out tool accuracy together.

## GAD

Use three explicit stages:

1. Generate aligned negatives from the SFT generator.
2. Warm the Bradley–Terry discriminator for a complete pass and save both checkpoint and manifest.
3. Start online GAD from the same SFT policy, with a running discriminator service.

The paper-like starting point is temperature 0.8, grouped sampling, low-variance KL coefficient 0.001, and discriminator/format/tool hybrid reward. Resource-constrained runs can use fewer samples, but must report the deviation.

Do not interpret one successful pair, accuracy 1.0 on one pair, or an all-equal reward group as learning evidence. Track discriminator loss, margin, accuracy, context truncation, response diversity, and generator gradients.

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
