# Failure catalog and promotion gates

## Contents

- Promotion ladder
- Health criteria
- Failure map
- Recovery rules
- Primary references

## Promotion ladder

Require these gates in order:

1. Worker: correct GPU count/model, host cgroup limit, disk, mounts, CUDA/Python stack, no conflicting jobs.
2. Source checkpoint: config/index present and every shard readable.
3. Conversion: atomic output, `release` tracker, nonempty `.distcp` files.
4. Load: target TP/PP/EP topology loads without bridge/key mismatch.
5. Compute: shortest sample completes forward/backward/optimizer with finite loss and grad norm.
6. Length: p50, p95, and maximum buckets pass; record peak per-rank HBM and host use.
7. Save: the intended checkpoint type serializes and reloads without exceeding host/HBM.
8. RL loop: rollout, reward, nonzero advantage, actor update, adapter/full-weight synchronization, rollout restore, and a second update all pass.
9. Method: ToolRL reward components or GAD discriminator metrics show useful variance and held-out quality does not regress.
10. Production: run multiple updates long enough to observe lazy optimizer initialization, memory reuse, sampling variability, and stage transitions.

For 122B LoRA, the real gate additionally required finite loss/grad, low first-step PPO KL/clip fraction under the Megatron old policy, exactly 384 adapter keys, no GDN keys, and successful SGLang adapter reload.

## Health criteria

Call a live stage healthy only when all apply:

- Ray job and the expected actor/rollout/service processes are alive.
- The latest stage log is fresh relative to measured step time and steps advance.
- All expected GPUs retain plausible memory and periodically show compute utilization.
- Loss and grad norm are finite; no runtime traceback, OOM, NCCL timeout, actor death, or SIGKILL exists in the current Ray submission segment.
- RL rewards are not persistently all equal; gradients are not zero for six consecutive updates.
- Response truncation is understood, adapter/full weights reload successfully, and stage/checkpoint markers match reality.
- Host memory and filesystem retain operational headroom.

Raw large grad norms can be valid when grad clipping is enabled; judge finiteness, clipping behavior, loss/quality trajectory, and updates together. `ppo_kl=0` is expected at the start of correctly recomputed old-policy updates and is not alone an error.

## Failure map

| Symptom | Proven or likely cause | Correct response |
|---|---|---|
| SSH fails for an old pod | Timestamped worker expired/replaced | Ask for or discover the current `rjob` pod; never hard-code historical hosts |
| Paths work on host but not worker | `/home/...` vs `/root/...` mount mapping | Resolve through `slime_env.sh`; validate both mounts in preflight |
| HF load fails midway | Missing indexed shard/partial download | Check every `weight_map` shard and byte size before conversion |
| TP4/TP8 actor assertion | Only two KV query groups | Use TP2 and spend remaining ranks on PP/EP/DP |
| Host OOM rises with CP | CP replicates optimizer state | Prefer PP/DP layout; CP is not a free long-context shard |
| One PP rank OOMs in CE/log-prob | LM head/loss stage heavier; vocab tile too large | Rebalance first/last layers and recompute/chunk vocab log-probs |
| 9B TP1 OOMs at the LM-head matmul despite chunk 64 | The 248,320-way logits tensor itself is materialized before the chunked loss path | Use TP2 or the measured TP4; loss/vocab recompute cannot shard an already-materialized TP1 output tensor |
| DP/NCCL appears hung on long data | Unequal sequence work across DP replicas | Increase GBS, enable token balancing, or bucket lengths; do not misdiagnose as memory |
| Half of a DP2 actor goes idle during backward | One replica received the long-tail sequence even after record balancing | Judge the full step and token distribution; increase distinct prompts/GBS or length-bucket future batches before changing parallelism |
| SFT launcher rejects a non-divisible corpus under TP4/DP2 | GBS must divide DP and epoch-mode rollout batches cannot drop a tail | Materialize a hash-audited derivative by appending the minimum unchanged shortest records, or use slower DP1; never silently drop canonical data |
| A 4K RL response cap truncates valid ranked outputs | `rollout-max-response-len` is SGLang `max_new_tokens`, not an SFT limit | Use 8K ordinary / 16K declared-long tiers with 131K context; monitor status and never reward an incomplete truncated decision |
| Raising every rollout to 16K makes loops four times costlier | Model context capacity was confused with a useful per-decision budget | Select the cap from task/decision metadata without reading gold length; retain repetition and truncation guards |
| GAD candidate cap is 16K but discriminator sees only 8K | Actor rollout and discriminator tokenization are independent | Gate discriminator 16K/32K separately or provide a structured compact state; do not infer RM capacity from actor success |
| 122B BF16 rollout loads but has no KV pool | Bare rollout weights consume about 31 GB/GPU | Use the official block-FP8 rollout checkpoint |
| FP8 KV lowers confidence/quality | Uncalibrated KV scales | Keep BF16/default KV unless calibrated equivalence passes |
| Full-param 122B online step computes then actor dies on pause | Whole actor backup plus optimizer exceeds 1-TiB host | Use LoRA/keep actor resident or external engines; do not disable Ray protection |
| CPUAdam estimate says it should fit but cgroup dies | Pinned HybridDeviceOptimizer CPU path uses four FP32 tensors, about 16 bytes/parameter | Budget actual implementation semantics, not requested low-precision dtype labels |
| `BlockwiseQTensor.view`/optimizer shard error | Blockwise FP8 incompatible with pinned full-param FP16 shard | Use delayed FP8 for full-param; blockwise remains valid for LoRA |
| SGLang CUDA invalid argument at TP rollout | Custom all-reduce with shared CUDA-graph inputs | Disable SGLang custom all-reduce; use NCCL |
| Correct pre-update output becomes corrupted after weight sync | CUDA graph/GDN state stale after in-place update | Disable CUDA graphs and validate post-update generation |
| First PPO update clips almost every token; log-prob diff ~15 | SGLang and Megatron log-probs are not numerically aligned | Set `USE_ROLLOUT_LOGPROBS=0` and recompute old policy in Megatron |
| LoRA reference policy duplicates 122B host state | Loading/backing up a full reference checkpoint | Disable adapters temporarily to evaluate the frozen base |
| LoRA hot reload misses modules or SGLang rejects adapter | Unsupported GDN targets or inconsistent export | Target QKV/proj/shared experts only; require 384-key/no-GDN gate |
| GRPO has zero advantage/gradient | Every response in a group received the same reward | Fix parsing/reward/discriminator or sampling; use a justified estimator/batch redesign, not a higher LR |
| GAD service silently uses weak/wrong model | Generic 0.8B fallback inherited | Pin and record the intended discriminator; final 122B path used Qwen3.5-4B |
| Error scanner reports traceback from a sample | Training trajectory contains literal failure text | Exclude full rollout-sample records; scan only current runtime segment |
| Scanner reports NaN because server args contain `nan_detection=False` | Overbroad regex | Match assignment-shaped metric values, not arbitrary `nan` substrings |
| First training step fits but save dies | Serialization has a separate memory transient | Add a dedicated save/reload gate before long training |
| Resume asserts scheduler total iterations differ | Initial and resumed commands declared different total rollout/train or LR/weight-decay schedule lengths | Declare the full plan on the first run; interrupt after a saved step and resume with identical `num_rollout`, decay iters, batch and scheduler settings |
| `ray job status` cannot reach GCS immediately after a short gate | The launcher completed and intentionally tore Ray down | Read the run's driver/Ray-submit tail and checkpoint marker before diagnosing a crash; a missing post-run GCS is not itself failure |
| Restart fixes one job but kills another | Unscoped `ray stop --force` | List Ray submissions/processes and obtain scope before restarting |
| Two resident RL copies appear after a tmux launch | Nested quoting expanded an empty run variable or launched twice | Resolve the complete remote command first, then check tmux, Ray submissions, owner PIDs and all GPU process maps before any update; stop both cleanly if ownership is ambiguous |
| SGLang health probes fail during startup but engines later become ready | Eight engines load/tokenize asynchronously and the router probes early | Treat as transient only while logs and model loading advance; require all engines healthy before rollout, and treat a repeated timeout after the startup budget as fatal |
| `cuda:7` is invalid for a pinned discriminator | `CUDA_VISIBLE_DEVICES=7` remaps physical GPU 7 to logical device 0 | Use `CUDA_VISIBLE_DEVICES=7` with device `cuda`, or expose all GPUs and explicitly use `cuda:7`; never combine both numbering schemes |
| GAD discriminator shows a raw grad norm in the hundreds | The logged value is pre-clip and long Bradley–Terry pairs amplify it | Require finite loss/margin and keep clip-grad 1.0; verify several updates instead of treating one large pre-clip value as divergence |
| Dynamic sampling never returns a GAD group | Discriminator/rule rewards have zero within-group variance | Inspect raw and normalized discriminator scores, service version and pair diversity; cap dropped groups and do not hide an uninformative reward behind endless resampling |
| A launcher prints a mangled command fragment only after its Ray job succeeds | The shared shell script was edited in place while bash was blocked inside a long child command, so its later read offset no longer matched the rewritten bytes | Never patch a launcher while any shell is executing that file; copy/version the next launcher first. Verify the Ray result separately because the completed child job may still be valid |

## Recovery rules

1. Preserve the failed run root and logs; create a new retry tag.
2. Identify the first causal error in the latest Ray submission segment.
3. Record GPU/host memory immediately before failure when available.
4. Form one hypothesis and change the smallest relevant control.
5. Rerun the smallest reproducer, then the next gate; do not jump directly to full training.
6. Add a regression test or fail-closed launcher check for every project-code fix.
7. Mark a stage complete only when its expected artifact exists and reloads.

Important local tests:

```bash
cd /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/slime
python -m pytest -q drug_agent/tests/test_large_model_profiles.py
MODEL_PROFILE=qwen35-122b-8xh200 \
  VALIDATE_LARGE_PROFILE_DATA=0 \
  bash drug_agent/scripts/validate_qwen3_large_profile.sh
```

The generic 122B validator intentionally rejects blockwise FP8 for the full-parameter profile. The dedicated LoRA launcher overrides that setting for a different optimizer regime; this is expected.

## Primary references

- Slime quick start and HF→torch_dist: <https://thudm.github.io/slime/get_started/quick_start.html>
- Slime low precision: <https://thudm.github.io/slime/advanced/low-precision.html>
- Megatron-Core distributed optimizer: <https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html>
- Megatron-Core CPU offload: <https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/optimizer_cpu_offload.html>
- Megatron-Core MoE: <https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html>
- Qwen3.5-122B-A10B-FP8: <https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8>
- ToolRL paper: <https://arxiv.org/html/2504.13958>
- GAD paper: <https://arxiv.org/html/2511.10643>
