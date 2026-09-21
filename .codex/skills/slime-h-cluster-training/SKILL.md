---
name: slime-h-cluster-training
description: "Design, gate, and diagnose Slime/Megatron/SGLang training runs for the drug-pipe project on PJLab H-cluster H200 workers. Covers parallelism topology (TP/PP/CP/EP/DP), HBM and host-memory sizing, long-context batching, HF-to-torch_dist conversion, SFT/ToolRL/GAD/LoRA method selection, FP8 interpretation, and root-causing training failures (OOM, NCCL, numerical, reward, checkpoint lifecycle) for Qwen3.5/3.6 9B, 27B, 35B-A3B and 122B-A10B-FP8. Use only when the task is about how a training run is configured, validated, or fixed. Do NOT use for composing, validating, or submitting the `rjob`/`rlaunch` command and choosing its resource pool or quota group (use `h-rjob-submit`); producing, cleaning, or auditing trajectory/rollout data (use `manage-drug-pipe-trajectories`); appending a fallback workload to keep a worker busy (use `h-long-sft`); pure evaluation, metric reporting, or dataset statistics with no training-configuration decision; or generic SSH/account/filesystem questions that do not change a training configuration."
---

# Slime H-Cluster Training

Apply the measured project workflow instead of treating model size, active MoE parameters, or a single successful forward as proof that a run is viable.

## Scope and hand-offs

This skill owns **training configuration and training failure diagnosis** only. When a request lands outside it, hand off instead of stretching this skill:

| If the request is about… | Use |
|---|---|
| Composing/validating/submitting an `rjob` or `rlaunch` command, per-replica resources, namespace, **which L1/L2 pool and quota group to occupy**, priority 9, queued-job scheduling diagnosis | `h-rjob-submit` |
| GPU allocation lifetime, foreground entrypoints, "never hold GPUs with `sleep`" | `h-rjob-submit` (authoritative), summarised in §5a below |
| Generating, resuming, monitoring, or auditing Drug-Pipe trajectory data | `manage-drug-pipe-trajectories` |
| Keeping an allocated worker productively busy after the primary command ends | `h-long-sft` |
| Sizing a run that has no GPU allocation yet | decide the pool first via `h-rjob-submit`, then return here |

A request that merely mentions the cluster, GPUs, or "drug-pipe" is **not** a trigger. Trigger on a configuration, sizing, compute-scheduling, or training-failure decision.

## Establish the source of truth

1. Locate the active repository. Prefer `/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/slime` on the login host and `/root/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/slime` inside workers.
2. Treat current launchers, tests, `resolved_config.env`, live logs, and checkpoint markers as newer than dated reports. The working tree contains critical uncommitted Qwen3.5/FP8/LoRA patches; never assume a clean upstream checkout is equivalent.
3. Resolve the current SSH target from the user or `rjob` output. Do not reuse a timestamped pod hostname from this skill.
4. Read [environment-and-data.md](references/environment-and-data.md) for paths, mounts, data contracts, conversion, and safe cluster operations.
5. Read only the method/model references needed for the task:
   - [profiles.md](references/profiles.md) for 4/8-GPU topology and memory profiles.
   - [rl-and-fp8.md](references/rl-and-fp8.md) for SFT, ToolRL, GAD, FP8, and the 122B LoRA path.
   - [failures-and-gates.md](references/failures-and-gates.md) for failure diagnosis and promotion gates.

When framework, model, or CUDA/SGLang/Megatron versions differ, browse current primary sources before reusing a low-precision or parallelism assumption. Prefer Slime, Megatron-Core, Transformer Engine, SGLang, and Qwen official documentation and the ToolRL/GAD papers.

## Follow the execution workflow

### 1. Inventory before changing state

- Inspect GPU count/type/HBM, host cgroup memory, disk space, mounts, environment versions, active Ray jobs, GPU processes, tmux panes, and dirty repository changes.
- Run the repository preflight and static profile validator before conversion or training.
- Inspect dataset counts and token-length quantiles with the actual tokenizer and chat-template settings. Do not infer workload from record count.
- Preserve unrelated processes and changes. Never run `ray stop --force`, kill a tmux session, or restart Ray until active submissions are identified and confirmed in scope.

### 2. Select the training regime explicitly

- Distinguish dense from MoE, total from active parameters, full-parameter from LoRA, train-only SFT from colocated online RL, and official FP8 weights from BF16 compute/KV tensors.
- For 9B/27B/35B, begin from the measured profile matching the exact H200
  count, then gate the actual length buckets and method. Do not assume the
  smaller 9B prefers TP1: its large vocabulary made TP4 the measured winner.
- For 122B on one 8×H200/1-TiB worker:
  - Use full-parameter training only for the already gated SFT path.
  - Use the official `Qwen/Qwen3.5-122B-A10B-FP8` lineage and LoRA for single-node ToolRL/GAD.
  - Do not retry the failed full-parameter colocated RL lifecycle by weakening Ray's OOM threshold.

### 3. Convert and prove the checkpoint

- Convert HF to torch distributed format into an empty, distinct directory.
- Verify every indexed safetensor shard before conversion; verify `latest_checkpointed_iteration.txt == release` and `.distcp` files afterward.
- Prove loading under the intended TP/PP/EP topology and complete a real forward/backward/optimizer step. A tracker file alone is insufficient.
- Keep actor and rollout lineage aligned. For the 122B production LoRA path, use the official FP8-derived SFT torch_dist actor and its SFT-aligned FP8 HF rollout view.

### 4. Promote through gates

Do not launch a full epoch merely because weights load or one short batch fits. Separate steady-state compute, train↔rollout transition, adapter/full-weight synchronization, and checkpoint serialization gates.

Treat every checkpoint produced by a smoke test, gate, probe, dry run, or other
non-production test as temporary. Write it under an explicitly scoped test run
directory; after the test finishes and required metrics/logs have been preserved,
delete all of its checkpoint files and verify that the storage was reclaimed. Never
apply this cleanup to production/resume checkpoints, source checkpoints, or reusable
HF-to-`torch_dist` conversions; resolve and validate the exact test path before
deleting anything.

### 5. Launch resumably

- Use a new timestamped run root and a dedicated tmux window.
- Save the resolved configuration, source model/checkpoint paths, dataset paths/counts, exact command, code revision/diff status, Ray job ID, and stage markers.
- Make SFT, ToolRL, and GAD resumable but keep their policy branches correct: ToolRL and GAD both start from SFT; GAD does not continue from ToolRL.
- Start production only after the matching gate has validated nonzero learning signal, weight synchronization, and memory headroom.

### 5a. Bind GPU allocation lifetime to the workload

The allocation is submitted by `h-rjob-submit`; the training design must not defeat it. Two rules are non-negotiable and are stated in full in `h-rjob-submit/references/resource-pools.md`:

- **Never hold GPUs with an idle command.** `sleep`, `tail -f`, an interactive shell, an infinite no-op loop, or a detached `tmux` is not a valid entrypoint. The allocation must be owned by the real bounded driver and released when that driver exits.
- **A finished or failed workload must release its allocation.** Do not keep a worker warm "for possible future work", and do not queue the next experiment by idling the current one — use `h-long-sft` inside the same allocation, or release it and submit fresh.

Before designing a run, resolve which pool and quota group it will occupy and how many GPUs it asks for (see `h-rjob-submit/references/resource-pools.md`). A run whose topology needs more GPUs than the chosen pool can provide must be re-scoped rather than silently submitted to the wrong pool.

### 6. Diagnose before fixing

- Capture the current Ray submission segment, GPU/host memory, process tree, latest metrics, and the first genuine traceback outside logged training samples.
- Classify the failure as HBM capacity, host/cgroup capacity, allocator/transient peak, collective imbalance, version incompatibility, stale rollout weights, numerical instability, data/reward failure, or checkpoint lifecycle failure.
- Change one causal dimension at a time and rerun the smallest gate that reproduces the issue.
- Never label a process healthy only because GPUs are allocated. Require fresh logs, advancing steps, finite metrics, nonzero useful gradients, reward diversity, and correct stage/checkpoint state.

Use the bundled read-only snapshot on a worker:

```bash
bash scripts/health_snapshot.sh RUN_ROOT [RAY_JOB_ID]
```

From the login host, stream it to a worker without installing files there:

```bash
ssh -CAXY WORKER 'bash -s -- /root/path/to/RUN_ROOT RAY_JOB_ID' \
  < /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/.codex/skills/slime-h-cluster-training/scripts/health_snapshot.sh
```

Stop only monitoring processes when monitoring is no longer requested; leave training, Ray, and stage services running.

## Preserve key invariants

- Keep `num_query_groups % TP == 0`. Read it from the exact checkpoint:
  Qwen3.5-9B has four groups and measured best at TP4, while the larger
  27B/35B/122B profiles here have two and are capped at TP2.
- Check both dense and expert grid divisibility. Prefer PP over CP when optimizer memory is the bottleneck because CP replicates optimizer state.
- Treat `max_tokens_per_gpu` as a dynamic packing target, not automatic truncation of an oversized sample.
- Balance long samples across DP ranks and account for the LM-head/loss pipeline stage separately.
- Keep Ray memory protection enabled. Capacity that requires disabling it is not a passed configuration.
- Interpret FP8 precisely: an FP8 checkpoint does not imply every compute path, KV cache, gradient, or optimizer tensor is FP8.
- Exclude logged rollout sample bodies and server argument dumps from naive error scans; they can contain literal failure words or `nan_detection=False` as data/config text.
