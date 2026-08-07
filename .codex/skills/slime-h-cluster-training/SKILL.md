---
name: slime-h-cluster-training
description: Plan, convert, launch, monitor, and diagnose Slime/Megatron/SGLang training on PJLab H-cluster 4-GPU or 8-GPU workers, especially Qwen3.5/3.6 9B, 27B, 35B-A3B, and 122B-A10B-FP8 SFT, ToolRL, GAD, full-parameter, and LoRA runs. Use for worker preflight, HF-to-torch_dist conversion, parallelism and memory sizing, long-context batching, FP8/LoRA decisions, serial SFT-to-RL workflows, Ray/tmux health checks, OOM/NCCL/numerical/reward diagnosis, and recovery of these drug-agent experiments.
---

# Slime H-Cluster Training

Apply the measured project workflow instead of treating model size, active MoE parameters, or a single successful forward as proof that a run is viable.

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

Advance in order: preflight → conversion → load → shortest step → p50 step → p95/max step → checkpoint-save → one online RL group → multi-update stability → production. Require every stage to pass the criteria in [failures-and-gates.md](references/failures-and-gates.md).

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
  < /home/sunxiangyu/slime_sxy/.codex/skills/slime-h-cluster-training/scripts/health_snapshot.sh
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
