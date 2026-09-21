# Drug-Pipe Codex skills — map and routing rules

Four project-scoped skills live in this directory. They are deliberately narrow: each owns one job, and each declares what it does **not** cover so that the router does not have to guess.

Read this file first when a request could plausibly match more than one skill.

## Ownership map

| Skill | Owns | Triggers on | Explicitly does NOT cover |
|---|---|---|---|
| `slime-h-cluster-training` | Training **configuration and diagnosis**: parallelism, memory/long-context sizing, HF→`torch_dist` conversion, SFT/ToolRL/GAD/LoRA method choice, FP8 semantics, gate promotion, OOM/NCCL/numerical/reward/checkpoint failure triage | "怎么配 TP/PP", "显存不够怎么办", "这个 loss/reward 异常", "转换 checkpoint", "该用 LoRA 还是全参", gate 设计 | the `rjob` command itself; pool/quota selection; allocation lifetime; trajectory data; fallback-SFT placement; pure evaluation or dataset statistics |
| `h-rjob-submit` | **GPU allocation**: pool and quota group choice (L1/L2), namespace, image, mounts, per-replica resources, priority 9, single/multi-node flags, foreground entrypoint, allocation lifetime, queued-job diagnosis | "起个 N 卡任务", "提交 rjob", "选哪个池子", "为什么一直排队", "优先级对不对", "别用 sleep 占卡" | training hyper-parameters and topology; what the workload computes; trajectory pipeline; fallback SFT |
| `h-long-sft` | **Worker utilisation**: serialise a continuous checkpoint-free Qwen3.5-9B SFT fallback behind a bounded primary command inside one allocation | "别浪费卡", "训练完接着跑", "append a backfill SFT", explicit keep-busy intent | submitting/sizing the allocation; anything when no keep-busy intent was expressed; keeping an allocation alive after its bounded work finished |
| `manage-drug-pipe-trajectories` | **Trajectory data**: Tool-KG sampling, schema conversion, Data-Pipe raw rollouts, Python/LLM cleaning, Claude/Data-Pipe concurrency, cc-switch provider preservation, stuck/duplicate controllers, raw+clean audits | rollouts, sessions, clean manifests, provider/rate-limit failures, duplicate attempts, checksum issues, MolClaw failure audits | GPU allocation; training configuration or training failures; GPU-count-based concurrency |

## Routing decision

```
Is the request about the GPU allocation itself
(command, pool/quota group, resources, priority, lifetime, scheduling)?   -> h-rjob-submit
        |
        no
        v
Is it about trajectory/rollout data production, cleaning, or audit?        -> manage-drug-pipe-trajectories
        |
        no
        v
Is it about training configuration, sizing, gates, or training failure?    -> slime-h-cluster-training
        |
        no
        v
Were you explicitly asked to keep an allocated worker busy?                -> h-long-sft
        |
        no
        v
None of the four. Do not load one "just in case".
```

Two cross-cutting notes:

- **Pool first, training second.** A request to size and launch a run that has no allocation yet needs `h-rjob-submit` to fix the pool and GPU count, then `slime-h-cluster-training` for the configuration. Say so explicitly rather than silently doing both under one skill.
- **Mentioning the cluster is not a trigger.** Words like "GPU", "H-cluster", "drug-pipe", "worker", "monitor", "preflight" appear in all four domains. Trigger on the *decision being made*, not on the vocabulary.

## Shared invariants and their single owner

Avoid duplicating authoritative rules. Each of these is owned in exactly one place; other skills cross-reference it.

| Invariant | Owner | Others |
|---|---|---|
| `--priority=9` on every normal RJob, verified from the created object's annotation | `h-rjob-submit` | `slime-h-cluster-training` no longer restates the mechanics |
| Pool/quota-group selection and the L1/L2 policy | `h-rjob-submit/references/resource-pools.md` | all other skills point here |
| Never hold GPUs with `sleep`/tmux/keepalive; a finished workload releases its allocation | `h-rjob-submit/references/resource-pools.md` | `slime-h-cluster-training` §5a summarises it |
| Measured topology/memory profiles per GPU count | `slime-h-cluster-training/references/profiles.md` | `h-rjob-submit` consumes the GPU count only |
| Provider must never be switched automatically | `manage-drug-pipe-trajectories` | — |

## Global vs project skills

- Project skills live here: `<repo>/.codex/skills/`.
- A global directory also exists at `~/.codex/skills/`. As of 2026-09-21 the duplicate global `slime-h-cluster-training` was **retired** (backed up under `~/.codex/skills_backup_<timestamp>/`) because it had diverged from this copy and its description also claimed "submit", which made it a second trigger source. Its unique assets — the `submit_normal_rjob.sh` helper and the §5a allocation-lifetime rules — were merged into `h-rjob-submit` here.
- **Rule: a skill exists in exactly one place.** If something must be global, make it a thin pointer, never a second divergent copy.
- `h-long-sft/` is listed in the repository `.gitignore`, so its edits are local-only and will not appear in `git status`. Treat it as an untracked local skill.

## Maintenance rules when editing these skills

1. **Keep `description` decisions-shaped, not keyword-shaped.** Enumerate the *decision* the skill makes, then an explicit `Do NOT use for:` clause naming the sibling skill that owns each excluded case. A long bare list of nouns is what caused `slime-h-cluster-training` to fire on nearly every request.
2. **Do not restate an invariant owned elsewhere.** Cross-reference the owner and keep a one-line summary at most.
3. **Negative scope must name the sibling skill.** "Do not use for data work" is weak; "…for trajectory data (use `manage-drug-pipe-trajectories`)" is actionable.
4. **One skill per job.** Adding a fifth skill is usually the wrong answer to an over-triggering problem — narrowing the description is. Splitting increases the number of routers and therefore the number of collisions.
5. **After any change, re-read `description` and check it against the other three.** Then confirm the change in a fresh session, because skill routing is what the model sees before it can read this file.
6. **Keep the skill map above in sync** — this file is the only place that states the whole picture.
