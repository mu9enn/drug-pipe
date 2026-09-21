---
name: h-rjob-submit
description: Build, validate, submit, and inspect `rjob`/`rlaunch` GPU allocations on PJLab H-cluster for the drug-pipe account — choosing the correct resource pool and quota group (L1 `ailab-agenttool` / L2 `ailab-ma4agismall`), namespace, image, mounts, per-replica GPU/CPU/memory, priority 9, single- vs multi-node flags, and a foreground entrypoint that releases the allocation when the workload ends. Use whenever an actual `rjob submit` or `rlaunch` command must be composed, reviewed, executed, or debugged — including "起个 N 卡任务", pool/quota-group selection, verifying the submitted priority, and diagnosing a queued or unschedulable job. Do NOT use for designing the training configuration, parallelism, or memory profile (use `slime-h-cluster-training`); for producing, resuming, or auditing trajectory data (use `manage-drug-pipe-trajectories`); for appending a fallback SFT workload inside an existing allocation (use `h-long-sft`); or for workloads that never need a GPU allocation.
---

# H-Cluster RJob Submit

Queue a reproducible workload whose RJob entrypoint is the real foreground task. Treat resource allocation and workload launch as one operation: a queued job must be ready to start useful work without a later SSH login.

## Scope and hand-offs

This skill owns **how a GPU allocation is requested and how long it lives**. It does not own what the workload computes.

| If the request is about… | Use |
|---|---|
| Which parallelism, batch, LR, precision, or memory profile the job should run | `slime-h-cluster-training` |
| What the trajectory-production pipeline should execute | `manage-drug-pipe-trajectories` |
| Keeping the worker busy after the primary command exits | `h-long-sft` |
| Whether an allocation can be released early | **this skill** |

Read [references/resource-pools.md](references/resource-pools.md) before every submission: it is the authoritative L1/L2 pool policy and holds the verified per-pool parameter set. Read [references/local-profiles.md](references/local-profiles.md) for the full field defaults and topology rules, and read [references/submission-template.md](references/submission-template.md) when rendering, checking, submitting, or monitoring a command.

## Kill the allocation when the work ends

**This is the single most expensive failure mode in this project.** An allocation that outlives its workload blocks other users, wastes a full node, and is hard to detect after the fact.

- The container command must remain in the foreground until the workload succeeds or fails. Use `exec` for the final process.
- **Never** use `sleep`, an interactive `bash`, `tail -f`, an infinite no-op loop, a dummy service, or a detached `tmux`/`nohup` to hold GPUs. This is prohibited even with a nominal timeout, and even "briefly" while preparing a real command.
- A launcher that starts `tmux`, `nohup`, or a background process and then exits is **not** a valid RJob entrypoint.
- If a bounded job has already finished, stop the allocation and verify its replica is no longer active. Never leave idle GPUs allocated "in case we need them".
- If a worker must stay productively busy after the primary command, use `h-long-sft` **inside the same foreground driver**, so the allocation still dies with the last real workload.
- Use a login-host tmux window only to own a foreground `rjob submit`/monitoring call. Never use worker-side tmux to detach the workload from its entrypoint.

## Resolve the job before submitting

1. Determine whether the user wants only a command/review or an actual submission. Running `rjob submit` consumes shared cluster resources; execute it only when the user has authorized submission in the current request. Never submit while merely creating or editing this skill.
2. **Resolve the target pool and quota group first** (see [references/resource-pools.md](references/resource-pools.md)). The cluster now has two GPU pools with different quotas; picking the wrong one either fails to schedule or wastes the scarce pool. State the chosen pool and the GPU count it implies before rendering the command.
3. Resolve the exact workload entrypoint from current repository files or a user-provided command. Prefer a versioned launcher on the mounted GPFS path. Do not invent a script path or training arguments.
4. Resolve the job name, image, namespace, charged group, mounts, replicas, GPUs/CPU/memory per replica, and any required environment variables. Read [local-profiles.md](references/local-profiles.md) for the known drug-pipe defaults.
5. Read [submission-template.md](references/submission-template.md) when rendering, checking, submitting, or monitoring a command.

If no concrete executable workload is available, stop at a clearly marked command template and identify what is missing. Do not replace the missing workload with an idle command.

## Preserve these submission invariants

- **Every normal RJob must explicitly pass `--priority=9`.** This applies to all
  GPU counts, probes, retries, clones and custom Python/shell submission paths.
  CLI omission defaults to 5 and is a pre-submit failure. Inspect the final
  executed arguments; a correct reference template alone is insufficient.
  Only an explicit user instruction overrides 9; never switch to idle mode to
  bypass it. After creation, read the exact job's
  `metadata.annotations["volcano.brainpp.cn/priority"]` and require `"9"` before
  declaring success; save the observed value and UID. Missing/different values
  require an explicit failure report and an authorized correction, not silent acceptance.
  Prefer the helper, which injects the priority and rejects a second priority or
  task-type flag:
  `bash <this-skill>/scripts/submit_normal_rjob.sh ...`

- Use `bash -lc 'set -euo pipefail; ...; exec ...'` when environment initialization or a working-directory change is needed. Keep secrets out of `-x` traces and command-line arguments.
- Use shared, container-visible paths below `/root/slime_sxy/group-space/...`; do not submit a launcher that exists only under the login host's `/home/...` path.
- Keep job names lowercase and use only letters, digits, and hyphens. Avoid underscores and uppercase letters.
- Interpret `--gpu`, `--cpu`, and `--memory` as resources for each replica. `--memory` is MiB on the documented/current CLI; verify with `rjob submit --help` if the installed version is uncertain.
- Use `-P 1` for a single node, including one 8-GPU node. For multi-node jobs, `-P` is the node/replica count and resources remain per replica.
- `--enable-sshd` may be retained for diagnosis, but SSH is never a substitute for a real entrypoint.
- Do not add `--delete`, delete a same-named job, change priority away from the required 9, select idle/preemptible mode, or add node tags merely to make scheduling succeed unless the user explicitly requests that change.
- Do not repeatedly resubmit a queued job. A successfully created Pending/Queued RJob is a successful submission, not a reason to create duplicates.

## Validate at the right depth

Before an authorized submission:

- Source `/etc/profile.d/ssh-init.sh` on the login/development host and confirm `rjob` is available.
- Inspect `rjob submit --help` for any uncertain or version-dependent flag. Do not silently upgrade `brainpp` or edit pip configuration unless the user asks.
- Confirm the host-side launcher and required config files exist on the mounted GPFS storage. Run focused static checks such as `bash -n` for a shell launcher when safe.
- Check that the command performs the intended real task, uses the expected container paths, does not daemonize, and propagates a nonzero exit code.
- Confirm the requested GPU count is actually available in the chosen pool. Aggregate cluster free resources do not prove that the chosen pool has a node that fits the replica.
- For multi-node training, confirm the launcher consumes the platform-injected distributed variables and that communication flags match the topology. Do not overwrite injected NCCL interface/HCA/GID variables.
- Show the fully resolved command to the user before execution when material values remain inferred or when the request is review-only.

After submission, capture the RJob identifier/output and query it once with an explicit namespace. Report whether it was created, its observed priority, and its current state. Do not wait for GPUs or poll continuously unless the user asks for monitoring.

## Diagnose without changing the request

For a job that stays queued, inspect the RJob state/events and compare the **per-node** free GPU, CPU, and memory against the per-replica request. Aggregate free resources do not prove that any one node can fit a replica. Also check namespace, charged-group permission, private-machine selection, replica/Gang requirements, mounts, image pull errors, and restrictive positive/negative tags. If the preferred pool cannot fit the job, report that and choose between waiting or falling back per the pool policy — do not silently switch pools.

For a job that starts and fails, inspect the job logs and events before proposing changes. The RJob result follows the foreground command's exit status, so preserve the first real workload error rather than replacing the command with a keepalive.
