# H-cluster GPU resource pools and quota groups

Authoritative policy for choosing **which pool every `rjob`/`rlaunch` GPU allocation must occupy** for the drug-pipe account, plus the allocation-lifetime rules. Re-read this before each submission.

## Contents

- The two pools
- Selection policy
- Verified per-pool submission parameters
- Pre-submit occupancy checks
- Allocation lifetime rules
- Evidence log
- Caveats

## The two pools

The account can now charge GPU jobs to two pools, referred to internally as **L1** and **L2**.

| Alias | Project / namespace | Charged quota group | GPU capacity (2026-09-21) | Public registry path |
|---|---|---|---|---|
| **L1** | `ailab-agenttool` | `agenttool_pool` | 16 GPUs — 2 × 8×H200 nodes | `registry.h.pjlab.org.cn/ailab-agenttool-agenttool_pool/…` |
| **L2** | `ailab-ma4agismall` | `ma4agismall_gpu` | 8 GPUs — 1 × 8×H200 node | `registry.h.pjlab.org.cn/ailab-ma4agismall-ma4agismall_gpu/…` |

The naming convention for both is `ailab-<project>-<quotagroup>`. Use it to sanity-check any new pair before submitting.

`L1`/`L2` are team shorthand, not CLI flags. They map to `--namespace` + `--charged-group`; there is no `--level` or `--pool` option on this CLI.

## Selection policy

Decide from the **total GPU count** the job needs (per replica × replicas), *before* rendering the command.

| GPUs needed | First choice | If it cannot schedule |
|---|---|---|
| `< 8` (a partial node) | **L1** `ailab-agenttool` / `agenttool_pool` | **L2** `ailab-ma4agismall` / `ma4agismall_gpu` |
| `= 8` (one full node) | **L2** `ailab-ma4agismall` / `ma4agismall_gpu` | **L1** `ailab-agenttool` / `agenttool_pool` |
| `> 8` (multi-node) | **L1** `ailab-agenttool` / `agenttool_pool` — L2 physically cannot fit | none; wait or ask the user |

Rationale, so the rule survives contact with edge cases:

- L2 is the small dedicated 8-card quota. A full-node job belongs there if it fits, because that keeps L1's 16 cards available for jobs that can only run there.
- Small jobs prefer L1 so they do not fragment the dedicated 8-card node.
- L2 has a hard ceiling of 8 GPUs, so a `> 8` request is not a preference question.

Rules around the choice:

- **State the chosen pool, the exact GPU count, and the reason** in the submission summary. Do not leave the pool implicit.
- **Never silently switch pools** to make scheduling succeed. If the first choice cannot schedule, report that and either wait or take the documented fallback; if the fallback also fails, ask.
- **Never relax the request** (fewer GPUs, different topology, `--preemptible`, extra tags) to force a fit unless the user asks.
- Pool occupancy is volatile. Re-check it rather than trusting the numbers in this file.

## Verified per-pool submission parameters

Every field below was confirmed working. `L2` values are the long-standing defaults from [local-profiles.md](local-profiles.md); `L1` values were verified by a live probe on 2026-09-21 (see Evidence log).

| Field | L1 (`ailab-agenttool`) | L2 (`ailab-ma4agismall`) |
|---|---|---|
| `--namespace` | `ailab-agenttool` | `ailab-ma4agismall` |
| `--charged-group` | `agenttool_pool` | `ma4agismall_gpu` |
| `--private-machine` | `group` | `group` |
| `--priority` | `9` (mandatory) | `9` (mandatory) |
| `--image` | `registry.h.pjlab.org.cn/ailab-ma4agismall-ma4agismall_gpu/slime-sxy:slime0529` (verified pullable in the L1 namespace) | `registry.h.pjlab.org.cn/ailab-ma4agismall-ma4agismall_gpu/slime-sxy:slime0529` |
| `--image-pull-policy` | `IfNotPresent` | `IfNotPresent` |
| User GPFS mount | `gpfs://gpfs1/sdpdev-fs/sunxiangyu:/root/slime_sxy/group-space/sunxiangyu` | same |
| Hugging Face mount | `gpfs://gpfs2/gpfs2-shared-public/huggingface:/root/slime_sxy/group-space/huggingface` | same |
| Distributed env | `-e DISTRIBUTED_JOB=true` | same |
| `NCCL_IB_DISABLE` | `1` for single-node (platform already injects `NCCL_SOCKET_IFNAME=eth0`, empty `NCCL_IB_HCA`) | same |
| `-P` | `1` single node; node count for multi-node | same |

Both pools expose H200 GPUs. Device count and topology are the same per node, so the measured 4/8-GPU training profiles in `slime-h-cluster-training` transfer unchanged between pools.

If an image is later published under the L1 project path, prefer it there; the L2 image is merely *verified to work*, not the canonical L1 image.

## Pre-submit occupancy checks

Check what already occupies the pool, then predict schedulability without mutating anything:

```bash
source /etc/profile.d/ssh-init.sh

# who is already in the pools
brainctl get rjobs -n ailab-agenttool
brainctl get rjobs -n ailab-ma4agismall

# non-mutating prediction for the exact command you are about to run
rjob submit --predict-only=True --dry-run=True \
  --name=<job-name> --namespace=<pool-namespace> --charged-group=<pool-group> \
  ... <remaining flags> -- bash -lc 'set -euo pipefail; exec <real workload>'
```

`--dry-run=True` prints the object it would create and does **not** create it; confirm the printed `metadata.namespace`, `quotagroup.brainpp.cn/quotagroup` label, and `volcano.brainpp.cn/priority` match your intent. `--predict-only=True` asks the platform whether resources exist. Verify flag syntax with `rjob submit --help` on the installed version.

Never use a real GPU submission as a probe unless its command does bounded real work and exits. See the next section.

## Allocation lifetime rules

These are the reason this file exists. Getting them wrong leases a whole node from someone else.

1. **The entrypoint must be the real bounded workload driver.** Shape it as `bash -lc 'set -euo pipefail; cd <dir>; exec <real command>'`, so the allocation dies with the workload and Kubernetes releases the worker automatically.
2. **Never hold GPUs with an idle command.** `sleep`, `tail -f`, an interactive shell, an infinite no-op loop, a dummy service, or a detached `tmux`/`nohup` is not a valid entrypoint — not even for a few minutes while you prepare the real command, and not with a nominal timeout.
3. **A finished or failed workload must release its allocation.** Confirm via `rjob get` and the worker process tree that the allocation is owned by the workload driver, not a keepalive. Stop the job and verify its replica is gone; do not leave idle GPUs allocated "in case we need them".
4. **Probes count as workloads.** A probe must do bounded real work (load the model, get an optimizer step, print a definitively resolved path) and exit. It must never be `sleep`-shaped.
5. **To keep a worker busy legitimately**, serialise real work inside the *same* foreground driver — a bounded test followed by the production run, or the `h-long-sft` fallback. The allocation then still ends with the last real command.
6. **Use login-host tmux only for foreground `rjob submit`/monitoring.** Do not use worker-side tmux to detach the workload from its entrypoint.
7. If you find an existing job holding GPUs with a keepalive, check its actual child workload first. Once the child has completed or failed, stop the rjob and verify the replica is inactive.

## Evidence log

**2026-09-21 — L1 (`ailab-agenttool` / `agenttool_pool`) parameters verified.**

```bash
rjob submit --priority=9 --name=skilltest-agenttool-params-0921a \
  --namespace=ailab-agenttool --charged-group=agenttool_pool --private-machine=group \
  --image=registry.h.pjlab.org.cn/ailab-ma4agismall-ma4agismall_gpu/slime-sxy:slime0529 \
  --image-pull-policy=IfNotPresent \
  --mount=gpfs://gpfs1/sdpdev-fs/sunxiangyu:/root/slime_sxy/group-space/sunxiangyu \
  -P 1 --gpu=1 --cpu=8 --memory=64000 -e DISTRIBUTED_JOB=true \
  -- bash -lc 'set -euo pipefail; echo TEST-OK; nvidia-smi -L; test -d /root/slime_sxy/group-space/sunxiangyu && echo MOUNT-OK; exit 0'
```

Observed: metadata name `skilltest-agenttool-params-0921a-2445012`; created successfully; annotation `volcano.brainpp.cn/priority == "9"`; phase `Starting` → `Succeeded`; logs printed `TEST-OK`, `GPU 0: NVIDIA H200`, `MOUNT-OK`; `ExitCode 0`. The job exited immediately, so no GPU stayed allocated — a bounded probe, as required above.

## Caveats

- The probe emitted `workspace.brainpp.cn/workspace-namspace: ailab-ma4agismall` even for the `ailab-agenttool` job, because that label reflects the login workspace, not the job's namespace. This is expected and did not block scheduling — do not "fix" it by changing the job namespace.
- Quota reported on 2026-09-21: L2 (`ma4agismall`) 8 GPUs total with 8 in use (0 free); L1 (`agenttool`) 16 GPUs total with 2 in use. At that moment every 8-GPU job necessarily fell back to L1. Numbers in the console are authoritative and change.
- L1's namespace is reachable from the login host (`brainctl get rjobs -n ailab-agenttool` works) even though the login workspace belongs to the L2 project.
- Job names must stay lowercase letters/digits/hyphens; the platform may append a timestamp to form the metadata name, so query by the metadata name it reports, not by the name you passed.
