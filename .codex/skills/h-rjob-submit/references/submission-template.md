# Submission template and checks

## Choose the pool first

Resolve `--namespace` and `--charged-group` from the job's total GPU count before rendering anything, using [resource-pools.md](resource-pools.md):

| GPUs needed | First choice namespace / charged group |
|---|---|
| `< 8` | `ailab-agenttool` / `agenttool_pool` |
| `= 8` | `ailab-ma4agismall` / `ma4agismall_gpu`, falling back to `ailab-agenttool` / `agenttool_pool` |
| `> 8` | `ailab-agenttool` / `agenttool_pool` only |

Everything else in the template is pool-independent. Do not copy the L2 pair below into an L1 job.

## Single-node template

Render a concrete command from this shape. Replace every placeholder before submission; never submit the template itself.

```bash
source /etc/profile.d/ssh-init.sh

JOB_NAME='<lowercase-job-name>'

# Prefer the helper: it injects --priority=9 and rejects a duplicated
# --priority/--task-type flag. Resolve <this-skill> to the actual skill directory.
bash <this-skill>/scripts/submit_normal_rjob.sh \
  --name="${JOB_NAME}" \
  --namespace=<pool-namespace> \
  --enable-sshd \
  --image=registry.h.pjlab.org.cn/ailab-ma4agismall-ma4agismall_gpu/slime-sxy:slime0529 \
  --image-pull-policy=IfNotPresent \
  --mount=gpfs://gpfs1/sdpdev-fs/sunxiangyu:/root/slime_sxy/group-space/sunxiangyu \
  --mount=gpfs://gpfs2/gpfs2-shared-public/huggingface:/root/slime_sxy/group-space/huggingface \
  --charged-group=<pool-charged-group> \
  --private-machine=group \
  -P 1 \
  --gpu=<gpus-per-node> \
  --cpu=<cpus-per-node> \
  --memory=<mib-per-node> \
  -e NCCL_IB_DISABLE=1 \
  -e DISTRIBUTED_JOB=true \
  -- bash -lc 'set -euo pipefail; cd /root/slime_sxy/group-space/sunxiangyu/drug-pipe/<workdir>; exec bash /root/slime_sxy/group-space/sunxiangyu/drug-pipe/<launcher>.sh <args>'
```

If you invoke `rjob submit` directly instead of through the helper, you must add `--priority=9` yourself.

Concrete pool values:

| Placeholder | L1 | L2 |
|---|---|---|
| `<pool-namespace>` | `ailab-agenttool` | `ailab-ma4agismall` |
| `<pool-charged-group>` | `agenttool_pool` | `ma4agismall_gpu` |

Prefer putting a complex training command in the mounted launcher rather than embedding several layers of quoting in `rjob submit`. The launcher should initialize the project environment, write outputs/checkpoints to intentional shared paths, start the real workload in the foreground, and propagate its exit code. It must never end in `sleep`, a detached `tmux`, or any other keepalive — see [resource-pools.md](resource-pools.md).

For a Python module whose environment is already established, the final part may instead be:

```bash
-- bash -lc 'set -euo pipefail; cd /root/slime_sxy/group-space/sunxiangyu/drug-pipe/<workdir>; exec python -m <module> <args>'
```

## Pre-submit checks

Require an explicit `--priority=9` in the fully resolved command, including any
custom submitter's subprocess argument list. The installed normal-job CLI defaults
to 5 when this is omitted. Do not rely on the template alone.

Use read-only checks appropriate to the command:

```bash
source /etc/profile.d/ssh-init.sh
command -v rjob
rjob submit --help
test -f /mnt/shared-storage-user/sdpdev-fs/sunxiangyu/drug-pipe/<launcher>.sh
bash -n /mnt/shared-storage-user/sdpdev-fs/sunxiangyu/drug-pipe/<launcher>.sh
```

The login-host path used for `test`/`bash -n` and the container path used in the RJob command refer to the same mounted storage but have different prefixes. Do not mechanically substitute one for the other without resolving the mount.

If supported by the installed CLI, `--predict-only` may be used for a non-mutating resource prediction and the current CLI's dry-run facility may be used to inspect generated configuration. Verify their syntax from `--help`; neither proves the training command itself works.

## Submission result and one-shot inspection

Verify the created object's actual priority, not just CLI success:

```bash
brainctl get rjobs -n <pool-namespace> \
  --field-selector "metadata.name=${JOB_NAME}" -o json
```

Require exactly one matching object and
`items[0].metadata.annotations["volcano.brainpp.cn/priority"] == "9"`, and that
`items[0].metadata.namespace` and the `quotagroup.brainpp.cn/quotagroup` label match the pool you chose.
Save name, UID, observed priority and state. An empty list is not success, even
when the command exits zero. On a mismatch, report failure and resolve it within
the user's authorization; do not blindly create replacement jobs.

Preserve the complete `rjob submit` output because the platform may normalize the Kubernetes metadata name. Then inspect with the installed CLI syntax, explicitly including the namespace. Common documented forms are:

```bash
rjob list --name "${JOB_NAME}" --namespace=<pool-namespace>
rjob get --name "${JOB_NAME}" --namespace=<pool-namespace>
```

Use `rjob events <metadata-name> --namespace=<pool-namespace>` when a created job cannot schedule or start. Use `rjob logs job <metadata-name> --namespace=<pool-namespace> -n 200` after a replica starts. Check the current CLI help before relying on syntax that differs across versions.

Do not automatically run `rjob delete` or resubmit when the job is Pending/Queued. Deletion is destructive and loses job metadata/log context; it requires explicit scope and an exact resolved target.
