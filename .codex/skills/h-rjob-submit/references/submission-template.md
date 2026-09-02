# Submission template and checks

## Single-node template

Render a concrete command from this shape. Replace every placeholder before submission; never submit the template itself.

```bash
source /etc/profile.d/ssh-init.sh

JOB_NAME='<lowercase-job-name>'

rjob submit \
  --priority=9 \
  --name="${JOB_NAME}" \
  --namespace=ailab-ma4agismall \
  --enable-sshd \
  --image=registry.h.pjlab.org.cn/ailab-ma4agismall-ma4agismall_gpu/slime-sxy:slime0529 \
  --image-pull-policy=IfNotPresent \
  --mount=gpfs://gpfs1/sdpdev-fs/sunxiangyu:/root/slime_sxy/group-space/sunxiangyu \
  --mount=gpfs://gpfs2/gpfs2-shared-public/huggingface:/root/slime_sxy/group-space/huggingface \
  --charged-group=ma4agismall_gpu \
  --private-machine=group \
  -P 1 \
  --gpu=<gpus-per-node> \
  --cpu=<cpus-per-node> \
  --memory=<mib-per-node> \
  -e NCCL_IB_DISABLE=1 \
  -e DISTRIBUTED_JOB=true \
  -- bash -lc 'set -euo pipefail; cd /root/slime_sxy/group-space/sunxiangyu/drug-pipe/<workdir>; exec bash /root/slime_sxy/group-space/sunxiangyu/drug-pipe/<launcher>.sh <args>'
```

Prefer putting a complex training command in the mounted launcher rather than embedding several layers of quoting in `rjob submit`. The launcher should initialize the project environment, write outputs/checkpoints to intentional shared paths, start the real workload in the foreground, and propagate its exit code.

For a Python module whose environment is already established, the final part may instead be:

```bash
-- bash -lc 'set -euo pipefail; cd /root/slime_sxy/group-space/sunxiangyu/drug-pipe/<workdir>; exec python -m <module> <args>'
```

## Pre-submit checks

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

Preserve the complete `rjob submit` output because the platform may normalize the Kubernetes metadata name. Then inspect with the installed CLI syntax, explicitly including the namespace. Common documented forms are:

```bash
rjob list --name "${JOB_NAME}" --namespace=ailab-ma4agismall
rjob get --name "${JOB_NAME}" --namespace=ailab-ma4agismall
```

Use `rjob events <metadata-name> --namespace=ailab-ma4agismall` when a created job cannot schedule or start. Use `rjob logs job <metadata-name> --namespace=ailab-ma4agismall -n 200` after a replica starts. Check the current CLI help before relying on syntax that differs across versions.

Do not automatically run `rjob delete` or resubmit when the job is Pending/Queued. Deletion is destructive and loses job metadata/log context; it requires explicit scope and an exact resolved target.
