# Drug-pipe H-cluster defaults

Use these as the current local defaults for Sun Xiangyu's drug-pipe jobs. Reconfirm with the user or current project configuration when a request supplies different values; these are not universal H-cluster settings.

## Shared fields

| Field | Current default |
|---|---|
| Namespace | `ailab-ma4agismall` |
| Charged group | `ma4agismall_gpu` |
| Private machine | `group` |
| Priority | `9` |
| Image | `registry.h.pjlab.org.cn/ailab-ma4agismall-ma4agismall_gpu/slime-sxy:slime0529` |
| Image pull policy | `IfNotPresent` |
| SSH | `--enable-sshd` |
| User GPFS mount | `gpfs://gpfs1/sdpdev-fs/sunxiangyu:/root/slime_sxy/group-space/sunxiangyu` |
| Hugging Face mount | `gpfs://gpfs2/gpfs2-shared-public/huggingface:/root/slime_sxy/group-space/huggingface` |
| Distributed env | `-e DISTRIBUTED_JOB=true` |

The user historically also set `-e NCCL_IB_DISABLE=1`. Retain it for the known single-node profiles unless the actual workload has a verified reason to differ. Do not copy it blindly to multi-node 8-GPU-per-replica training, where disabling IB can defeat the intended inter-node transport.

## Known single-node profiles

These values are per replica. Both profiles use `-P 1`.

| Profile | `--gpu` | `--cpu` | `--memory` |
|---|---:|---:|---:|
| Small/debug workload | 2 | 64 | 530000 |
| Full 8-GPU worker | 8 | 108 | 1400000 |

Do not infer GPU count from an old job-name fragment. In particular, a historical name containing `4g64c` was paired with `--gpu=2`; name the new job from the resolved resources instead.

## Multi-node additions

For `N` nodes, use `-P N`, keep resource flags per node, and normally add `--gang-start=true` so all replicas are scheduled together. Keep `-e DISTRIBUTED_JOB=true`; the platform injects variables including `NODE_RANK`, `NODE_COUNT`, `MASTER_ADDR`, `PROC_PER_NODE`, and `JOB_ID`.

The training launcher must use those values (or a project wrapper that does) and must set a consistent master port. Do not overwrite platform-injected `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, or `NCCL_IB_GID_INDEX`.

For verified multi-node RoCE/RDMA workloads, the platform guide documents:

```text
--custom-resources rdma/mlnx_shared=8
--custom-resources mellanox.com/mlnx_rdma=1
```

Only add these after confirming the selected machines/workload require them. The guide separately notes that multi-node jobs using fewer than 8 GPUs per node may need `NCCL_IB_DISABLE=1`; resolve that topology deliberately instead of combining contradictory defaults.
