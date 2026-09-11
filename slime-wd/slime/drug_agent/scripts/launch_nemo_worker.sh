#!/usr/bin/env bash
set -euo pipefail
run_root="${1:?run root required}"
mode="${2:?smoke or full required}"
eps="${3:?explicit eps required}"
[[ "$mode" == smoke || "$mode" == full || "$mode" == experiment ]]
[[ -f "$run_root/prepared/preparation_manifest.json" ]]
[[ -x "$run_root/venv/bin/python" && -x "$run_root/bootstrap/bin/uv" ]]
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /etc/profile.d/ssh-init.sh
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
exec rlaunch \
  --namespace=ailab-ma4agismall --enable-sshd \
  --mount=gpfs://gpfs1/sdpdev-fs/sunxiangyu:/home/sunxiangyu/slime_sxy/group-space/sunxiangyu \
  --mount=gpfs://gpfs1/sdpdev-fs/sunxiangyu:/mnt/shared-storage-user/sdpdev-fs/sunxiangyu \
  --mount=gpfs://gpfs2/gpfs2-shared-public/huggingface:/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface \
  --charged-group=ma4agismall_gpu --private-machine=group \
  --gpu=1 --cpu=24 --memory=98304 \
  -e NCCL_IB_DISABLE=1 -e DISTRIBUTED_JOB=true \
  -- bash "$script_dir/run_nemo_worker.sh" "$run_root" "$mode" "$eps"
