#!/usr/bin/env bash
set -Eeuo pipefail
: "${RELEASE_ROOT:?}" "${RUN_ROOT:?}" "${JOB_NAME:?}"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python3 "$script_dir/validate_release.py" "$RELEASE_ROOT"
namespace=ailab-ma4agismall
worker_prefix=/root/slime_sxy/group-space/sunxiangyu
worker_release=$worker_prefix/drug_wd/$(basename "$RELEASE_ROOT")
worker_run=$worker_release/experiments/$(basename "$RUN_ROOT")
mkdir -p "$RUN_ROOT"
[[ ! -e "$RUN_ROOT/worker.exit" ]] || { echo 'Use a new run directory'; exit 1; }
rjob submit --name="$JOB_NAME" --metadata-name="$JOB_NAME" --namespace="$namespace" \
 --task-type=normal --priority=9 --restart-policy=never --preemptible=no --enable-sshd \
 --image=registry.h.pjlab.org.cn/ailab-ma4agismall-ma4agismall_gpu/slime-sxy:slime0529 \
 --image-pull-policy=IfNotPresent \
 --mount=gpfs://gpfs1/sdpdev-fs/sunxiangyu:$worker_prefix \
 --charged-group=ma4agismall_gpu --private-machine=group --gpu=8 --cpu=108 --memory=1060000 \
 -e NCCL_IB_DISABLE=1 -e DISTRIBUTED_JOB=true -e RJOB_NAME="$JOB_NAME" \
 -e RELEASE_ROOT="$worker_release" -e RUN_ROOT="$worker_run" \
 -e LR="${LR:-5e-6}" -e MIN_LR="${MIN_LR:-5e-7}" \
 -- bash -lc "exec bash $worker_prefix/drug-pipe/slime-wd/experiments/aligned_sft/run_worker.sh" \
 > "$RUN_ROOT/submission.log" 2>&1
while [[ ! -f "$RUN_ROOT/worker.exit" ]]; do
 rjob get "$JOB_NAME" --namespace="$namespace" > "$RUN_ROOT/rjob_state.txt" 2>&1
 if rg -q 'Failed|Stopped' "$RUN_ROOT/rjob_state.txt"; then cat "$RUN_ROOT/rjob_state.txt"; exit 1; fi
 sleep 60
done
cat "$RUN_ROOT/worker.exit"
[[ "$(cat "$RUN_ROOT/worker.exit")" == 0 && -f "$RUN_ROOT/training.complete" ]]
rjob get "$JOB_NAME" --namespace="$namespace" > "$RUN_ROOT/rjob_final_state.txt" 2>&1
