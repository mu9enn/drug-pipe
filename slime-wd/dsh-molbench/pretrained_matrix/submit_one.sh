#!/usr/bin/env bash
set -Eeuo pipefail

: "${JOB_NAME:?}" "${MODEL_DIR:?}" "${MODEL_ID:?}" "${MODEL_NAME:?}"
: "${GPU_COUNT:?}" "${TP_SIZE:?}" "${WORKSPACE_VARIANT:?}" "${RUN_NAME:?}" "${INFRA_NAME:?}"
: "${LIMIT_PER_SUITE:=0}"
: "${SUITES:=ms1 ms2}"
: "${EVAL_SEED:=42}"
read -r -a selected_suites <<< "$SUITES"
suite_args=()
for suite in "${selected_suites[@]}"; do suite_args+=(--suite "$suite"); done

namespace=ailab-ma4agismall
login_root=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd
infra_dir=$login_root/outputs/$INFRA_NAME
run_dir=$login_root/outputs/dsh_molbench_evals/$RUN_NAME
worker_driver=${WORKER_DRIVER:-/root/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/dsh-molbench/pretrained_matrix/run_worker.sh}
worker_model_dir=${MODEL_DIR/\/home\/sunxiangyu\/slime_sxy/\/root\/slime_sxy}
worker_model_dir=${worker_model_dir/\/mnt\/shared-storage-user\/sdpdev-fs\/sunxiangyu/\/root\/slime_sxy\/group-space\/sunxiangyu}
cpu_count=$((GPU_COUNT * 16))
memory_mib=$((GPU_COUNT * 132500))
tunnel_pid=
submitted=0

mkdir -p "$infra_dir"
[[ ! -e "$run_dir/run_manifest.json" ]] || { echo "run already exists: $run_dir"; exit 1; }
log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$infra_dir/orchestrator.log"; }
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ "$rc" != 0 && "$submitted" == 1 ]]; then
    rjob stop "$JOB_NAME" --namespace="$namespace" >> "$infra_dir/orchestrator.log" 2>&1 || true
  fi
  if [[ -n "${tunnel_pid:-}" ]] && kill -0 "$tunnel_pid" 2>/dev/null; then
    kill "$tunnel_pid" 2>/dev/null || true
    wait "$tunnel_pid" 2>/dev/null || true
  fi
  printf '%s\n' "$rc" > "$infra_dir/orchestrator.exit"
  exit "$rc"
}
trap cleanup EXIT INT TERM

log "submitting job=$JOB_NAME gpu=$GPU_COUNT tp=$TP_SIZE model=$MODEL_DIR variant=$WORKSPACE_VARIANT"
rjob submit --name="$JOB_NAME" --metadata-name="$JOB_NAME" --namespace="$namespace" \
  --task-type=normal --priority=9 --restart-policy=never --preemptible=no --enable-sshd \
  --image=registry.h.pjlab.org.cn/ailab-ma4agismall-ma4agismall_gpu/slime-sxy:slime0529 \
  --image-pull-policy=IfNotPresent \
  --mount=gpfs://gpfs1/sdpdev-fs/sunxiangyu:/root/slime_sxy/group-space/sunxiangyu \
  --mount=gpfs://gpfs2/gpfs2-shared-public/huggingface:/root/slime_sxy/group-space/huggingface \
  --charged-group=ma4agismall_gpu --private-machine=group \
  --gpu="$GPU_COUNT" --cpu="$cpu_count" --memory="$memory_mib" \
  -e NCCL_IB_DISABLE=1 -e DISTRIBUTED_JOB=true -e RJOB_NAME="$JOB_NAME" \
  -e MODEL_DIR="$worker_model_dir" -e MODEL_ID="$MODEL_ID" -e MODEL_NAME="$MODEL_NAME" \
  -e GPU_COUNT="$GPU_COUNT" -e TP_SIZE="$TP_SIZE" \
  -e WORKSPACE_VARIANT="$WORKSPACE_VARIANT" -e RUN_NAME="$RUN_NAME" -e INFRA_NAME="$INFRA_NAME" \
  -e LIMIT_PER_SUITE="$LIMIT_PER_SUITE" -e SUITES="$SUITES" -e EVAL_SEED="$EVAL_SEED" \
  -- bash -lc "exec $worker_driver" 2>&1 | tee "$infra_dir/rjob_submit.log"
submitted=1

log waiting_for_replica
for attempt in $(seq 1 8640); do
  state=$(rjob get "$JOB_NAME" --namespace "$namespace" 2>&1 || true)
  replica=$(sed -nE 's/.*replica ([^:]+):.*/\1/p' <<<"$state" | head -1)
  if [[ -n "$replica" ]]; then
    printf '%s\n' "$replica" > "$infra_dir/replica.txt"
    printf '%s\n' "$state" > "$infra_dir/rjob_state_after_submit.txt"
    break
  fi
  sleep 5
done
[[ -s "$infra_dir/replica.txt" ]] || { log replica_timeout; exit 1; }
replica=$(<"$infra_dir/replica.txt")
target=${replica}.sunxiangyu+root.ailab-ma4agismall.pod@h.pjlab.org.cn
ssh_opts=(-o BatchMode=yes -o ClearAllForwardings=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)

log "waiting_for_worker target=$target"
while ! ssh "${ssh_opts[@]}" "$target" true >> "$infra_dir/bootstrap.log" 2>&1; do
  [[ ! -f "$infra_dir/worker.exit" ]] || exit 1
  sleep 15
done
ssh "${ssh_opts[@]}" "$target" 'install -d -m 700 /root/.dsh'
scp "${ssh_opts[@]}" /home/sunxiangyu/.dsh/molclaw.env "$target:/root/.dsh/molclaw.env"
ssh "${ssh_opts[@]}" "$target" 'chmod 600 /root/.dsh/molclaw.env'
log secret_installed

(
  while [[ ! -f "$infra_dir/worker.exit" ]]; do
    ssh -NT -o BatchMode=yes -o ConnectTimeout=15 -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new \
      -R 127.0.0.1:13208:127.0.0.1:13208 "$target" >> "$infra_dir/bootstrap.log" 2>&1 || true
    [[ -f "$infra_dir/worker.exit" ]] || sleep 15
  done
) &
tunnel_pid=$!

for attempt in $(seq 1 2880); do
  [[ -f "$infra_dir/worker.exit" ]] && break
  sleep 60
done
[[ -f "$infra_dir/worker.exit" ]] || { log worker_timeout; exit 1; }
wait "$tunnel_pid" || true
tunnel_pid=
worker_rc=$(<"$infra_dir/worker.exit")
log "worker_exit=$worker_rc"
[[ "$worker_rc" == 0 && -f "$infra_dir/rollout.complete" ]] || exit 1

if [[ "${PAIR_EVALUATION:-0}" == 1 ]]; then
  bash "$login_root/experiments/aligned_sft/score_pair_runs.sh" "$RUN_NAME"
else
skill_source=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/workdir-skills/molclaw-l1-workspace
[[ "$WORKSPACE_VARIANT" == l1-flat ]] || skill_source=$infra_dir/workspace_template
cd "$login_root"
"${SCORE_PYTHON:-python3}" dsh-molbench/run_dsh_molbench.py "${suite_args[@]}" --score-only \
  --run-dir "$run_dir" --agent-preset molclaw-v8-eval --skill-source "$skill_source" \
  --limit-per-suite "$LIMIT_PER_SUITE" \
  > "$infra_dir/login-score.log" 2>&1
if [[ "$SUITES" == *mo-opt* || "$SUITES" == *mo-edit* || "$SUITES" == *ms3* ]]; then
  "${SCORE_PYTHON:-python3}" dsh-molbench/score_format_sensitivity.py --run-dir "$run_dir" \
    > "$infra_dir/format-sensitivity-score.log" 2>&1
fi
touch "$infra_dir/evaluation.complete"

fi

for attempt in $(seq 1 120); do
  state=$(rjob get "$JOB_NAME" --namespace "$namespace" 2>&1 || true)
  if ! grep -q "active': 1" <<<"$state"; then
    printf '%s\n' "$state" > "$infra_dir/rjob_final_state.txt"
    break
  fi
  sleep 5
done
log EVALUATION_COMPLETE
