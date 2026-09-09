#!/usr/bin/env bash
set -Eeuo pipefail
: "${RUN_NAME:?}" "${INFRA_NAME:?}"
stamp=${RUN_NAME#aligned-v4-sft-pair-}
wd=/root/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd
parent=$wd/outputs/$INFRA_NAME
mkdir -p "$parent"
trap 'rc=$?; chmod -R a+rwX "$parent"; printf "%s\n" "$rc" > "$parent/worker.exit"' EXIT
[[ "$GPU_COUNT" == 4 && "$TP_SIZE" == 4 ]]
source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
CUDA_VISIBLE_DEVICES= python "$wd/experiments/aligned_sft/verify_checkpoint_reload.py" \
 "$MODEL_DIR" "$parent/final_checkpoint_reload_validation.json"
for variant in l1-flat; do
 for phase in smoke full; do
  limit=0; [[ "$phase" == full ]] || limit=1
  name=aligned-v4-9b-sft-${variant}-${phase}-${stamp}
  WORKSPACE_VARIANT="$variant" RUN_NAME="$name" INFRA_NAME="infra-$name" LIMIT_PER_SUITE="$limit" \
   bash "$wd/dsh-molbench/pretrained_matrix/run_worker.sh"
 done
done
touch "$parent/rollout.complete"
