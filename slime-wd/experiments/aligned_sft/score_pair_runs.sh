#!/usr/bin/env bash
set -Eeuo pipefail
stamp=${1#aligned-v4-sft-pair-}
wd=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
for variant in l1-flat; do
 for phase in smoke full; do
  limit=0; [[ "$phase" == full ]] || limit=1
  name=aligned-v4-9b-sft-${variant}-${phase}-${stamp}
  infra=$wd/outputs/infra-$name
  skill=$wd/../workdir-skills/molclaw-l1-workspace
  [[ "$variant" == l1-flat ]] || skill=$infra/workspace_template
  python3 "$wd/dsh-molbench/run_dsh_molbench.py" --suite ms1 --suite ms2 --score-only \
    --run-dir "$wd/outputs/dsh_molbench_evals/$name" --agent-preset molclaw-v8-eval \
    --skill-source "$skill" --limit-per-suite "$limit" > "$infra/login-score.log" 2>&1
  python3 "$wd/dsh-molbench/pretrained_matrix/wait_existing.py" "$wd/outputs" "$name" qwen3.5-9b-sft-local "$limit"
  touch "$infra/evaluation.complete"
 done
done
