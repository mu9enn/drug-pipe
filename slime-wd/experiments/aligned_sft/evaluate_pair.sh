#!/usr/bin/env bash
set -Eeuo pipefail
: "${SFT_MODEL_DIR:?}" "${PAIR_STAMP:?}"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
wd=$(cd "$script_dir/../.." && pwd)
cmp "$wd/data/Qwen3.5-9B/tokenizer_config.json" "$SFT_MODEL_DIR/tokenizer_config.json"
JOB_NAME="av4-sft-pair-${PAIR_STAMP}" MODEL_DIR="$SFT_MODEL_DIR" MODEL_ID=qwen3.5-9b-sft-local \
 MODEL_NAME="Qwen3.5-9B SFT aligned v4" GPU_COUNT=4 TP_SIZE=4 WORKSPACE_VARIANT=l1-flat \
 RUN_NAME="aligned-v4-sft-pair-${PAIR_STAMP}" INFRA_NAME="infra-aligned-v4-sft-pair-${PAIR_STAMP}" \
 PAIR_EVALUATION=1 WORKER_DRIVER=/root/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/experiments/aligned_sft/run_eval_pair_worker.sh \
 bash "$wd/dsh-molbench/pretrained_matrix/submit_one.sh"
for variant in l1-flat; do
 python3 "$wd/dsh-molbench/pretrained_matrix/wait_existing.py" "$wd/outputs" \
  "aligned-v4-9b-original-${variant}-full-${BASELINE_STAMP:-$PAIR_STAMP}" qwen3.5-9b-original-local 0
done
python3 "$script_dir/summarize_pair.py" --stamp "$PAIR_STAMP" --baseline-stamp "${BASELINE_STAMP:-$PAIR_STAMP}" --runs-root "$wd/outputs/dsh_molbench_evals"
