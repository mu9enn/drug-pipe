#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
round=${MATRIX_ROUND:-v4}
: "${MATRIX_STAMP:?Set a new experiment stamp}"
: "${MODEL_TAGS:=9b}"
stamp=$MATRIX_STAMP
: "${EXPERIMENT_MANIFEST:?Final experiment manifest required}"
python3 - "$EXPERIMENT_MANIFEST" "$MODEL_TAGS" <<'PY_VALIDATE'
import json,pathlib,sys,hashlib
m=json.load(open(sys.argv[1]));p=pathlib.Path(m['release_manifest']);r=json.loads(p.read_text())
assert hashlib.sha256(p.read_bytes()).hexdigest()==m['release_manifest_sha256']
assert r['context_gate']=='passed'
a=json.loads((p.parent/'experiments/release_validation.json').read_text())
assert a['release_ready'] and a['release_manifest_sha256']==m['release_manifest_sha256']
assert m['stage_a']['models']==sys.argv[2].split(',')
assert m['selected_questions']==87 and m['tools']==88 and m['reserved_questions']==112
PY_VALIDATE
job_date=${MATRIX_JOB_DATE:-$(date +%m%d)}
session=${1:-dsh-pretrained-matrix-${job_date}-${round}}
login_root=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd
serial_dir=$login_root/outputs/dsh_pretrained_matrix_serial_$stamp

models=(
  "9b|Qwen3.5 9B original released checkpoint|/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/data/Qwen3.5-9B|1|1"
  "27b|Qwen3.5 27B original released checkpoint|/home/sunxiangyu/slime_sxy/group-space/huggingface/zskj-hub/models-Qwen-Qwen3.5-27B|2|2"
  "35b-a3b|Qwen3.5 35B-A3B original released checkpoint|/home/sunxiangyu/slime_sxy/group-space/huggingface/zskj-hub/models-Qwen-Qwen3.5-35B-A3B|2|2"
  "122b-a10b|Qwen3.5 122B-A10B original released checkpoint|/home/sunxiangyu/slime_sxy/group-space/huggingface/zskj-hub/models-Qwen-Qwen3.5-122B-A10B|4|4"
)

run_queue() {
  local variant=$1 short_variant=flat model tag model_name model_dir gpu_count tp_size
  local job_name run_name infra_name model_id rc failed=0
  [[ "$variant" == l1-flat ]] || short_variant=hier
  mkdir -p "$serial_dir"

  for model in "${models[@]}"; do
    IFS='|' read -r tag model_name model_dir gpu_count tp_size <<<"$model"
    [[ ",$MODEL_TAGS," == *",$tag,"* ]] || continue
    model_id="qwen3.5-${tag}-original-local"
    rc=0
    for phase in smoke full; do
      run_name="aligned-v4-${tag}-original-${variant}-${phase}-${stamp}"
      job_name="av4-${tag}-orig-${short_variant}-${phase}-${stamp}"
      infra_name="infra-$run_name"
      if [[ -f "$login_root/outputs/$infra_name/worker.exit" && ! -f "$login_root/outputs/dsh_molbench_evals/$run_name/run_manifest.json" ]]; then
        infra_name="${infra_name}-bootstrap-r1"
        job_name="${job_name}-r1"
        [[ ! -e "$login_root/outputs/$infra_name/worker.exit" ]] || return 1
      fi
      limit=0; [[ "$phase" == full ]] || limit=1
      if [[ -f "$login_root/outputs/dsh_molbench_evals/$run_name/run_manifest.json" ]]; then
        if ! python3 "$script_dir/wait_existing.py" "$login_root/outputs" "$run_name" "$model_id" "$limit"; then rc=1; break; fi
      elif ! JOB_NAME="$job_name" MODEL_DIR="$model_dir" MODEL_ID="$model_id" \
        MODEL_NAME="$model_name" GPU_COUNT="$gpu_count" TP_SIZE="$tp_size" \
        WORKSPACE_VARIANT="$variant" RUN_NAME="$run_name" INFRA_NAME="$infra_name" \
        LIMIT_PER_SUITE="$limit" "$script_dir/submit_one.sh"; then rc=1; break
      fi
    done
    [[ "$rc" == 0 ]] || failed=1
    printf '[%s] END job=%s rc=%s\n' "$(date --iso-8601=seconds)" "$job_name" "$rc" \
      | tee -a "$serial_dir/${short_variant}.log"
    printf '%s\t%s\n' "$job_name" "$rc" >> "$serial_dir/${short_variant}.status.tsv"
  done

  printf '%s\n' "$failed" > "$serial_dir/${short_variant}.exit"
  return "$failed"
}

if [[ "${1:-}" == --run-queue ]]; then
  [[ $# -eq 2 ]] || { echo "usage: $0 --run-queue {l1-flat|legacy-hierarchy}" >&2; exit 2; }
  case "$2" in
    l1-flat|legacy-hierarchy) run_queue "$2" ;;
    *) echo "unknown queue variant: $2" >&2; exit 2 ;;
  esac
  exit $?
fi

tmux has-session -t "$session" 2>/dev/null && { echo "tmux session already exists: $session"; exit 1; }
mkdir -p "$serial_dir"
: > "$serial_dir/flat.status.tsv"
: > "$serial_dir/hier.status.tsv"
printf -v flat_command 'MODEL_TAGS=%q EXPERIMENT_MANIFEST=%q MATRIX_ROUND=%q MATRIX_STAMP=%q MATRIX_JOB_DATE=%q exec %q --run-queue l1-flat' \
  "$MODEL_TAGS" "$EXPERIMENT_MANIFEST" "$round" "$stamp" "$job_date" "$script_dir/submit_matrix.sh"
printf -v hier_command 'MODEL_TAGS=%q EXPERIMENT_MANIFEST=%q MATRIX_ROUND=%q MATRIX_STAMP=%q MATRIX_JOB_DATE=%q exec %q --run-queue legacy-hierarchy' \
  "$MODEL_TAGS" "$EXPERIMENT_MANIFEST" "$round" "$stamp" "$job_date" "$script_dir/submit_matrix.sh"
tmux new-session -d -s "$session" -n flat "$flat_command"
tmux new-window -t "$session" -n hier "$hier_command"
printf -v summary_command 'MODEL_TAGS=%q MATRIX_STAMP=%q exec %q' "$MODEL_TAGS" "$stamp" "$script_dir/follow_summary.sh"
tmux new-window -t "$session" -n summary "$summary_command"
tmux list-windows -t "$session"
