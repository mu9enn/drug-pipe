#!/usr/bin/env bash
set -Eeuo pipefail
: "${RELEASE_ROOT:?}" "${RUN_ROOT:?}"
export LR="${LR:-5e-6}" MIN_LR="${MIN_LR:-5e-7}"
project=/root/slime_sxy/group-space/sunxiangyu/drug-pipe
slime=$project/slime-wd/slime
model=$project/slime-wd/data/Qwen3.5-9B
ref=$project/slime-wd/data/Qwen3.5-9B_torch_dist
mkdir -p "$RUN_ROOT"
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN_ROOT/worker.exit"' EXIT
exec > >(tee -a "$RUN_ROOT/driver.log") 2>&1
source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
export PYTHONPATH=$project/data-pipe:$slime:${PYTHONPATH:-}
python "$project/slime-wd/experiments/aligned_sft/validate_release.py" "$RELEASE_ROOT"
python - "$RELEASE_ROOT" "$model" "$RUN_ROOT" <<'PY'
import hashlib,json,pathlib,sys,subprocess,os
release,model,run=map(pathlib.Path,sys.argv[1:]);m=json.loads((release/'training/context_gate_manifest.json').read_text());pub=json.loads((release/'release_manifest.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert m['ok'] and pub['context_gate']=='passed'
assert sha(release/'training/qwen35_sft_train.jsonl')==m['output_sha256']
assert sha(model/'tokenizer_config.json')==m['tokenizer_config_sha256']
count=m['accepted_count'];gbs=2 if count%2==0 else 1
rbs=max(n for n in range(gbs,min(86,count)+1,gbs) if count%n==0)
(run/'batch.env').write_text(f'GLOBAL_BATCH_SIZE={gbs}\nROLLOUT_BATCH_SIZE={rbs}\n')
lr, min_lr = float(os.environ['LR']), float(os.environ['MIN_LR'])
assert 0 < min_lr <= lr
(run/'resolved_config.json').write_text(json.dumps({'release_sha256':sha(release/'release_manifest.json'),'train_sha256':m['output_sha256'],'count':count,'global_batch_size':gbs,'rollout_batch_size':rbs,'epochs':1,'lr':lr,'min_lr':min_lr,'tp':4,'pp':2,'gpu_count':8,'model':str(model),'checkpoint_selection':'final_epoch_without_test_feedback'},indent=2)+'\n')
PY
python "$project/slime-wd/experiments/aligned_sft/capture_runtime.py" --model "$model" --output "$RUN_ROOT/runtime_snapshot.json"
EXPECTED_GPUS=8 HF_CHECKPOINT="$model" bash "$slime/drug_agent/scripts/preflight_large_model_worker.sh"
source "$RUN_ROOT/batch.env"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUM_GPUS=8 TENSOR_MODEL_PARALLEL_SIZE=4 PIPELINE_MODEL_PARALLEL_SIZE=2 CONTEXT_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1 EXPERT_TENSOR_PARALLEL_SIZE=1
export HF_CHECKPOINT=$model REF_LOAD=$ref MAX_TOKENS_PER_GPU=16384
export RECOMPUTE_FULL=1 RECOMPUTE_NUM_LAYERS=1 RECOMPUTE_LOSS_FUNCTION=1 RECOMPUTE_VOCAB_LOG_PROBS=1 LOG_PROBS_CHUNK_SIZE=64 BALANCE_DATA=1
export SFT_DEBUG_TRAIN_ONLY=1 SFT_DISABLE_OFFLOAD=1 LR_WARMUP_FRACTION=0.05
launcher=$slime/drug_agent/scripts/run_qwen3_5_9b_drug_sft_full.sh
probe_count=$(wc -l < "$RELEASE_ROOT/training/probes.jsonl")
# The probe proves real learning and checkpoint serialization across length buckets.
env PROMPT_DATA="$RELEASE_ROOT/training/probes.jsonl" NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE="$probe_count" GLOBAL_BATCH_SIZE=1 \
  SFT_EPOCH_ONLY=0 SAVE_INTERVAL=1 SFT_METRICS_DIR="$RUN_ROOT/probe_metrics" SAVE_DIR="$RUN_ROOT/probe" RAY_SUBMIT_LOG="$RUN_ROOT/probe_submit.log" \
  bash "$launcher"
test -f "$RUN_ROOT/probe/latest_checkpointed_iteration.txt"
python "$project/slime-wd/experiments/aligned_sft/verify_metrics.py" "$RUN_ROOT/probe_metrics" "$probe_count"
probe_iteration=$(cat "$RUN_ROOT/probe/latest_checkpointed_iteration.txt")
printf -v probe_checkpoint '%s/iter_%07d' "$RUN_ROOT/probe" "$probe_iteration"
python "$slime/tools/convert_torch_dist_to_hf.py" --input-dir "$probe_checkpoint" --output-dir "$RUN_ROOT/probe_reload_hf" \
  --origin-hf-dir "$model" --add-missing-from-origin-hf
CUDA_VISIBLE_DEVICES= python "$project/slime-wd/experiments/aligned_sft/verify_checkpoint_reload.py" \
  "$RUN_ROOT/probe_reload_hf" "$RUN_ROOT/probe_reload_validation.json"
touch "$RUN_ROOT/probe.complete"
env PROMPT_DATA="$RELEASE_ROOT/training/qwen35_sft_train.jsonl" NUM_EPOCH=1 \
  GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" ROLLOUT_BATCH_SIZE="$ROLLOUT_BATCH_SIZE" SFT_EPOCH_ONLY=1 \
  SAVE_INTERVAL=1 CHECKPOINT_KEEP_LAST=2 SFT_METRICS_DIR="$RUN_ROOT/train_metrics" SAVE_DIR="$RUN_ROOT/checkpoint" RAY_SUBMIT_LOG="$RUN_ROOT/train_submit.log" \
  bash "$launcher"
test -f "$RUN_ROOT/checkpoint/latest_checkpointed_iteration.txt"
expected_steps=$(python -c 'import json,sys;m=json.load(open(sys.argv[1]));print(m["count"]//m["global_batch_size"])' "$RUN_ROOT/resolved_config.json")
python "$project/slime-wd/experiments/aligned_sft/verify_metrics.py" "$RUN_ROOT/train_metrics" "$expected_steps"
iteration=$(cat "$RUN_ROOT/checkpoint/latest_checkpointed_iteration.txt")
printf -v checkpoint '%s/iter_%07d' "$RUN_ROOT/checkpoint" "$iteration"
cd "$slime"
python tools/convert_torch_dist_to_hf.py --input-dir "$checkpoint" --output-dir "$RUN_ROOT/hf" \
  --origin-hf-dir "$model" --add-missing-from-origin-hf
python - "$RUN_ROOT/hf" "$model" <<'PY'
import json,pathlib,sys
out,base=map(pathlib.Path,sys.argv[1:]);index=json.loads((out/'model.safetensors.index.json').read_text());original=json.loads((base/'model.safetensors.index.json').read_text())
assert set(index['weight_map'])==set(original['weight_map'])
for shard in set(index['weight_map'].values()):assert (out/shard).stat().st_size>0
assert (out/'tokenizer_config.json').read_bytes()==(base/'tokenizer_config.json').read_bytes()
PY
touch "$RUN_ROOT/training.complete"
