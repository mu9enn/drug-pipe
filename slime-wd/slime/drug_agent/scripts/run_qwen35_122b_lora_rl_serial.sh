#!/usr/bin/env bash
# Single-node 8xH200 production sequence for the SFT-aligned
# Qwen3.5-122B-A10B-FP8 policy: ToolRL, then GAD stages 2/3.
set -euo pipefail

SLIME_ENV=${SLIME_ENV:-/root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh}
[[ -f "$SLIME_ENV" ]] || SLIME_ENV=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
source "$SLIME_ENV"
cd "$SLIME"

MODEL_PROFILE=qwen35-122b-8xh200
source drug_agent/scripts/qwen3_large_profile.sh

BASE_RUN=${QWEN122_BASE_RUN:-$DRUG_AGENT_RUNS_ROOT/qwen35-122b-fp8official-8xh200_serial_20260803_prod1}
SFT_LOAD=${QWEN122_SFT_LOAD:-$BASE_RUN/sft}
SFT_HF_FP8=${QWEN122_SFT_HF_FP8:-$BASE_RUN/sft_hf_fp8_v2}
TOOLRL_DATA=${QWEN122_TOOLRL_DATA:-$DRUG_AGENT_DATA_ROOT/live_tool_catalog_v1/toolrl/toolrl_steps_ctx10240.jsonl}
GAD_DATA=${QWEN122_GAD_DATA:-$DRUG_AGENT_DATA_ROOT/live_tool_catalog_v1/gad/gad_steps_ctx10240.jsonl}
# Do not inherit the generic large-model profile's 0.8B fallback.  The 4B
# discriminator is the measured GAD quality/capacity point for this 8xH200 run.
DISCRIMINATOR_MODEL_PATH=${QWEN122_DISCRIMINATOR_MODEL_PATH:-$DATA/Qwen3.5-4B}
RUN_ROOT=${QWEN122_LORA_RUN_ROOT:-$DRUG_AGENT_RUNS_ROOT/qwen35-122b-lora-rl_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RUN_ROOT/logs"

for path in "$SFT_LOAD/latest_checkpointed_iteration.txt" "$SFT_HF_FP8/config.json" \
  "$SFT_HF_FP8/model.safetensors.index.json" "$TOOLRL_DATA" "$GAD_DATA" "$DISCRIMINATOR_MODEL_PATH/config.json"; do
  [[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done

export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NUM_GPUS=8
export ROLLOUT_HF_CHECKPOINT=$SFT_HF_FP8
export REF_LOAD=$SFT_LOAD
export ROLLOUT_NUM_GPUS_PER_ENGINE=8
export TENSOR_MODEL_PARALLEL_SIZE=2 PIPELINE_MODEL_PARALLEL_SIZE=4 CONTEXT_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=2 EXPERT_TENSOR_PARALLEL_SIZE=1
export COLOCATE_OFFLOAD_TRAIN=0 COLOCATE_OFFLOAD_ROLLOUT=0
export SGLANG_MEM_FRACTION_STATIC=0.25 GAD_SGLANG_MEM_FRACTION_STATIC=0.25
export SGLANG_DISABLE_CUDA_GRAPH=1 SGLANG_DISABLE_CUSTOM_ALL_REDUCE=1 SGLANG_DISABLE_OVERLAP_SCHEDULE=1
# BF16 KV cache is retained for reward fidelity; official SGLang guidance
# warns that uncalibrated FP8 KV scales can reduce accuracy.
unset SGLANG_KV_CACHE_DTYPE
export ROLLOUT_MAX_PROMPT_LEN=10240 ROLLOUT_MAX_RESPONSE_LEN=2048 ROLLOUT_MAX_CONTEXT_LEN=12288
export MAX_TOKENS_PER_GPU=${QWEN122_MAX_TOKENS_PER_GPU:-10240}
export RECOMPUTE_FULL=1 RECOMPUTE_VOCAB_LOG_PROBS=1 LOG_PROBS_CHUNK_SIZE=512
export OFFLOAD_OPTIMIZER_STATES=0 OPTIMIZER_CPU_OFFLOAD=0
export MAIN_GRADS_DTYPE=fp32 MAIN_PARAMS_DTYPE=fp32 EXP_AVG_DTYPE=fp32 EXP_AVG_SQ_DTYPE=fp32
# The pinned full-parameter profile needs delayed FP8 because blockwise tensors
# cannot back an FP16 optimizer shard.  LoRA only optimizes the tiny FP32
# adapter, so blockwise is valid here and was 2.3x faster in the real 122B gate.
export FP8_RECIPE=${QWEN122_LORA_FP8_RECIPE:-blockwise}
export MEGATRON_LORA=1 MEGATRON_LORA_RANK=32 MEGATRON_LORA_ALPHA=64 MEGATRON_LORA_DROPOUT=0.0
export MEGATRON_LORA_SKIP_INITIAL_BASE_SYNC=1
export DISABLE_CHECKPOINT_SAVE=1 NO_SAVE_OPTIM=1
# SGLang and Megatron differ by ~14-15 nats/token for this hybrid GDN/MoE
# checkpoint even after FP8/BF16 and temperature alignment gates.  Recompute
# the frozen old policy in Megatron (standard two-policy PPO/GSPO) so the first
# update starts at ratio=1 instead of clipping most tokens immediately.
export USE_ROLLOUT_LOGPROBS=0

cat > "$RUN_ROOT/resolved_config.env" <<EOF
BASE_RUN=$BASE_RUN
SFT_LOAD=$SFT_LOAD
SFT_HF_FP8=$SFT_HF_FP8
TOOLRL_DATA=$TOOLRL_DATA
GAD_DATA=$GAD_DATA
DISCRIMINATOR_MODEL_PATH=$DISCRIMINATOR_MODEL_PATH
LORA_RANK=$MEGATRON_LORA_RANK
LORA_ALPHA=$MEGATRON_LORA_ALPHA
TP=$TENSOR_MODEL_PARALLEL_SIZE
PP=$PIPELINE_MODEL_PARALLEL_SIZE
EP=$EXPERT_MODEL_PARALLEL_SIZE
PROMPT_LEN=$ROLLOUT_MAX_PROMPT_LEN
RESPONSE_LEN=$ROLLOUT_MAX_RESPONSE_LEN
KV_CACHE_DTYPE=bf16
FP8_RECIPE=$FP8_RECIPE
MAX_TOKENS_PER_GPU=$MAX_TOKENS_PER_GPU
USE_ROLLOUT_LOGPROBS=$USE_ROLLOUT_LOGPROBS
EOF

echo "[122B LoRA serial] run_root=$RUN_ROOT"

if [[ ! -f "$RUN_ROOT/TOOLRL_DONE" ]]; then
  export PROMPT_DATA=$TOOLRL_DATA LOAD=$SFT_LOAD SAVE_DIR=$RUN_ROOT/toolrl
  export MEGATRON_LORA_SYNC_DIR=$SAVE_DIR/adapter_current
  export ROLLOUT_BATCH_SIZE=${TOOLRL_RBS:-8} N_SAMPLES_PER_PROMPT=1 GLOBAL_BATCH_SIZE=${TOOLRL_GBS:-8}
  DATASET_SIZE=$(wc -l < "$PROMPT_DATA")
  export NUM_ROLLOUT=${TOOLRL_NUM_ROLLOUT:-$(((DATASET_SIZE + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE))}
  export ADVANTAGE_ESTIMATOR=reinforce_plus_plus NORMALIZE_ADVANTAGES=1
  export TOOLRL_REWARD_MODE=molclaw USE_KL_LOSS=0
  export LR=${TOOLRL_LORA_LR:-2e-7} MIN_LR=${TOOLRL_LORA_MIN_LR:-2e-8}
  export LR_DECAY_STYLE=cosine LR_WARMUP_FRACTION=0.03 WEIGHT_DECAY=0.01 ROLLOUT_TEMPERATURE=0.8
  mkdir -p "$SAVE_DIR"
  bash drug_agent/toolrl/scripts/run_toolrl_grpo.sh 2>&1 | tee "$RUN_ROOT/logs/toolrl.log"
  touch "$RUN_ROOT/TOOLRL_DONE"
fi

GAD_ROOT=$RUN_ROOT/gad
NEGATIVE_CACHE=$GAD_ROOT/stage2_negatives.jsonl
WARMUP_DIR=$GAD_ROOT/discriminator_warmup
mkdir -p "$GAD_ROOT"

if [[ ! -f "$RUN_ROOT/GAD_NEGATIVES_DONE" ]]; then
  export PROMPT_DATA=$GAD_DATA GAD_NEGATIVE_CACHE=$NEGATIVE_CACHE
  export STUDENT_LOAD=$SFT_LOAD GAD_NEGATIVE_ROLLOUT_ONLY=1
  export ROLLOUT_BATCH_SIZE=${GAD_NEGATIVE_RBS:-8}
  DATASET_SIZE=$(wc -l < "$PROMPT_DATA")
  export NUM_ROLLOUT=${GAD_NEGATIVE_NUM_ROLLOUT:-$(((DATASET_SIZE + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE))}
  bash drug_agent/gad/scripts/generate_stage2_negatives.sh 2>&1 | tee "$RUN_ROOT/logs/gad_stage2_negatives.log"
  [[ $(wc -l < "$NEGATIVE_CACHE") -ge $DATASET_SIZE ]] || {
    echo "GAD negative cache is incomplete" >&2; exit 3;
  }
  touch "$RUN_ROOT/GAD_NEGATIVES_DONE"
fi

if [[ ! -f "$RUN_ROOT/GAD_WARMUP_DONE" ]]; then
  export PAIRS=$NEGATIVE_CACHE GENERATOR_WARMUP_LOAD=$SFT_LOAD
  export DISCRIMINATOR_OUTPUT_DIR=$WARMUP_DIR
  export DISCRIMINATOR_EPOCHS=1 DISCRIMINATOR_BATCH_SIZE=2 DISCRIMINATOR_LR=1e-5
  export DISCRIMINATOR_MAX_LENGTH=4096 DISCRIMINATOR_SAVE_INTERVAL=100
  bash drug_agent/gad/scripts/run_stage2_discriminator_warmup.sh 2>&1 | tee "$RUN_ROOT/logs/gad_discriminator_warmup.log"
  touch "$RUN_ROOT/GAD_WARMUP_DONE"
fi

if [[ ! -f "$RUN_ROOT/GAD_DONE" ]]; then
  DISCRIMINATOR_WARMUP_LOAD=$WARMUP_DIR/latest
  GAD_WARMUP_MANIFEST=$WARMUP_DIR/warmup_manifest.json
  SERVICE_DIR=$GAD_ROOT/discriminator_online
  SERVICE_PORT=${GAD_SERVICE_PORT:-8100}
  CUDA_VISIBLE_DEVICES=0 python -m drug_agent.gad.service \
    --model-path "$DISCRIMINATOR_MODEL_PATH" --resume "$DISCRIMINATOR_WARMUP_LOAD" \
    --output-dir "$SERVICE_DIR" --host 127.0.0.1 --port "$SERVICE_PORT" \
    --device cuda:0 --lr 1e-5 --max-length 4096 --update-steps 1 \
    --save-interval 50 --keep-last-checkpoints 2 --offload-after-request \
    > "$RUN_ROOT/logs/gad_discriminator_service.log" 2>&1 &
  SERVICE_PID=$!
  trap 'kill "$SERVICE_PID" 2>/dev/null || true' EXIT
  for _ in $(seq 1 180); do
    curl -fsS "http://127.0.0.1:$SERVICE_PORT/health" >/dev/null 2>&1 && break
    sleep 2
  done
  curl -fsS "http://127.0.0.1:$SERVICE_PORT/health" >/dev/null

  export PROMPT_DATA=$GAD_DATA STUDENT_WARMUP_LOAD=$SFT_LOAD
  export DISCRIMINATOR_WARMUP_LOAD GAD_WARMUP_MANIFEST
  export GAD_DISCRIMINATOR_URL=http://127.0.0.1:$SERVICE_PORT
  export GAD_REWARD_MODE=hybrid GAD_REWARD_COEF=0.8 GAD_FORMAT_REWARD_COEF=0.1 GAD_TOOL_REWARD_COEF=0.1
  export SAVE_DIR=$GAD_ROOT/policy MEGATRON_LORA_SYNC_DIR=$GAD_ROOT/policy/adapter_current
  export ROLLOUT_BATCH_SIZE=${GAD_RBS:-4} N_SAMPLES_PER_PROMPT=${GAD_N_SAMPLES:-2} GLOBAL_BATCH_SIZE=${GAD_GBS:-8}
  DATASET_SIZE=$(wc -l < "$PROMPT_DATA")
  export NUM_ROLLOUT=${GAD_NUM_ROLLOUT:-$(((DATASET_SIZE + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE))}
  export ADVANTAGE_ESTIMATOR=gspo
  export DYNAMIC_SAMPLING_FILTER_PATH=slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
  export DYNAMIC_SAMPLING_MAX_DROPPED_GROUPS=32
  export USE_KL_LOSS=1 KL_LOSS_COEF=0.001
  export STUDENT_LR=${GAD_LORA_LR:-1e-7} STUDENT_MIN_LR=${GAD_LORA_MIN_LR:-1e-8}
  export LR_DECAY_STYLE=cosine LR_WARMUP_FRACTION=0.03 WEIGHT_DECAY=0.01 ROLLOUT_TEMPERATURE=0.8
  bash drug_agent/gad/scripts/run_stage3_gad_grpo.sh 2>&1 | tee "$RUN_ROOT/logs/gad_stage3.log"
  curl -fsS -X POST "http://127.0.0.1:$SERVICE_PORT/checkpoint" -H 'Content-Type: application/json' -d '{}' >/dev/null
  touch "$RUN_ROOT/GAD_DONE"
fi

echo "[122B LoRA serial] complete: $RUN_ROOT"
