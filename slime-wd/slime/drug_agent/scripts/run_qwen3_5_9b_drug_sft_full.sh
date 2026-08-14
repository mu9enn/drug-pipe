#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/resolve_slime_env.sh"
source "$SLIME_ENV"

export MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-scripts/models/qwen3.5-9B.sh}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-$DATA/Qwen3.5-9B}"
export REF_LOAD="${REF_LOAD:-$DATA/Qwen3.5-9B_torch_dist}"
export NUM_GPUS="${NUM_GPUS:-8}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
export LR="${LR:-5e-6}"
export MIN_LR="${MIN_LR:-5e-7}"
export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.05}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
export RECOMPUTE_FULL="${RECOMPUTE_FULL:-1}"
export RECOMPUTE_LOSS_FUNCTION="${RECOMPUTE_LOSS_FUNCTION:-1}"
export RECOMPUTE_VOCAB_LOG_PROBS="${RECOMPUTE_VOCAB_LOG_PROBS:-1}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-64}"
export BALANCE_DATA="${BALANCE_DATA:-1}"

exec bash "$SLIME/drug_agent/scripts/run_qwen3_5_4b_drug_sft_full.sh"
