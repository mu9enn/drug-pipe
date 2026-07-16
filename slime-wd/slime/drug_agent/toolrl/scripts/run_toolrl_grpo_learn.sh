#!/bin/bash
set -ex

export MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-0.8B.sh}
export NUM_GPUS=${NUM_GPUS:-2}
export NUM_ROLLOUT=${NUM_ROLLOUT:-8}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-4}
export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
export NUM_EPOCH=${NUM_EPOCH:-1}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
export SAVE_INTERVAL=${SAVE_INTERVAL:-5}
export LR=${LR:-1e-6}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/run_toolrl_grpo.sh"
