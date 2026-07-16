#!/bin/bash
set -ex

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh

cd $SLIME

unset RAY_ADDRESS || true
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray python 2>/dev/null || true
sleep 3

export PYTHONBUFFERED=16
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

SCRIPT_DIR="$SLIME/examples/retool"

source scripts/models/qwen3.5-0.8B.sh

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
CKPT_ARGS=(
  --hf-checkpoint $DATA/Qwen3.5-0.8B
  --ref-load $DATA/Qwen3.5-0.8B_torch_dist
  --save $OUTPUTS_ROOT/slime_runs/Qwen3.5-0.8B_retool_grpo_smoke
  --save-interval 9999
)

# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
ROLLOUT_ARGS=(
  --prompt-data $DATA/dapo-math-17k/dapo-math-17k.jsonl
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle

  --reward-key score

  --num-rollout 3
  --rollout-batch-size 4
  --n-samples-per-prompt 4
  --rollout-max-response-len 1024
  --rollout-temperature 1

  --global-batch-size 16
  --balance-data
)

CUSTOM_ARGS=(
  --custom-generate-function-path generate_with_retool.generate
  --custom-rm-path generate_with_retool.reward_func
)

# ---------------------------------------------------------------------------
# Parallel / performance
# ---------------------------------------------------------------------------
PERF_ARGS=(
  --tensor-model-parallel-size 1
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1

  --use-dynamic-batch-size
  --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.55
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

# ---------------------------------------------------------------------------
# Ray
# ---------------------------------------------------------------------------
NUM_GPUS=${NUM_GPUS:-2}
REAL_CPU=${REAL_CPU:-$(nproc)}

ray start --head \
  --node-ip-address "$MASTER_ADDR" \
  --num-gpus "$NUM_GPUS" \
  --num-cpus "$REAL_CPU" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHON_CPU_FIX_DIR}:/root/Megatron-LM/:${SLIME}:${SCRIPT_DIR}:${PYTHONPATH:-}\",
    \"PYTHON_CPU_COUNT\": \"${REAL_CPU}\",
    \"PATH\": \"${PATH}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"CUDA_HOME\": \"${CUDA_HOME:-/usr/local/cuda}\",
    \"NVIDIA_VISIBLE_DEVICES\": \"${NVIDIA_VISIBLE_DEVICES:-all}\",
    \"NVIDIA_DRIVER_CAPABILITIES\": \"${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_IB_DISABLE\": \"${NCCL_IB_DISABLE:-1}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$NUM_GPUS" \
  --colocate \
  ${MODEL_ARGS[@]} \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}"
