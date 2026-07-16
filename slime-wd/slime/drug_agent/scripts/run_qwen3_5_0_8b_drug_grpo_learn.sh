#!/bin/bash
set -ex

if [ -f /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh ]; then
  source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
else
  source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi

cd "$SLIME"
source drug_agent/scripts/reject_legacy_online_training.sh

unset RAY_ADDRESS || true
pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
pkill -9 ray python 2>/dev/null || true
sleep 2

export PYTHONBUFFERED=16
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

NUM_GPUS=${NUM_GPUS:-2}
REAL_CPU=${REAL_CPU:-$(nproc)}
SCRIPT_DIR="$SLIME/drug_agent"
OUTPUTS_ROOT=${OUTPUTS_ROOT:-${WD:-$GROUP_SPACE/slime-wd}/outputs}
DRUG_AGENT_DATA_ROOT=${DRUG_AGENT_DATA_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_data}
DRUG_AGENT_RUNS_ROOT=${DRUG_AGENT_RUNS_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_runs}
mkdir -p "$DRUG_AGENT_DATA_ROOT" "$DRUG_AGENT_RUNS_ROOT"

PROMPT_DATA=${PROMPT_DATA:-$DRUG_AGENT_DATA_ROOT/grpo/mixed.jsonl}
if [ ! -f "$PROMPT_DATA" ]; then
  echo "PROMPT_DATA not found: $PROMPT_DATA"
  exit 2
fi

NUM_ROLLOUT=${NUM_ROLLOUT:-10}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-4}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-2048}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.55}
LR=${LR:-1e-6}

SAVE_DIR=${SAVE_DIR:-$DRUG_AGENT_RUNS_ROOT/Qwen3.5-0.8B_drug_grpo_learn}
SAVE_INTERVAL=${SAVE_INTERVAL:-5}
HF_CHECKPOINT=${HF_CHECKPOINT:-$DATA/Qwen3.5-0.8B}
REF_LOAD=${REF_LOAD:-$DATA/Qwen3.5-0.8B_torch_dist}
LOAD=${LOAD:-}

export DRUG_AGENT_RUN_NAME=${DRUG_AGENT_RUN_NAME:-qwen3.5_0.8b_drug_grpo_learn}
export DRUG_AGENT_ALLOWLIST_PATH=${DRUG_AGENT_ALLOWLIST_PATH:-$SLIME/drug_agent/tools/allowlist_v0.json}
export DRUG_AGENT_ALLOW_ALL=${DRUG_AGENT_ALLOW_ALL:-0}
export DRUG_AGENT_ROLLOUT_MODE=${DRUG_AGENT_ROLLOUT_MODE:-train_strict}
export DRUG_AGENT_ALLOW_PARSE_RECOVERY=${DRUG_AGENT_ALLOW_PARSE_RECOVERY:-0}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

source scripts/models/qwen3.5-0.8B.sh

CKPT_ARGS=(
  --hf-checkpoint "$HF_CHECKPOINT"
  --ref-load "$REF_LOAD"
  --save "$SAVE_DIR"
  --save-interval "$SAVE_INTERVAL"
)
if [ -n "$LOAD" ]; then
  CKPT_ARGS+=(--load "$LOAD")
fi

ROLLOUT_ARGS=(
  --prompt-data "$PROMPT_DATA"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --rollout-shuffle

  --reward-key score

  --num-rollout "$NUM_ROLLOUT"
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
  --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1}"

  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --balance-data
)

PERF_ARGS=(
  --tensor-model-parallel-size 1
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef "${KL_LOSS_COEF:-0.00}"
  --kl-loss-type "${KL_LOSS_TYPE:-low_var_kl}"
  --entropy-coef "${ENTROPY_COEF:-0.00}"
  --eps-clip "${EPS_CLIP:-0.2}"
  --eps-clip-high "${EPS_CLIP_HIGH:-0.28}"
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "$LR"
  --lr-decay-style "${LR_DECAY_STYLE:-constant}"
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --adam-beta1 "${ADAM_BETA1:-0.9}"
  --adam-beta2 "${ADAM_BETA2:-0.98}"
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC"
)

CUSTOM_ARGS=(
  --custom-generate-function-path drug_agent.rollout.generate_with_drug_agent.generate
  --custom-rm-path drug_agent.rollout.reward_func.reward_func
  --custom-rollout-log-function-path drug_agent.rollout.trajectory_logger.log_rollout_data
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

ray start --head \
  --node-ip-address "$MASTER_ADDR" \
  --num-gpus "$NUM_GPUS" \
  --num-cpus "$REAL_CPU" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHON_CPU_FIX_DIR}:/root/Megatron-LM/:${SLIME}:${SCRIPT_DIR:-}:${PYTHONPATH:-}\",
    \"PYTHON_CPU_COUNT\": \"${REAL_CPU}\",
    \"PATH\": \"${PATH}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"CUDA_HOME\": \"${CUDA_HOME:-/usr/local/cuda}\",
    \"NVIDIA_VISIBLE_DEVICES\": \"${NVIDIA_VISIBLE_DEVICES:-all}\",
    \"NVIDIA_DRIVER_CAPABILITIES\": \"${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_IB_DISABLE\": \"${NCCL_IB_DISABLE:-1}\",
    \"OUTPUTS_ROOT\": \"${OUTPUTS_ROOT}\",
    \"DRUG_AGENT_DATA_ROOT\": \"${DRUG_AGENT_DATA_ROOT}\",
    \"DRUG_AGENT_RUNS_ROOT\": \"${DRUG_AGENT_RUNS_ROOT}\",
    \"DRUG_AGENT_ALLOWLIST_PATH\": \"${DRUG_AGENT_ALLOWLIST_PATH:-}\",
    \"DRUG_AGENT_ALLOW_ALL\": \"${DRUG_AGENT_ALLOW_ALL:-0}\",
    \"DRUG_AGENT_ROLLOUT_MODE\": \"${DRUG_AGENT_ROLLOUT_MODE:-train_strict}\",
    \"DRUG_AGENT_ALLOW_PARSE_RECOVERY\": \"${DRUG_AGENT_ALLOW_PARSE_RECOVERY:-0}\",
    \"DRUG_AGENT_RUN_NAME\": \"${DRUG_AGENT_RUN_NAME:-}\",
    \"MOLCLAW_SCP_SERVER_URL\": \"${MOLCLAW_SCP_SERVER_URL:-}\",
    \"MOLCLAW_SCP_API_KEY\": \"${MOLCLAW_SCP_API_KEY:-}\",
    \"MOLCLAW_CONNECT_TIMEOUT_SEC\": \"${MOLCLAW_CONNECT_TIMEOUT_SEC:-}\",
    \"MOLCLAW_LIST_TOOLS_TIMEOUT_SEC\": \"${MOLCLAW_LIST_TOOLS_TIMEOUT_SEC:-}\",
    \"MOLCLAW_TOOL_TIMEOUT_SEC\": \"${MOLCLAW_TOOL_TIMEOUT_SEC:-}\",
    \"MOLCLAW_TOOL_HEARTBEAT_SEC\": \"${MOLCLAW_TOOL_HEARTBEAT_SEC:-}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$NUM_GPUS" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}"
