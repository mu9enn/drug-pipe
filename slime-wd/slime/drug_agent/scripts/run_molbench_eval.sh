#!/usr/bin/env bash
set -euo pipefail

if [[ -f /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh ]]; then
  source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
else
  source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi
cd "$SLIME"

: "${MODEL_CHECKPOINT:?Set MODEL_CHECKPOINT to a Slime torch-distributed checkpoint directory}"
EVAL_MODE=${EVAL_MODE:-molbench}
MOLBENCH_ROOT=${MOLBENCH_ROOT:-$GROUP_SPACE/drug_wd/MolClaw/molbench}
PROMPT_FILE=${PROMPT_FILE:-}
PROMPT_SUITE_FILE=${PROMPT_SUITE_FILE:-}
TASK_TYPE=${TASK_TYPE:-e2e}
TASK_ID=${TASK_ID:-manual_prompt_001}
MAX_WORKERS=${MAX_WORKERS:-2}
MAX_STEPS=${MAX_STEPS:-0}
TEMPERATURE=${TEMPERATURE:-0.0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-32768}
TASK_TIMEOUT_SEC=${TASK_TIMEOUT_SEC:-10800}
RUN_NAME=${RUN_NAME:-molbench_$(basename "$MODEL_CHECKPOINT")_$(date +%Y%m%d_%H%M%S)}
DRUG_AGENT_EVAL_ROOT=${DRUG_AGENT_EVAL_ROOT:-${OUTPUTS_ROOT:-$WD/outputs}/slime_drug_agent_evals}
DRUG_AGENT_EVAL_RUN_DIR=${DRUG_AGENT_EVAL_RUN_DIR:-$DRUG_AGENT_EVAL_ROOT/$RUN_NAME}
DRUG_AGENT_L1_SKILLS_ROOT=${DRUG_AGENT_L1_SKILLS_ROOT:-$GROUP_SPACE/drug-pipe/molclaw-skills/.claude/skills/L1_tools}

case "$MODEL_CHECKPOINT" in
  *9B*|*9b*)
    MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-9B.sh}
    HF_CHECKPOINT=${HF_CHECKPOINT:-$DATA/Qwen3.5-9B}
    NUM_GPUS=${NUM_GPUS:-8}
    TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-4}
    PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
    ;;
  *4B*|*4b*)
    MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-4B.sh}
    HF_CHECKPOINT=${HF_CHECKPOINT:-$DATA/Qwen3.5-4B}
    NUM_GPUS=${NUM_GPUS:-4}
    TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-4}
    PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
    ;;
  *)
    : "${MODEL_ARGS_FILE:?Cannot infer model architecture; set MODEL_ARGS_FILE}"
    : "${HF_CHECKPOINT:?Cannot infer tokenizer/config source; set HF_CHECKPOINT}"
    : "${NUM_GPUS:?Cannot infer GPU topology; set NUM_GPUS}"
    : "${TENSOR_MODEL_PARALLEL_SIZE:?Set TENSOR_MODEL_PARALLEL_SIZE}"
    : "${PIPELINE_MODEL_PARALLEL_SIZE:?Set PIPELINE_MODEL_PARALLEL_SIZE}"
    ;;
esac
REF_LOAD=${REF_LOAD:-${HF_CHECKPOINT}_torch_dist}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
REAL_CPU=${REAL_CPU:-$(nproc)}

REQUIRED_PATHS=("$MODEL_CHECKPOINT" "$HF_CHECKPOINT" "$REF_LOAD" "$DRUG_AGENT_L1_SKILLS_ROOT")
case "$EVAL_MODE" in
  molbench) REQUIRED_PATHS+=("$MOLBENCH_ROOT") ;;
  single_prompt)
    : "${PROMPT_FILE:?Set PROMPT_FILE for EVAL_MODE=single_prompt}"
    REQUIRED_PATHS+=("$PROMPT_FILE")
    ;;
  prompt_suite)
    : "${PROMPT_SUITE_FILE:?Set PROMPT_SUITE_FILE for EVAL_MODE=prompt_suite}"
    REQUIRED_PATHS+=("$PROMPT_SUITE_FILE")
    ;;
  *) echo "Unsupported EVAL_MODE=$EVAL_MODE (expected molbench, single_prompt, or prompt_suite)" >&2; exit 2 ;;
esac
for path in "${REQUIRED_PATHS[@]}"; do
  [[ -e "$path" ]] || { echo "Required path not found: $path" >&2; exit 2; }
done
[[ -f "$MODEL_ARGS_FILE" ]] || { echo "MODEL_ARGS_FILE not found: $MODEL_ARGS_FILE" >&2; exit 2; }
source "$MODEL_ARGS_FILE"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
ENV_FILES=()
for candidate in "${MOLCLAW_ENV_FILE:-}" "$REPO_ROOT/data-pipe/.env" "$REPO_ROOT/tool-kg/.env"; do
  [[ -n "$candidate" && -f "$candidate" ]] || continue
  ENV_FILES+=("$candidate")
  set -a
  # shellcheck disable=SC1090
  source "$candidate"
  set +a
done
export MOLCLAW_SCP_SERVER_URL=${MOLCLAW_SCP_SERVER_URL:-${MOLCLAW_SCP_MCP_URL:-}}
export MOLCLAW_SCP_API_KEY=${MOLCLAW_SCP_API_KEY:-${MOLCLAW_SCP_MCP_AUTH:-}}
export MOLCLAW_SCP_AUTH_HEADER=${MOLCLAW_SCP_AUTH_HEADER:-${MOLCLAW_SCP_MCP_AUTH_HEADER:-SCP-HUB-API-KEY}}
export MOLCLAW_CONNECT_TIMEOUT_SEC=${MOLCLAW_CONNECT_TIMEOUT_SEC:-60}
export MOLCLAW_LIST_TOOLS_TIMEOUT_SEC=${MOLCLAW_LIST_TOOLS_TIMEOUT_SEC:-60}
export MOLCLAW_TOOL_TIMEOUT_SEC=${MOLCLAW_TOOL_TIMEOUT_SEC:-14400}
: "${MOLCLAW_SCP_SERVER_URL:?MolClaw server URL is missing from environment/.env}"
: "${MOLCLAW_SCP_API_KEY:?MolClaw API key is missing from environment/.env}"

# GPU workers without egress can route the unchanged MolClaw URL through the
# no-GPU relay.  Keep this evaluation-only; formal training remains offline.
if [[ -n "${MOLCLAW_PROXY_URL:-}" ]]; then
  [[ "$MOLCLAW_PROXY_URL" == http://* || "$MOLCLAW_PROXY_URL" == https://* ]] || {
    echo "MOLCLAW_PROXY_URL must be an http(s) proxy URL" >&2
    exit 2
  }
  export HTTP_PROXY="$MOLCLAW_PROXY_URL"
  export HTTPS_PROXY="$MOLCLAW_PROXY_URL"
  export http_proxy="$MOLCLAW_PROXY_URL"
  export https_proxy="$MOLCLAW_PROXY_URL"
fi

mkdir -p "$DRUG_AGENT_EVAL_RUN_DIR"
ENV_FILE_ARGS=()
for path in "${ENV_FILES[@]}"; do ENV_FILE_ARGS+=(--env-file "$path"); done
export DRUG_AGENT_ALLOW_TOOL_ENV=1
export DRUG_AGENT_TRAINING_OFFLINE=0
export DRUG_AGENT_MAX_WORKERS="$MAX_WORKERS"
export DRUG_AGENT_MAX_STEPS="$MAX_STEPS"
export DRUG_AGENT_TASK_TIMEOUT_SEC="$TASK_TIMEOUT_SEC"
export DRUG_AGENT_WORKSPACES_ROOT="$DRUG_AGENT_EVAL_RUN_DIR/workspaces"
export DRUG_AGENT_EVAL_RUN_DIR MOLBENCH_ROOT DRUG_AGENT_L1_SKILLS_ROOT DRUG_AGENT_WORKSPACES_ROOT

PREFLIGHT_INPUT_ARGS=()
if [[ "$EVAL_MODE" == "molbench" ]]; then
  PREFLIGHT_INPUT_ARGS+=(--molbench-root "$MOLBENCH_ROOT")
elif [[ "$EVAL_MODE" == "prompt_suite" ]]; then
  PREFLIGHT_INPUT_ARGS+=(--prompt-suite-file "$PROMPT_SUITE_FILE")
else
  PREFLIGHT_INPUT_ARGS+=(--prompt-file "$PROMPT_FILE" --task-type "$TASK_TYPE" --task-id "$TASK_ID")
fi
python -m drug_agent.evaluation.preflight \
  --checkpoint "$MODEL_CHECKPOINT" \
  --run-dir "$DRUG_AGENT_EVAL_RUN_DIR" \
  --max-workers "$MAX_WORKERS" \
  --max-steps "$MAX_STEPS" \
  --temperature "$TEMPERATURE" \
  --task-timeout-sec "$TASK_TIMEOUT_SEC" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-context-len "$MAX_CONTEXT_LEN" \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --model-args-file "$MODEL_ARGS_FILE" \
  --num-gpus "$NUM_GPUS" \
  --tensor-model-parallel-size "$TENSOR_MODEL_PARALLEL_SIZE" \
  --pipeline-model-parallel-size "$PIPELINE_MODEL_PARALLEL_SIZE" \
  "${PREFLIGHT_INPUT_ARGS[@]}" \
  "${ENV_FILE_ARGS[@]}"

export DRUG_AGENT_EXPECTED_TOOL_CATALOG="$DRUG_AGENT_EVAL_RUN_DIR/tool_catalog.json"

EVAL_CONFIG="$DRUG_AGENT_EVAL_RUN_DIR/eval_config.yaml"
if [[ "$EVAL_MODE" == "molbench" ]]; then
  EVAL_DATASET="$DRUG_AGENT_EVAL_RUN_DIR/molbench_eval.jsonl"
  EVAL_LOGGER=drug_agent.evaluation.logger.log_eval_rollout_data
  EVAL_TASK_COUNT=186
else
  EVAL_DATASET="$DRUG_AGENT_EVAL_RUN_DIR/prompt_eval.jsonl"
  EVAL_LOGGER=drug_agent.evaluation.prompt_logger.log_eval_rollout_data
  EVAL_TASK_COUNT=$(wc -l < "$EVAL_DATASET")
fi
python - "$EVAL_CONFIG" "$EVAL_DATASET" "$TEMPERATURE" "$MAX_NEW_TOKENS" <<'PY'
import sys
from pathlib import Path
path, dataset, temperature, max_tokens = sys.argv[1:]
Path(path).write_text(
    "eval:\n"
    "  defaults:\n"
    f"    temperature: {float(temperature)}\n"
    "    top_p: 1.0\n"
    "    n_samples_per_eval_prompt: 1\n"
    f"    max_response_len: {int(max_tokens)}\n"
    "    input_key: prompt\n"
    "    label_key: label\n"
    "    metadata_key: metadata\n"
    "  datasets:\n"
    "    - name: drug_agent_online_eval\n"
    f"      path: {dataset}\n"
    "      rm_type: drug_agent_eval\n"
    "      custom_generate_function_path: drug_agent.rollout.generate_with_drug_agent.generate\n",
    encoding="utf-8",
)
PY

if [[ "${RESET_RAY:-1}" == "1" ]]; then
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 sglang >/dev/null 2>&1 || true
fi
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
# The MCP endpoint uses HTTP(S)_PROXY on egress-restricted GPU workers, but
# Ray control-plane traffic must stay local. Without this exclusion, the Ray
# Jobs client or SGLang health checks can send loopback/node-local traffic
# through the relay and either fail or stall before model startup.
NODE_LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
LOCAL_NO_PROXY="127.0.0.1,localhost,::1,$MASTER_ADDR${NODE_LOCAL_IP:+,$NODE_LOCAL_IP}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$LOCAL_NO_PROXY"
export no_proxy="${no_proxy:+$no_proxy,}$LOCAL_NO_PROXY"
ray start --head --node-ip-address "$MASTER_ADDR" --num-gpus "$NUM_GPUS" --num-cpus "$REAL_CPU" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

export PYTHONBUFFERED=16
export PYTHONPATH="${PYTHON_CPU_FIX_DIR:+$PYTHON_CPU_FIX_DIR:}/root/Megatron-LM/:$SLIME:${PYTHONPATH:-}"
RUNTIME_ENV_JSON=$(python - <<'PY'
import json, os
keys = [
    "PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHON_CPU_FIX_DIR", "CUDA_HOME",
    "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "NCCL_IB_DISABLE",
    "MOLCLAW_SCP_SERVER_URL", "MOLCLAW_SCP_API_KEY", "MOLCLAW_SCP_AUTH_HEADER",
    "MOLCLAW_CONNECT_TIMEOUT_SEC", "MOLCLAW_LIST_TOOLS_TIMEOUT_SEC", "MOLCLAW_TOOL_TIMEOUT_SEC",
    "MOLCLAW_PROXY_URL", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "NO_PROXY", "no_proxy",
    "DRUG_AGENT_EVAL_RUN_DIR", "MOLBENCH_ROOT", "DRUG_AGENT_L1_SKILLS_ROOT",
    "DRUG_AGENT_WORKSPACES_ROOT", "DRUG_AGENT_MAX_WORKERS", "DRUG_AGENT_MAX_STEPS",
    "DRUG_AGENT_TASK_TIMEOUT_SEC", "DRUG_AGENT_EXPECTED_TOOL_CATALOG",
]
env = {key: value for key in keys if (value := os.environ.get(key))}
env.update({
    "DRUG_AGENT_ALLOW_TOOL_ENV": "1", "DRUG_AGENT_TRAINING_OFFLINE": "0",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_NVLS_ENABLE": "0",
})
print(json.dumps({"env_vars": env}, separators=(",", ":")))
PY
)

PERF_ARGS=(
  --tensor-model-parallel-size "$TENSOR_MODEL_PARALLEL_SIZE"
  --pipeline-model-parallel-size "$PIPELINE_MODEL_PARALLEL_SIZE"
  --context-parallel-size "$CONTEXT_PARALLEL_SIZE"
  --expert-model-parallel-size "$EXPERT_MODEL_PARALLEL_SIZE"
  --expert-tensor-parallel-size "$EXPERT_TENSOR_PARALLEL_SIZE"
  --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
)
[[ "$TENSOR_MODEL_PARALLEL_SIZE" -gt 1 ]] && PERF_ARGS+=(--sequence-parallel)

# Evaluation favors a reliable startup over CUDA-graph throughput.  In particular,
# SGLang's TP=8 custom all-reduce graph capture is not supported by every H200
# worker/runtime combination.  Training launchers are intentionally unaffected.
SGLANG_RUNTIME_ARGS=()
if [[ "${SGLANG_DISABLE_CUDA_GRAPH:-1}" == "1" ]]; then
  SGLANG_RUNTIME_ARGS+=(--sglang-disable-cuda-graph)
fi
# Custom all-reduce has hung during the first health generation on TP=4 and
# failed CUDA-graph capture on TP=8 on the current H200 image. Evaluation is
# latency-insensitive, so prefer NCCL's stable path. Training is unaffected.
if [[ "${SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-1}" == "1" ]]; then
  SGLANG_RUNTIME_ARGS+=(--sglang-disable-custom-all-reduce)
fi

RAY_SUBMIT_LOG="$DRUG_AGENT_EVAL_RUN_DIR/ray_submit.log"
echo "[drug-agent eval] mode=$EVAL_MODE checkpoint=$MODEL_CHECKPOINT run_dir=$DRUG_AGENT_EVAL_RUN_DIR tasks=$EVAL_TASK_COUNT workers=$MAX_WORKERS"
set +e
ray job submit --address=http://127.0.0.1:8265 --runtime-env-json="$RUNTIME_ENV_JSON" -- \
  python3 train.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node "$NUM_GPUS" --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_CHECKPOINT" --ref-load "$REF_LOAD" --load "$MODEL_CHECKPOINT" \
  --no-load-optim --no-load-rng \
  --num-rollout 0 --rollout-batch-size 1 --n-samples-per-prompt 1 --global-batch-size 1 \
  --lr-decay-iters 1 --eval-interval 1 --eval-config "$EVAL_CONFIG" \
  --custom-rm-path drug_agent.evaluation.reward.reward \
  --custom-eval-rollout-log-function-path "$EVAL_LOGGER" \
  --rollout-num-gpus "$NUM_GPUS" --rollout-num-gpus-per-engine "$NUM_GPUS" \
  --sglang-server-concurrency "$MAX_WORKERS" --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.60}" \
  "${SGLANG_RUNTIME_ARGS[@]}" \
  --eval-max-prompt-len "$MAX_CONTEXT_LEN" --eval-max-context-len "$MAX_CONTEXT_LEN" \
  --attention-dropout 0.0 --hidden-dropout 0.0 --attention-backend flash \
  --optimizer adam --lr 1e-6 --lr-decay-style constant \
  "${PERF_ARGS[@]}" 2>&1 | tee "$RAY_SUBMIT_LOG"
status=${PIPESTATUS[0]}
set -e
exit "$status"
