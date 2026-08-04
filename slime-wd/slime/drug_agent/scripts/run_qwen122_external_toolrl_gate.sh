#!/usr/bin/env bash
# Two-update correctness/capacity gate for Qwen3.5-122B-A10B-FP8 with a
# dedicated external SGLang worker.  This is intentionally not the production
# dataset pass: update 0 proves cross-node weight sync + optimizer execution;
# rollout 1 proves generation remains valid after a real in-place update.
set -euo pipefail

SLIME_ENV=${SLIME_ENV:-/root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh}
if [[ ! -f "$SLIME_ENV" ]]; then
  SLIME_ENV=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi
source "$SLIME_ENV"
cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh

MODEL_PROFILE=qwen35-122b-8xh200
source drug_agent/scripts/qwen3_large_profile.sh

: "${ROLLOUT_EXTERNAL_ENGINE_ADDRS:?Set ROLLOUT_EXTERNAL_ENGINE_ADDRS to the dedicated SGLang host:port}"
QWEN122_RUN_ROOT=${QWEN122_RUN_ROOT:-$DRUG_AGENT_RUNS_ROOT/qwen35-122b-fp8official-8xh200_serial_20260803_prod1}
QWEN122_SFT_LOAD=${QWEN122_SFT_LOAD:-$QWEN122_RUN_ROOT/sft}
[[ -s "$QWEN122_SFT_LOAD/latest_checkpointed_iteration.txt" ]] || {
  echo "Completed 122B SFT checkpoint is missing: $QWEN122_SFT_LOAD" >&2
  exit 2
}

GATE_ID=${QWEN122_EXTERNAL_GATE_ID:-qwen35-122b-fp8_external_toolrl_gate_$(date +%Y%m%d_%H%M%S)}
GATE_DIR=${QWEN122_EXTERNAL_GATE_DIR:-$DRUG_AGENT_RUNS_ROOT/$GATE_ID}
if [[ -e "$GATE_DIR" ]]; then
  echo "Gate output already exists: $GATE_DIR" >&2
  exit 2
fi
mkdir -p "$GATE_DIR"

export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export ROLLOUT_EXTERNAL=1
export ROLLOUT_EXTERNAL_NUM_GPUS=8
export SKIP_RAY_RESTART=1
export RAY_DASHBOARD_ADDRESS=${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}
export NUM_GPUS=8
export PROMPT_DATA=$TOOLRL_DATA
export LOAD=$QWEN122_SFT_LOAD
export SAVE_DIR=$GATE_DIR/checkpoint_unused
export DISABLE_CHECKPOINT_SAVE=1

# Two prompts x four candidates preserves per-prompt GRPO groups while giving
# the gate two chances to obtain non-equal dense MolClaw rewards.  Dynamic
# batching retains the measured 6,144-token per-GPU actor ceiling.
export NUM_ROLLOUT=2
export ROLLOUT_BATCH_SIZE=2
export N_SAMPLES_PER_PROMPT=4
export GLOBAL_BATCH_SIZE=8
export ADVANTAGE_ESTIMATOR=grpo
export TOOLRL_REWARD_MODE=molclaw
export USE_ROLLOUT_LOGPROBS=1
export USE_KL_LOSS=0
export LR=${QWEN122_EXTERNAL_GATE_LR:-1e-8}
export LR_DECAY_STYLE=constant
export ROLLOUT_TEMPERATURE=${QWEN122_EXTERNAL_GATE_TEMPERATURE:-1.0}
export ROLLOUT_MAX_PROMPT_LEN=${QWEN122_EXTERNAL_GATE_MAX_PROMPT_LEN:-8192}
export ROLLOUT_MAX_RESPONSE_LEN=${QWEN122_EXTERNAL_GATE_MAX_RESPONSE_LEN:-256}
export ROLLOUT_MAX_CONTEXT_LEN=$((ROLLOUT_MAX_PROMPT_LEN + ROLLOUT_MAX_RESPONSE_LEN))
export MAX_TOKENS_PER_GPU=${QWEN122_EXTERNAL_GATE_MAX_TOKENS_PER_GPU:-6144}
export SGLANG_DISABLE_CUDA_GRAPH=1
export SGLANG_DISABLE_CUSTOM_ALL_REDUCE=1
export SGLANG_DISABLE_OVERLAP_SCHEDULE=1
export SGLANG_MEM_FRACTION_STATIC=${QWEN122_EXTERNAL_GATE_MEM_FRACTION:-0.25}
export SGLANG_KV_CACHE_DTYPE=fp8_e4m3

cat > "$GATE_DIR/gate_config.env" <<EOF
MODEL_PROFILE=$MODEL_PROFILE
QWEN122_SFT_LOAD=$QWEN122_SFT_LOAD
ROLLOUT_EXTERNAL_ENGINE_ADDRS=$ROLLOUT_EXTERNAL_ENGINE_ADDRS
NUM_ROLLOUT=$NUM_ROLLOUT
ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE
N_SAMPLES_PER_PROMPT=$N_SAMPLES_PER_PROMPT
GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE
LR=$LR
ROLLOUT_MAX_PROMPT_LEN=$ROLLOUT_MAX_PROMPT_LEN
ROLLOUT_MAX_RESPONSE_LEN=$ROLLOUT_MAX_RESPONSE_LEN
MAX_TOKENS_PER_GPU=$MAX_TOKENS_PER_GPU
EOF

echo "[qwen122-external-gate] output=$GATE_DIR sft=$QWEN122_SFT_LOAD engine=$ROLLOUT_EXTERNAL_ENGINE_ADDRS"
bash drug_agent/toolrl/scripts/run_toolrl_grpo.sh 2>&1 | tee "$GATE_DIR/gate.log"

python3 - "$GATE_DIR/gate.log" <<'PY'
import ast
import math
import re
import sys

path = sys.argv[1]
ansi = re.compile(r"\x1b\[[0-9;]*m")
steps = {}
perfs = {}
with open(path, encoding="utf-8", errors="replace") as f:
    for raw in f:
        line = ansi.sub("", raw)
        m = re.search(r"model\.py:\d+ - step (\d+): (\{.*\})", line)
        if m:
            steps[int(m.group(1))] = ast.literal_eval(m.group(2))
        m = re.search(r"rollout\.py:\d+ - perf (\d+): (\{.*\})", line)
        if m:
            perfs[int(m.group(1))] = ast.literal_eval(m.group(2))

missing = [i for i in (0, 1) if i not in steps or i not in perfs]
if missing:
    raise SystemExit(f"gate is incomplete; missing rollout/step metrics for {missing}")
for i in (0, 1):
    vals = (float(steps[i]["train/loss"]), float(steps[i]["train/grad_norm"]))
    if not all(math.isfinite(x) for x in vals):
        raise SystemExit(f"gate step {i} has non-finite loss/grad_norm: {vals}")
    if float(perfs[i]["rollout/repetition_frac"]) >= 1.0:
        raise SystemExit(f"gate rollout {i} is fully repetitive")
    if float(perfs[i]["rollout/truncated_ratio"]) >= 1.0:
        raise SystemExit(f"gate rollout {i} is fully truncated")
if all(float(steps[i]["train/grad_norm"]) == 0.0 for i in (0, 1)):
    raise SystemExit("both gate updates have zero gradient; rerun with fresh prompts before production")
print("qwen122_external_toolrl_gate=PASS")
PY

touch "$GATE_DIR/PASS"
