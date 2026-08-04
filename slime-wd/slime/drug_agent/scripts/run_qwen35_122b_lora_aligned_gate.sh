#!/usr/bin/env bash
# One real long-context update against an independently aligned SFT-FP8
# rollout base.  Production may start only after this gate succeeds.
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
GATE_DIR=${QWEN122_LORA_GATE_DIR:-$DRUG_AGENT_RUNS_ROOT/qwen35-122b-lora-aligned-gate_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$GATE_DIR"

export NUM_GPUS=8
export PROMPT_DATA=${QWEN122_GATE_PROMPT_DATA:-$DRUG_AGENT_DATA_ROOT/live_tool_catalog_v1/toolrl/toolrl_steps_ctx10240.jsonl}
[[ -f "$PROMPT_DATA" ]] || { echo "Missing gate data: $PROMPT_DATA" >&2; exit 2; }
export ROLLOUT_HF_CHECKPOINT=$SFT_HF_FP8
export LOAD=$SFT_LOAD REF_LOAD=$SFT_LOAD SAVE_DIR=$GATE_DIR
export DISABLE_CHECKPOINT_SAVE=1 NO_SAVE_OPTIM=1
export MEGATRON_LORA=1 MEGATRON_LORA_RANK=32 MEGATRON_LORA_ALPHA=64 MEGATRON_LORA_DROPOUT=0.0
export MEGATRON_LORA_SYNC_DIR=$GATE_DIR/adapter_current MEGATRON_LORA_SKIP_INITIAL_BASE_SYNC=1
export NUM_ROLLOUT=${QWEN122_GATE_NUM_ROLLOUT:-1}
export ROLLOUT_BATCH_SIZE=${QWEN122_GATE_ROLLOUT_BATCH_SIZE:-8}
export N_SAMPLES_PER_PROMPT=${QWEN122_GATE_N_SAMPLES_PER_PROMPT:-1}
export GLOBAL_BATCH_SIZE=${QWEN122_GATE_GLOBAL_BATCH_SIZE:-8}
export ADVANTAGE_ESTIMATOR=reinforce_plus_plus NORMALIZE_ADVANTAGES=1
export TOOLRL_REWARD_MODE=${TOOLRL_REWARD_MODE:-molclaw}
export USE_ROLLOUT_LOGPROBS=${USE_ROLLOUT_LOGPROBS:-1}
export USE_KL_LOSS=${USE_KL_LOSS:-0}
export LR=2e-7 MIN_LR=2e-8 LR_DECAY_STYLE=cosine LR_WARMUP_FRACTION=0
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.8}
export ROLLOUT_MAX_PROMPT_LEN=${QWEN122_GATE_ROLLOUT_MAX_PROMPT_LEN:-10240}
export ROLLOUT_MAX_RESPONSE_LEN=${QWEN122_GATE_ROLLOUT_MAX_RESPONSE_LEN:-2048}
export ROLLOUT_MAX_CONTEXT_LEN=12288 MAX_TOKENS_PER_GPU=${QWEN122_GATE_MAX_TOKENS_PER_GPU:-6144}
export COLOCATE_OFFLOAD_TRAIN=0 COLOCATE_OFFLOAD_ROLLOUT=0
export SGLANG_MEM_FRACTION_STATIC=${QWEN122_GATE_SGLANG_MEM_FRACTION_STATIC:-0.25} SGLANG_DISABLE_CUDA_GRAPH=1
export SGLANG_DISABLE_CUSTOM_ALL_REDUCE=1 SGLANG_DISABLE_OVERLAP_SCHEDULE=1
unset SGLANG_KV_CACHE_DTYPE
export OFFLOAD_OPTIMIZER_STATES=0 OPTIMIZER_CPU_OFFLOAD=0
export MAIN_GRADS_DTYPE=fp32 MAIN_PARAMS_DTYPE=fp32 EXP_AVG_DTYPE=fp32 EXP_AVG_SQ_DTYPE=fp32
export RECOMPUTE_FULL=1 RECOMPUTE_VOCAB_LOG_PROBS=1 LOG_PROBS_CHUNK_SIZE=512

bash drug_agent/toolrl/scripts/run_toolrl_grpo.sh 2>&1 | tee "$GATE_DIR/gate.log"

python - "$GATE_DIR" <<'PY'
import ast, json, math, os, re, sys
from pathlib import Path
from safetensors import safe_open

root = Path(sys.argv[1])
ansi = re.compile(r"\x1b\[[0-9;]*m")
step = perf = None
for raw in (root / "gate.log").open(errors="replace"):
    line = ansi.sub("", raw)
    match = re.search(r"model\.py:\d+ - step 0: (\{.*\})", line)
    if match:
        step = ast.literal_eval(match.group(1))
    match = re.search(r"rollout\.py:\d+ - perf 0: (\{.*\})", line)
    if match:
        perf = ast.literal_eval(match.group(1))
if not step or not perf:
    raise SystemExit("gate lacks step/rollout metrics")
for key in ("train/loss", "train/grad_norm"):
    if not math.isfinite(float(step[key])):
        raise SystemExit(f"non-finite {key}: {step[key]}")
if float(step["train/grad_norm"]) <= 0:
    raise SystemExit("gate update has zero gradient")
max_ppo_kl = float(os.environ.get("QWEN122_GATE_MAX_PPO_KL", "0.1"))
max_clipfrac = float(os.environ.get("QWEN122_GATE_MAX_PG_CLIPFRAC", "0.1"))
if abs(float(step["train/ppo_kl"])) > max_ppo_kl:
    raise SystemExit(f"train/ppo_kl exceeds {max_ppo_kl}: {step['train/ppo_kl']}")
if float(step["train/pg_clipfrac"]) > max_clipfrac:
    raise SystemExit(f"train/pg_clipfrac exceeds {max_clipfrac}: {step['train/pg_clipfrac']}")
adapter = root / "adapter_current" / "adapter_model.safetensors"
with safe_open(adapter, framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
    gdn = [key for key in keys if ".linear_attn." in key]
    if len(keys) != 384 or gdn:
        raise SystemExit(f"incomplete adapter export: all={len(keys)} gdn={len(gdn)}")
summary = {"step": step, "perf": perf, "adapter_keys": len(keys), "gdn_keys": len(gdn)}
(root / "gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
PY
touch "$GATE_DIR/PASS"
