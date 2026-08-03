#!/usr/bin/env bash
# Source-only defaults for the large Qwen drug-agent experiments.
# Usage: MODEL_PROFILE=qwen36-35b-4xh200 source drug_agent/scripts/qwen3_large_profile.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file after slime_env.sh; do not execute it directly." >&2
  exit 2
fi
if [[ -z "${SLIME:-}" || -z "${GROUP_SPACE:-}" ]]; then
  echo "Source slime_env.sh before qwen3_large_profile.sh" >&2
  return 2
fi

MODEL_PROFILE=${MODEL_PROFILE:?Set MODEL_PROFILE to qwen35-27b-4xh200, qwen36-35b-4xh200, or qwen35-122b-8xh200}
GROUP_SPACE_ROOT=$(dirname "$GROUP_SPACE")
GPFS2_PUBLIC=${GPFS2_PUBLIC:-$GROUP_SPACE_ROOT/gpfs2-shared-public}
# Worker images expose GPFS2 through group-space/huggingface, while some login
# hosts use the explicit group-space/gpfs2-shared-public mount.
if [[ ! -d "$GPFS2_PUBLIC/huggingface" && -d "$GROUP_SPACE_ROOT/huggingface" ]]; then
  GPFS2_PUBLIC=$GROUP_SPACE_ROOT
fi
LIVE_DATA_ROOT=${LIVE_DATA_ROOT:-$WD/outputs/slime_drug_agent_data/live_tool_catalog_v1}

export CANONICAL_DATA=${CANONICAL_DATA:-$LIVE_DATA_ROOT/react_trajectories.jsonl}
export TOOLRL_DATA=${TOOLRL_DATA:-$LIVE_DATA_ROOT/toolrl/toolrl_steps.jsonl}
export GAD_DATA=${GAD_DATA:-$LIVE_DATA_ROOT/gad/gad_steps.jsonl}
export DISCRIMINATOR_MODEL_PATH=${DISCRIMINATOR_MODEL_PATH:-$GROUP_SPACE/drug-pipe/cached/archive_20260730_150855/models/slime-wd/data/Qwen3.5-0.8B}

# Shared memory-safety defaults.  Optimizer state is the dominant cost for
# full-parameter tuning; H200 capacity does not make unsharded Adam affordable.
export OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD:-1}
export OVERLAP_CPU_OPTIMIZER_D2H_H2D=${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-1}
export USE_PRECISION_AWARE_OPTIMIZER=${USE_PRECISION_AWARE_OPTIMIZER:-1}
export RECOMPUTE_FULL=${RECOMPUTE_FULL:-1}
export RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
export LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-1024}
export RECOMPUTE_LOSS_FUNCTION=${RECOMPUTE_LOSS_FUNCTION:-1}
export ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-98304}
export ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-4096}
export ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-102400}
# The dataset median is about 10K tokens. Left truncation keeps the candidate,
# and 8K is the first H200-tested discriminator window; track truncation before
# attempting 16K/32K in the shared-GPU online phase.
export DISCRIMINATOR_MAX_LENGTH=${DISCRIMINATOR_MAX_LENGTH:-8192}
# Keep the Megatron actor resident during colocated RL. Slime's actor sleep is
# a separate whole-CUDA-allocation CPU backup; combining it with optimizer CPU
# offload exceeded 517 GiB after an otherwise successful 27B GRPO update.
export COLOCATE_OFFLOAD_TRAIN=${COLOCATE_OFFLOAD_TRAIN:-0}

case "$MODEL_PROFILE" in
  qwen35-27b-4xh200)
    export MODEL_NAME=Qwen3.5-27B
    export MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-27B.sh}
    export HF_CHECKPOINT=${HF_CHECKPOINT:-$GROUP_SPACE/drug-pipe/cached/archive_20260730_150855/models/slime-wd/data/Qwen3.5-27B}
    export REF_LOAD=${REF_LOAD:-$WD/data/Qwen3.5-27B_torch_dist}
    export NUM_GPUS=${NUM_GPUS:-4}
    export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
    export PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
    export CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
    export EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
    export EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
    export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}
    export MOE_ENABLE_DEEPEP=0
    export TRAINABLE_PARAM_BILLIONS=${TRAINABLE_PARAM_BILLIONS:-26.896}
    # A real 47.5k-token step succeeded with TP2/PP2, full recompute and 40%
    # optimizer offload.  TP2/CP2 replicated optimizer state and crossed the
    # 517 GiB worker's Ray memory threshold at both 70% and 80% offload.
    export OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0.4}
    # The colocated one-group gate completed end-to-end at 70% optimizer
    # offload and a 0.18 SGLang pool.  60% left too much actor state on HBM;
    # 80% drove the node to 512.82/517.58 GiB during CPUAdam.
    export SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.18}
    # RL must leave enough HBM for the colocated SGLang weight + hybrid cache
    # to resume while the actor remains resident. SFT does not need this.
    export TOOLRL_OPTIMIZER_OFFLOAD_FRACTION=${TOOLRL_OPTIMIZER_OFFLOAD_FRACTION:-0.7}
    export GAD_OPTIMIZER_OFFLOAD_FRACTION=${GAD_OPTIMIZER_OFFLOAD_FRACTION:-0.7}
    export GAD_SGLANG_MEM_FRACTION_STATIC=${GAD_SGLANG_MEM_FRACTION_STATIC:-0.18}
    export ACCUMULATE_ALLREDUCE_GRADS_IN_FP32=${ACCUMULATE_ALLREDUCE_GRADS_IN_FP32:-1}
    export MAIN_GRADS_DTYPE=${MAIN_GRADS_DTYPE:-fp32}
    export MAIN_PARAMS_DTYPE=${MAIN_PARAMS_DTYPE:-fp32}
    export EXP_AVG_DTYPE=${EXP_AVG_DTYPE:-fp32}
    export EXP_AVG_SQ_DTYPE=${EXP_AVG_SQ_DTYPE:-fp32}
    export MIN_HOST_MEMORY_GIB=${MIN_HOST_MEMORY_GIB:-480}
    # The 0.8B discriminator plus the 27B CPUAdam update reached
    # 507.31/517.58 GiB and was killed even with Ray's threshold at 98%.
    # Keep the ordinary SFT/ToolRL floor above, but reject an online GAD
    # generator probe unless the worker has meaningful transient headroom.
    export GAD_MIN_HOST_MEMORY_GIB=${GAD_MIN_HOST_MEMORY_GIB:-600}
    export HOST_MEMORY_RESERVE_GIB=${HOST_MEMORY_RESERVE_GIB:-96}
    # CPUAdam transiently reaches about 492/517.58 GiB. Keep Ray's OOM
    # protection enabled, but move it above the measured 95% false-positive
    # boundary; 97% still reserves roughly 15.5 GiB for the node.
    export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.97}
    export SFT_LR=${SFT_LR:-1e-6}
    export SFT_MIN_LR=${SFT_MIN_LR:-1e-7}
    export TOOLRL_LR=${TOOLRL_LR:-1e-7}
    export GAD_LR=${GAD_LR:-5e-8}
    ;;
  qwen36-35b-4xh200)
    export MODEL_NAME=Qwen3.6-35B-A3B
    # Qwen3.6-35B-A3B retains the qwen3_5_moe HF architecture and uses the
    # same Megatron model spec as Qwen3.5-35B-A3B.
    export MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-35B-A3B.sh}
    export HF_CHECKPOINT=${HF_CHECKPOINT:-$GPFS2_PUBLIC/huggingface/zskj-hub/models--Qwen--Qwen3.6-35B-A3B}
    export REF_LOAD=${REF_LOAD:-$WD/data/Qwen3.6-35B-A3B_torch_dist}
    export NUM_GPUS=${NUM_GPUS:-4}
    export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
    export PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
    export CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
    export EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-2}
    export EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
    export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}
    export MOE_ENABLE_DEEPEP=${MOE_ENABLE_DEEPEP:-1}
    export TRAINABLE_PARAM_BILLIONS=${TRAINABLE_PARAM_BILLIONS:-34.661}
    export OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0.4}
    export SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.35}
    export TOOLRL_OPTIMIZER_OFFLOAD_FRACTION=${TOOLRL_OPTIMIZER_OFFLOAD_FRACTION:-0.6}
    export GAD_OPTIMIZER_OFFLOAD_FRACTION=${GAD_OPTIMIZER_OFFLOAD_FRACTION:-0.7}
    export GAD_SGLANG_MEM_FRACTION_STATIC=${GAD_SGLANG_MEM_FRACTION_STATIC:-0.20}
    export ACCUMULATE_ALLREDUCE_GRADS_IN_FP32=${ACCUMULATE_ALLREDUCE_GRADS_IN_FP32:-1}
    export MAIN_GRADS_DTYPE=${MAIN_GRADS_DTYPE:-fp32}
    export MAIN_PARAMS_DTYPE=${MAIN_PARAMS_DTYPE:-fp32}
    export EXP_AVG_DTYPE=${EXP_AVG_DTYPE:-fp32}
    export EXP_AVG_SQ_DTYPE=${EXP_AVG_SQ_DTYPE:-fp32}
    export MIN_HOST_MEMORY_GIB=${MIN_HOST_MEMORY_GIB:-480}
    # 35B GAD is not yet measured. Its larger FP32 CPUAdam state plus the
    # colocated discriminator must not inherit the already-tight 480 GiB gate.
    export GAD_MIN_HOST_MEMORY_GIB=${GAD_MIN_HOST_MEMORY_GIB:-700}
    export HOST_MEMORY_RESERVE_GIB=${HOST_MEMORY_RESERVE_GIB:-96}
    export SFT_LR=${SFT_LR:-2e-6}
    export SFT_MIN_LR=${SFT_MIN_LR:-2e-7}
    export TOOLRL_LR=${TOOLRL_LR:-2e-7}
    export GAD_LR=${GAD_LR:-1e-7}
    ;;
  qwen35-122b-8xh200)
    export MODEL_NAME=Qwen3.5-122B-A10B
    export MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-122B-A10B.sh}
    export HF_CHECKPOINT=${HF_CHECKPOINT:-$GPFS2_PUBLIC/huggingface/zskj-hub/models--Qwen--Qwen3.5-122B-A10B}
    export REF_LOAD=${REF_LOAD:-$WD/data/Qwen3.5-122B-A10B_torch_dist}
    export NUM_GPUS=${NUM_GPUS:-8}
    export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
    export PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-4}
    export CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
    export EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-2}
    export EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
    export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}
    # At DP=1 and on-node NVLink, ordinary all-to-all avoids DeepEP's extra
    # communication buffers and leaves more room for the first Adam update.
    export MOE_ENABLE_DEEPEP=${MOE_ENABLE_DEEPEP:-0}
    export TRAINABLE_PARAM_BILLIONS=${TRAINABLE_PARAM_BILLIONS:-122.112}
    export SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.25}
    # A 1 TiB host cannot use HybridDeviceOptimizer: its CPU path forces four
    # FP32 tensors and needs >1.27 TiB at 70% offload. The H200 probe instead
    # uses TE FusedAdam with FP8 primary weights and two-byte optimizer tensors,
    # offloading complete optimizer states between updates. Keep the original
    # CPUAdam path available only as an explicit override.
    export OPTIMIZER_CPU_OFFLOAD=${QWEN122_OPTIMIZER_CPU_OFFLOAD:-0}
    export OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0}
    export TOOLRL_OPTIMIZER_OFFLOAD_FRACTION=${TOOLRL_OPTIMIZER_OFFLOAD_FRACTION:-0}
    export GAD_OPTIMIZER_OFFLOAD_FRACTION=${GAD_OPTIMIZER_OFFLOAD_FRACTION:-0}
    export OFFLOAD_OPTIMIZER_STATES=${OFFLOAD_OPTIMIZER_STATES:-1}
    # Train-only SFT can retain the two-byte main weights in HBM (the measured
    # optimizer peak still leaves ~8.5 GiB) and offload only FP8 moments. A
    # second CPU copy of the main weights pushed this 1 TiB pod to 1016 GiB.
    # Colocated RL launchers do not consume this SFT-specific setting.
    export SFT_OFFLOAD_OPTIMIZER_MASTER_WEIGHTS=${SFT_OFFLOAD_OPTIMIZER_MASTER_WEIGHTS:-0}
    export FP8_PARAM_GATHER=${FP8_PARAM_GATHER:-1}
    export FP8_FORMAT=${FP8_FORMAT:-e4m3}
    # In this pinned MCore, BlockwiseQTensor cannot be viewed into an FP16
    # optimizer shard (shard_float16_groups becomes None). Delayed Float8Tensor
    # supports that path and lets TE retain a two-byte main parameter copy.
    export FP8_RECIPE=${FP8_RECIPE:-delayed}
    export GAD_SGLANG_MEM_FRACTION_STATIC=${GAD_SGLANG_MEM_FRACTION_STATIC:-0.18}
    export ACCUMULATE_ALLREDUCE_GRADS_IN_FP32=${ACCUMULATE_ALLREDUCE_GRADS_IN_FP32:-0}
    export MAIN_GRADS_DTYPE=${MAIN_GRADS_DTYPE:-bf16}
    # This pinned MCore accepts BF16 gradients/moments but only FP16 (not
    # BF16) for the two-byte main parameter copy.
    export MAIN_PARAMS_DTYPE=${MAIN_PARAMS_DTYPE:-fp16}
    # BF16 moments reached 139.98/140.06 GiB while allocating the first Adam
    # state on H200. TE's scaled FP8 moments reduce this peak by about 29 GiB.
    export EXP_AVG_DTYPE=${EXP_AVG_DTYPE:-fp8}
    export EXP_AVG_SQ_DTYPE=${EXP_AVG_SQ_DTYPE:-fp8}
    # TE lazily creates hundreds of scaled optimizer tensors on the first
    # update. Expandable allocator segments reduce otherwise-fatal 2--3 GiB
    # fragmentation at the H200 capacity boundary.
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
    export MIN_HOST_MEMORY_GIB=${MIN_HOST_MEMORY_GIB:-900}
    export GAD_MIN_HOST_MEMORY_GIB=${GAD_MIN_HOST_MEMORY_GIB:-1000}
    export HOST_MEMORY_RESERVE_GIB=${HOST_MEMORY_RESERVE_GIB:-128}
    export SFT_LR=${SFT_LR:-5e-7}
    export SFT_MIN_LR=${SFT_MIN_LR:-5e-8}
    export TOOLRL_LR=${TOOLRL_LR:-5e-8}
    export GAD_LR=${GAD_LR:-2e-8}
    ;;
  *)
    echo "Unknown MODEL_PROFILE: $MODEL_PROFILE" >&2
    return 2
    ;;
esac

echo "[large-profile] profile=$MODEL_PROFILE model=$MODEL_NAME gpus=$NUM_GPUS"
echo "[large-profile] TP=$TENSOR_MODEL_PARALLEL_SIZE PP=$PIPELINE_MODEL_PARALLEL_SIZE CP=$CONTEXT_PARALLEL_SIZE EP=$EXPERT_MODEL_PARALLEL_SIZE ETP=$EXPERT_TENSOR_PARALLEL_SIZE rollout_tp=$ROLLOUT_NUM_GPUS_PER_ENGINE"
echo "[large-profile] optimizer_offload=$OPTIMIZER_OFFLOAD_FRACTION main_grad=$MAIN_GRADS_DTYPE main_param=$MAIN_PARAMS_DTYPE exp_avg=$EXP_AVG_DTYPE exp_avg_sq=$EXP_AVG_SQ_DTYPE min_host_gib=$MIN_HOST_MEMORY_GIB"
echo "[large-profile] hf=$HF_CHECKPOINT ref=$REF_LOAD"
