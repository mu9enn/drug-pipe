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

MODEL_PROFILE=${MODEL_PROFILE:?Set MODEL_PROFILE to qwen35-27b-4xh200, qwen35-27b-8xh200, qwen36-35b-4xh200, qwen36-35b-8xh200, or qwen35-122b-8xh200}
GROUP_SPACE_ROOT=$(dirname "$GROUP_SPACE")
GPFS2_PUBLIC=${GPFS2_PUBLIC:-$GROUP_SPACE_ROOT/gpfs2-shared-public}
# Worker images expose GPFS2 through group-space/huggingface, while some login
# hosts use the explicit group-space/gpfs2-shared-public mount.
if [[ ! -d "$GPFS2_PUBLIC/huggingface" && -d "$GROUP_SPACE_ROOT/huggingface" ]]; then
  GPFS2_PUBLIC=$GROUP_SPACE_ROOT
fi
LIVE_DATA_ROOT=${LIVE_DATA_ROOT:-$WD/outputs/slime_drug_agent_data/live_tool_catalog_v2}

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
export BALANCE_DATA=${BALANCE_DATA:-1}
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
    export COLOCATED_TOOLRL_SUPPORTED=${COLOCATED_TOOLRL_SUPPORTED:-1}
    export COLOCATED_GAD_SUPPORTED=${COLOCATED_GAD_SUPPORTED:-1}
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
    # These are unverified starting profiles, not a claim that the online
    # actor/rollout memory gate has passed.
    export COLOCATED_TOOLRL_SUPPORTED=${COLOCATED_TOOLRL_SUPPORTED:-1}
    export COLOCATED_GAD_SUPPORTED=${COLOCATED_GAD_SUPPORTED:-1}
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
  qwen35-27b-8xh200)
    export MODEL_NAME=Qwen3.5-27B
    export MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-27B.sh}
    export HF_CHECKPOINT=${HF_CHECKPOINT:-$GROUP_SPACE/drug-pipe/cached/archive_20260730_150855/models/slime-wd/data/Qwen3.5-27B}
    export REF_LOAD=${REF_LOAD:-$WD/data/Qwen3.5-27B_torch_dist}
    export NUM_GPUS=${NUM_GPUS:-8}
    # TP2/PP2 leaves DP2, which is faster than PP4 on this workload and shards
    # Adam across two replicas.  FP32 moments repeatedly missed H200 capacity
    # in the second update; BF16 moments plus the uneven pipeline below passed
    # two real updates without CPU offload at 2368.7 tokens/s.
    export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
    export PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
    export CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
    export EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
    export EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
    export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}
    export COLOCATED_TOOLRL_SUPPORTED=${COLOCATED_TOOLRL_SUPPORTED:-1}
    export COLOCATED_GAD_SUPPORTED=${COLOCATED_GAD_SUPPORTED:-1}
    export MOE_ENABLE_DEEPEP=0
    export TRAINABLE_PARAM_BILLIONS=${TRAINABLE_PARAM_BILLIONS:-26.896}
    export OPTIMIZER_CPU_OFFLOAD=0
    export OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0}
    # Move two transformer layers from the LM-head/loss pipeline rank to the
    # much lighter embedding rank. Equal 32/32 stages left the final rank at
    # 140.00 GiB and OOMed on a 478 MiB allocation during the second update.
    export NUM_LAYERS_IN_FIRST_PIPELINE_STAGE=${NUM_LAYERS_IN_FIRST_PIPELINE_STAGE:-34}
    export NUM_LAYERS_IN_LAST_PIPELINE_STAGE=${NUM_LAYERS_IN_LAST_PIPELINE_STAGE:-30}
    # vocab/TP=124160; a 1024-token FP32 log-prob tile needs ~485 MiB and was
    # the exact failing allocation. 512 still missed by 5.5 MiB after Adam
    # initialization, so use a 256-token (~121 MiB) tile for steady state.
    export LOG_PROBS_CHUNK_SIZE=${QWEN27_8_LOG_PROBS_CHUNK_SIZE:-256}
    # NVIDIA's supported precision-aware Adam path halves moment storage while
    # retaining FP32 master parameters and accumulated gradients.
    export SFT_EXP_AVG_DTYPE=${SFT_EXP_AVG_DTYPE:-bf16}
    export SFT_EXP_AVG_SQ_DTYPE=${SFT_EXP_AVG_SQ_DTYPE:-bf16}
    export TOOLRL_OPTIMIZER_OFFLOAD_FRACTION=${TOOLRL_OPTIMIZER_OFFLOAD_FRACTION:-0.35}
    export GAD_OPTIMIZER_OFFLOAD_FRACTION=${GAD_OPTIMIZER_OFFLOAD_FRACTION:-0.4}
    export TOOLRL_OPTIMIZER_CPU_OFFLOAD=${TOOLRL_OPTIMIZER_CPU_OFFLOAD:-1}
    export GAD_OPTIMIZER_CPU_OFFLOAD=${GAD_OPTIMIZER_CPU_OFFLOAD:-1}
    export SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.22}
    export GAD_SGLANG_MEM_FRACTION_STATIC=${GAD_SGLANG_MEM_FRACTION_STATIC:-0.18}
    # Custom all-reduce v2 fails while sharing CUDA-graph inputs at TP4 in the
    # pinned H200/SGLang image. NCCL all-reduce preserves CUDA graphs and is the
    # measured-safe fallback for both ToolRL and GAD rollout engines.
    export SGLANG_DISABLE_CUSTOM_ALL_REDUCE=${SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-1}
    # Qwen3.5 GDN/Mamba CUDA graphs can retain stale state after Slime's
    # in-place tensor weight update.  The 27B production run reproduced
    # corrupted post-update generations with graphs enabled and remained
    # correct in eager mode.  External 122B rollout also receives an update
    # every optimizer step, so keep the same correctness-first default until
    # tensor-update graph recapture is supported and validated end to end.
    export SGLANG_DISABLE_CUDA_GRAPH=${SGLANG_DISABLE_CUDA_GRAPH:-1}
    # DP overlap is a faster short/mid-context mode, but its communication
    # buffers caused a measured 139.68 GiB first-step OOM on the long bucket.
    export OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-0}
    export OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-0}
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
    export ACCUMULATE_ALLREDUCE_GRADS_IN_FP32=${ACCUMULATE_ALLREDUCE_GRADS_IN_FP32:-1}
    export MAIN_GRADS_DTYPE=${MAIN_GRADS_DTYPE:-fp32}
    export MAIN_PARAMS_DTYPE=${MAIN_PARAMS_DTYPE:-fp32}
    export EXP_AVG_DTYPE=${EXP_AVG_DTYPE:-fp32}
    export EXP_AVG_SQ_DTYPE=${EXP_AVG_SQ_DTYPE:-fp32}
    export MIN_HOST_MEMORY_GIB=${MIN_HOST_MEMORY_GIB:-700}
    export GAD_MIN_HOST_MEMORY_GIB=${GAD_MIN_HOST_MEMORY_GIB:-800}
    export HOST_MEMORY_RESERVE_GIB=${HOST_MEMORY_RESERVE_GIB:-128}
    export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.97}
    export SFT_LR=${SFT_LR:-1e-6}
    export SFT_MIN_LR=${SFT_MIN_LR:-1e-7}
    export TOOLRL_LR=${TOOLRL_LR:-1e-7}
    export GAD_LR=${GAD_LR:-5e-8}
    ;;
  qwen36-35b-8xh200)
    export MODEL_NAME=Qwen3.6-35B-A3B
    export MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-35B-A3B.sh}
    export HF_CHECKPOINT=${HF_CHECKPOINT:-$GPFS2_PUBLIC/huggingface/zskj-hub/models--Qwen--Qwen3.6-35B-A3B}
    export REF_LOAD=${REF_LOAD:-$WD/data/Qwen3.6-35B-A3B_torch_dist}
    export NUM_GPUS=${NUM_GPUS:-8}
    export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
    export PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
    export CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
    export EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-2}
    export EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
    export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}
    export COLOCATED_TOOLRL_SUPPORTED=${COLOCATED_TOOLRL_SUPPORTED:-1}
    export COLOCATED_GAD_SUPPORTED=${COLOCATED_GAD_SUPPORTED:-1}
    # DeepEP's largest gains are cross-node; ordinary NVLink all-to-all uses
    # fewer buffers on this single-node EP2 profile and is the baseline gate.
    export MOE_ENABLE_DEEPEP=${MOE_ENABLE_DEEPEP:-0}
    export TRAINABLE_PARAM_BILLIONS=${TRAINABLE_PARAM_BILLIONS:-34.661}
    export OPTIMIZER_CPU_OFFLOAD=0
    export OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0}
    export NUM_LAYERS_IN_FIRST_PIPELINE_STAGE=${NUM_LAYERS_IN_FIRST_PIPELINE_STAGE:-22}
    export NUM_LAYERS_IN_LAST_PIPELINE_STAGE=${NUM_LAYERS_IN_LAST_PIPELINE_STAGE:-18}
    export LOG_PROBS_CHUNK_SIZE=${QWEN35_8_LOG_PROBS_CHUNK_SIZE:-256}
    export SFT_EXP_AVG_DTYPE=${SFT_EXP_AVG_DTYPE:-bf16}
    export SFT_EXP_AVG_SQ_DTYPE=${SFT_EXP_AVG_SQ_DTYPE:-bf16}
    export TOOLRL_OPTIMIZER_OFFLOAD_FRACTION=${TOOLRL_OPTIMIZER_OFFLOAD_FRACTION:-0.4}
    export GAD_OPTIMIZER_OFFLOAD_FRACTION=${GAD_OPTIMIZER_OFFLOAD_FRACTION:-0.5}
    export TOOLRL_OPTIMIZER_CPU_OFFLOAD=${TOOLRL_OPTIMIZER_CPU_OFFLOAD:-1}
    export GAD_OPTIMIZER_CPU_OFFLOAD=${GAD_OPTIMIZER_CPU_OFFLOAD:-1}
    export SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.22}
    export GAD_SGLANG_MEM_FRACTION_STATIC=${GAD_SGLANG_MEM_FRACTION_STATIC:-0.18}
    export OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-0}
    export OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-0}
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
    export ACCUMULATE_ALLREDUCE_GRADS_IN_FP32=${ACCUMULATE_ALLREDUCE_GRADS_IN_FP32:-1}
    export MAIN_GRADS_DTYPE=${MAIN_GRADS_DTYPE:-fp32}
    export MAIN_PARAMS_DTYPE=${MAIN_PARAMS_DTYPE:-fp32}
    export EXP_AVG_DTYPE=${EXP_AVG_DTYPE:-fp32}
    export EXP_AVG_SQ_DTYPE=${EXP_AVG_SQ_DTYPE:-fp32}
    export MIN_HOST_MEMORY_GIB=${MIN_HOST_MEMORY_GIB:-800}
    export GAD_MIN_HOST_MEMORY_GIB=${GAD_MIN_HOST_MEMORY_GIB:-900}
    export HOST_MEMORY_RESERVE_GIB=${HOST_MEMORY_RESERVE_GIB:-128}
    export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.97}
    export SFT_LR=${SFT_LR:-2e-6}
    export SFT_MIN_LR=${SFT_MIN_LR:-2e-7}
    export TOOLRL_LR=${TOOLRL_LR:-2e-7}
    export GAD_LR=${GAD_LR:-1e-7}
    ;;
  qwen35-122b-8xh200)
    export MODEL_NAME=Qwen3.5-122B-A10B-FP8
    export MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-122B-A10B.sh}
    # Use the downloaded official Qwen FP8 release as the single source for
    # both actor conversion and rollout.  REF_LOAD is a distinct torch_dist
    # checkpoint produced by the Qwen3.5 FP8-aware bridge; never reuse the
    # older BF16-derived actor checkpoint under this profile.
    export HF_CHECKPOINT=${HF_CHECKPOINT:-$GROUP_SPACE/slime_wd/data/Qwen3.5-122B-A10B-FP8}
    export ROLLOUT_HF_CHECKPOINT=${ROLLOUT_HF_CHECKPOINT:-$HF_CHECKPOINT}
    export REF_LOAD=${REF_LOAD:-$WD/data/Qwen3.5-122B-A10B-FP8_torch_dist}
    export NUM_GPUS=${NUM_GPUS:-8}
    export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
    export PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-4}
    export CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
    export EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-2}
    export EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
    # The streamed FP8-state path now releases each FP32 expansion immediately.
    # Under the old optimizer, uniform 12/12/12/12 made the embedding stage the
    # late bottleneck; under the streamed path, 11/12/13/12 instead made the
    # 13-layer third stage miss a 24 MiB expansion. Re-test the only layout with
    # no 13-layer stage. Backslashes protect pipes when Ray reconstructs argv.
    export NUM_LAYERS_IN_FIRST_PIPELINE_STAGE=${NUM_LAYERS_IN_FIRST_PIPELINE_STAGE:-12}
    export NUM_LAYERS_IN_LAST_PIPELINE_STAGE=${NUM_LAYERS_IN_LAST_PIPELINE_STAGE:-12}
    export PIPELINE_MODEL_PARALLEL_LAYOUT=${PIPELINE_MODEL_PARALLEL_LAYOUT:-'Et*12\|t*12\|t*12\|t*12L'}
    export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}
    # Keep the online launchers gated until a complete rollout + actor update
    # succeeds.  The old BF16 rollout could not create a KV pool, whereas the
    # block-FP8 rollout below has ample KV capacity on the same eight H200s.
    export COLOCATED_TOOLRL_SUPPORTED=${COLOCATED_TOOLRL_SUPPORTED:-0}
    export COLOCATED_GAD_SUPPORTED=${COLOCATED_GAD_SUPPORTED:-0}
    # BF16 rollout could not allocate any KV cache. FP8 rollout compute fits,
    # but pausing the 132 GiB/GPU actor creates a host backup which, together
    # with optimizer state, exceeds this worker's 1 TiB cgroup.
    export COLOCATE_OFFLOAD_TRAIN=1
    # At DP=1 and on-node NVLink, ordinary all-to-all avoids DeepEP's extra
    # communication buffers and leaves more room for the first Adam update.
    export MOE_ENABLE_DEEPEP=${MOE_ENABLE_DEEPEP:-0}
    export TRAINABLE_PARAM_BILLIONS=${TRAINABLE_PARAM_BILLIONS:-122.112}
    export SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.25}
    export SGLANG_KV_CACHE_DTYPE=${SGLANG_KV_CACHE_DTYPE:-fp8_e4m3}
    # SGLang custom-all-reduce v2 fails while sharing CUDA-graph inputs at TP8
    # in this exact H200 image (CUDA invalid argument). NCCL all-reduce keeps
    # CUDA graphs enabled and avoids falling back to graphless inference.
    export SGLANG_DISABLE_CUSTOM_ALL_REDUCE=${SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-1}
    # The dedicated external SGLang server disables overlap scheduling so an
    # in-flight decode cannot race Slime's in-place full-weight update.  The
    # external-engine contract compares this field strictly, so every 122B
    # ToolRL/GAD client must declare the same value.
    export SGLANG_DISABLE_OVERLAP_SCHEDULE=${SGLANG_DISABLE_OVERLAP_SCHEDULE:-1}
    # TorchMemorySaver normally reserves 1 GiB.  In the 122B colocated probe
    # all four forward/backward microbatches completed, but that artificial
    # floor rejected a 24 MiB transient in TE FusedAdam.  Retain a measured
    # 512 MiB guard while making the otherwise free HBM usable by the update.
    export TRAIN_MEMORY_MARGIN_BYTES=${TRAIN_MEMORY_MARGIN_BYTES:-536870912}
    # Colocated SGLang retains roughly 4.7 GiB/GPU of CUDA runtime state even
    # while its weights/KV are asleep.  Bound each TE Adam group more tightly
    # than train-only SFT so its lazy FP8-state initialization and FP32
    # expansion complete before the group is evicted to CPU.
    export SLIME_FP8_OPTIMIZER_MAX_GROUP_NUMEL=${SLIME_FP8_OPTIMIZER_MAX_GROUP_NUMEL:-33554432}
    # The fused CE retains every chunk's softmax clone until backward, so
    # smaller chunks alone merely postpone the same OOM. Recompute softmax in
    # backward. The recompute Function uses one full-tensor autograd node but
    # performs FP32 normalization in bounded tiles during both passes.
    export RECOMPUTE_VOCAB_LOG_PROBS=${RECOMPUTE_VOCAB_LOG_PROBS:-1}
    # Full logits remain BF16; each 512-token tile uses about 243 MiB for its
    # FP32 normalization workspace. This is well inside the measured ~2 GiB
    # steady-state margin and avoids the 8x loop overhead of 64-token tiles.
    export LOG_PROBS_CHUNK_SIZE=${QWEN122_LOG_PROBS_CHUNK_SIZE:-512}
    # A 1 TiB host cannot use HybridDeviceOptimizer: its CPU path forces four
    # FP32 tensors and needs >1.27 TiB at 70% offload. The H200 probe instead
    # uses TE FusedAdam with FP8 primary weights and two-byte optimizer tensors.
    # The successful SFT path evicts pageable FP8 moments between updates while
    # retaining FP16 master weights in HBM. Keep the original CPUAdam path
    # available only as an explicit override.
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
    # The same moments-only rule is required by colocated RL. TorchMemorySaver
    # already backs up the resident actor allocation while SGLang runs; adding
    # a second FP16 master-weight CPU copy reproduced the >1 TiB failure.
    export OFFLOAD_OPTIMIZER_MASTER_WEIGHTS=${OFFLOAD_OPTIMIZER_MASTER_WEIGHTS:-0}
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
    # Native allocation retained 2.89 GiB as fragmented reserve and failed on
    # an earlier 20 MiB master-weight expansion. Expandable segments progressed
    # through initialization and all four microbatches, so retain that measured
    # allocator while the uniform layout removes the former 13-layer hotspot.
    export PYTORCH_CUDA_ALLOC_CONF=${QWEN122_PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
    export MIN_HOST_MEMORY_GIB=${MIN_HOST_MEMORY_GIB:-900}
    export GAD_MIN_HOST_MEMORY_GIB=${GAD_MIN_HOST_MEMORY_GIB:-1000}
    export HOST_MEMORY_RESERVE_GIB=${HOST_MEMORY_RESERVE_GIB:-128}
    # The optimizer computation itself fits, but the following TorchMemorySaver
    # pause backs up roughly 120 GiB from each actor and drove the 1 TiB cgroup
    # above even Ray's 99% threshold. Do not hide that real post-step host peak
    # by weakening/disabling the monitor; colocated 122B RL remains fail-closed.
    export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.95}
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
