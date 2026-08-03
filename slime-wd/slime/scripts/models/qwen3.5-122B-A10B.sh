NLAYERS=48
FIRST_K_DENSE_REPLACE=0

arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done

printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

# Qwen3.5-122B-A10B text-tower configuration.  Values are intentionally
# explicit so Megatron validates them against the nested HF text_config before
# loading any weights.  The HF source has one MTP layer, but this config omits
# it intentionally; conversion and training therefore exclude MTP unless a
# separately audited launch config adds both MTP construction and training.
MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 32
   --num-query-groups 2
   --kv-channels 256
   --num-layers 48
   --hidden-size 3072
   --ffn-hidden-size 1024
   --use-gated-attention

   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 248320

   --rotary-base 10000000

   # MoE: 256 total experts, 8 routed experts plus one shared expert.
   --moe-ffn-hidden-size 1024
   --moe-shared-expert-intermediate-size 1024
   --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall
   --moe-router-topk 8
   --moe-layer-freq "$MOE_LAYER_FREQ"
   --num-experts 256
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-router-dtype fp32
   --moe-permute-fusion
   --moe-aux-loss-coeff 0

   # Qwen3.5-specific gates.
   --attention-output-gate
   --moe-shared-expert-gate
)
