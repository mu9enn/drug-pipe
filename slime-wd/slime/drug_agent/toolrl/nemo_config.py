"""Pinned encoder/runtime settings shared by preparation and GPU execution."""
from pathlib import Path

REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
MODEL = "Qwen/Qwen3-Embedding-8B"
MODEL_CACHE = Path("/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub")
MODEL_PATH = MODEL_CACHE / "models--Qwen--Qwen3-Embedding-8B/snapshots" / REVISION
ENGINE = {"max_model_len":32768, "tensor_parallel_size":1, "dtype":"bfloat16",
          "runner":"pooling", "enforce_eager":True, "seed":42,
          "gpu_memory_utilization":0.80, "max_num_seqs":4, "max_num_batched_tokens":32768,
          "pooler_config":{"seq_pooling_type":"LAST", "use_activation":True,
                           "enable_chunked_processing":False}}
