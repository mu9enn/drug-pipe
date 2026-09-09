#!/usr/bin/env bash
set -euo pipefail

# Run inside the persistent one-GPU development worker created with rlaunch.
# The worker itself may use a sleep keepalive; this script is the bounded,
# auditable workload and writes all state to the shared dataset directory.
worker_root=/root/slime_sxy/group-space/sunxiangyu
repo_root="$worker_root/drug-pipe/slime-wd/slime"
dataset_root="$worker_root/drug_wd/drug_pipe_605_local_copy_20260904/v8_toolrl_20260907"
input_path="$dataset_root/01_all_decisions/all_decisions.jsonl"
embedding_root="$dataset_root/03_embedding_cache"
length_root="$dataset_root/02_length_audit_all"
embedding_model=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3-Embedding-0.6B
tokenizer_model="$worker_root/drug-pipe/slime-wd/data/Qwen3.5-9B"
status_path="$dataset_root/rlaunch_workload_status.json"

mkdir -p "$embedding_root" "$length_root"
cd "$repo_root"

on_exit() {
  code=$?
  STATUS_CODE="$code" STATUS_PATH="$status_path" python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

Path(os.environ["STATUS_PATH"]).write_text(json.dumps({
    "exit_code": int(os.environ["STATUS_CODE"]),
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "worker_kind": "one_gpu_rlaunch_development_worker",
    "workload_is_bounded": True,
}, indent=2) + "\n")
PY
  exit "$code"
}
trap on_exit EXIT

python - <<'PY'
import json
from pathlib import Path

import jinja2
import sentence_transformers
import sklearn
import torch
import transformers

assert torch.cuda.is_available(), "CUDA is not available"
assert tuple(int(part) for part in jinja2.__version__.split(".")[:2]) >= (3, 1)
embedding_model = Path("/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3-Embedding-0.6B")
tokenizer_model = Path("/root/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/data/Qwen3.5-9B")
assert (embedding_model / "model.safetensors").is_file()
assert (tokenizer_model / "tokenizer_config.json").is_file()
Path("/root/slime_sxy/group-space/sunxiangyu/drug_wd/drug_pipe_605_local_copy_20260904/v8_toolrl_20260907/rlaunch_preflight.json").write_text(
    json.dumps({
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "scikit_learn": sklearn.__version__,
        "jinja2": jinja2.__version__,
        "embedding_model": str(embedding_model),
        "tokenizer_model": str(tokenizer_model),
    }, indent=2) + "\n"
)
PY

python -m pytest -q \
  drug_agent/tests/test_toolrl_v8_dataset.py \
  drug_agent/tests/test_trajectory_batching.py \
  drug_agent/tests/test_materialize_toolrl_training_view.py

# Token counting is CPU-heavy while embedding is GPU-heavy. Run them together
# so the single requested GPU is not held idle solely for the length audit.
python -m drug_agent.scripts.audit_toolrl_v8_lengths \
  --input "$input_path" \
  --model "$tokenizer_model" \
  --output-root "$length_root" \
  --context-limit 262144 \
  --generation-budgets 16384 32768 65536 \
  > "$length_root/worker.log" 2>&1 &
length_pid=$!

python -m drug_agent.scripts.select_toolrl_v8 encode \
  --input "$input_path" \
  --output-root "$embedding_root" \
  --model "$embedding_model" \
  --batch-size 32 \
  > "$embedding_root/worker.log" 2>&1

wait "$length_pid"
