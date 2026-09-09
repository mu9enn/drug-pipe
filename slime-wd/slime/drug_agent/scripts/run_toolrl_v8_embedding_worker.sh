#!/usr/bin/env bash
set -euo pipefail

worker_root=/root/slime_sxy/group-space/sunxiangyu
repo_root="$worker_root/drug-pipe/slime-wd/slime"
dataset_root="$worker_root/drug_wd/drug_pipe_605_local_copy_20260904/v8_toolrl_20260907"
input_path="$dataset_root/01_all_decisions/all_decisions.jsonl"
output_root="$dataset_root/03_embedding_cache"
status_path="$output_root/worker_status.json"

mkdir -p "$output_root"
exec > >(tee "$output_root/worker.log") 2>&1
on_exit() {
  code=$?
  STATUS_CODE="$code" STATUS_PATH="$status_path" python - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
Path(os.environ["STATUS_PATH"]).write_text(json.dumps({
    "exit_code": int(os.environ["STATUS_CODE"]),
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "entrypoint_is_foreground_workload": True,
}, indent=2) + "\n")
PY
  exit "$code"
}
trap on_exit EXIT

source "$worker_root/slime_env/slime_env.sh"
cd "$repo_root"
export HF_HOME="$worker_root/drug-pipe/slime-wd/data/huggingface_cache"
dependency_root="$dataset_root/python_deps"
mkdir -p "$dependency_root"
export PYTHONPATH="$dependency_root${PYTHONPATH:+:$PYTHONPATH}"
if ! python -c 'import sentence_transformers' >/dev/null 2>&1; then
  python -m pip install \
    --no-index \
    --find-links "$dataset_root/wheels" \
    --no-deps \
    --target "$dependency_root" \
    -r "$repo_root/drug_agent/requirements_toolrl_v8.txt"
fi

python - <<'PY'
import json
import torch
import sentence_transformers
from pathlib import Path
assert torch.cuda.is_available(), "CUDA is not available"
Path("/root/slime_sxy/group-space/sunxiangyu/drug_wd/drug_pipe_605_local_copy_20260904/v8_toolrl_20260907/03_embedding_cache/worker_preflight.json").write_text(
    json.dumps({
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
    }, indent=2) + "\n"
)
PY

python -m pytest -q \
  drug_agent/tests/test_toolrl_v8_dataset.py \
  drug_agent/tests/test_trajectory_batching.py \
  drug_agent/tests/test_materialize_toolrl_training_view.py

python -m drug_agent.scripts.select_toolrl_v8 encode \
  --input "$input_path" \
  --output-root "$output_root" \
  --model Qwen/Qwen3-Embedding-0.6B \
  --batch-size 32
