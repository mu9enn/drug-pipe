#!/usr/bin/env bash
set -euo pipefail

worker_root=/root/slime_sxy/group-space/sunxiangyu
repo_root="$worker_root/drug-pipe/slime-wd/slime"
dataset_root="$worker_root/drug_wd/drug_pipe_605_local_copy_20260904/v8_toolrl_20260907"
input_path="$dataset_root/01_all_decisions/all_decisions.jsonl"
output_root="$dataset_root/02_length_audit_all"
model_path="$worker_root/drug-pipe/slime-wd/data/Qwen3.5-9B"
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
python - <<'PY'
import jinja2
from pathlib import Path
from transformers import AutoTokenizer
assert tuple(int(part) for part in jinja2.__version__.split(".")[:2]) >= (3, 1)
model = Path("/root/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/data/Qwen3.5-9B")
assert (model / "tokenizer_config.json").is_file()
AutoTokenizer.from_pretrained(model, trust_remote_code=True)
PY

python -m drug_agent.scripts.audit_toolrl_v8_lengths \
  --input "$input_path" \
  --model "$model_path" \
  --output-root "$output_root" \
  --context-limit 262144 \
  --generation-budgets 16384 32768 65536
