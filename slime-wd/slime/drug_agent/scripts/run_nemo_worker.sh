#!/usr/bin/env bash
set -euo pipefail

# The allocation runs this foreground workload and exits on success or failure.
run_root="${1:?run root required}"
mode="${2:?smoke or full required}"
[[ "$mode" == smoke || "$mode" == full ]]
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export PATH="$run_root/venv/bin:$PATH"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
unset PYTHONPATH RAY_ADDRESS
exec > "$run_root/${mode}_worker.log" 2>&1
on_exit() {
  code=$?
  trap - EXIT
  printf '%s\n' "$code" > "$run_root/${mode}_worker.exitcode"
  exit "$code"
}
trap on_exit EXIT
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_properties(0))'
"$run_root/bootstrap/bin/uv" pip freeze --python "$run_root/venv/bin/python" > "$run_root/environment_frozen.txt"
extra=()
if [[ "$mode" == smoke ]]; then extra+=(--smoke); fi
python -m drug_agent.scripts.run_nemo_toolrl encode --run-root "$run_root" "${extra[@]}"
python -m drug_agent.scripts.run_nemo_toolrl dedup --run-root "$run_root" "${extra[@]}"
