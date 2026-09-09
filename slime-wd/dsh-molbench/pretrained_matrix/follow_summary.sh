#!/usr/bin/env bash
set -Eeuo pipefail
: "${MATRIX_STAMP:?}"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
wd=$(cd "$script_dir/../.." && pwd)
serial=$wd/outputs/dsh_pretrained_matrix_serial_$MATRIX_STAMP
trap 'rc=$?; printf "%s\n" "$rc" > "$serial/summary.exit"' EXIT
for ((attempt=0; attempt<10080; attempt++)); do
 [[ -f "$serial/flat.exit" && -f "$serial/hier.exit" ]] && break
 sleep 60
done
[[ -f "$serial/flat.exit" && -f "$serial/hier.exit" ]]
python3 "$script_dir/summarize_matrix.py" --stamp "$MATRIX_STAMP" --models "${MODEL_TAGS:-9b}" \
 --outputs-root "$wd/outputs/dsh_molbench_evals" --output-json "$serial/matrix_summary.json" \
 --output-markdown "$serial/matrix_summary.md"
python3 - "$serial/matrix_summary.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))['all_publishable']
PY
