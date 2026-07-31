#!/usr/bin/env bash
set -euo pipefail

if [[ -f /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh ]]; then
  source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
else
  source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi
cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh

: "${TOOL_CATALOG:?Set TOOL_CATALOG to tool_catalog.json captured by online eval preflight}"
INPUT=${INPUT:-$DRUG_AGENT_DATA_ROOT/react_trajectories.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-$DRUG_AGENT_DATA_ROOT/live_tool_catalog_v1}
mkdir -p "$OUTPUT_ROOT"

python -m drug_agent.data.migrate_live_tool_catalog \
  --input "$INPUT" --tool-catalog "$TOOL_CATALOG" --output-root "$OUTPUT_ROOT"
export DRUG_AGENT_TOOL_CATALOG="$TOOL_CATALOG"
python -m drug_agent.toolrl.convert_react_to_toolrl_steps \
  --input "$OUTPUT_ROOT/react_trajectories.jsonl" \
  --output "$OUTPUT_ROOT/toolrl/toolrl_steps.jsonl" \
  --skipped-report "$OUTPUT_ROOT/toolrl/toolrl_steps.skipped.jsonl" \
  --report "$OUTPUT_ROOT/toolrl/toolrl_steps.report.json"
python -m drug_agent.gad.data \
  --input "$OUTPUT_ROOT/react_trajectories.jsonl" \
  --output "$OUTPUT_ROOT/gad/gad_steps.jsonl" \
  --skipped-report "$OUTPUT_ROOT/gad/gad_steps.skipped.jsonl" \
  --report "$OUTPUT_ROOT/gad/gad_steps.report.json"
python -m drug_agent.data.generate_format_examples \
  --input "$OUTPUT_ROOT/react_trajectories.jsonl" \
  --output-root "$OUTPUT_ROOT/format_examples"
python - "$INPUT" "$TOOL_CATALOG" "$OUTPUT_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source, catalog, root = map(Path, sys.argv[1:])
artifacts = {
    "canonical_react": root / "react_trajectories.jsonl",
    "toolrl": root / "toolrl/toolrl_steps.jsonl",
    "gad": root / "gad/gad_steps.jsonl",
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def count(path):
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

manifest = {
    "schema_version": "drug_agent_live_tool_derived_data_v1",
    "source": {"path": str(source), "sha256": digest(source)},
    "tool_catalog": {"path": str(catalog), "sha256": digest(catalog)},
    "artifacts": {
        name: {"path": str(path), "sha256": digest(path), "records": count(path)}
        for name, path in artifacts.items()
    },
}
(root / "derived_data_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY
