#!/usr/bin/env bash
set -euo pipefail
export DRUG_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "$DRUG_PROJECT/runtime/molclaw.local.env" ]]; then
  set -a
  source "$DRUG_PROJECT/runtime/molclaw.local.env"
  set +a
fi
node_bin="${MOLCLAW_NODE_BIN:-$DRUG_PROJECT/slime-wd/outputs/dsh_eval_runtime/node-v24.19.0/bin/node}"
exec "$node_bin" "$DRUG_PROJECT/scripts/molclaw/preflight.mjs"
