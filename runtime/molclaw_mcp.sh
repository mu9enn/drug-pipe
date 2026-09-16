#!/usr/bin/env bash
# Shared standard MCP entry: stdout is reserved for MCP JSON-RPC.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DRUG_PROJECT="${DRUG_PROJECT:-$project_dir}"
export SECRET_FILE="${SECRET_FILE:-$HOME/.dsh/molclaw.env}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-http://httpproxy-headless.kubebrain.svc.pjlab.local:3128}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-$HTTP_PROXY}}"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
export no_proxy="${no_proxy:-$NO_PROXY}"
export NODE_USE_ENV_PROXY=1
node_bin="${MOLCLAW_NODE_BIN:-$DRUG_PROJECT/slime-wd/outputs/dsh_eval_runtime/node-v24.19.0/bin/node}"
if [[ ! -x "$node_bin" ]]; then
  echo 'MolClaw adapter Node runtime missing; set MOLCLAW_NODE_BIN to an installed Node runtime.' >&2
  exit 1
fi
exec "$node_bin" "$DRUG_PROJECT/slime-wd/molclaw-mcp-relay/polling-adapter/stdio_configurable.mjs"
