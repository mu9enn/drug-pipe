#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MCP_RELAY_LISTEN_HOST="${MCP_RELAY_LISTEN_HOST:-0.0.0.0}"
export MCP_RELAY_LISTEN_PORT="${MCP_RELAY_LISTEN_PORT:-13208}"
export MCP_RELAY_TARGET_HOST="${MCP_RELAY_TARGET_HOST:-httpproxy-headless.kubebrain.svc.pjlab.local}"
export MCP_RELAY_TARGET_PORT="${MCP_RELAY_TARGET_PORT:-3128}"
MOLCLAW_RELAY_LOG="${MOLCLAW_RELAY_LOG:-${HOME}/mcp_http_proxy_relay.log}"

exec python -u "${SCRIPT_DIR}/relay.py" 2>&1 | tee -a "$MOLCLAW_RELAY_LOG"
