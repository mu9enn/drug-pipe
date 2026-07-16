#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MCP_RELAY_LISTEN_PORT="${MCP_RELAY_LISTEN_PORT:-13208}"

python -u "${SCRIPT_DIR}/molclaw_tcp_proxy.py" 2>&1 | tee -a "${HOME}/mcp_http_proxy_relay.log"