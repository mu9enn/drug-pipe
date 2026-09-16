#!/usr/bin/env bash
# Explicit opt-in for an Internet-capable collector; no tunnels or system changes.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
deps="$root/runtime/molclaw-deps"
mkdir -p "$deps"
cp "$root/scripts/molclaw/package.json" "$root/scripts/molclaw/package-lock.json" "$deps/"
npm ci --prefix "$deps" --no-audit --no-fund
if [[ ! -e "$root/runtime/molclaw.local.env" ]]; then
  (umask 077
   printf 'MOLCLAW_NETWORK_MODE=direct\nMOLCLAW_NODE_BIN=%q\nMOLCLAW_DEPS_ROOT=%q\nSECRET_FILE=%q\nMOLCLAW_POLL_INTERVAL_SECONDS=60\nMOLCLAW_HEADERS_TIMEOUT_MS=14400000\n' \
    "$deps/node_modules/node/bin/node" "$deps" "$root/data-pipe/.env" > "$root/runtime/molclaw.local.env")
fi
bash "$root/scripts/molclaw/preflight.sh"
