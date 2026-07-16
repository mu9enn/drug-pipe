#!/usr/bin/env bash

# Source on no-gpu worker.

export MOLCLAW_SCP_SERVER_URL="http://180.184.86.2:32208/mcp"

# Credentials must be injected by the caller. Never commit a real key here.
: "${MOLCLAW_SCP_API_KEY:?Export MOLCLAW_SCP_API_KEY before sourcing this template}"
export MOLCLAW_SCP_API_KEY

# no-gpu worker reaches MolClaw through PJLab proxy.
export MOLCLAW_PROXY_URL="${MOLCLAW_PROXY_URL:-http://httpproxy-headless.kubebrain.svc.pjlab.local:3128}"

export HTTP_PROXY="$MOLCLAW_PROXY_URL"
export HTTPS_PROXY="$MOLCLAW_PROXY_URL"
export http_proxy="$MOLCLAW_PROXY_URL"
export https_proxy="$MOLCLAW_PROXY_URL"

# Keep origin endpoint out of NO_PROXY.
export NO_PROXY="${NO_PROXY:-10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn,127.0.0.1,localhost}"
export no_proxy="$NO_PROXY"
