#!/usr/bin/env bash

# Source on gpu worker.

export MOLCLAW_SCP_SERVER_URL="http://180.184.86.2:32208/mcp"

# Credentials must be injected by the caller. Never commit a real key here.
: "${MOLCLAW_SCP_API_KEY:?Export MOLCLAW_SCP_API_KEY before sourcing this template}"
export MOLCLAW_SCP_API_KEY

# no-gpu relay endpoint (update if worker IP changes)
export MOLCLAW_PROXY_HOST="${MOLCLAW_PROXY_HOST:-100.100.245.233}"
export MOLCLAW_PROXY_PORT="${MOLCLAW_PROXY_PORT:-13208}"
export MOLCLAW_PROXY_URL="http://${MOLCLAW_PROXY_HOST}:${MOLCLAW_PROXY_PORT}"

export HTTP_PROXY="$MOLCLAW_PROXY_URL"
export HTTPS_PROXY="$MOLCLAW_PROXY_URL"
export http_proxy="$MOLCLAW_PROXY_URL"
export https_proxy="$MOLCLAW_PROXY_URL"

# Do not include 180.184.86.2 here.
export NO_PROXY="${NO_PROXY:-10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn,127.0.0.1,localhost}"
export no_proxy="$NO_PROXY"
