#!/usr/bin/env bash
set -eu
export DRUG_PROJECT=${DRUG_PROJECT:-/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe}
export SECRET_FILE=${SECRET_FILE:-/home/sunxiangyu/.dsh/molclaw.env}
export HTTP_PROXY=${HTTP_PROXY:-http://httpproxy-headless.kubebrain.svc.pjlab.local:3128}
export HTTPS_PROXY=${HTTPS_PROXY:-$HTTP_PROXY}
export http_proxy=${http_proxy:-$HTTP_PROXY} https_proxy=${https_proxy:-$HTTPS_PROXY}
export NODE_USE_ENV_PROXY=1
export NO_PROXY=${NO_PROXY:-localhost,127.0.0.1} no_proxy=${no_proxy:-localhost,127.0.0.1}
export MOLCLAW_POLL_INTERVAL_SECONDS=${MOLCLAW_POLL_INTERVAL_SECONDS:-300}
adapter_dir=$(cd -- "$(dirname -- "$0")" && pwd)
exec "$DRUG_PROJECT/slime-wd/outputs/dsh_eval_runtime/node-v24.19.0/bin/node" "$adapter_dir/stdio_adapter.mjs"
