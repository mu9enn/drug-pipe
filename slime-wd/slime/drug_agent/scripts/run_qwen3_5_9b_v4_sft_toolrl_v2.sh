#!/usr/bin/env bash
# Canonical Qwen3.5-9B two-stage launcher: completed ReAct SFT -> ToolRL v2.
# The former filename is retained as a compatibility entrypoint only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_qwen3_5_9b_v4_plan_sft_toolrl_v2.sh" "$@"
