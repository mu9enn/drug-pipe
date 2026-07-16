#!/bin/bash
# Source from every formal SFT / offline ToolRL / step-level GAD training entry.
export DRUG_AGENT_TRAINING_OFFLINE=1
export DRUG_AGENT_ALLOW_TOOL_ENV=0

unset MOLCLAW_SCP_SERVER_URL
unset MOLCLAW_SCP_API_KEY
unset MOLCLAW_CONNECT_TIMEOUT_SEC
unset MOLCLAW_LIST_TOOLS_TIMEOUT_SEC
unset MOLCLAW_TOOL_TIMEOUT_SEC
unset MOLCLAW_TOOL_HEARTBEAT_SEC

python -m drug_agent.offline_guard --check-offline-training
