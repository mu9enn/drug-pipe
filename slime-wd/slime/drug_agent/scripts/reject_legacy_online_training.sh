#!/bin/bash
cat >&2 <<'EOF'
This legacy action-json training entry executes real MolClaw/MCP tools and is disabled.

Project policy:
  SFT, offline ToolRL-style RL, and step-level GAD train on fixed offline states.
  Student rollout samples next-action tokens but never executes the action.
  Real agent-environment interaction is reserved for explicitly named online evaluation/debug.

Use drug_agent/toolrl/scripts/* or drug_agent/gad/scripts/* for RL training.
EOF
exit 2
