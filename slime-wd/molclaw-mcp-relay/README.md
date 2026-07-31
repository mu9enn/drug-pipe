# MolClaw MCP Relay

This directory owns the network relay used by online Drug-Agent evaluation when
the GPU worker cannot reach the MolClaw endpoint directly.

```text
GPU worker HTTP(S)_PROXY
→ no-GPU worker relay:13208
→ PJLab HTTP proxy:3128
→ configured MolClaw MCP endpoint
```

The relay is protocol-agnostic: it forwards raw TCP bytes to the PJLab HTTP
proxy. It does not parse MCP traffic and does not require or store the MolClaw
API key. Formal SFT, ToolRL and GAD training do not use this service.

Start it on the no-GPU worker:

```bash
cd /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd

tmux new-session -d -s molclaw-relay-13208 \
  "MCP_RELAY_LISTEN_HOST=0.0.0.0 MCP_RELAY_LISTEN_PORT=13208 \
   bash molclaw-mcp-relay/run_relay.sh"
```

Point an online evaluation on the GPU worker at it:

```bash
MOLCLAW_PROXY_URL=http://<no-gpu-ip>:13208 \
MODEL_CHECKPOINT=/path/to/checkpoint \
bash drug_agent/scripts/run_molbench_eval.sh
```

The second command is run from `slime-wd/slime`.
