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

## MCP response-header timeout

`pretrained_matrix/run_worker.sh` preloads `install_mcp_headers_timeout.mjs` in new DSH processes. It sets only requests to `https://scp.intern-ai.org.cn` to a 14400000 ms (four-hour) headers timeout (override: `MOLCLAW_HEADERS_TIMEOUT_MS`). The native fetch, existing environment proxy dispatcher, body timeout, signals, SDK four-hour budget and retries are preserved. The public `getGlobalDispatcher` API comes from the installed harness Undici dependency; the preload fails if that dependency is missing. Startup logs `[mcp-http-policy]` with the effective value; no credentials are logged.

This fixes the reproduced `UND_ERR_HEADERS_TIMEOUT` at approximately 300 seconds. It does not establish a fix for historical fast CONNECT failures. Disable by removing the policy preload from the worker launch; restart workers for either change to take effect.
