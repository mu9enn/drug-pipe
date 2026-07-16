# MolClaw preflight

This is the standalone 81-tool MolClaw MCP reachability and schema preflight
suite used by slime's DrugAgent work. It does not import or require verl.

## Runtime

Install the small client dependency set in the active slime environment:

```bash
python -m pip install -r drug_agent/tools_debug/molclaw_preflight/requirements.txt
```

Credentials are never stored in this directory. Export
`MOLCLAW_SCP_API_KEY` before sourcing either worker template.

When a GPU worker cannot reach the proxy directly, start the relay on the
no-GPU worker:

```bash
cd "$SLIME"
bash drug_agent/tools_debug/molclaw_preflight/relay/run_molclaw_tcp_proxy.sh
```

Run the suites from the slime repository root:

```bash
bash drug_agent/tools_debug/molclaw_preflight/no_gpu_worker/run_no_gpu_suite.sh
bash drug_agent/tools_debug/molclaw_preflight/gpu_worker/run_gpu_suite.sh
```

Generated reports are written to this suite's ignored `reports/` directory.
The May 2026 baseline is archived under
`$OUTPUTS_ROOT/molclaw_preflight/history/20260525/` and may contain historical
absolute paths.

The suite distinguishes successful execution (`pass_ok`) from a reachable
tool that returns a business or validation error (`pass_reached`). Registration,
route, connectivity, timeout, and unknown failures remain separate statuses.
