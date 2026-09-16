# Internet-capable collection host: MolClaw runtime

The Data-Pipe Claude and DeepSeek launch entry, legacy canonical SCP HTTP configs,
and direct `run_claude.py` calls with no MCP config route through the standard
stdio adapter. Explicit custom MCP servers are not rewritten. Ambient user MCP
configuration is not used by the default direct runner.

## Install on an Internet-capable collector

From the repository root, run:

```bash
bash scripts/molclaw/setup_direct.sh
```

This installs the pinned Node 24.19.0, MCP SDK 1.29.0, Undici 7.16.0 and Zod 4.4.3
under ignored `runtime/molclaw-deps/`. It creates ignored, mode-0600
`runtime/molclaw.local.env` only when absent. Existing settings are preserved.
Credentials are read from existing `data-pipe/.env`; no credentials are in this
package or copied from another host. The system Node and Claude binary are unchanged.
Do not use the direct setup on a network-restricted GPU worker. That environment
must use its approved routing configuration.

## Before starting a collection batch

```bash
bash scripts/molclaw/preflight.sh
# Only proceed when this exits zero, then invoke the normal Data-Pipe launch script.
```

The check starts the actual stdio adapter, lists tools and validates `CCO` using
`is_valid_smiles(smiles_list=["CCO"])`. It requires a successful scientific result,
no pending envelope and no progress/log notifications. No LLM is invoked.

On this collector the adapter clears inherited HTTP/HTTPS/ALL proxy variables for
its own process and connects directly to the unchanged SCP HTTPS endpoint. Other
processes and LLM API routing are unaffected. Polls wait at most 60 seconds. Running
envelopes are consumed internally; the harness receives only the final result.
Each tool has a four-hour adapter budget; transient transport recovery reuses the
same job ID and does not rerun the trajectory. Ordinary tool failures and invalid
arguments are not silently retried. This is not a promise of zero network failures.

Local diagnostics are in `slime-wd/outputs/molclaw_adapter_logs/` with private file
permissions. A recorded recovered transport error is not by itself a failed tool.
Backend compute limits, downstream ColabFold timeouts, invalid files and model
mistakes remain possible. Examine the final tool result, including structured
`status`/`diagnostics`; a successful Claude process exit is not scientific success.
Do not extend every backend timeout, discard failed reruns, or relabel a trajectory
budget exhaustion as a network failure. Keep originals and new attempts separately.
