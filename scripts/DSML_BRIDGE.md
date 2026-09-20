# Local DeepSeek bridge

`dsml_bridge.py` listens on `127.0.0.1:18765` and reads the normal configured
`dsv4flash` provider from cc-switch's database. Its current main model is
`deepseek-v4.1-flash`. No API key is embedded in this script or its logs.

Start the bridge with the base Python environment (httpx + jsonschema):

```bash
python scripts/dsml_bridge.py --provider dsv4flash
```

Run Claude through the existing concurrency gate:

```bash
DRUG_PIPE_DSML_BRIDGE=1 /home/sunxiangyu/sunxiangyu/drug-pipe/runtime/claude
```

The wrapper sets explicit per-invocation settings, one model for all model tiers,
and direct loopback routing. It removes inherited HTTP proxies for this child.
Ordinary `claude` remains governed by cc-switch; use the command above for the
bridge. Stop/restart the bridge after changing the upstream provider configuration.

The bridge translates Anthropic message/tool histories to OpenAI chat completions.
It preserves native tool calls and converts complete DSML invoke/parameter blocks
into schema-validated `tool_use` blocks with `stop_reason=tool_use`. It buffers
one upstream completion and emits Anthropic SSE with heartbeat pings. It does not
delete arbitrary marker strings or execute tools itself. Tool execution and
permissions remain in Claude Code. Raw thinking is carried in thinking blocks and
returned as `reasoning_content` on subsequent requests.

Malformed/truncated DSML, unadvertised tools, duplicate parameters, mixed native
and DSML calls, and unsupported modalities fail explicitly. Images/documents are
not silently discarded. Upstream errors surface as API errors, never synthetic
successful scientific observations. The current scope is text tool workflows.

Tests: `python -m unittest discover -s scripts -p test_dsml_bridge.py`.
Live validation on 2026-09-20: Read of the 605-data README, successful MolClaw
`is_valid_smiles(["CCO"])`, and multi-call production trajectories.
These live calls were native; the DSML fallback is covered by deterministic tests.

Reference: https://github.com/vllm-project/vllm/blob/main/vllm/parser/deepseek_v4.py
