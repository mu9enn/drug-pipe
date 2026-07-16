# ToolRL-style MolClaw Reward

This package adds a step-level ToolRL path on top of the cleaned ReAct data.

## What it does

- Converts cleaned ReAct sessions into step-level RL samples.
- Extracts and normalizes `<tool_call>...</tool_call>` blocks.
- Scores model responses against reference tool calls with a structured reward.

## Input / Output

- Input: cleaned ReAct JSON or JSONL records.
- Output: JSONL with:
  - `prompt`
  - `label`
  - `metadata`
  - `target_assistant`
  - `target_tool_calls`

## Runtime model

Use the conversion output as `--prompt-data` and keep the slime-native `train.py` path.
The custom reward hook is:

`drug_agent.toolrl.molclaw_reward.reward_func`

The rollout function stays slime native:

`slime.rollout.sglang_rollout.generate_rollout`

This path is offline ToolRL-style: training does not instantiate `MCPToolExecutor`,
does not require `MOLCLAW_SCP_SERVER_URL` / `MOLCLAW_SCP_API_KEY`, and does not call
MolClaw tools during rollout or reward computation.
The rollout and reward hook are fixed in the formal script and cannot be overridden back to an
online generator. See [`../OFFLINE_TRAINING_POLICY.md`](../OFFLINE_TRAINING_POLICY.md).

## Validation

Validate step-level data before training:

```bash
python drug_agent/toolrl/validate_toolrl_offline_data.py \
  --input $DRUG_AGENT_DATA_ROOT/toolrl/mcp_sft_all.toolrl_steps.jsonl
```

Verify the offline reward path without MCP credentials:

```bash
python drug_agent/tools_debug/debug_toolrl_offline_no_tool_call.py
```

## Notes

- Offline step-level only.
- No slime core / trainer changes.
- No online tool-execution rollout is added in this round.
