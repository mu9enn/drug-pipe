# Offline ToolRL Audit

## Official ToolRL Pattern

The upstream ToolRL repository uses offline rule rewards for tool learning.
Its dataset stores a prompt plus `reward_model.ground_truth`; rollout generates
assistant text once, and the reward manager decodes that text and calls
`verl.utils.reward_score.rlla.compute_score`.

The official reward in `ToolRL/verl/utils/reward_score/rlla.py` compares
generated `<tool_call>...</tool_call>` text against the ground-truth tool calls.
It scores:

- output format
- tool name overlap
- argument key overlap
- exact argument value matches
- optional length / schedule variants

It does not call an external tool runtime during training.

## Current drug_agent Paths

The legacy `drug_agent/scripts/run_qwen3_5_0_8b_drug_grpo_smoke.sh`,
`run_qwen3_5_0_8b_drug_grpo_learn.sh`, and
`run_qwen3_5_0_8b_drug_ppo_smoke.sh` contain the historical online-agent implementation. They use:

```text
--custom-generate-function-path drug_agent.rollout.generate_with_drug_agent.generate
```

That generator instantiates `MCPToolExecutor` and calls `executor.execute(...)` after parsing a
valid tool call. Because formal training must now be offline, all three legacy scripts are disabled
before Ray starts and are retained only as historical reference.

The offline ToolRL path is separate and lives under `drug_agent/toolrl/`.
Its training script uses:

```text
--rollout-function-path slime.rollout.sglang_rollout.generate_rollout
--custom-rm-path drug_agent.toolrl.molclaw_reward.reward_func
```

This path produces one assistant response per prompt and computes reward by
comparing generated ReAct `<tool_call>` blocks to reference tool calls in
`label` / `metadata`.

## Data Requirements

Each offline ToolRL row should contain:

- `prompt`: messages containing the state before the target assistant step
- `label.target_tool_calls`: reference MolClaw tool calls
- `metadata.target_tool_calls`: same reference calls for reward/debug access
- `metadata.allowed_tool_names` or `metadata.allowed_tools`
- `metadata.task_id` / `source_id`
- `metadata.task_type`
- `metadata.assistant_index`

`drug_agent/toolrl/convert_react_to_toolrl_steps.py` extracts these fields from
cleaned ReAct SFT trajectories. Every assistant turn with retained MolClaw
tool calls becomes one step-level sample.

## Checks

Use these commands before launching an offline ToolRL run:

```bash
python drug_agent/toolrl/validate_toolrl_offline_data.py \
  --input $DRUG_AGENT_DATA_ROOT/toolrl/mcp_sft_all.toolrl_steps.jsonl

python drug_agent/tools_debug/debug_toolrl_offline_no_tool_call.py
```

The no-tool-call debug script unsets all `MOLCLAW_*` variables, runs parser and
reward computation, and checks that the ToolRL training script does not use the
online custom generator or MCP runtime env.
