# Drug Agent Offline Training Boundary

> Training uses fixed, offline-built ReAct decision states. Student rollout means sampling the next
> action from the current policy; training never executes that action. Real agent-environment
> interaction is reserved for explicitly named online evaluation/debug.

## Formal training chains

| Chain | State source | Generation/loss | Reward/update | Real tool access |
|---|---|---|---|---|
| ReAct-SFT | Fixed teacher messages and observations | `slime.rollout.sft_rollout` + `sft_loss` | teacher forcing | Forbidden |
| Offline ToolRL-style RL | Fixed step state | slime-native SGLang next-action generation | reference tool-call rule reward + GRPO | Forbidden |
| GAD Stage 2 | Fixed step state | current SFT student next-action generation | zero student reward; cache negatives; BT warmup | Forbidden |
| GAD Stage 3 | Fixed step state | current student next-action generation | discriminator `/score-and-update` + rule reward + GRPO | Forbidden |

Observations in these datasets are immutable historical context. Generated tool calls are scored as
text/actions and never used to request another observation.

Actual call chains:

```text
ReAct-SFT script
  -> train.py
  -> slime.rollout.sft_rollout.generate_rollout
  -> sft_loss

Offline ToolRL script
  -> train.py
  -> slime.rollout.sglang_rollout.generate_rollout
  -> drug_agent.toolrl.molclaw_reward.reward_func
  -> GRPO

GAD Stage 2 negatives
  -> train.py
  -> slime-native current-student SGLang generation
  -> drug_agent.gad.negative_cache.zero_reward/log_negative_cache
  -> discriminator BT warmup

GAD Stage 3
  -> train.py
  -> slime-native current-student SGLang generation
  -> drug_agent.gad.reward.reward_func
  -> discriminator /score-and-update
  -> GRPO
```

Slime-native OPD is not a Drug Agent formal chain. It requires response-aligned per-token teacher
log-probabilities from a distinct compatible teacher checkpoint or service. Those are unavailable in
the current setting; GAD is the supported teacher-response distillation route.

No formal chain references `drug_agent.rollout.generate_with_drug_agent.generate`; that historical
generator is the code path that executes actions.

## Runtime enforcement

All formal training scripts source `drug_agent/scripts/offline_training_env.sh`. It:

- sets `DRUG_AGENT_TRAINING_OFFLINE=1`;
- sets `DRUG_AGENT_ALLOW_TOOL_ENV=0`;
- removes all `MOLCLAW_*` credentials/timeouts;
- validates the environment before launch.

`MCPClient.connect()` and `MCPToolExecutor` fail closed. A process may access the real tool
environment only when it is not offline training and explicitly sets:

```bash
export DRUG_AGENT_ALLOW_TOOL_ENV=1
```

The old action-json GRPO/PPO training scripts executed tools and are now disabled before Ray starts.

## Network requests

Allowed during formal training:

- local Ray dashboard/job submission;
- SGLang generation/router traffic;
- GAD discriminator `/health`, `/score-and-update`, `/metrics`, `/checkpoint`;
- checkpoint, logging, and metrics infrastructure.

Forbidden during formal training:

- MolClaw/MCP endpoints;
- sandbox/tool executors;
- agent environments;
- any request that executes a generated action or obtains a new observation.

## Online tool interaction

Current explicitly named tool-interaction utilities are debug/evaluation helpers:

- `drug_agent/tools_debug/debug_one_task.py`
- `drug_agent/tools_debug/debug_replay_trajectory.py`
- `drug_agent/tools_debug/debug_mcp_tools.py`

They require explicit `DRUG_AGENT_ALLOW_TOOL_ENV=1`. They are not training entrypoints. Any future
formal online evaluation script must include `online_eval` in its name, print a prominent startup
warning, and require the same explicit opt-in.

There is currently no production online-evaluation launcher; the three utilities above are the only
explicitly permitted real-tool entrypoints.

## Audit

```bash
python drug_agent/tools_debug/audit_offline_training.py
python -m unittest discover -s drug_agent/tests -p 'test_*.py' -v
```
