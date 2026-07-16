# Slime-Native Drug Agent

This directory is a thin plugin layer for migrating MolClaw drug-agent training onto slime native interfaces.

## Progress + Runbook

- Chinese runbook (implementation + principles + staged commands): `drug_agent/SLIME_DRUG_AGENT_RUNBOOK_zh.md`
- Worker path rule: on GPU worker prefer `/root/slime_sxy/...` paths.
- Debug model default/fallback:
  - default: `$DATA/Qwen3.5-122B-A10B`
  - fallback when default missing: `$DATA/Qwen3.5-27B`

## Design Principles

- Use slime-native training stack: `train.py` + SGLang rollout + Megatron actor.
- Use slime-native hooks: `--custom-generate-function-path`, `--custom-rm-path`, and `--custom-rollout-log-function-path`.
- Keep all new logic under `$SLIME/drug_agent`.
- Do not port verl-agent trainer/EnvManager/VectorEnv adapters.
- Formal training is offline-state only. Current-policy rollout samples a next action but never
  executes it. See [`OFFLINE_TRAINING_POLICY.md`](OFFLINE_TRAINING_POLICY.md).

## Directory Layout

```text
drug_agent/
  protocol/
    action_schema.py
    action_parser.py
    parse_policy.py
    react_protocol.py
    prompts.py
  data/
    inspect_pipelined_data.py
    convert_pipelined_to_slime_grpo.py
    convert_pipelined_to_slime_sft.py
  toolrl/
    convert_react_to_toolrl_steps.py
    normalization.py
    parse_tool_calls.py
    molclaw_reward.py
    tool_schema_config.yaml
  common.py
  tools/
    mcp_client.py
    tool_executor.py
    tool_success.py
    tool_registry.py
    allowlist_v0.json
  rollout/
    generate_with_drug_agent.py
    reward_func.py
    trajectory_logger.py
  scripts/
    check_env.sh
    run_qwen3_5_0_8b_drug_grpo_smoke.sh
    run_qwen3_5_0_8b_drug_grpo_learn.sh
    run_qwen3_5_0_8b_drug_ppo_smoke.sh
    run_qwen3_5_0_8b_drug_sft_smoke.sh
  tools_debug/
    debug_mcp_tools.py
    sglang_launcher.py
    debug_sglang_launch.py
    debug_replay_trajectory.py
    debug_one_task.py
    debug_reward.py
    debug_training_compliance.py  # deprecated helper only
```

## Environment

Always source environment first:

```bash
source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
cd $SLIME
```

Quick check:

```bash
bash drug_agent/scripts/check_env.sh
```

## Prepare Qwen3.5-4B torch_dist

The SFT and ToolRL runs for Qwen3.5-4B expect a Megatron `torch_dist` checkpoint at:

```bash
$DATA/Qwen3.5-4B_torch_dist
```

If that directory does not exist yet, prepare it with:

```bash
bash drug_agent/scripts/prepare_qwen3_5_4B_torch_dist.sh
```

You can override the input/output paths and GPU count with environment variables such as
`HF_CHECKPOINT`, `SAVE_DIR`, `NUM_GPUS`, and `MEGATRON_LM_PATH`.

MCP environment variables are only for explicitly opted-in online evaluation/debug; formal training
removes them and never requires them:

- `MOLCLAW_SCP_SERVER_URL`
- `MOLCLAW_SCP_API_KEY`
- `MOLCLAW_CONNECT_TIMEOUT_SEC`
- `MOLCLAW_LIST_TOOLS_TIMEOUT_SEC`
- `MOLCLAW_TOOL_TIMEOUT_SEC`
- `MOLCLAW_TOOL_HEARTBEAT_SEC`

Optional plugin controls:

- `DRUG_AGENT_ALLOWLIST_PATH` (default: `drug_agent/tools/allowlist_v0.json`)
- `DRUG_AGENT_ALLOW_ALL` (`0` or `1`)
- `DRUG_AGENT_ROLLOUT_MODE` (default: `train_strict`; optional `debug_permissive`)
- `DRUG_AGENT_ALLOW_PARSE_RECOVERY` (default: `0`; set `1` only for diagnostics)
- `DRUG_AGENT_UNKNOWN_SEMANTIC_AS_FAILURE` (default: `1`)
- `DRUG_AGENT_TRAINING_OFFLINE` (`1` in every formal training entry)
- `DRUG_AGENT_ALLOW_TOOL_ENV` (default denied; set `1` only for named online evaluation/debug)
- `DRUG_AGENT_RUN_NAME`
- `OUTPUTS_ROOT` (default: `$WD/outputs`)
- `DRUG_AGENT_DATA_ROOT` (default: `$OUTPUTS_ROOT/slime_drug_agent_data`)
- `DRUG_AGENT_RUNS_ROOT` (default: `$OUTPUTS_ROOT/slime_drug_agent_runs`)

Debug model path strategy:

```bash
export DEBUG_MODEL_PATH=${DEBUG_MODEL_PATH:-$DATA/Qwen3.5-122B-A10B}
if [ ! -d "$DEBUG_MODEL_PATH" ]; then
  export DEBUG_MODEL_PATH=$DATA/Qwen3.5-27B
fi
```

## Data Pipeline

### 1) Inspect pipelined data

```bash
python drug_agent/data/inspect_pipelined_data.py
```

### 2) Convert GRPO prompt-data (Hybrid join)

```bash
python drug_agent/data/convert_pipelined_to_slime_grpo.py \
  --max-samples-per-task-type 100
```

Outputs:

- `ac.jsonl`
- `pf.jsonl`
- `vs.jsonl`
- `mixed.jsonl`
- `skipped_report.jsonl`
- `manifest.json`

### 3) Convert SFT messages

```bash
python drug_agent/data/convert_pipelined_to_slime_sft.py \
  --max-samples-per-task-type 100
```

Outputs:

- `ac.jsonl`
- `pf.jsonl`
- `vs.jsonl`
- `mixed.jsonl`
- `skipped_report.jsonl`
- `manifest.json`

Note:

- This converter is legacy compatibility only and still emits action-json SFT.
- Canonical ReAct-style SFT now comes from the upstream `pipeline/postprocess` path.

### 4) Convert ToolRL step-level data

ToolRL consumes cleaned ReAct trajectories and splits them into one sample per MolClaw tool-calling assistant turn:

```bash
python drug_agent/toolrl/convert_react_to_toolrl_steps.py \
  --input /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime-wd/data/mcp_sft_all \
  --output /tmp/toolrl_steps.jsonl \
  --skipped-report /tmp/toolrl_skipped.jsonl \
  --report /tmp/toolrl_report.json
```

The output rows contain:

- `prompt`
- `label`
- `metadata`
- `target_assistant`
- `target_tool_calls`

The custom reward hook is:

- `drug_agent.toolrl.molclaw_reward.reward_func`

ToolRL training scripts live under:

- `drug_agent/toolrl/scripts/run_toolrl_grpo_smoke.sh`
- `drug_agent/toolrl/scripts/run_toolrl_grpo_learn.sh`
- `drug_agent/toolrl/scripts/run_toolrl_grpo.sh`
- `drug_agent/toolrl/scripts/run_qwen3_5_4b_toolrl_smoke.sh`
- `drug_agent/toolrl/scripts/run_qwen3_5_4b_toolrl_full.sh`

ToolRL is offline by design. It uses slime's native single-response rollout and the custom reward hook
`drug_agent.toolrl.molclaw_reward.reward_func`; it does not call MCP or MolClaw tools during training.
Use these checks before launching a ToolRL run:

```bash
python drug_agent/toolrl/validate_toolrl_offline_data.py \
  --input $DRUG_AGENT_DATA_ROOT/toolrl/mcp_sft_all.toolrl_steps.jsonl

python drug_agent/tools_debug/debug_toolrl_offline_no_tool_call.py
```

Native OPD is intentionally not exposed as a `drug_agent` training path. Slime OPD requires
response-aligned, per-token teacher log-probabilities, either from a locally loadable compatible
teacher checkpoint or an external teacher service that returns them. The current Drug Agent setting
has neither a distinct compatible teacher checkpoint nor teacher log-prob access; using the same
base model as both student and teacher produces no meaningful distillation signal. GAD is the
supported distillation path because it learns from teacher responses through a scalar discriminator
and does not require teacher logits or log-probabilities.

## Protocol

### Legacy action-json online-agent protocol

The historical online-agent implementation uses action-json for rollout:

- one assistant turn contains one strict JSON object
- supported actions are `tool_call` and `final_answer`
- no markdown fences, XML, or prose wrappers

Tool call:

```json
{"type":"tool_call","tool_name":"...","arguments":{}}
```

Final answer:

```json
{"type":"final_answer","answer":{"summary":"...","evidence":[],"result":{},"ranked_molecules":[]}}
```

Parser rejects:

- Markdown fenced JSON
- XML
- Natural language wrappers around JSON

### ReAct SFT protocol

The canonical SFT data path now consumes ReAct-style tagged messages produced upstream by `pipeline/postprocess`:

- `<thought>...</thought>`
- `<tool_call>...</tool_call>`
- `<observation tool_name="...">...</observation>`
- `<final_answer>...</final_answer>`

Use the validator with the ReAct protocol once the upstream postprocess output is available:

```bash
python drug_agent/data/validate_sft_messages.py \
  --input "$DATA/mcp_sft_all.train.jsonl" \
  --protocol react_json \
  --tokenizer "$DATA/Qwen3.5-0.8B"
```

If you still need to inspect legacy action-json SFT, use `--protocol action_json` or `--protocol auto`.
If you are inspecting the current legacy compatibility `mixed.jsonl`, `--protocol auto` may report mixed/legacy records by design.

Self-test:

```bash
python -m drug_agent.protocol.action_parser
```

Validate SFT messages before training:

```bash
python drug_agent/data/validate_sft_messages.py \
  --input "$DATA/mcp_sft_all.train.jsonl" \
  --protocol react_json \
  --tokenizer "$DATA/Qwen3.5-0.8B"
```

Run compliance audit:

```bash
python drug_agent/tools_debug/debug_training_compliance.py
```

This helper is deprecated and informational only. It is no longer the training authority.

## Train Entrypoints

The old action-json GRPO/PPO scripts executed tools during training and are now disabled before Ray
starts. Formal RL training uses the offline ToolRL-style or step-level GAD scripts.

SFT smoke (slime native SFT: `sft_rollout + sft_loss`):

```bash
bash drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh
```

The SFT smoke script accepts either a `.jsonl` file or a directory of per-sample `.json` files via `PROMPT_DATA`. If a directory is provided, it will materialize a stable `*.train.jsonl` next to the source directory before validation and training.

For Qwen3.5 ReAct-SFT, the script explicitly uses `--loss-mask-type qwen3_5`. Using slime's default `qwen` loss-mask path will fail on multi-turn ReAct observation data with `No user query found in messages`.

The same SFT script can be reused across model sizes by overriding `MODEL_ARGS_FILE`, for example `scripts/models/qwen3.5-4B.sh` or `scripts/models/qwen3.5-27B.sh`.

SFT batch and parallel settings must satisfy:

- `ROLLOUT_BATCH_SIZE >= GLOBAL_BATCH_SIZE`
- `ROLLOUT_BATCH_SIZE % GLOBAL_BATCH_SIZE == 0`
- `NUM_GPUS % (TP * PP * CP * EP) == 0`
- `GLOBAL_BATCH_SIZE % DATA_PARALLEL_SIZE == 0`
- For the current dense Qwen setup, `DATA_PARALLEL_SIZE = NUM_GPUS / (TP * PP * CP * EP)`.

`MAX_TOKENS_PER_GPU` is a dynamic microbatch packing budget. It does not truncate an individual
sample that is longer than the budget. The current ReAct SFT data contains samples around 10k
tokens, so larger models may require tensor parallelism to shard the vocabulary logits and their
temporary copies.

For Qwen3.5-4B on four GPUs, use the dedicated conservative smoke entrypoint:

```bash
bash drug_agent/scripts/run_qwen3_5_4b_drug_sft_smoke.sh
```

It defaults to `TP=4`, `DP=1`, and `ROLLOUT_BATCH_SIZE=GLOBAL_BATCH_SIZE=1`. This is intentionally
more conservative than `TP=2`: the observed TP=1 failure occurred while cloning full-vocabulary
logits for an approximately 10k-token sample, not because of Ray scheduling or host memory.

Do not enable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` or
`PYTORCH_ALLOC_CONF=expandable_segments:True` on this colocated SGLang + Megatron path.
SGLang uses `TorchMemorySaver`, which currently rejects expandable allocator segments. The SFT
entrypoint removes inherited incompatible values before starting Ray and does not inject them into
the Ray runtime environment.

After smoke passes, launch a complete one-epoch 4B SFT run with:

```bash
bash drug_agent/scripts/run_qwen3_5_4b_drug_sft_full.sh
```

The full entrypoint uses slime's epoch-only mode, so it does not pass `--num-rollout` alongside
`--num-epoch`. With the current 516-row materialized dataset and `RBS=GBS=4`, slime runs exactly
129 optimizer steps per epoch. The wrapper refuses to start if the dataset size is not divisible
by the rollout batch size, preventing an unnoticed dropped tail.

To resume an interrupted full run, explicitly set `RESUME_DIR` to its checkpoint directory. The
wrapper then uses the same path for both `--load` and `--save`.

The smoke scripts also guard against an even smaller but easy-to-miss failure mode: if your chosen `NUM_ROLLOUT`, `ROLLOUT_BATCH_SIZE`, and `GLOBAL_BATCH_SIZE` combination would make the derived `train_iters` zero, the script will automatically bump `LR_DECAY_ITERS` to `1` so Megatron's scheduler stays valid instead of crashing deep inside initialization.

ToolRL smoke / learn use the same model-specific `MODEL_ARGS_FILE` pattern, but they expect a step-level ToolRL JSONL produced by `drug_agent/toolrl/convert_react_to_toolrl_steps.py`. The reward is the structured dict returned by `drug_agent.toolrl.molclaw_reward.reward_func`, and `--reward-key score` is passed through slime's native reward extraction path.

Optional debug-only switch (default off):

```bash
SFT_DEBUG_TRAIN_ONLY=1 bash drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh
```

Mask debug:

```bash
python drug_agent/tools_debug/debug_sft_mask_qwen.py \
  --input "$GROUP_SPACE/slime-wd/data/mcp_sft_all" \
  --tokenizer "$DATA/Qwen3.5-0.8B"
```

## Rollout + Reward + Trace

- Custom generate: `drug_agent.rollout.generate_with_drug_agent.generate`
- Custom reward: `drug_agent.rollout.reward_func.reward_func`
- Custom rollout log hook: `drug_agent.rollout.trajectory_logger.log_rollout_data`

Trajectory JSONL path:

```text
$DRUG_AGENT_RUNS_ROOT/<run_name>/trajectories.jsonl
```

Logged metrics include:

- `action_valid_rate`
- `strict_success_rate`
- `recovery_success_rate`
- `tool_success_rate`
- `tool_execution_success_rate`
- `tool_semantic_unknown_rate`
- `final_success_rate`

Strict/diagnostic parsing policy:

- `train.py` rollout defaults to `DRUG_AGENT_ROLLOUT_MODE=train_strict`.
- In `train_strict`, model output must be a full valid JSON object; parse failure becomes `invalid_action` and no tool is executed.
- `debug_permissive` is only for debug tools. It allows JSON extraction recovery and records `parse_recovery=true` in trace.
- Even when recovery is explicitly enabled (`DRUG_AGENT_ALLOW_PARSE_RECOVERY=1`), reward includes `parse_recovery_penalty`.
- Tool success is semantic-level by default (not transport-level). Unknown semantic status is treated as failure by default.

## Tool Debug

The following commands perform real tool-environment interaction. They are not training entrypoints
and require an explicit opt-in:

```bash
export DRUG_AGENT_ALLOW_TOOL_ENV=1

python drug_agent/tools_debug/debug_mcp_tools.py --list-tools
python drug_agent/tools_debug/debug_mcp_tools.py --tool is_valid_smiles --args '{"smiles_list":["CCO"]}'

python drug_agent/tools_debug/debug_sglang_launch.py \
  --model-path "$DEBUG_MODEL_PATH" \
  --tp-size 2 \
  --port 30000 \
  --context-length 8192 \
  --mem-fraction-static 0.80 \
  --dry-run

# non-dry-run starts SGLang in background and returns JSON with pid/process_alive
python drug_agent/tools_debug/debug_sglang_launch.py \
  --model-path "$DEBUG_MODEL_PATH" \
  --tp-size 2 \
  --port 30000 \
  --context-length 8192 \
  --mem-fraction-static 0.80

# add --auto-stop if you only want launch verification
# python drug_agent/tools_debug/debug_sglang_launch.py ... --auto-stop

python drug_agent/tools_debug/debug_replay_trajectory.py \
  --input-jsonl $DATA/mcp_sft_all.train.jsonl \
  --index 0 \
  --max-tool-calls 3 \
  --run-name gate_replay_$(date +%Y%m%d_%H%M%S)

python drug_agent/tools_debug/debug_one_task.py \
  --input-jsonl $DRUG_AGENT_DATA_ROOT/grpo/mixed.jsonl \
  --index 0 \
  --sglang-base-url http://127.0.0.1:30000 \
  --max-steps 1 \
  --temperature 0.2 \
  --max-new-tokens 1024 \
  --run-name gate_27b_step1_$(date +%Y%m%d_%H%M%S)
```

Notes:
- We do not add tool-argument alias auto-fixes in gate stage.
- `Qwen3.5-122B-A10B` / `Qwen3.5-27B` are used for rollout debug or teacher usage, not current actor training.
- `debug_one_task.py` is permissive debug tooling, not a training reward pipeline.
- Formal offline audit: `python drug_agent/tools_debug/audit_offline_training.py`.

## Notes

- v1 defaults to mini allowlist, not all 81 tools.
- Use `DRUG_AGENT_ALLOW_ALL=1` only for debugging.
- Keep rollout strict for RL/SFT evaluation: `DRUG_AGENT_ROLLOUT_MODE=train_strict`, `DRUG_AGENT_ALLOW_PARSE_RECOVERY=0`.
- `reward_func` returns a dict score payload; scripts pass `--reward-key score`.
- `pipelined_data` converters emit `skipped_report.jsonl` for auditability.
- Step-level adversarial distillation lives in [`drug_agent/gad/README.md`](gad/README.md). It starts
  from an existing SFT checkpoint and keeps its Bradley-Terry discriminator outside slime core.
