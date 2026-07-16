# Drug Agent GAD

This directory implements step-level Generative Adversarial Distillation without modifying slime core.
By default the student starts from the original converted model checkpoint; `STUDENT_LOAD` can
optionally select a prior student checkpoint. Stage 1 SFT is intentionally out of scope.
It follows [`../OFFLINE_TRAINING_POLICY.md`](../OFFLINE_TRAINING_POLICY.md): states and historical
observations are fixed, while current-policy next actions are generated but never executed.

## Architecture

- Teacher positives come from cleaned ReAct trajectories.
- Each assistant decision is converted into an aligned `(state, teacher_response)` sample. Tool-call
  and final-answer decisions are both retained; future observations and the target response never
  enter the student prompt.
- Stage 2 generates student negatives with a frozen slime-native student rollout, then warms up a Qwen3.5-0.8B
  discriminator using Bradley-Terry pairwise loss.
- Stage 3 keeps the student on slime-native SGLang + GRPO. An independent discriminator worker
  receives each current on-policy group, scores it with the pre-update version, then immediately
  trains on `teacher > current student`.
- No stage requires live MCP execution or teacher logits.

On the current 516-trajectory cleaned dataset, conversion retains all 3,126 aligned decisions:
2,610 MolClaw tool-call steps and all 516 final-answer steps. Bare MolClaw names are accepted because
the cleaned source is already the authority for MolClaw filtering; explicit non-MolClaw MCP prefixes
and known local engineering tools such as Bash/Read/Write remain rejected.

The independent service is deliberate. Slime's native critic has a scalar head and independent
checkpoint, but its training path is fixed to PPO value loss. Reusing it for BT loss would require
changing slime core and its actor/critic data flow.

## Data

```bash
bash drug_agent/gad/scripts/prepare_gad_step_data.sh
```

Outputs are written under `$DRUG_AGENT_DATA_ROOT/gad/`. Each row contains:

- `prompt` / `state_messages`
- `teacher_response`
- `label`
- `metadata`

## Stage 2: Discriminator Warmup

Generate negatives from the default original model or an optional prior student checkpoint:

```bash
STUDENT_LOAD=/optional/path/to/student_slime_checkpoint \
bash drug_agent/gad/scripts/generate_stage2_negatives.sh
```

This deliberately does **not** use slime's `--debug-rollout-only`: that mode skips Megatron actor
initialization and cannot load a slime SFT checkpoint into SGLang. The script loads the SFT actor
normally, pushes those weights to SGLang, fixes actor LR to zero, uses zero reward, and does not save
actor checkpoints. The student stays frozen while the negative cache is produced.
Unless `NUM_ROLLOUT` is explicitly set, the script derives enough rollout batches to cover the
entire GAD step dataset once.
If the final rollout batch wraps around the dataset boundary, discriminator warmup deduplicates the
cache by `sample_id` before training.

Warm up the discriminator on its own GPU worker:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash drug_agent/gad/scripts/run_stage2_discriminator_warmup.sh
```

The discriminator logs BT loss, accuracy, positive/negative means, margin, and grad norm. Resume by
setting `DISCRIMINATOR_RESUME=/path/to/checkpoint`.
It uses a hidden-state-only `AutoModel` plus scalar head (not an LM head), and left-truncates long
inputs so the appended teacher/student candidate is never discarded.

## Stage 3: Adversarial Training

On the independent discriminator worker:

```bash
CUDA_VISIBLE_DEVICES=0 \
DISCRIMINATOR_RESUME=$DRUG_AGENT_RUNS_ROOT/gad_discriminator_warmup/latest \
bash drug_agent/gad/scripts/serve_discriminator.sh
```

On the student worker:

```bash
STUDENT_LOAD=/optional/path/to/student_slime_checkpoint \
GAD_DISCRIMINATOR_URL=http://DISCRIMINATOR_HOST:8100 \
bash drug_agent/gad/scripts/run_stage3_gad_grpo_smoke.sh
```

For one complete pass over the GAD step data, use `run_stage3_gad_grpo_full.sh`. The default reward is:

```text
clip(0.8 * normalized_discriminator_score
   + 0.1 * format_reward
   + 0.1 * tool_schema_reward, -2, 2)
```

All coefficients are configurable with `GAD_*_REWARD_COEF`. The trajectory log records student
weight versions, discriminator pre/post-update versions, raw/normalized GAD scores, and rule
components. The discriminator service serializes score/update requests with one lock, ensuring a
reward always identifies the exact pre-update discriminator version.

By default, `STUDENT_LOAD` points to the original converted model and is treated as an initialization
checkpoint: only weights are loaded and GAD starts at rollout 0 with fresh optimizer/RNG state. To resume an existing GAD student
checkpoint, set `STUDENT_RESUME=1`, point `STUDENT_LOAD` at it, and set `NUM_ROLLOUT` greater than
its saved rollout index.

Both colocated student scripts remove inherited `expandable_segments` allocator settings because
slime's SGLang `TorchMemorySaver` path does not support them.

Service endpoints:

- `GET /health`
- `GET /metrics`
- `POST /score-and-update`
- `POST /checkpoint`

## Checks

```bash
python -m unittest discover -s drug_agent/gad/tests -p 'test_*.py'
python -m py_compile drug_agent/gad/*.py
bash -n drug_agent/gad/scripts/*.sh
```

GPU smoke is intentionally not launched automatically. Stage 2 and Stage 3 require GPU workers;
Stage 3 additionally requires network reachability from the student worker to the discriminator worker.
