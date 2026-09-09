> **September 9 scope update:** User authorized the latest regular-v1 596-row release for 8-GPU SFT, automatic 4-GPU SFT paired evaluation, and original **9B only** in both skill environments. Larger-model matrix remains paused. Active dispatch: `drug_pipe_regular_v1_20260908/experiments/experiment_manifest_0909a.json`. The September 8 pause below is historical.

# Post-alignment Qwen training and evaluation handoff

> **Paused by user, September 8:** cluster capacity is reserved for an internal
> delivery. Do not submit rjobs. Automatic queues have been stopped. The 596-row
> release and Stage A/B/C code remain prepared; interrupted runs are not completed
> results. Resume only after the user lifts this pause, using fresh run/job names.


Do not launch this workflow until the data/contract/harness-alignment work has
published its final release manifest and declared the release ready. Historical
`602`-row paths, the old 105-tool snapshot, 83-question assertions, and old
MolBench exclusions are not valid defaults for the new experiment.

## Required inputs from the alignment work

Record these immutable values in a new experiment manifest before submission:

- final training JSONL path, row count, SHA-256, context-gate manifest, and
  tokenizer/loss-mask validation report;
- frozen MS-1/MS-2 question snapshot and protocol hash: 50 + 37 = 87 selected
  questions, unless the final alignment report explicitly changes this;
- final DSH preset and exported tool-schema hash: 81 MolClaw tools plus the
  seven allowed local/harness tools, with `tool-skill` enabled and `ask-user`
  absent;
- hashes/counts for the 52-skill L1 workspace and the 68-skill hierarchy
  workspace after their final normalization;
- final common decoding/runtime settings and the current MolClaw relay port.

Fail closed if any input is missing or if a launcher still embeds the previous
training path/count, 105 tools, 83 questions, or a test-question exclusion.

## Stage A: eight pretrained Qwen evaluations

Run two login-host queues concurrently. Each queue submits and waits for four
GPU rjobs serially, so at most two evaluation workers are active at once:

1. `l1-flat`: 52 L1 skills from
   `workdir-skills/molclaw-l1-workspace/.agents/skills`.
2. `hierarchy`: the finalized 68-skill L1/L2/L3 workspace, exposed at
   `.agents/skills` without modifying DeepSeek Harness itself.

Each queue evaluates these checkpoints in order:

| Tag | Checkpoint on login host | GPUs | SGLang TP |
| --- | --- | ---: | ---: |
| `9b` | `drug-pipe/slime-wd/data/Qwen3.5-9B` | 1 | 1 |
| `27b` | `huggingface/zskj-hub/models-Qwen-Qwen3.5-27B` | 2 | 2 |
| `35b-a3b` | `huggingface/zskj-hub/models-Qwen-Qwen3.5-35B-A3B` | 2 | 2 |
| `122b-a10b` | `huggingface/zskj-hub/models-Qwen-Qwen3.5-122B-A10B` | 4 | 4 |

Use one bounded foreground worker driver per rjob. Do not allocate a keepalive
and SSH-start the workload. Every question gets a fresh workdir populated from
the selected workspace. Both variants use the same DSH executable, preset,
question snapshot, tool schemas, prompt, MolClaw endpoint, task budget, and
scorer. Fixed settings are native thinking, temperature 0, context 262144, and
response limit 16384. Only infrastructure failures may be retried, at most
twice; format and scientific failures remain in the denominator.

The existing `slime-wd/dsh-molbench/pretrained_matrix` scripts are useful
scaffolding, but must be updated and smoke-tested against the final manifest
before reuse. In particular, replace their `expected = 83` assertion and old
preset/tool-count assumptions. Use a new round/stamp and new output roots; do
not resume or overwrite historical runs.

Publish per-run protocol manifests plus one matrix summary. A run is publishable
only with 87 selected questions accounted for, no unresolved infrastructure
failure, matching protocol hashes, and upstream MS-1/MS-2 scores computed over
the full selected denominator.

## Stage B: latest Qwen3.5-9B SFT

This may run alongside Stage A after the final release exists. Start from the
same local Qwen3.5-9B checkpoint listed above and the final training JSONL from
the release manifest—not a path copied from this document.

- request one 8×H200 worker, `--cpu=108`, memory about `1060000` MiB;
- use the existing measured Qwen3.5-9B full-parameter 8-GPU SFT profile unless
  the final length distribution invalidates it;
- bind the rjob lifetime to the foreground training driver;
- run real-tokenizer validation and short/p50(or p95)/maximum optimizer gates,
  then a checkpoint-save/reload gate before the full epoch;
- derive expected row/exclusion counts and hashes from the final manifest;
- use a new timestamped training root and record base-model, data, code, tool,
  skill, and prompt hashes plus resolved settings and final checkpoint marker.

Do not edit the existing hard-coded
`run_worker_sft_8gpu_skill_native.sh` in place and silently call it current.
Create a new bounded launcher or parameterize it so its data path, counts,
probe set, run root, and hashes resolve from the finalized release.

## Stage C: wait for SFT, then evaluate it

Start one login-host tmux orchestrator immediately after submitting Stage B.
The tmux process may monitor and submit jobs; it must not own GPU work inside a
worker. It waits for both the SFT completion marker and a readable/reloadable
final checkpoint. If the training rjob terminates without them, it exits
nonzero and does not submit evaluation.

After successful SFT, submit **one bounded 4×H200 evaluation rjob** (`--cpu=64`,
about `530000` MiB). Convert/export and validate the final checkpoint as needed,
then evaluate MS-1 and MS-2 with the same frozen protocol used by the original
checkpoint runs. Run the 52-skill L1 and finalized 68-skill hierarchy variants
**serially inside this worker**, with fresh question workdirs and separate outputs.
Release the worker when both variants succeed or either fails.

The orchestrator then scores both runs, verifies all 87 selected questions are
in each denominator, writes the paired pretrained-vs-SFT summary, records final
rjob states, and exits. It must not choose checkpoints or retries based on test
scores.

## Minimum smoke and acceptance gates

Before the full matrix or post-SFT evaluation:

1. replay the actual first DSH request and skill result against the SFT adapter;
2. run one MS-1 and one MS-2 question in each workspace variant;
3. verify the tool catalog/preset hashes, fresh-workdir rule, exact JSON parser,
   and upstream scorer projection;
4. confirm model/format/scientific failures are counted and only infrastructure
   failures are retryable;
5. confirm the foreground worker exits and the rjob releases its allocation.

Keep historical runs as historical evidence, but do not merge their scores into
the new aligned matrix.
