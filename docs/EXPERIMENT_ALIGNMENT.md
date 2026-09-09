> **September 9 scope update:** User authorized the latest regular-v1 596-row release for 8-GPU SFT, automatic 4-GPU SFT paired evaluation, and original **9B only** in both skill environments. Larger-model matrix remains paused. Active dispatch: `drug_pipe_regular_v1_20260908/experiments/experiment_manifest_0909a.json`. The September 8 pause below is historical.

# Dataset and evaluation protocol

> **Paused by user, September 8:** cluster capacity is reserved for an internal
> delivery. Do not submit rjobs. Automatic queues have been stopped. The 596-row
> release and Stage A/B/C code remain prepared; interrupted runs are not completed
> results. Resume only after the user lifts this pause, using fresh run/job names.


The protocol authority is `data-pipe/pipeline/output_contracts.py` (`molbench_answer_contract_v4`). The public task specifies the exact terminal JSON fields; scientific candidate identifiers are copied verbatim. Protocol validation never receives ground truth. An empty PF selection is a valid answer for filtering, whereas a single-selection task requires one candidate. VS requires the complete candidate permutation. Scientific correctness remains a separate evaluator result.

## Benchmark release

The source is InternScience/MolClaw commit `180abb7679ffc5b1ca9974e7390e141d8098f642`. MS-1 has 50 questions, MS-2 has 37, and MS-3 has 25. `python -m pipeline.benchmark_release` (with `PYTHONPATH=data-pipe`) verifies pinned source hashes, preserves upstream bytes, and publishes the JSON-output question CSVs and per-task constraints under `slime-wd/molbench/aligned`. Only output instructions change. The scorer implementations and scientific labels remain upstream definitions.

The DSH runner consumes these published prompts directly. Its default suites are MS-1/MS-2 (87 questions). `--suite ms3` opts into the 25 expensive full-ranking tasks. No training-overlap exclusions are applied to evaluation. MS-3 is included in training isolation even when it is not run.

## Training publication

`pipeline.cleaning.release_aligned` consumes immutable mother data and a raw-session
provenance index. The September 8 user-authorized reconstruction accepts reviewed
whole-trajectory evidence through `--reviewed-reconstructions`. Raw bytes, task,
answer constraints and evidence locations are bound by hashes. The live parser
remains strict and never uses historical recovery. Reasoning appendices disclose
reconstruction rules; these labels are not claimed to be untouched teacher finals.

AC isolation groups use the normalized target and unordered isomeric candidate pair, independent of question direction. PF conservatively groups candidate sets; VS groups target and candidate set. All 112 held-out groups are reserved. Generated training task CSVs receive the shared output instructions and held-out filtering before collection; collection validates rather than rewrites them. Cleaning and SFT export independently reject contract violations.

`pipeline.cleaning.gate_release` uses the checkpoint tokenizer and Slime `qwen3_5` mask for every record. Complete trajectories over 245760 tokens remain in the audit release and are excluded from the training view. There is no truncation, padding by duplicate examples, or hardcoded training count. The manifest binds tokenizer, source and output hashes.

The current release is `/mnt/shared-storage-user/sdpdev-fs/sunxiangyu/drug_wd/drug_pipe_aligned_v4_596_20260908`. Its 605 mother trajectories yield
599 accepted semantic trajectories: 3 reserved-test overlaps, 1 explicit answer-key
reference, and 2 interrupted trajectories without conclusions are excluded.
Three whole overlength KG trajectories are excluded by the context gate, leaving
**596 training trajectories** (AC 100, PF 144, VS 97, KG 237, E2E 18).

Of the previously quarantined records, 84 are restored: AC 5, PF 4, VS 75.
Historical numeric predictions are sorted when their exact/isomeric candidate
identity is available; otherwise explicit report order is retained. In 51 VS
records, candidates absent from usable measurements and the report occupy an
explicit unresolved-priority tail. Lexical ordering breaks ties within that tail;
it is not a measured affinity ranking. This coarse teacher-label reconstruction
was authorized by the user. It is not a claim of scientifically verified labels.
Structurally different or mistyped ligand scores are not transferred to candidates.
Every restoration has a reasoning appendix, original raw path/hash and evidence
locations in `reviewed_reconstructions.jsonl` and `answer_audit.jsonl`.

Training SHA-256: `724106d3027ab240eb723c385df5c8097707987f2770bba9d89024abbdd484f0`.
The old 512-row release and interrupted training are historical only; training
restarts from the original released checkpoint. No old optimizer/checkpoint is reused.

## Harness inputs and comparisons

The checked tool manifest is exported from a real dedicated-preset request: 81 MolClaw tools plus `bash`, `read`, `write`, `edit`, `grep`, `glob`, and `skill`. Exact descriptions and schemas are checked at request assembly. Recorded catalog and skill-result fixtures check the adapter against the harness, normalizing only the workspace prefix. The adapter parses YAML descriptions and supplies skill bodies without frontmatter, matching DSH. Skill projections are explicitly marked synthetic runtime adaptation in provenance.

Both the original Qwen3.5-9B released checkpoint and its SFT descendant use the same launcher. They are evaluated separately in L1-only (52 skills) and L1/L2/L3 (68 skills) environments. Training is aligned to the former; the latter intentionally adds methodological knowledge. Compare model differences within each environment. Larger original Qwen checkpoints are reference baselines unless a corresponding trained checkpoint exists. “Original” does not mean a bare pretraining Base checkpoint.

Thinking remains enabled, decoding is greedy, context is 262144 tokens and the per-response limit is 16384. Task timeout is 14400 seconds and concurrency is two. Only classified infrastructure faults receive up to two retries. Model/format failures stay in the denominator. Missing, unfinished or protocol-unverified runs are not publishable. JSON-format rate, task-constraint compliance and upstream scientific scores are distinct measures; `strict_format_rate` retains its historical meaning of full task-constraint compliance. Upstream auxiliary metrics keep their definitions, including specificity for empty predictions; they must not be described as task success.

Runtime manifests bind task prompts, tool schemas, skill files, tokenizer, model identity and actual sampling observations. Resuming after changes fails. The final epoch checkpoint is chosen without test-set feedback. Earlier 83-question runs and contaminated training checkpoints remain historical results and cannot be pooled with this release.

## Execution

Use `slime-wd/experiments/aligned_sft/submit.sh` with explicit `RELEASE_ROOT`, a new `RUN_ROOT` under that release's `experiments/`, and `JOB_NAME`. The bounded GPU worker validates the release, runs length probes including checkpoint saving, then trains one full epoch and exports the HF checkpoint. Its rjob exits with the driver; no idle GPU keepalive is used.

`evaluate_pair.sh` requires `SFT_MODEL_DIR` and `PAIR_STAMP`. It submits one bounded 4-H200 / TP4 worker for the SFT checkpoint, running both skill environments serially, with a one-question-per-suite smoke gate before each full run. `summarize_pair.py` refuses incomplete or protocol-mismatched pairs. MS-3 is supported by the runner and upstream scorer, but is not launched by the paired experiment.

`follow_training.sh` runs on the login host with `TRAINING_RUN_ROOT`,
`TRAINING_JOB_NAME` and `PAIR_STAMP`. It stops if training fails and otherwise
launches the paired pipeline after the exported checkpoint exists. Existing
completed results are reused; existing live runs are awaited. Model failures are
never rerun to improve a score. Paired summaries include all question-level
scientific scores and a paired bootstrap interval, with a fixed reporting seed.

Stage A runs all eight original-model evaluations (9B, 27B, 35B-A3B and 122B-A10B in both environments). The matrix uses `pretrained_matrix/submit_matrix.sh`
with an explicit new `MATRIX_STAMP`; each model/environment has its own smoke gate.
These reference models are separate from the 9B SFT paired experiment.

The reviewable upstream artifact and application instructions are under
[`dsh-molbench/upstream`](../slime-wd/dsh-molbench/upstream/README.md).

## Validation status on September 8

All 112 frozen tasks pass integrity and contract checks. All 599 semantic records
pass the shared answer and tool-binding checks; full tokenizer/mask gating retains
596 without truncation. The new reconstruction and adapter checks pass 12 tests.
The actual harness replay is referenced with its hash because input protocols did
not change. New optimization, save/reload and full-epoch gates execute in order
inside the bounded 8-H200 / 108-CPU worker before evaluation is permitted.

Stage A uses two serial original-checkpoint queues (eight full evaluations),
Stage B trains the 596-row release, and Stage C uses one 4-H200 / TP4 worker for
serial L1 and hierarchy smoke/full runs. Original 9B uses TP1 per the handoff;
that hardware difference is explicit. Runtime configuration and protocol hashes
are compared before paired results can be published.

The dispatch manifest is `/mnt/shared-storage-user/sdpdev-fs/sunxiangyu/drug_wd/drug_pipe_aligned_v4_596_20260908/experiments/experiment_manifest.json`.
Login sessions: `drug-aligned-v4-matrix-0908b`, `drug-aligned-v4-train-0908b`,
and `drug-aligned-v4-follow-0908b`. No full paired score or expensive MS-3
scientific run is claimed. ToolRL/GAD remain outside this work.
