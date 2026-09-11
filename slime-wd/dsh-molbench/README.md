# DSH MolBench MS-1/MS-2/MS-3 evaluation

This runner evaluates the persistent source-built DSH Web host and its configured
local `qwen3.5-9b-local` model. It does not start `train.py`, Ray, SLIME rollout,
or another SGLang server.

The runner defaults to the dedicated `molclaw-v8-eval` preset. Its complete
system prompt is exactly `data-pipe/pipeline/cleaning/prompts/qwen35_system.md`;
DSH runtime context and coding-agent instructions cannot append to it. The only
local model-facing tools are `Read`, `Write`, `Edit`, `Bash`, `Grep`, `Glob`,
and `skill`, alongside the 81 MolClaw MCP tools.

The authoritative publication and paired experiment instructions are in
[`EXPERIMENT_ALIGNMENT.md`](../../docs/EXPERIMENT_ALIGNMENT.md). Historical
83-question results are not comparable with the new isolated release.

## Evaluation contract

- MS-1: all 50 held-out RDKit filtering questions.
- MS-2: all 37 questions. Train/test overlap is removed from training.
- MS-3: 25 complete-ranking questions, enabled explicitly with `--suite ms3`.
- The default run contains 87 MS-1/MS-2 questions.
- One DSH session and one project-root-marked workspace per question.
- No ground-truth answer is written into a task workspace.
- `workdir-skills/molclaw-l1-workspace/` is copied verbatim into every fresh
  task workspace. Its 52 deployed-tool skills live directly under
  `.agents/skills/`; there is no L1/L2/L3 hierarchy or top-level wrapper.
- Each catalog description names its exact `mcp__molclaw-scp__...` tools. The
  model loads the matching instructions through native `skill` before first use.
- `prompt_prefix.md` is concatenated with the frozen JSON-contract benchmark question as
  one user message. No custom `AGENTS.md` or `question.json` is injected.
- Claude/Python SDK examples are mapped explicitly to DSH's native MCP naming:
  a bare server tool `x` is invoked as `mcp__molclaw-scp__x`, never through a
  generic `.call_tool_call` wrapper.
- The unattended runner answers DSH approval requests with `allowed-once` and
  records every decision outside the task workspace under `results/`.
- Official `RdkitBenchEval`, `ACNetCuratedEval` and `MolbenchVsEval` classes compute the metrics.
- The terminal format is the shared structured-v8 bare JSON contract from
  `data-pipe/pipeline/output_contracts.py`. All MS-1 rows use
  `selected_smiles: list`; single-selection questions carry an explicit one-item constraint in the frozen
  question manifest. MS-2 uses `answer_smiles: string`. Each payload has exactly that answer
  field plus `evidence`; `task_type`, prose, Markdown fences, XML envelopes,
  extra fields, and candidate recovery are rejected.

## Smoke test

```bash
python dsh-molbench/run_dsh_molbench.py \
  --suite ms1 --suite ms2 \
  --limit-per-suite 1
```

The command prints the run directory. A prepared run can be resumed with:

```bash
python dsh-molbench/run_dsh_molbench.py \
  --suite ms1 --suite ms2 \
  --limit-per-suite 1 \
  --run-dir <RUN_DIR> \
  --resume --retry-infra-failed
```

## Full evaluation

```bash
python dsh-molbench/run_dsh_molbench.py
```

For a worker image that does not contain MolBench scoring dependencies, keep
rollout and scoring separate:

```bash
# On the worker; writes trajectories and predictions only.
python dsh-molbench/run_dsh_molbench.py \
  --suite ms2 --rollout-only --run-dir <SHARED_RUN_DIR>

# On the login host after all rollouts finish.
python dsh-molbench/run_dsh_molbench.py \
  --suite ms2 --score-only --run-dir <SHARED_RUN_DIR>
```

`--rollout-only` never imports the official evaluator and updates
`rollout_summary.json` after every task. An unattended task that calls
`ask_user_question` fails fast instead of blocking the entire batch.

Key outputs are `run_manifest.json`, `results/*/record.json`,
`preds/`, `metrics.json`, and `evaluation_summary.json`. Failed or missing tasks
receive empty predictions, so the denominator always remains the complete
selected set.

The pretrained matrix uses Qwen native thinking, the `qwen3` reasoning parser,
the `qwen3_coder` tool parser, a 262144-token context and 16384-token response
cap. SGLang enforces greedy decoding through preferred sampling parameters;
DSH's stream-idle timeout is 30 minutes. Only classified transport/stream/MCP
connection failures are eligible for two task-level retries.

MS-3 scoring uses `top3_list_v2`: accept a JSON object's `ranked_smiles` string
list without requiring 60 entries, uniqueness, or candidate membership.
The original exact fields (`ranked_smiles`, `evidence`) and their types remain required.
The upstream Top-3 metrics inspect the original first three positions, including
repeated strings; unknown strings occupy their positions and do not match GT.
No filtering, deduplication, or backfilling occurs. Full-list average rank remains
an auxiliary upstream metric, not an acceptance gate. Published question text
and SFT data validation are unchanged. The scoring policy is saved in the evaluation
summary. Wrapper sensitivity scoring still reports its three extraction policies.
`EVAL_MAX_WORKERS` controls task concurrency in the matrix launcher (default 2).

### New evaluations with infrastructure recovery

Use `python pretrained_matrix/submit_recover.py` in place of
`bash pretrained_matrix/submit_one.sh`, with the same JOB_NAME, MODEL_DIR,
MODEL_ID, MODEL_NAME, GPU_COUNT, TP_SIZE, WORKSPACE_VARIANT, RUN_NAME,
INFRA_NAME and SCORE_PYTHON environment variables. Use a new RUN_NAME.
No existing job or queued historical experiment is opted in automatically.

The new entrypoint fixes task and tool-call concurrency to 1. A client lock
prevents two recovery entrypoints running together. Keep using the serial
queue when mixing with older launchers. Model/token/decoding settings do not
change. The worker requires the existing Landlock binary to be installed
(`deepseek-harness/native/landlock-run`: `pnpm build:native`, musl-tools required)
and checks actual confined shell execution before loading the model.

A task retries only unresolved transient failures: fetch/connection errors,
HTTP 408/429/500/502/503/504, request timeouts, and explicit out-of-memory
errors in actual tool or turn error observations. A later success of the same
tool with the same JSON arguments (key order ignored) resolves an earlier
failure. Business errors with `isError=false` are read from their status and
diagnostics. Wrong answers, malformed final JSON, max-tokens, invalid tool
inputs and generic scientific-program errors do not trigger retries.

Each task executes at most three times, with 60/180-second retry delays.
Attempts and their workspaces are retained under `results/<task>/attempts/`.
First non-infrastructure-failed attempt wins, even if scientifically wrong;
exhaustion retains the last answer. Scoring keeps all tasks in the denominator
and reports `retry_exhausted_count`; unresolved faults make `publishable=false`,
without discarding the last completed answer. Operational completion still
allows the serial queue to advance.

Local model/DSH request failure or unexpected service/worker death delegates
recovery to the login entrypoint. It confirms release before rebuilding the
worker, at most twice, without resetting task budgets or rerunning completed
clean tasks. `worker_attempts.json` records the allocation history. Setup
failures without a recovery marker stop; remote MCP servers are never restarted.
An interrupted controller can resume only after the previous allocation is
confirmed released; it will not allocate a duplicate worker.

The normal runner stays compatible with existing experiments. Only
`--recover-infra` runs use this policy, recorded as `infra_v1`; changing policy
on resume is rejected. No automatic tool-level request replay is added.

For diagnostic reruns, `SAMPLE_IDS_FILE` may point to a JSON list of task IDs
under the shared `/home/sunxiangyu/slime_sxy` tree. The runner's
`--sample-ids-file` selects those tasks; resume and scoring use the manifest's
saved IDs. This is marked `selected_questions_diagnostic`. Selection based on
previous wrong answers is not a fresh full-benchmark accuracy estimate.
