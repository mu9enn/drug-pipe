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
