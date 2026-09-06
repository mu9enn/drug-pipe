# DSH MolBench MS-1/MS-2 evaluation

This runner evaluates the persistent source-built DSH Web host and its configured
local `qwen3.5-9b-local` model. It does not start `train.py`, Ray, SLIME rollout,
or another SGLang server.

The runner uses DSH's `standard` agent preset. The `code` preset is intentionally
not used because it renders MCP descriptions into a templated Code Mode SDK
prompt; literal double braces in upstream MolClaw examples are valid tool
documentation but are not valid DSH prompt-variable names. `standard` sends the
same MCP tools as native OpenAI tool schemas, matching SGLang's Qwen tool parser.

## Evaluation contract

- MS-1: all 50 held-out RDKit filtering questions.
- MS-2: 33 questions after applying the repository's existing four-item
  training-overlap exclusion list.
- One DSH session and one project-root-marked workspace per question.
- No ground-truth answer is written into a task workspace.
- The Claude Code MolClaw hierarchy is exposed as one DSH-native
  `.dsh/skills/execute-molclaw-trajectory/SKILL.md` entry. L3/L2/L1 documents
  remain on-demand resources instead of bloating the initial DSH skill catalog.
- Claude/Python SDK examples are mapped explicitly to DSH's native MCP naming:
  a bare server tool `x` is invoked as `mcp__molclaw-scp__x`, never through a
  generic `.call_tool_call` wrapper.
- The unattended runner answers DSH approval requests with `allowed-once` and
  records every decision in the task `record.json`. Task instructions direct
  the model to use DSH filesystem tools and avoid needless shell escalation.
- Official `RdkitBenchEval` and `ACNetCuratedEval` classes compute the metrics.

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
  --resume --retry-failed
```

To evaluate a student model with only the tool-level skill catalog exposed,
matching deployment without workflow or methodology skills, add:

```bash
--skill-visibility l1-only
```

The resulting skill snapshot contains only `L1_tools/`; its manifest records
the visibility mode so a run cannot be resumed with a different skill set. In
this mode DSH exposes its native namespaced tool schemas, and the model returns
the benchmark's plain SMILES answer; no legacy XML ReAct bridge is requested.

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

Key outputs are `run_manifest.json`, `workspaces/*/record.json`,
`preds/`, `metrics.json`, and `evaluation_summary.json`. Failed or missing tasks
receive empty predictions, so the denominator always remains the complete
selected set.

The runner first projects the answer from the final assistant message. If that
message violates the benchmark output contract but the completed MolClaw skill
wrote a valid answer to its required `result.md`, the runner uses that artifact
and records `answer_source: result.md`. `final_output_conforming_count` remains a
separate strict harness-format metric; artifact recovery never hides that defect.
