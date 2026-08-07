# Drug-Pipe Operations Reference

## Contents

1. Environment and preflight
2. Task preparation
3. Raw execution and resume
4. Cleaning
5. Concurrency
6. Monitoring and progress
7. Runtime expectations
8. Safe termination and handoff

## 1. Environment and preflight

Resolve the repository and use the configured environment. Connect to the
worker as ordinary `sunxiangyu`; do not use the `sunxiangyu+root` SSH identity.
The repository remains mounted under `/root/...` but is user-accessible:

```bash
export REPO=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe
# worker: export REPO=/root/slime_sxy/group-space/sunxiangyu/drug-pipe
export TOOL_KG="$REPO/tool-kg"
export DATA_PIPE="$REPO/data-pipe"
export PATH="/home/sunxiangyu/.local/bin:$PATH"
export CLAUDE_BIN="$(command -v claude)"

# These are admission targets, not proof that a gate wrapper exists.
export CLAUDE_GATE_MAX_CONCURRENCY=4
export CLAUDE_GATE_DATA_PIPE_MAX_CONCURRENCY=2  # set explicitly to 3 or 4 only when requested

export API_TIMEOUT_MS=1800000
export CLAUDE_CODE_MAX_RETRIES=10
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000
export LLM_CLEAN_TIMEOUT_SEC=3600
export MOLCLAW_MCP_TOOL_TIMEOUT_MS=14400000

cd "$DATA_PIPE"
set -a
source .env
set +a
test -n "$MOLCLAW_SCP_MCP_URL" -a -n "$MOLCLAW_SCP_MCP_AUTH"
python -c 'import pipeline.cleaning.llm_clean, pipeline.cleaning.python_clean'
```

Require `claude` and `cc-switch` to exist on the worker. On 2026-08-07 they were
available through ordinary-user SSH but absent through the root-style endpoint;
SSH identity is therefore part of preflight. Do not install provider state,
copy credentials, or choose a provider implicitly.

For Tool-KG, use the isolated worker environment at
`/home/sunxiangyu/slime_sxy/.venv-tool-kg` when present. Require
`mcp>=1.0,<2.0`: MCP 2.0 renamed and changed the streamable HTTP client API used
by the current snapshot code. Never pass MCP/API keys through `--api-key` or
another argv field; source `.env` and let the Python config read the environment
so secrets do not appear in `ps`.

Non-interactive tmux shells return from `.bashrc` before its PJLab proxy setup.
For external MCP endpoints, inject the platform egress proxy into that tmux
only:

```bash
export http_proxy=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128
export https_proxy="$http_proxy"
export no_proxy=127.0.0.1,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn,.pjh-service.org.cn,35.220.164.252
```

Keep `token.pjlab.org.cn` on the direct PJLab route. Validate the MCP initialize
handshake without printing its URL token or auth value before a long launch.

Preserve the active provider unconditionally. Read the current provider ID with
`cc-switch --app claude provider list`, and check
`cc-switch --app claude failover show`.
Require automatic failover, proxy takeover, and provider scheduling to be off.
Do not call a mutation to make them so; ask the user to change them manually.
Avoid logging `provider current` output because it includes a masked API-key
prefix. See [concurrency-and-provider.md](concurrency-and-provider.md) for the
full invariant and capacity gates. If an interactive shell succeeds but tmux
fails, compare `HTTP_PROXY` and `HTTPS_PROXY` before blaming the provider.

Confirm live work:

```bash
pgrep -af 'pipeline/claude_agent/run_claude.py|pipeline/kg/run_kg_pipeline.sh|pipeline.cleaning.llm_clean'
```

Resolve every Data-Pipe process's results root. Use one KG controller for a
batch; its tool-aware admission owns concurrency within that run.

## 2. Task preparation

### Sample from an existing graph

Do not resample merely because a downstream stage failed. When sampling is actually required:

```bash
cd "$TOOL_KG"
bash scripts/run_sample_questions.sh \
  "$KG_RUN_ID" \
  --sampling-profile simple_default \
  --target-successes 100 \
  --max-attempts 500 \
  --min-hops 2 \
  --max-hops 4 \
  --semantic-repair-rounds 1 \
  --grounding-selection random_seeded \
  --max-repeat-target 2 \
  --max-repeat-compound 2 \
  --seed "$SEED"
```

Choose smoke questions by reading their text. Avoid generation-100 + ADMET + all-candidate Boltz2 tasks for connectivity tests.

### Convert Tool-KG schema

Tool-KG canonical `results/tasks.jsonl` commonly uses `id` and `public_question_text`; Data-Pipe requires `task_id` and `question`.

```bash
cd "$DATA_PIPE"
python pipeline/kg/scripts/build_kg_task_dataset.py \
  --kg-run-dir "$TOOL_KG/runs/$KG_RUN_ID" \
  --output-dir "$TOOL_KG/runs/$KG_RUN_ID/data_pipe_input"

KG_TASK_FILE="$TOOL_KG/runs/$KG_RUN_ID/data_pipe_input/kg_sampled_tasks.jsonl"
```

Verify row count, nonempty fields, and unique IDs before execution.

## 3. Raw execution and resume

Start one batch:

```bash
cd "$DATA_PIPE"
bash pipeline/kg/run_kg_pipeline.sh \
  --kg-task-file "$KG_TASK_FILE" \
  --n-cases 100 \
  --claude-bin "$CLAUDE_BIN" \
  --num-rollouts 1 \
  --parallel-rollouts 1 \
  --max-workers 2 \
  --results-root "$RAW_RESULTS_ROOT" \
  --skip-provider-switch 1
```

Resume the same run in place:

```bash
bash pipeline/kg/run_kg_pipeline.sh \
  --kg-task-file "$KG_TASK_FILE" \
  --n-cases 100 \
  --claude-bin "$CLAUDE_BIN" \
  --num-rollouts 1 \
  --parallel-rollouts 1 \
  --max-workers 2 \
  --results-root "$RAW_RESULTS_ROOT" \
  --resume-run-dir "$RAW_RUN_DIR" \
  --skip-provider-switch 1
```

Never start a second controller for the same `RAW_RUN_DIR`. The entry holds `.data_pipe_resume.lock`, but inspect old controllers started before a lock change.

The canonical raw session is:

```text
<RAW_RUN_DIR>/row*/complete_session.jsonl
```

An active or failed attempt remains under:

```text
<RAW_RUN_DIR>/row*/attempts/attempt_NNNN/complete_session.jsonl
```

Do not treat an attempt session as canonical until `run_meta.json` selects it and the promoted canonical hash matches.

## 4. Cleaning

Run Python clean and LLM clean without `--only-molclaw-tool`:

```bash
cd "$DATA_PIPE"
bash scripts/run_cleaning.sh \
  --results-root "$RAW_RUN_DIR" \
  --work-root "$CLEAN_ROOT/python" \
  --output-root "$CLEAN_ROOT/final" \
  --claude-bin "$CLAUDE_BIN" \
  --timeout-sec 3600 \
  --max-workers "$CLEAN_WORKERS" \
  --resume-valid-debug
```

Reuse the exact same Python/final roots when resuming. `--resume-valid-debug` validates source/context/prompt/schema fingerprints, the raw session, patch JSON, immutable fields, final schema, and residual prose. It creates a new attempt for an untrusted cache entry.

The only formal training output is:

```text
<CLEAN_ROOT>/final/react_trajectories.jsonl
```

Do not publish a `debug/` directory or patch file as final output.

## 5. Concurrency

Keep aggregate Claude concurrency at 4 unless a real global admission gate and
the canary criteria in [concurrency-and-provider.md](concurrency-and-provider.md)
prove a higher setting. KG Data-Pipe defaults to 2 workers and may be explicitly
raised to at most 4 when both admission capacities match. The runner serializes
tasks whose expected toolchains share the same limit-4 MolClaw tool; limit-30
tools do not create task-level exclusions.

| Active Data-Pipe invocations | Maximum LLM-clean workers | Total Claude |
|---:|---:|---:|
| 0 | 4 | 4 |
| 1 | 3 | 4 |
| 2 | 2 | 4 |

Use one controller per raw run. Never split the same run between controllers.
The controller may leave a worker idle while all pending tasks conflict with a
limit-4 tool claimed by an active task.

Tool-KG stages and LLM clean do not consume scientific MCP capacity but still
consume global Claude slots. Treat 6 as a canary target and 8 as an experimental
Claude-only ceiling, not as permission to run more than four raw MolClaw tasks.
If no global gate implementation is present, enforce the aggregate limit by
running one controller and not overlapping independent cleaners/controllers.

## 6. Monitoring and progress

Inspect state and logs:

```bash
date --iso-8601=seconds
cat "$RUN_ROOT/state/status" 2>/dev/null
cat "$RUN_ROOT/state/current_stage" 2>/dev/null
tail -n 100 "$RUN_ROOT/serial.log"
test ! -f "$REPO/.runtime/claude_gate/events.log" || \
  tail -n 30 "$REPO/.runtime/claude_gate/events.log"
```

Count strict successful raw rows:

```bash
valid=0; bad=0
for meta in "$RAW_RUN_DIR"/row*/run_meta.json; do
  [[ -f "$meta" ]] || continue
  if jq -e '.return_code == 0 and .timed_out == false and .raw_session_valid == true' "$meta" >/dev/null; then
    valid=$((valid+1))
  else
    bad=$((bad+1))
  fi
done
printf 'strict_valid=%s terminal_bad=%s\n' "$valid" "$bad"
```

Map the real Claude process to an attempt:

```bash
for pid in $(pgrep -f '(^|/)claude( |$)'); do
  printf 'pid=%s cwd=' "$pid"
  readlink -f "/proc/$pid/cwd"
done
```

Compare the active stream's size and mtime twice, then inspect descendants and recent workdir files. A stream may pause while a MolClaw request, Boltz2, FoldX, or an intentional wait is active.

Check gate events for:

```text
start slot=<1..4> class_slot=<1..configured_data_pipe_limit> workload=data_pipe
```

## 7. Runtime expectations

Empirical successful-invocation durations from 139 production samples:

- combined median 18.1 min, P75 38.0 min, P90 73.3 min, P95 121.6 min;
- the newer heavier batch median 28.1 min, P75 42.4 min, P90 87.8 min, P95 127.5 min, maximum 175.6 min;
- an older workload had a 9.41-hour outlier.

These values describe the final successful attempt and can undercount earlier retry cost. Do not terminate solely because elapsed time exceeds the median. Require evidence of no stream/artifact activity, no live scientific child operation, or a terminal failure.

## 8. Safe termination and handoff

Before terminating anything:

1. Resolve PID, PPID, PGID, SID, cwd, command, and raw row.
2. Read `run_meta.json` and attempt metadata.
3. Hash selected attempt and canonical session.
4. Identify the highest ancestor that can restart the redundant process.
5. Send `SIGTERM` to the exact redundant process group; wait; use `SIGKILL` only if required.
6. Verify the intended controller remains alive and canonical hashes did not change.
7. Preserve all attempt files and write a diagnosis report for material incidents.

For ordinary recovery, let the process-group and invocation-marker cleanup in
`session_capture.py` and the active launcher do its work. Never use broad
`pkill python`, `pkill claude`, or deletion-based recovery.
