---
name: manage-drug-pipe-trajectories
description: "Operate the local Drug-Pipe trajectory-production system end to end: inspect environment and live state, sample Tool-KG questions, convert task schemas, run or safely resume Data-Pipe raw rollouts, run Python and LLM cleaning, size and enforce Claude/Data-Pipe concurrency, preserve the manually selected cc-switch provider, diagnose stuck or duplicate processes, and strictly audit raw scientific-tool execution and final training JSONL quality. Use for worker preflight, status checks, production starts, resumptions, monitoring, concurrency or rate-limit planning, timeout/provider failures, duplicate attempts, session checksum issues, MolClaw failure audits, or handoff documentation in the mounted drug-pipe repository."
---

# Manage Drug-Pipe Trajectories

Manage this pipeline as a resumable production system, not as a sequence of ad hoc shell commands. Preserve raw evidence, prove state before acting, and distinguish structural cleaning success from scientific task completion.

## Bootstrap every new context

1. Resolve and verify the repository paths. Use the first path on the login
   workspace and the second inside an rlaunch/rjob worker:

   ```bash
   export REPO=/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe
   # worker: export REPO=/root/slime_sxy/group-space/sunxiangyu/drug-pipe
   export TOOL_KG="$REPO/tool-kg"
   export DATA_PIPE="$REPO/data-pipe"
   export CLAUDE_BIN="$(command -v claude)"
   test -d "$TOOL_KG" -a -d "$DATA_PIPE" -a -n "$CLAUDE_BIN" -a -x "$CLAUDE_BIN"
   ```

   Connect to workers with the ordinary `sunxiangyu` SSH identity, not the
   `sunxiangyu+root` identity. The ordinary identity exposes the user's Claude,
   cc-switch, and provider state while the GPFS repository still lives under
   `/root/slime_sxy/...` and is accessible to that user. Verify all paths and
   commands; if any is missing, fail preflight rather than copy credentials or
   select a provider for the user.

2. Inspect before mutating:

   ```bash
   date --iso-8601=seconds
   pgrep -af 'run_kg_pipeline|pipeline/claude_agent/run_claude.py|pipeline.cleaning.llm_clean|runtime/claude'
   find "$DATA_PIPE/results" -path '*/state/status' -o -path '*/state/current_stage'
   test ! -f "$REPO/.runtime/claude_gate/events.log" || \
     tail -n 30 "$REPO/.runtime/claude_gate/events.log"
   ```

3. Resolve the exact task file, run root, raw run directory, clean roots, active provider, and expected counts from manifests/state/process arguments. Never infer completion from directory names or row-directory counts.

4. Read [references/operations.md](references/operations.md) before starting,
   resuming, parallelizing, or monitoring work. Read
   [references/concurrency-and-provider.md](references/concurrency-and-provider.md)
   before choosing concurrency, preparing a new worker, or handling provider
   state. Read [references/quality-and-recovery.md](references/quality-and-recovery.md)
   before accepting outputs, killing processes, or diagnosing a failure.

5. Treat `docs/END_TO_END_PIPELINE_RUNBOOK.md` in the repository as the current installation runbook. Reconcile it with live code and this skill; live code and explicit user instructions win over stale historical paths.

## Enforce the production invariants

- Keep aggregate production Claude concurrency at 4 by default. Count raw,
  cleaning, adjudication, and interactive Claude processes together. Increase
  to 6 only after the worker/provider canary gates pass; treat 8 as an
  experimental ceiling, never a default. Do not claim a higher limit from CPU
  count or TPM alone.
- Run one KG Data-Pipe controller with tool-aware admission. Use `--max-workers 2`
  by default and never exceed 4. Tasks whose expected toolchains share the same
  limit-4 MolClaw tool must not overlap; different compute tools may run in
  parallel.
- Never run two controllers against the same `--resume-run-dir`. Resume in place and honor `.data_pipe_resume.lock`.
- Allocate LLM-clean workers from Claude slots not currently occupied by active
  Data-Pipe invocations; never exceed the configured aggregate limit, which is
  4 by default.
- Preserve the provider selected manually by the user. Automation must never
  call any cc-switch/config mutation, choose a provider, enable failover, or
  perform peak/off-peak switching, even after an API failure. Require
  `--skip-provider-switch 1`, snapshot the current provider without secrets,
  detect drift, and fail closed rather than switch or mix providers.
- Use `API_TIMEOUT_MS=1800000`, `LLM_CLEAN_TIMEOUT_SEC=3600`, `MOLCLAW_MCP_TOOL_TIMEOUT_MS=14400000`, `CLAUDE_CODE_MAX_RETRIES=10`, and `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000` unless current code/user requirements supersede them.
- Do not pass `--only-molclaw-tool` during cleaning. Retain supported local tool calls alongside MolClaw calls.
- Never delete or overwrite raw sessions or old attempts. A recovery creates a new attempt or validates and reuses an immutable existing result.

## Choose the workflow

- For environment/status questions: inspect processes, state, logs, canonical metadata, stream growth, and current child work. Report evidence; do not restart healthy long-running science.
- For new production: preflight provider/environment, validate or build the Tool-KG task adapter, run raw, then Python clean and LLM clean, then strict audits.
- For interruption: identify the existing `molbench_kg_*` directory and use `--resume-run-dir`; reuse the same clean work/output roots with `--resume-valid-debug`.
- For a suspected hang: compare stream size/mtime over a short interval and inspect descendants. A deliberate sleep or long MolClaw/Boltz/FoldX call is not a hang.
- For duplicate attempts: prove which attempt is canonical and successful using `run_meta.json` plus SHA256, then terminate only the redundant controller branch that can restart it.
- For quality review: run the bundled raw audit, validate clean manifests, and perform task-completion review on flagged high-failure samples.

## Apply two independent acceptance layers

1. **Structural/cleaning acceptance:** schema, pairing, immutable fields, prose cleanup, counts, fallback, and residual findings.
2. **Scientific execution acceptance:** required MolClaw calls actually succeeded and the final answer completed the requested scientific deliverables.

Never equate `accepted_count == N` with scientific completeness. Hard-reject a raw trajectory with zero successful MolClaw results. Review samples with at least 50% failed MolClaw results or explicit fallback from a required method; reject or rerun when the prompt's required-N outputs are missing.

Run the deterministic audit:

```bash
python "$REPO/.codex/skills/manage-drug-pipe-trajectories/scripts/audit_raw_sessions.py" \
  /path/to/molbench_kg_run_or_historical_raw_root \
  --expected 100 \
  --fail-on-hard
```

Use `--json-output /path/to/audit.json` for a durable report. The script excludes `attempts/` and audits canonical `row*/complete_session.jsonl` files.

## Monitor with evidence

- Count completed raw rows only from strict terminal `run_meta.json`, not precreated directories.
- Identify the real Claude cwd through `/proc/<pid>/cwd`; map it to the active row/attempt.
- Check stream growth and recent workspace artifacts before declaring a stall.
- Use empirical runtime as context, not a timeout: recent heavy questions had median about 28 minutes, P75 about 42 minutes, P90 about 88 minutes, P95 about 128 minutes, and observed maximum about 176 minutes. Older workloads contained a 9.4-hour outlier.
- When a gate implementation is present, confirm current Data-Pipe starts
  appear in its logs with `workload=data_pipe` and a `class_slot`. Do not infer
  enforcement merely from gate environment variables.
- Never use broad `pkill` commands. Match PID, PPID, PGID, cwd, invocation marker, and terminal state.

## Hand off clearly

Report:

- exact canonical raw and final clean paths;
- strict completed/failed/active counts;
- active row, attempt, elapsed time, last stream activity, and child operation;
- provider and concurrency-slot state;
- remaining raw and clean work;
- hard quality failures and review-only warnings separately;
- exact resume command and immutable identifiers when another agent must continue.

Do not claim “all good” merely because processes exist, a JSONL has the expected number of lines, or cleaning reports every row accepted.
