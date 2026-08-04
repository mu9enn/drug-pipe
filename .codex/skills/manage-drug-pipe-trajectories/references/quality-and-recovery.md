# Drug-Pipe Quality and Recovery Reference

## Contents

1. Acceptance model
2. Raw structural gate
3. Scientific execution gate
4. Clean gate
5. Failure triage
6. Known incidents and durable fixes
7. Regression tests

## 1. Acceptance model

Require three separate conclusions:

1. **Process health:** the controller and current work are alive or terminal as expected.
2. **Trajectory structure:** raw/clean JSONL is parseable, paired, immutable, and schema-valid.
3. **Scientific completion:** the required tools and required-N deliverables succeeded.

Passing one layer does not imply the others. In particular, cleaning removes or repairs teacher prose and validates structure; it is not a scientific evaluator.

## 2. Raw structural gate

For a completed batch require:

- exactly the expected number of unique `(row_number, rollout_index)` entries in `run_summary.jsonl`;
- `return_code == 0`, `timed_out == false`, no parse error, and nonempty answer;
- nonempty `question.json`, `parsed_answer.json`, `run_meta.json`, and canonical `complete_session.jsonl`;
- `run_meta.raw_session_valid == true`;
- no malformed stream events;
- every retained tool call paired with a result;
- at least one successful `mcp__molclaw-*` result.

Do not count precreated `row*` directories. Exclude `attempts/` when auditing canonical sessions.

Use:

```bash
python "$HOME/.codex/skills/manage-drug-pipe-trajectories/scripts/audit_raw_sessions.py" \
  "$RAW_RUN_DIR" --expected "$EXPECTED" --fail-on-hard
```

Hard failures include missing/corrupt session, bad terminal result, strict `run_meta` failure, unpaired calls, and zero successful MolClaw results.

## 3. Scientific execution gate

The current raw quality gate's “at least one successful MolClaw result” is necessary but insufficient.

Flag for task-level review when:

- MolClaw failure ratio is at least 50%;
- a required method is replaced by a fallback;
- a prompt requests N computations but fewer than N succeed;
- the final answer labels a major requested deliverable as unavailable;
- results come from pre-existing workspace artifacts rather than successful calls in the canonical session;
- the final answer claims completion despite tool errors or missing artifacts.

Hard-reject or rerun when the sample is intended as a successful trajectory and required scientific work is missing. Keep an honest failure/recovery trajectory only if the dataset specification explicitly wants such examples.

Historical evidence demonstrates the gap:

- one transferred trajectory had 2/2 MolClaw calls fail and reused prior workspace artifacts, yet cleaning accepted it;
- one old trajectory had 611/758 MolClaw calls fail, obtained only 1 of 100 required Boltz2 predictions, and substituted QuickVina; its report disclosed the gap, but it did not fulfill the original required method.

Do not use a failure-ratio threshold as an automatic rejection by itself: retries and valid recovery can produce many failures. Use it to trigger prompt-versus-deliverable review.

## 4. Clean gate

For expected count N, require `final/run_manifest.json`:

```text
input_count = N
processed_count = N
accepted_count = N
rejected_count = 0
llm_fallback_count = 0
residual_prose_finding_count = 0
```

Require `llm_clean_status_hist` to contain only successful `cleaned` or `not_required` counts. Treat `failed_fallback`, `unsafe_patch_fallback`, and `incomplete_clean_fallback` as incomplete production.

Require exactly N nonblank lines in both:

```text
final/react_trajectories.jsonl
final/curation_audit.jsonl
```

Do not add `--only-molclaw-tool`. Out-of-order observations from concurrent tools are warnings with sidecar detail, not automatic corruption, when call/result pairing remains valid.

## 5. Failure triage

### API timeout or HTTP error

Do not infer provider overload from `API Error: operation timed out` alone. Check provider health, proxy inheritance, stream events, exit code, and retry state. Keep `API_TIMEOUT_MS=1800000` and bounded retries. Preserve failed attempts.

### LLM-clean local timeout

Use 3600 seconds. Admission/slot waiting must not consume this clock. On timeout, terminate the complete process group, wait up to 10 seconds, escalate to `SIGKILL`, reap it, close the stream, then hash/copy.

### Session checksum mismatch

Assume a descendant may still be writing. Do not bypass the immutable hash check. Verify the process tree, stop the complete invocation, and rerun into a new attempt. Never copy a changing file into canonical output.

### Output truncation

Use `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`. A 32K guard failure requires a fresh attempt; do not publish a partial patch.

### Detached MCP helpers

Claude-authored Bash can daemonize helpers outside the launcher PGID. Use the unique `DRUG_PIPE_CLAUDE_INVOCATION_ID` marker and same-user `/proc/*/environ` cleanup. Handle normal exit, nonzero exit, missing executable, and timeout. Do not scan by process name alone.

### Duplicate resume controller

Two controllers can read the same pending state and both create attempts. Use `.data_pipe_resume.lock`. If a duplicate already exists, prove the earlier attempt is successful and canonical before terminating the redundant restart-capable branch.

### Provider mismatch

Generic scripts should keep `CLAUDE_PROVIDER=`. A provider named in an old run is historical configuration, not authority to switch the current user selection. Check the current provider directly.

### NumPy/RDKit warnings

Judge `chemistry_available`, `chemistry_error`, and `python_valid_count` in the Python manifest. Do not stop solely on a warning banner.

### Tool-KG/Data-Pipe schema mismatch

Convert canonical Tool-KG tasks with `pipeline/kg/scripts/build_kg_task_dataset.py`. Do not manually edit or resample because Data-Pipe reported no valid `task_id/question` rows.

## 6. Known incidents and durable fixes

- **Checksum mismatch:** wrapper timeout killed only the shell; real Claude kept writing. Fixed with process-group lifecycle and stable hash selection.
- **Detached MCP orphans:** normal Claude exit left independently sessioned helpers under PID 1. Fixed with invocation IDs and `/proc` cleanup.
- **Duplicate row attempt:** two workflow branches resumed the same raw run. Fixed with per-run lock and separate Data-Pipe class slots.
- **Concurrent observation order:** tool results may return out of call order. Preserve and annotate ordering instead of failing valid concurrent execution.
- **Heavy smoke sample:** the first question requested generation 100, broad property work, Boltz2, and visualization. Read the prompt and select a light sample deliberately.
- **Accepted but scientifically incomplete:** structural cleaning accepted samples with zero successful MolClaw calls or major required-method gaps. Always run the scientific execution gate.

Keep material diagnosis reports near the affected run and include evidence, root cause, impact, repair, and regression results.

## 7. Regression tests

After modifying capture, runner, cleaning, cache, or concurrency:

```bash
cd /home/sunxiangyu/sunxiangyu/drug-pipe/data-pipe
PYTHONPATH=. python -m unittest discover -s pipeline/claude_agent/tests -p 'test_*.py' -v
PYTHONPATH=. python -m unittest discover -s pipeline/cleaning/tests -p 'test_*.py' -v
PYTHONPATH=. python -m unittest discover -s pipeline/kg/tests -p 'test_*.py' -v
```

Cover normal exit, empty output, nonzero exit, parent/child/grandchild timeout, TERM-ignore then KILL, independent-session descendants, stable SHA256, attempt non-overwrite, safe cache resume, out-of-order observations, global concurrency 4, and Data-Pipe concurrency 2.
