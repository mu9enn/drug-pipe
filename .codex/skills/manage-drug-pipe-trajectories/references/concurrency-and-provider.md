# Claude Concurrency and Provider Invariants

## Contents

1. Decision summary
2. Evidence and limits
3. Capacity calculation
4. Promotion and monitoring
5. Provider preservation
6. Worker bootstrap boundary

## 1. Decision summary

For the 16-vCPU, 33,000-MiB CPU worker and the shared 100-RPM / 50,000,000-TPM
token.pjlab.org.cn account:

| Workload | Default | Canary | Initial ceiling |
|---|---:|---:|---:|
| All production Claude processes combined | 4 | 6 | 8 |
| Raw Drug-Pipe tasks sharing limit-4 MolClaw tools | 2 workers | 4 | 4 |
| Claude-only cleaning/adjudication with no raw work | 4 | 6 | 8 |

Keep 4 as the production default. Promote to 6 only after a representative
30-minute canary. Treat 8 as an experimental ceiling requiring a real admission
gate, staggered starts, and live rate/memory monitoring. Do not exceed 8 until a
new measurement under the exact provider, model, prompt, and tool mix proves all
formulas below with at least 20% headroom.

This is not a vendor-published maximum. Claude Code exposes no fixed
"conversations per machine" limit, and 100 RPM is not 100 simultaneous
conversations. Each agentic tool round may create another model request.

## 2. Evidence and limits

### Worker and current-version measurements

Observed on 2026-08-07:

- The rlaunch worker exposes 16 CPUs and 33,792,000 kB total memory, equivalent
  to about 32.23 GiB and consistent with the requested 33,000 MiB.
- The root-style SSH identity had the repository mount but not the user's
  `claude`, `cc-switch`, or provider state. Ordinary-user SSH exposed all three
  while retaining access to the `/root/slime_sxy/...` GPFS mount. Always use the
  ordinary identity for production.
- A temporary credential-free copy of Claude Code 2.1.215 used about 228 MiB
  peak RSS for `--version` and 286 MiB for `--help`, with about 0.10-0.29 CPU
  seconds. Treat these as cold-start lower bounds only, not active-session
  sizing.

Anthropic's [official installation requirements](https://code.claude.com/docs/en/installation#system-requirements)
specify 4 GB or more RAM for the machine. This is a system requirement, not a
promise that each session consumes 4 GB and not a basis for dividing RAM by 4.

Reports in Anthropic's official Claude Code issue tracker provide risk bounds,
not guarantees:

- One report measured roughly 1.5-2 GB active memory per session and only 3-4
  concurrent sessions on a 16-GB Mac under pressure
  ([issue #9604](https://github.com/anthropics/claude-code/issues/9604)).
- A large-response failure held about 1.2 GB, peaked near 1.6 GB, and saturated
  one CPU core during JSON parsing
  ([issue #10479](https://github.com/anthropics/claude-code/issues/10479)).
- A pathological 3.8-GB session transcript caused one Linux process to reach
  12.8 GB RSS and about 91% CPU
  ([issue #22365](https://github.com/anthropics/claude-code/issues/22365)).

Use 1.5-2 GiB only as a planning prior. Measure the entire Claude process tree,
including MCP and scientific-tool children; long sessions and large JSONL files
can invalidate the prior.

### Drug-Pipe request-rate evidence

A 2026-08-07 read-only analysis of 39 successful archived Drug-Pipe attempts
used `result.num_turns / result.duration_ms` as a conservative proxy for model
request cadence. It counted input, cache-creation, cache-read, and output tokens
together for a conservative combined-TPM estimate:

| Metric per active session | P50 | P75 | P90 | P95 | Maximum |
|---|---:|---:|---:|---:|---:|
| Request-turn proxy/min | 4.24 | 5.69 | 7.46 | 9.06 | 12.12 |
| Total tokens/min | 0.341M | 0.457M | 0.588M | 0.744M | 0.787M |

`num_turns` is not a provider access log, so confirm against gateway metrics or
429 responses when available. A faster model such as DeepSeek V4 Flash may
complete tool loops faster than the archive and raise RPM. Do not assume that
Anthropic's cache-aware ITPM accounting applies to the PJLab gateway; its public
100-RPM / 50M-TPM limits are the authority. Anthropic's documentation is useful
only for general semantics: rate limits can use continuously replenished token
buckets and can reject short bursts even when a one-minute average appears safe
([rate-limit documentation](https://platform.claude.com/docs/en/api/rate-limits#about-rate-limits)).

## 3. Capacity calculation

Reserve 20% of the API allowance and about 6 GiB of worker RAM for the OS,
controller, filesystem cache, and non-Claude tools. Calculate:

```text
C_rpm  = floor(80 / P95_session_requests_per_minute)
C_tpm  = floor(40,000,000 / P95_session_total_tokens_per_minute)
C_mem  = floor((worker_memory_mib - 6144) / P95_process_tree_rss_mib)
C_cpu  = capacity that keeps sustained load below 12 on the 16-vCPU worker
C_tool = concurrency of the narrowest shared scientific tool
C_safe = min(C_rpm, C_tpm, C_mem, C_cpu, C_tool)
```

Using the archived P95 values gives `C_rpm = floor(80 / 9.06) = 8` and
`C_tpm = floor(40M / 0.744M) = 53`. RPM therefore binds far before TPM. The
historical maximum cadence at eight sessions is about 97 RPM, leaving almost no
burst margin; this is why 8 is experimental and 6 is the preferred promoted
setting.

Memory alone would suggest about 13 sessions if every complete process tree
stayed below 2 GiB, but that assumption ignores scientific children and known
long-session outliers. CPU alone is also not authoritative: sessions are often
I/O-bound but a response parse or local tool can consume a full core. For raw
Drug-Pipe work, the shared MolClaw limit of 4 remains the binding tool cap.

## 4. Promotion and monitoring

1. Start at aggregate concurrency 4 and stagger new Claude starts by 5-10
   seconds; do not release all slots simultaneously.
2. Run representative prompts for at least 30 minutes before moving to 6, then
   repeat before testing 8. Include long-context and MCP-heavy cases.
3. Measure the whole invocation tree. Record peak RSS, CPU seconds, duration,
   request-turn cadence, total tokens/min, 429 count, retries, and tool-child
   peaks per session.
4. Require all of the following for promotion:
   - worker memory stays below 80% and no OOM kill or sustained swap occurs;
   - sustained one-minute load stays below 12 and tool latency does not regress
     materially;
   - aggregate P95 RPM stays at or below 80 and total TPM at or below 40M;
   - no rate-limit retry storm occurs and the 429 rate is below 1%;
   - raw work still obeys every scientific-tool concurrency limit.
5. Reduce concurrency on any violated gate. Do not compensate for 429s by
   switching provider. Use same-provider bounded exponential backoff with jitter
   and honor `retry-after` when the gateway supplies it.

Recalculate after a Claude Code upgrade, provider/model change, prompt/tool-set
change, or a material increase in transcript/context size.

## 5. Provider preservation

Treat the user's manual cc-switch choice as immutable automation input, never as
something the pipeline owns.

- Never call `cc-switch use`, `cc-switch provider switch`, `cc-switch start`,
  `cc-switch provider set-default`, provider add/edit/delete, or any direct
  write to `~/.cc-switch` or `~/.claude/settings.json`.
- Never enable or use cc-switch automatic failover, proxy takeover, a provider
  queue, or peak/off-peak scheduling. On the login workspace, automatic
  failover was observed disabled on 2026-08-07; recheck rather than assume.
- Always pass `--skip-provider-switch 1` through Drug-Pipe launchers. Treat a
  `--provider` value in a legacy interface or run directory name as metadata
  only.
- Read and record only the current provider ID/name, public base URL/model, and a
  redacted fingerprint at run start. Never log keys or tokens. Prefer the
  existing `canonical-reclean/reclean.py` provider snapshot logic, which redacts
  secret-bearing fields.
- Recheck the fingerprint before every new Claude invocation or batch admission.
  If the user changes the provider during a run, stop admitting new work and
  create an explicit new run/resume boundary. Do not switch back and do not mix
  providers silently.
- If the selected provider is unhealthy, exhaust bounded retries against that
  same provider and fail closed. Report the failure to the user.

The current code had provider-switch calls commented out on 2026-08-07, but the
invariant must not depend on comments remaining in place. Audit executable code
for `cc-switch`, config writes, failover, and provider scheduling after changes.

## 6. Worker bootstrap boundary

Inside a worker, use:

```bash
export REPO=/root/slime_sxy/group-space/sunxiangyu/drug-pipe
export PATH="/home/sunxiangyu/.local/bin:$PATH"
test -d "$REPO/data-pipe" -a -d "$REPO/tool-kg"
command -v claude
command -v cc-switch
cc-switch --app claude provider list
cc-switch --app claude failover show
```

Connect as ordinary `sunxiangyu`, not `sunxiangyu+root`. If either executable or
the manually selected provider state is missing, stop.
Do not copy `~/.claude`, the cc-switch database, or API credentials from another
host as an automated repair. Ask the user to install/authenticate and manually
select the provider on that worker, then rerun the read-only preflight.
