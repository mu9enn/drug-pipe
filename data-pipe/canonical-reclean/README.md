# Canonical ReAct v2 prose reclean

This standalone project rewrites only the prose portions of an existing
canonical ReAct dataset. It is intended for historical canonical datasets for
which the raw Claude Code sessions are no longer available.

The immutable authority remains the input trajectory. The cleaner may replace
or delete existing `<thought>` segments and may replace an existing
`final_answer.summary`. It cannot change messages, tool calls, observations,
structured predictions, artifacts, or scientific values.

## Run

```bash
INPUT=../slime-wd/outputs/slime_drug_agent_data/live_tool_catalog_v2/react_trajectories.jsonl \
OUTPUT=../slime-wd/outputs/slime_drug_agent_data/live_tool_catalog_v3 \
MAX_WORKERS=2 \
bash data-pipe/canonical-reclean/run.sh
```

The launcher uses the provider currently selected for Claude Code by
`cc-switch`. It does not switch or hard-code a provider. Every Claude
invocation is archived as an unmodified merged stdout/stderr stream-json file.

Useful overrides:

```bash
RUN_ROOT=/path/to/run \
TIMEOUT_SEC=900 \
MAX_ATTEMPTS=3 \
TOKENIZER=/path/to/Qwen3.5-9B \
bash data-pipe/canonical-reclean/run.sh
```

For a targeted smoke test, set `RECORD_ID=<canonical-id>`; `LIMIT=N` limits the
selected input in source order.

Resume an interrupted run with the exact same input, output, prompts,
tokenizer, and provider:

```bash
RESUME=/path/to/run bash data-pipe/canonical-reclean/run.sh
```

Records that still fail after all attempts are preserved in
`unresolved.jsonl`; they are not silently copied into the train-ready output.
Global provider failures such as HTTP 401/403 or exhausted quota stop the whole
run immediately and write `fatal_error.json`; they are not converted into
hundreds of per-record unresolved results.
After restoring quota for the same provider, resume the existing run. If a
different provider is selected, start a new run so one dataset release never
mixes providers.

## Tests

```bash
cd data-pipe
python -m unittest discover -s canonical-reclean/tests -v
```
