# ReAct SFT Schema v1

This document describes the cleaned SFT format emitted by `post_process_sft.py`.

## Message shape

Each record is a JSON object with the minimal training payload:

- `schema_version`
- `id`
- `messages`

Provenance and audit fields live in sidecar reports, not in the per-sample training JSON.

### Supported roles

- `system`
- `user`
- `assistant`

Default observation role:

- `user`

### ReAct tags

Assistant content is encoded with plain text tags:

- `<thought>...</thought>`
- `<tool_call>...</tool_call>`
- `<final_answer>{...task-aware final_answer...}</final_answer>`

Observation content is encoded as:

```json
{
  "role": "user",
  "content": "<observation tool_name=\"...\">{...}</observation>"
}
```

## Cleaning rules

- Keep `mcp__molclaw-scp__*` and `mcp__molclaw-vs__*` tool calls
- Drop non-MCP tools and orphan tool results
- Strip only outer triple-backtick wrappers
- Preserve inner content
- Replace strict local absolute paths rooted at `/root`, `/home`, `/tmp`, `/mnt`, or `/workspace` with stable plain-text `<artifact:...>` placeholders
- Compress `fpocket_toolkit` observations to a top-pocket summary
- Keep multi-tool-call sequences by default
- Split multi-tool-call sequences only when `--split-multi-tool-calls` is enabled

## Metadata

Important audit fields are written to:

- `cleaning_reports/*.json`
- `cleaning_report_index.jsonl`
- `schema_validation_report.json`

The final answer schema is task-aware:

- `ac`: `answer_smiles` / `short_reason` / `evidence`
- `vs`: `ranked_smiles` / `selected_smiles` / `short_reason` / `evidence`
- `pf`: `selected_smiles` / `labels`(optional) / `short_reason` / `evidence`
- `kg` / `e2e`: minimal task answer structure

## Validation

The validator checks:

- JSON parseability of ReAct payloads
- task-aware final answer schema
- message role compatibility
- fence stripping statistics
- assistant / tool / observation counts
