---
name: summarize-react-step-context
description: Summarize an oversized, history-only Drug-Pipe ReAct decision prefix into grounded chronological JSON events. Use in an isolated context-budget workdir containing context_request.json, omitted_history.json, source_inventory.json, and the react_context_summary_v1 schema; never infer or expose the withheld current assistant target.
---

# Summarize an oversized ReAct prefix

## Read the runtime contract

Read `context_request.json`, `omitted_history.json`, and `source_inventory.json`. Then read `references/output_schema.json` and `references/examples.md`. Runtime files are authoritative.

When `context_request.json` contains a non-null `output_max_tokens`, make the JSON substantially shorter than that budget. Reduce older event detail first while preserving exact state identifiers and recent events.

The current assistant response is deliberately absent. Never search for it, predict it, or add a recommended next tool call.

## Produce grounded chronological events

- Preserve chronological order and cite every contributing message in `source_message_indices`.
- Summarize rationale as the purpose of recorded actions, not as a new plan.
- Copy tool names exactly from the inventory.
- Keep compact arguments only when they identify state needed later. Omit bulk arrays, blobs, file bodies, and repeated defaults.
- Copy statuses, paths, artifacts, IDs, and error text exactly from the inventory. Do not normalize or invent them.
- Set status to `unknown` unless the recorded observation explicitly establishes success or failure.
- Put unresolved recorded failures or pending state in `unresolved_state`; do not resolve them speculatively.
- Keep free-text summaries concise and free of protocol tags.

## Write one result

Write exactly `context_summary.json` and do not modify inputs. Set `source_context_sha256` exactly to the value in `context_request.json`. Validate against `references/output_schema.json`. Output JSON to the file only; the conversational response is ignored.
