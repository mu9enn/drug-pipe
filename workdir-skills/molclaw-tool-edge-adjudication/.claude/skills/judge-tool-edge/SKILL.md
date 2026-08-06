---
name: judge-tool-edge
description: Adjudicate one directed MolClaw Tool-KG edge from canonical tool skills, MCP schema facts, taxonomy, and the runtime edge contract. Use for an isolated edge-judge workdir containing task_context.json, pair_spec.json, stage_taxonomy.json, edge_contract.json, source_manifest.json, source_tool_card.json, target_tool_card.json, and output_schema.json.
---

# Judge a directed MolClaw tool edge

## Read evidence in order

Runtime files in the workdir are authoritative for the current pair. Files under `references/` document stable contracts and representative shapes only.

1. Read `task_context.json` and `pair_spec.json`.
2. Read `stage_taxonomy.json`, `edge_contract.json`, and `output_schema.json`.
3. Read `source_manifest.json`, then search only the listed canonical skill paths.
4. Read `source_tool_card.json` and `target_tool_card.json` only as structured aids.

Treat canonical L1/L2/L3 skills and MCP schema facts as primary evidence. Do not use `pair_spec.json`, `task_context.json`, or generated tool-card prose as the sole evidence for a positive decision.

Bundled shapes are available in `references/task_context.example.json`, `references/pair_spec.example.json`, `references/source_manifest.example.json`, `references/source_tool_card.example.json`, `references/target_tool_card.example.json`, `references/stage_taxonomy.example.json`, `references/edge_contract.example.json`, and `references/output_schema.json`.

## Decide the direction

- Apply only relation statuses and edge types defined in `edge_contract.json`.
- Judge the supplied source-to-target direction, not an undirected association.
- Allow `valid` when a source satisfies at least one meaningful connectable target input or precondition, even if it cannot satisfy every required input.
- Never reject solely because other target inputs are missing. Record them in `unsatisfied_required_inputs`.
- Use `alternative` only when supported by the runtime contract and evidence.
- Prefer uncertainty over hallucinated mappings, conditions, or evidence.

For `valid` and `alternative`, cite canonical paths from `source_manifest.json` or `snapshot::<tool_id>` whenever possible.

## Return the decision

Return exactly one object matching `output_schema.json`, including the required pair identifier, relation status, direct-transition flag, typed edges, negative reason, context, satisfied mappings, unsatisfied required inputs, evidence, rationale, confidence, and model fields. Use an empty string for `context` when no extra dependency applies.

Return strict JSON only as the final response. Do not add Markdown, comments, extra text, or an output file.
