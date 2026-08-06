---
name: annotate-tool-card
description: Annotate one deterministic MolClaw MCP tool card with evidence-grounded semantics while preserving all MCP schema and taxonomy facts. Use for an isolated Tool-KG tool-card workdir containing task_context.json, deterministic_base_tool_card.json, tool_snapshot_row.json, source_manifest.json, and output_schema.json.
---

# Annotate a MolClaw tool card

## Read the task

Runtime files in the workdir are authoritative for the current invocation. JSON files under `references/` are stable schemas or representative shapes only; never use them in place of current runtime data.

1. Read `task_context.json` and `output_schema.json`.
2. Read `deterministic_base_tool_card.json` as the immutable starting point and `tool_snapshot_row.json` as canonical MCP schema evidence.
3. Read `source_manifest.json`, then inspect only the canonical skill paths listed there that are needed to resolve semantics.

Bundled resource shapes are documented by `references/task_context.example.json`, `references/tool_snapshot_row.example.json`, `references/deterministic_base_tool_card.example.json`, `references/source_manifest.example.json`, and `references/output_schema.json`.

## Preserve source authority

- Never rename, remove, add, or change a schema slot's `slot_path`, `name`, `direction`, `raw_type`, `required`, `requirement_status`, `default`, `enum`, description, or source.
- Never return `primary_stage` or `scheduling_stages`; taxonomy owns them.
- Use `unknown` when evidence is insufficient. Retain every real schema slot even when its semantics remain unknown.
- Put outputs, preconditions, side effects, or conditions described only by skills in `skill_derived_slots`; never disguise them as MCP schema slots.

## Build the annotation patch

Return only the fields allowed by `output_schema.json`:

- `tool_id`
- `description_summary`
- `aliases`
- `slot_annotations`, keyed by an existing schema `slot_path`
- `skill_derived_slots`
- `skill_derived_requirement_sets`
- `needs_review`

For each schema annotation, provide only its semantic type, format, parameter kind, connectability, evidence references, and confidence. Cite nontrivial annotations and every skill-derived fact with canonical references such as `snapshot::<tool_id>` or the canonical paths supplied by `source_manifest.json`.

Use `skill::...` IDs for skill-derived requirement sets. Reference only known schema inputs or skill-derived preconditions. Treat these sets as supplements to MCP schema requirements, never replacements.

## Return the result

Validate the object mentally against `output_schema.json`. Return exactly one strict JSON object as the final response, with no Markdown, comments, or extra text. Do not write an output file.
