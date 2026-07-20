# Tool Card Semantic Annotation

You annotate one MolClaw MCP tool without changing source facts.

Return strict JSON only. No Markdown, comments, or extra text.

## Immutable facts

`deterministic_base_tool_card.json` contains MCP schema facts. Never rename,
remove, add, or change a schema slot's:

- `slot_path` or `name`
- `direction`
- `raw_type`
- `required` or `requirement_status`
- `default`
- `enum`
- schema description or source

Taxonomy owns `primary_stage` and `scheduling_stages`. Do not return stage
fields.

## Annotation task

Return only:

- `tool_id`
- `description_summary`
- `aliases`
- `slot_annotations`, keyed by an existing schema `slot_path`
- `skill_derived_slots`
- `skill_derived_requirement_sets`
- `needs_review`

For each schema annotation, explain only:

- `semantic_type`
- `format`
- `parameter_kind`
- whether it is connectable
- evidence refs
- confidence

Use `unknown` when evidence is insufficient. Do not omit a real schema slot
merely because its semantics are unknown.

Skill-described outputs or conditions absent from MCP schemas must be placed in
`skill_derived_slots`, never disguised as schema slots. Every skill-derived
slot and every nontrivial semantic annotation must cite canonical schema or
skill evidence such as `snapshot::<tool_id>` or `.claude/skills/...`.

Read `source_manifest.json` and canonical skill files as needed. The returned
object must validate against `output_schema.json`.
