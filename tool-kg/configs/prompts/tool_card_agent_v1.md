# Tool Card Extraction (Fixed Taxonomy + Lean Main Card)

You create one evidence-grounded tool card for one MolClaw MCP tool.

Return strict JSON only. No Markdown. No comments. No extra text.

## Stage Constraints (Hard)
- Use the provided `fixed_primary_stage` exactly.
- Never change, rename, or reinterpret `fixed_primary_stage`.
- `secondary_stages` must only come from provided `allowed_stages`.
- Do not invent stages.

## Evidence Priority
1. deterministic schema-derived facts
2. raw MCP schema
3. canonical skills/docs (`.claude/skills`)
4. MCP description
5. domain knowledge
6. tool name (weak)

## Output Contract (Main ToolCard)
Must be compatible with ToolCard model and include:
- `tool_id`, `title`, `description_summary`
- `primary_stage`, `secondary_stages`, `aliases`
- `inputs`, `outputs`, `preconditions`, `side_effects`
- `connectable_inputs`, `connectable_outputs`, `input_requirement_sets`
- `needs_review`

## Slot Shape (Strict)
Each item in `inputs/outputs/preconditions/side_effects/connectable_*` must use:

{
  "name": "string",
  "raw_type": "string",
  "semantic_type": "string",
  "format": "string",
  "unit": null,
  "cardinality": "single|list|map|unknown",
  "parameter_kind": "data|config|control|unknown",
  "requirement_status": "required|optional|conditional",
  "required": false,
  "description": "string",
  "source": "input_schema|output_schema|description|inferred|doc",
  "confidence": 0.0
}

Rules:
- `format` must be string; unknown use `"unknown"` (never `null`).
- For nested useful outputs (e.g. `best_model.cif_path`), flatten into explicit connectable output slots.
- `connectable_inputs` should focus on upstream-satisfiable data/precondition slots.
- `input_requirement_sets` should model alternative/conditional invocation modes when present.

## Confidence and Review
- Set `needs_review=true` when evidence conflicts or key semantics are unclear.

## Optional Extra Fields
You may output additional analysis fields (for debug sidecar), but they are optional and must not break the main ToolCard fields.
