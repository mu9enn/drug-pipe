You are repairing a MolClaw Stage3 grounded workflow proposal.

Read `previous_proposal.json` and `validation_feedback.json` first. Preserve valid parts, but fix every reported error. You may query the read-only Science-KB MCP and read ToolKG/skills context.
Search only inside the current sample workdir, especially `.claude/skills`; never inspect other runs or the wider project tree.

Hard requirements:
- The anchor is inspiration only.
- Use only real Science-KB identifiers and values.
- Keep KB discovery queries small (`limit` 1-3), and fetch full records only after selecting them.
- Never use file-path placeholders or invented scientific entities.
- Remove every unresolved placeholder, including values described as "to be specified by user".
- Add or remove tools when needed for dependency closure and scientific necessity.
- ToolKG edges must exist in `kg_context.json`.
- For ToolKG edges, copy the exact `pair_id` shown in `kg_context.json` into `support_ref`; never synthesize a reference.
- Skills may clarify tool usage, but every workflow transition must be supported by a canonical ToolKG edge in `kg_context.json`.
- The public question must request actual execution, expose all starting values, hide tool names, and avoid explicit tool order.
- Return the complete repaired compact proposal as strict JSON matching `output_schema.json`.
- If the feedback cannot be repaired credibly, return `status="reject"`.
