# Drug-Pipe restricted prose cleaning

Work only inside the current isolated directory.

Read these files before doing anything:

1. `source_trajectory.json` — the record to clean;
2. `react_trajectory_v1.example.json` — a valid complete trajectory example;
3. `llm_clean_patch_v1.example.json` — the exact output shape;
4. `llm_clean_patch_v1.schema.json` — the machine-enforced patch contract;
5. `repair_hints.json` — deterministic hints about prose that may need repair.

Write exactly one file named `llm_clean_patch.json`. Do not modify any input
file. Your conversational response is ignored.

The patch may edit only:

- prose inside an existing `<thought>...</thought>` segment;
- the existing string value of `summary` inside `<final_answer>`.

Use `message_index` to identify the message and `segment_index` to identify the
zero-based occurrence of that segment type inside the message. For
`final_summary`, `segment_index` must be `0`.

Repair only the segments listed in `repair_hints.json`. Address every entry in
`editable_findings` when it can be repaired without changing an immutable fact.
If one entry cannot be repaired safely, omit that edit; the pipeline will keep
the source prose and record an unresolved audit finding.

In the default mode, remove explicit L2/L3 workflow or methodology
orchestration and narration for teacher-only runtime sidecars or non-L1 skill
catalog inspection. Teacher-only sidecars include `question.json`,
`parsed_answer.json`, `run_meta.json`, `complete_session.jsonl`, `prompt.txt`,
and `CLAUDE.md`. L1 tool-level skills and real task file operations are part of
the student runtime and must be preserved. In particular, do not remove or
rewrite a segment merely because it uses Read, Write, Edit, Bash, Grep, Glob, an
L1 Skill, `run_log.md`, `result.md`, `results.md`, an execution log, a result
report, or a file inventory.

If `repair_hints.json` sets `only_molclaw_tool` to true, the Python step has
removed local tool calls. In that mode only, a reason named
`local_tool_narration_removed_in_only_molclaw_mode` means the corresponding
narration should also be removed. Preserve any scientific decision, MolClaw
tool motivation, parameters, failure diagnosis, replanning, uncertainty, and
evidence-grounded conclusion from a mixed thought.

Do not deduplicate thoughts merely because they are similar. Do not impose a
thought-count or length target, and do not generally compress scientific
reasoning.

For a thought that contains only flagged L2/L3 orchestration, or only
mode-specific local-tool narration in `only_molclaw_tool` mode, set
`replacement` to the empty string. Use a non-empty replacement for a mixed
thought and keep its scientific content. Never delete a thought that contains
useful scientific reasoning, a real MolClaw tool motivation, failure diagnosis,
or evidence.

When a hint says `premature_completion_before_tool`, the thought appears before
the named tool call but claims that operation already completed. Rewrite it as a
reason or intention for making the upcoming call. Do not perform a general
timeline rewrite and do not remove valid success conclusions that occur after
their recorded observation.

In the final summary, refer only to canonical `<artifact:...>` references that
already exist in `source_trajectory.json`. Never introduce a server path, an
unobserved workspace filename, or an engineering report as the task result.

Before writing the patch, check that every replacement resolves the exact
reason listed for that segment. Do not turn L2/L3 orchestration into a
paraphrased hierarchy instruction. Do not broadly delete phases, skills, files,
logs, reports, or local-tool operations: those are allowed unless the hint
explicitly identifies L2/L3 orchestration, a teacher-only sidecar/non-L1
catalog, or file narration in `only_molclaw_tool` mode.

Never modify or restate as an edit:

- `schema_version`, sample ID, system message, or user message;
- message roles, order, or loss masks;
- tool names, calls, arguments, or order;
- observations, statuses, values, paths, or artifact identities;
- task predictions, result/ranking/SMILES fields, or evidence;
- execution/task/trace validity or acceptance status.

Do not add scientific claims or numbers not supported by the source. If no prose
needs improvement, write a valid patch with an empty `edits` array.

The output file must contain JSON only and must validate against
`llm_clean_patch_v1.schema.json`.
