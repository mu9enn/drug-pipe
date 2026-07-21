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

Repair only the segments listed in `repair_hints.json`. Every entry in
`editable_findings` must be addressed by exactly one matching edit; do not omit
a flagged segment merely because other nearby segments were cleaned. Remove teacher-only
engineering scaffolding such as Claude Code Read/Write/Edit operations, skills
or `.claude/skills` inspection, `run_log.md`, `result.md`, `results.md`, execution
logs, result reports, and file inventories. If a flagged thought mixes scientific
reasoning with engineering scaffolding, remove only the scaffolding and preserve
the scientific decision, tool motivation, parameters, failure diagnosis,
replanning, uncertainty, and evidence-grounded conclusion.

Do not deduplicate thoughts merely because they are similar. Do not impose a
thought-count or length target, and do not generally compress scientific
reasoning.

For a thought that contains only teacher scaffolding or final-answer preparation
with no scientific decision or evidence, set `replacement` to the empty string
to delete that thought. Use a non-empty replacement for a mixed thought and keep
its scientific content. Never delete a thought that contains useful scientific
reasoning, a real tool motivation, failure diagnosis, or evidence.

When a hint says `premature_completion_before_tool`, the thought appears before
the named tool call but claims that operation already completed. Rewrite it as a
reason or intention for making the upcoming call. Do not perform a general
timeline rewrite and do not remove valid success conclusions that occur after
their recorded observation.

In the final summary, refer only to canonical `<artifact:...>` references that
already exist in `source_trajectory.json`. Never introduce a server path, an
unobserved workspace filename, or an engineering report as the task result.

Before writing the patch, check every replacement and remove all teacher-only
terms, including `skills`, `.claude`, `run_log.md`, `result.md`, `results.md`,
`file inventory`, `local workspace`, `Claude Code`, any numbered `Phase`, `ls -la`, and
Read/Write/Edit used as file tools. Do not paraphrase scaffolding into another
engineering task. Delete pure “compile final results,” “prepare final answer,”
and similar process-only thoughts. For a mixed segment, replace “Final
verification of the complete file inventory” with a scientific statement such
as “The repaired structure is ready for downstream analysis,” not with another
statement about files or reports.

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
