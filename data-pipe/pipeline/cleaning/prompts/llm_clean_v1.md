# Drug-Pipe restricted prose cleaning

Work only inside the current isolated directory.

Read these files before doing anything:

1. `source_trajectory.json` — the record to inspect and clean;
2. `cleaning_context.json` — the active tool-retention mode;
3. `react_trajectory_v1.example.json` — a valid complete trajectory example;
4. `llm_clean_patch_v1.example.json` — the exact output shape;
5. `llm_clean_patch_v1.schema.json` — the machine-enforced patch contract.

Write exactly one file named `llm_clean_patch.json`. Do not modify any input
file. Your conversational response is ignored.

Inspect every existing thought and, only when present, the optional final summary. The patch may
edit only:

- prose inside an existing `<thought>...</thought>` segment;
- the existing string value of `summary` inside `<final_answer>`; never create a new summary.

Use `message_index` to identify the message and `segment_index` to identify the
zero-based occurrence of that segment type inside the message. For
`final_summary`, `segment_index` must be `0`. Emit only edits that are actually
needed; use an empty `edits` array when the prose is already suitable.

Remove or rewrite:

- explicit L2/L3 workflow or methodology orchestration;
- teacher-only skill hierarchy/catalog inspection;
- narration about teacher runtime sidecars such as `question.json`,
  `parsed_answer.json`, `run_meta.json`, `complete_session.jsonl`, `prompt.txt`,
  and `CLAUDE.md`.

L1 tool-level skill documents and real task file operations are part of the
student runtime and must be preserved. The student reads L1 `SKILL.md` files
with Read; there is no separate Skill tool. Do not remove or rewrite prose merely because it
uses Read, Write, Edit, Bash, Grep, Glob, `run_log.md`,
`result.md`, `results.md`, an execution log, a result report, or a file
inventory.

If `cleaning_context.json` sets `only_molclaw_tool` to true, Python has removed
all local-tool calls. In that mode only, remove pure narration whose sole
purpose was a removed local call. Preserve scientific decisions, MolClaw tool
motivation, parameters, failure diagnosis, replanning, uncertainty, and
evidence-grounded conclusions from mixed thoughts.

For a thought containing only removable scaffolding, set `replacement` to the
empty string. For a mixed thought, use a non-empty replacement that removes
only the scaffolding and retains the scientific content.

Merge adjacent thoughts when they repeat the same planned action, observation,
progress update, or conclusion with no new scientific information. Keep one
concise statement containing the union of any concrete target names, parameters,
measurements, uncertainty, and scientific rationale from both thoughts. Delete
the later thought when it is an exact duplicate. Similar topic alone is not
enough: preserve a later thought when it adds a new observation, parameter,
failure diagnosis, alternative hypothesis, or replanning decision.

Do not impose a thought-count or length target, and do not generally compress
scientific reasoning. A tool failure is valid execution evidence and may be
followed by diagnosis, replanning, and eventual success; never rewrite the
trajectory merely to hide such a failure.

When an optional final summary exists, use only canonical `<artifact:...>` references already
present in `source_trajectory.json`. Never introduce a server path, an
unobserved workspace filename, a teacher sidecar, or an engineering report as
the scientific task result.

Never modify or restate as an edit:

- `schema_version`, sample ID, system message, or user message;
- message roles, order, or loss masks;
- tool names, calls, arguments, or order;
- observations, statuses, values, paths, or artifact identities;
- task predictions, result/ranking/SMILES fields, or evidence;
- execution/task/trace validity or acceptance status.

Do not add scientific claims or numbers not supported by the source. The output
file must contain JSON only and must validate against
`llm_clean_patch_v1.schema.json`.
