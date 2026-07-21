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

Remove engineering chatter, repeated planning narration, file-management
narration, and incoherent wording. Preserve useful scientific reasoning,
failure diagnosis, replanning, uncertainty, and evidence-grounded conclusions.

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
