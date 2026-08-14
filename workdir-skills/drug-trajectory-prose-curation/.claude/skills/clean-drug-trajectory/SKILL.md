---
name: clean-drug-trajectory
description: Produce a machine-validated restricted patch that cleans editable Drug-Pipe trajectory prose and creates one grounded high-level planning thought for the first assistant decision. Use in an isolated LLM-clean workdir containing source_trajectory.json, cleaning_context.json, editable_segments.json, prose_findings.json, and the v2 patch contract.
---

# Clean restricted Drug-Pipe prose

## Read the contract

Read these workdir files before editing anything:

1. `source_trajectory.json`
2. `cleaning_context.json`
3. `editable_segments.json`
4. `prose_findings.json`

Read `.claude/skills/clean-drug-trajectory/references/react_trajectory_v1.example.json`, `.claude/skills/clean-drug-trajectory/references/llm_clean_patch_v2.example.json`, and `.claude/skills/clean-drug-trajectory/references/llm_clean_patch_v2.schema.json`. Current workdir files are runtime authority; bundled files are stable contracts and representative examples.

Representative runtime shapes also live in `references/source_trajectory.example.json`, `references/cleaning_context.example.json`, `references/editable_segments.example.json`, and `references/prose_findings.example.json`.

## Create the initial high-level plan

- Inspect the complete successful teacher trajectory before deciding how to write the plan.
- Target exactly the first assistant message containing a tool call or final answer.
- Choose `rewrite_first_thought` only when its first thought already describes the task-level objective and ordered major subgoals; set reason to `existing_thought_is_plan_like`.
- Otherwise choose `prepend_planning_thought`. Use reason `existing_thought_is_step_local` when a thought exists, or `no_existing_thought` when none exists.
- Describe scientific subgoals and their order. Do not copy concrete parameters, paths, artifact IDs, long results, or the known final answer. Do not claim steps absent from the teacher trajectory.
- When prepending, preserve the existing local rationale. When rewriting, do not also emit a prose edit for thought segment zero of that message.

## Restrict every prose edit

- Edit only prose in an existing `<thought>...</thought>` segment or the existing string value of `summary` inside `<final_answer>`.
- Copy every target coordinate exactly from `editable_segments.json`. Never infer or recount coordinates.
- Cover every coordinate listed by `prose_findings.json`, then inspect all remaining allowlisted prose for the same prohibited material.
- Never modify inputs, roles, message order, loss masks, tool calls, arguments, observations, statuses, values, paths, artifact identities, predictions, results, rankings, SMILES, evidence, or validity decisions.
- Never create a final summary. Never add scientific claims or numbers unsupported by `source_trajectory.json`.

## Clean prose conservatively

Remove or rewrite explicit L2/L3 orchestration, teacher-only skill hierarchy inspection, and narration about teacher sidecars such as `question.json`, `parsed_answer.json`, `run_meta.json`, `complete_session.jsonl`, `prompt.txt`, and `CLAUDE.md`.

Preserve L1 tool-skill reads, real task-file operations, scientific decisions, parameters, failure diagnosis, replanning, uncertainty, and evidence-grounded conclusions. Do not remove prose merely because it mentions Read, Write, Edit, Bash, Grep, Glob, `run_log.md`, `result.md`, `results.md`, an execution log, a result report, or a file inventory.

If `only_molclaw_tool` is true, remove pure narration whose sole purpose was a removed local-tool call, while preserving scientific content from mixed thoughts.

Set `replacement` to an empty string only for fully removable scaffolding. For mixed thoughts, retain the scientific content. Merge adjacent thoughts only when they repeat the same action, observation, update, or conclusion without new scientific information. Preserve later text that adds a parameter, observation, failure diagnosis, alternative hypothesis, or replanning decision.

For an existing final summary, use only canonical `<artifact:...>` references already present in the source. Never introduce server paths, unseen workspace files, teacher sidecars, or engineering reports as scientific results.

## Write the patch

Write exactly one file named `llm_clean_patch.json`; do not modify any input file. Always include `planning_action`; use an empty `edits` array when no other prose needs cleaning. Validate the patch against the runtime `llm_clean_patch_v2.schema.json`. Write JSON only to the file; the conversational response is ignored.
