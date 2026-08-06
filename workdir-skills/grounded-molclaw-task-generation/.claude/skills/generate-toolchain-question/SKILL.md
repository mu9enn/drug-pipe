---
name: generate-toolchain-question
description: Generate, semantically repair, or JSON-repair one grounded MolClaw drug-discovery question while keeping it self-contained and rolloutable. Use in an isolated Tool-KG sample-question workdir containing output_schema.json plus either simple_context.json, previous_output.json and semantic_feedback.json, or raw_output.txt.
---

# Generate or repair a grounded MolClaw question

## Select the mode

Runtime files are authoritative for the current invocation. Bundled files under `references/` are stable schemas or representative examples only.

Choose exactly one mode from the files present:

1. If `raw_output.txt` exists, perform JSON repair.
2. Otherwise, if `previous_output.json` and `semantic_feedback.json` exist, perform semantic repair.
3. Otherwise, perform initial generation from `simple_context.json`.

Always read `output_schema.json` before producing the answer.

Use `references/output_schema.json`, `references/simple_context.example.json`, `references/previous_output.example.json`, `references/semantic_feedback.example.json`, and `references/raw_output.example.txt` to understand the supported runtime shapes when needed.

## Generate an initial question

Read `simple_context.json`. Treat its hidden toolchain, compact tool cards, and edge evidence as inspiration rather than a required public execution contract. Ground every entity and scientific value in `grounding_facts` or in a read-only Science-KB result for the same selected protein or compound.

- Write one natural, closed, executable request that asks for actions and results.
- Include every required starting value inline. Never ask a future user for missing information.
- Never invent identifiers, facts, files, paths, Base64, warheads, attachment points, trajectories, docking poses, or other run artifacts.
- Never use placeholders such as `user_provided_file`, `target_protein`, `ligand_smiles`, `/path/to/file`, or `to_be_specified`.
- Do not expose internal tool IDs, pair IDs, or the hidden blueprint. Avoid explicit tool-order narration.
- Use `target_fanout_runtime_minutes` only as a light fan-out sizing hint. Ignore it when the task has no repeated expensive step; never mention it publicly or trade away required science to fit it.

Before rejecting, try in order: query the selected entity's full Science-KB record; substitute a real PDB for a missing sequence or a complete sequence for a missing PDB when appropriate; add a standard enabling prerequisite supported by a real inline input; omit an infeasible blueprint prefix and retain a useful feasible subset.

## Repair semantics

Read `simple_context.json`, `previous_output.json`, and `semantic_feedback.json`. Preserve the same grounded scientific entities and apply the same constraints as initial generation. Fix the reported rolloutability problem using the recovery order above. Return `reject` only if no useful self-contained task remains.

## Repair JSON

Read `raw_output.txt` and `output_schema.json`. Convert the prior answer into one valid object without changing its scientific meaning or adding tools, identifiers, values, facts, or missing inputs. Remove formatting and internal toolchain leakage. Return `reject` if the original task is not self-contained.

## Return the result

Return exactly one compact JSON object matching `output_schema.json`. For success, populate `public_question_text`, `question_payload.task`, `question_payload.inputs`, `question_payload.expected_output`, and `rationale`. For rejection, use an empty public question and payload plus a non-empty rationale.

Return JSON only in the final response. Do not write files, start agents, or use shell tools to emit the answer.
