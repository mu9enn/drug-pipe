# LLM Clean Prompt for MolBench ReAct SFT Trajectories

You are running inside a local Claude Code workdir.

This workdir contains exactly one source trajectory JSON file:

`{{SOURCE_FILENAME}}`

Your normal output file, if this trajectory is worth keeping, must be:

`{{CLEANED_FILENAME}}`

Variable meanings:

* `{{SOURCE_FILENAME}}`: the original source JSON file, for example `000238__mcp_sft_ac_41821aab498c.json`
* `{{SOURCE_STEM}}`: the source filename without the `.json` suffix
* `{{CLEANED_FILENAME}}`: the cleaned output filename, for example `000238__mcp_sft_ac_41821aab498c-cleaned.json`

Your job is to perform semantic cleanup of this single ReAct SFT trajectory.

Do not use MCP.
Do not rely on `.claude`.
Do not read files outside the current workdir.
Do not create extra JSON files.
Do not modify `{{SOURCE_FILENAME}}`.
Do not print the cleaned JSON to stdout instead of writing the file.
Do not wrap the cleaned sample in `status`, `cleaned_sample`, `audit`, or any other outer object.

The final `{{CLEANED_FILENAME}}`, when created, must be the complete cleaned training sample itself, with top-level structure like:

```json
{
  "schema_version": "...",
  "id": "...",
  "messages": [...]
}
```

## First decide whether this trajectory should be kept

Before creating `{{CLEANED_FILENAME}}`, inspect `{{SOURCE_FILENAME}}`.

Create `{{CLEANED_FILENAME}}` only if the trajectory can be cleaned into a useful training sample without inventing evidence or tool results.

Skip this trajectory and do not create `{{CLEANED_FILENAME}}` if any of the following conditions hold:

1. Most substantive tool calls failed, timed out, or returned unusable errors, and the final answer is essentially a fallback guess that is not grounded in successful observations.

2. The final answer depends on metrics, docking scores, binding affinity values, property values, protein structures, pockets, rankings, or evidence that do not appear in the trajectory.

3. The trajectory contains too little successful tool evidence to support the final answer for its task type.

4. The final answer conflicts with the available evidence, and the correct answer cannot be determined unambiguously from the existing observations.

5. Fixing the sample would require inventing missing values, rerunning tools, fabricating evidence, or assuming external domain knowledge not present in the trajectory.

6. The JSON is too malformed to safely edit while preserving the original sample structure.

Important: repeated tool failures alone are not an automatic reason to skip. Keep the sample if there are still successful, task-relevant observations that support a clear final answer. Skip only when the final answer is basically unsupported by the trajectory.

If you decide to skip:

* Do not create `{{CLEANED_FILENAME}}`.
* If `{{CLEANED_FILENAME}}` already exists for any reason, delete it.
* You may print one short line explaining the skip reason.
* Do not create any alternative JSON output.

## If the trajectory should be kept

If the trajectory is worth keeping, first copy the source file:

```bash
cp "{{SOURCE_FILENAME}}" "{{CLEANED_FILENAME}}"
```

Then edit only:

`{{CLEANED_FILENAME}}`

Do not edit `{{SOURCE_FILENAME}}`.

## What to clean

This is a semantic cleanup task, not a broad mechanical reformatter.

Your main responsibilities are:

1. Remove engineering chatter from assistant thoughts.
2. Preserve useful scientific reasoning.
3. Make the final answer, short reason, and evidence consistent with the actual trajectory.
4. Keep the ReAct structure intact.
5. Avoid fabricating anything.

## Engineering chatter to remove or rewrite

Assistant `<thought>` blocks may contain traces of coding-agent workflow or project operations. Remove or rewrite such content.

Examples of engineering chatter include:

* reading skill documentation
* reading L2/L3 methodology files
* checking available workflows
* initializing or updating `run_log.md`
* updating todo lists
* inspecting `.claude/skills`
* mentioning Claude Code internal workflow
* saying “I found the skills directory”
* saying “the auto-generated skills index is empty”
* discussing file management rather than molecular reasoning
* generic execution bookkeeping unrelated to the scientific task

Do not delete the entire assistant message if it also contains valid `<tool_call>` blocks. Instead, rewrite only the `<thought>` content and preserve the tool calls.

A cleaned thought should be concise, task-relevant, and scientifically grounded.

Example style:

```xml
<thought>This is a pairwise affinity-comparison task. I need to compare the two candidate molecules against the target using available structure, docking, affinity-prediction, or property evidence from the trajectory.</thought>
```

or:

```xml
<thought>The candidate SMILES are valid. I will use the prepared protein structure and predicted pocket information to run docking and rank the molecules by the available docking scores.</thought>
```

## Scientific reasoning to preserve

Preserve or rewrite reasoning that is directly relevant to the molecular task, such as:

* identifying the task type in scientific terms
* validating SMILES
* choosing a protein structure or pocket based on observations
* interpreting docking scores
* interpreting binding-affinity predictions
* interpreting property descriptors such as MolWt, LogP, HBD, HBA, TPSA, RotB, QED, or Lipinski violations
* explaining why one molecule, ranking, or filtered set follows from the tool observations
* explaining how failed tools limit the conclusion, if the sample is still supported by other successful observations

## Evidence rules

Use only evidence already present in the trajectory.

You may use:

* successful tool observations
* numerical values present in observations
* existing valid scientific reasoning already present in the trajectory
* explicit user task requirements
* exact SMILES strings from the user prompt or observations

You must not invent:

* tool calls
* tool outputs
* docking scores
* affinity values
* property values
* protein structures
* pockets
* rankings
* labels
* evidence
* SMILES
* target metadata

If the final answer currently has empty evidence but the trajectory contains real successful observations that support it, add concise evidence derived only from those observations.

If the final answer's `short_reason` conflicts with the evidence, rewrite the reason.

If the final answer itself conflicts with the evidence, you may change the final answer only when all of the following are true:

1. the user task objective is clear;
2. the metric direction is clear from the task, the trajectory, or standard wording in the sample;
3. the observations unambiguously support the corrected answer;
4. no external knowledge is needed.

If these conditions are not all satisfied, skip the trajectory instead of guessing.

## Task-general guidance

The sample may belong to any MolBench-style task family. Do not overfit to AC, VS, PF, or any single example.

For pairwise comparison tasks:

* Ensure the selected molecule matches the evidence.
* Preserve exact SMILES.
* If both molecule A and molecule B have comparable metrics, include those metrics in evidence when available.
* If all tools failed and the final answer is based only on heuristic structural speculation, skip the trajectory.

For ranking or virtual-screening tasks:

* Preserve the ranked list order unless the observed scores clearly require a correction.
* Do not invent missing scores for molecules.
* If the ranking is mostly unsupported because docking/rescoring failed for most candidates and no other evidence supports the order, skip the trajectory.
* If a subset has valid scores but the final answer ranks all candidates using unsupported assumptions, skip unless the original ranking can be justified from available observations.

For property-filtering tasks:

* Preserve exact input SMILES.
* Use only observed property values and explicit constraints.
* If the selected set can be verified from available property observations, keep and clean.
* If required properties are missing and the final set cannot be justified, skip.

## ReAct structure preservation

Preserve the overall sample structure.

Keep:

* top-level `schema_version`
* top-level `id`
* top-level `messages`
* message order
* message roles
* `step_loss_mask` fields
* `<thought>...</thought>` tags where thoughts remain useful
* `<tool_call>...</tool_call>` blocks
* `<observation tool_name="...">...</observation>` blocks
* `<final_answer>...</final_answer>` block

Do not reorder tool calls or observations.

Do not move observations from user role to assistant role.

Do not delete successful observations merely because they are long.

Do not rewrite tool-call JSON unless necessary to preserve valid JSON syntax after editing surrounding text.

Do not rewrite observation JSON except for minimal edits needed to maintain consistency with the cleaned final answer. Prefer leaving observations unchanged.

## Final answer cleanup

The final answer must be internally consistent.

It should contain:

* the task type, if already present or inferable from existing fields;
* the selected SMILES, ranked SMILES, or selected set, depending on task type;
* a concise `short_reason` grounded in existing observations;
* concise `evidence` when real evidence exists in the trajectory.

Do not wrap the final output sample in an audit object.

Do not add a separate report.

Inside the sample, preserve the `<final_answer>...</final_answer>` tag format if present.

If the existing final answer schema is already used by the pipeline, keep it compatible. You may simplify duplicated fields only when doing so does not change meaning and does not risk breaking the sample.

## Quality bar for keeping a sample

A kept sample should teach a model to:

* reason scientifically about the molecule/protein task;
* use real tool observations;
* avoid engineering workflow chatter;
* produce a final answer consistent with the trajectory;
* avoid unsupported claims.

A skipped sample is better than a misleading cleaned sample.

In particular, if the trajectory is dominated by timeouts/errors and the final answer is effectively unsupported, do not generate `{{CLEANED_FILENAME}}`.

## Final checks before finishing

Before finishing, verify:

1. `{{SOURCE_FILENAME}}` is unchanged.
2. If kept, `{{CLEANED_FILENAME}}` exists.
3. If skipped, `{{CLEANED_FILENAME}}` does not exist.
4. If kept, `{{CLEANED_FILENAME}}` is valid JSON.
5. If kept, the top-level object contains `schema_version`, `id`, and `messages`.
6. If kept, there is no outer `status / cleaned_sample / audit` wrapper.
7. If kept, the final answer does not contradict the cleaned reasoning or the available observations.
8. If kept, no tool result or scientific evidence has been fabricated.

Now perform the cleanup decision and edit the file if appropriate.
