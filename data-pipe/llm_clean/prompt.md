# Semantic Cleaner for ReAct SFT Trajectories

You are operating inside an isolated local Claude Code workdir.

The workdir initially contains exactly one source trajectory JSON file:

`{{SOURCE_FILENAME}}`

If the trajectory is suitable for training after semantic cleanup, you must create:

`{{CLEANED_FILENAME}}`

Variable definitions:

* `{{SOURCE_FILENAME}}`: the immutable source trajectory JSON
* `{{SOURCE_STEM}}`: the source filename without the `.json` suffix
* `{{CLEANED_FILENAME}}`: the cleaned trajectory output filename

Your task is to decide whether the trajectory can become a high-quality ReAct SFT training sample and, if so, edit it into one.

---

## 1. Non-negotiable file contract

You must obey all of the following:

1. Do not modify `{{SOURCE_FILENAME}}`.
2. Do not use MCP.
3. Do not rely on `.claude`.
4. Do not read or assume files outside the current workdir.
5. Do not rerun any scientific tool.
6. Do not create any JSON file other than `{{CLEANED_FILENAME}}`.
7. Do not create reports, audit files, manifests, logs, or sidecar files.
8. Do not print the cleaned JSON to stdout instead of writing it.
9. Do not wrap the cleaned sample inside `status`, `cleaned_sample`, `audit`, or any other outer object.
10. Do not create `{{CLEANED_FILENAME}}` until you have decided that the trajectory should be kept.

If the trajectory should be kept, first copy the source:

```bash
cp "{{SOURCE_FILENAME}}" "{{CLEANED_FILENAME}}"
```

Then edit only `{{CLEANED_FILENAME}}`.

If the trajectory should be skipped:

* do not create `{{CLEANED_FILENAME}}`;
* delete `{{CLEANED_FILENAME}}` if it already exists;
* do not create an alternative output file;
* optionally print one concise skip reason to stdout.

---

## 2. Required output shape

When created, `{{CLEANED_FILENAME}}` must be the complete cleaned training sample itself.

Its top-level structure must remain compatible with the source and must contain:

```json
{
  "schema_version": "...",
  "id": "...",
  "messages": [...]
}
```

Preserve:

* the sample identity;
* message roles;
* chronological order;
* `step_loss_mask` fields;
* the ReAct protocol;
* the original user task;
* exact molecular strings and identifiers;
* real tool calls and real tool observations.

---

## 3. Core quality standard

A kept trajectory must teach a model to:

* understand the user objective;
* choose actions relevant to that objective;
* reason from real observations;
* recover from useful failures without repeating unproductive actions;
* avoid unsupported scientific claims;
* produce a final answer grounded in the trajectory;
* avoid irrelevant coding-agent or workflow behavior.

Every retained message, reasoning statement, tool interaction, and final-answer claim must contribute to understanding or solving the user task.

A misleading or unsupported trajectory must be skipped rather than cosmetically cleaned.

---

## 4. Inspect the trajectory before editing

Before creating the cleaned file, inspect the complete source trajectory and construct an internal understanding of:

1. the user objective;
2. the required output contract;
3. the sequence of assistant reasoning, tool calls, and observations;
4. which tool calls succeeded, failed, or returned unusable content;
5. which observations are relevant to the final answer;
6. the actual basis of the final answer;
7. any contradictions, unsupported claims, structural corruption, or incomplete reasoning.

Do not decide based only on the final message. Evaluate the trajectory as a whole.

---

## 5. Keep-or-skip decision

Keep the trajectory only when its final answer can be made fully supportable using information already present in the sample.

Skip the trajectory if any of the following applies:

### 5.1 Unsupported outcome

The final result materially depends on values, evidence, rankings, properties, labels, structures, or conclusions that do not appear in the user task or real observations.

### 5.2 Evidence-insufficient fallback

The trajectory is dominated by failed or unusable tool calls, and the final answer is primarily an unsupported fallback, guess, or heuristic conclusion.

Repeated failures alone do not require skipping. Keep the trajectory when remaining successful observations still provide sufficient support for the answer.

### 5.3 Irreconcilable contradiction

The final answer conflicts with available observations or earlier valid reasoning, and the correct result cannot be determined unambiguously from the existing trajectory.

### 5.4 Incomplete task fulfillment

The user requires a complete comparison, selection, ranking, filtering result, or other structured output, but the trajectory only contains enough evidence to support an incomplete result.

### 5.5 Unsupported interpretation

The conclusion requires assuming the meaning, direction, unit, scale, or interpretation of an opaque metric that is not defined by the user task, observation, or trajectory.

### 5.6 Critical structural corruption

Important content has been corrupted in a way that prevents reliable reconstruction of the task, tool interaction, evidence, or final answer.

### 5.7 Fabrication would be required

Repairing the sample would require rerunning tools, inventing missing outputs, using external knowledge, or making an uncertain scientific judgment.

### 5.8 Unsafe JSON repair

The JSON or ReAct structure is too malformed to repair confidently without changing the original trajectory semantics.

When uncertain whether the final answer is supportable, skip the trajectory.

---

## 6. Semantic cleanup procedure

For a kept trajectory, clean the sample according to the following rules.

### 6.1 Remove non-task operational chatter

Remove or rewrite assistant reasoning that describes the process of operating as a coding agent rather than solving the user task.

This includes any reasoning primarily about:

* locating or reading instructions;
* browsing project structure;
* inspecting helper files;
* managing logs, todos, plans, phases, or internal workflow state;
* loading skills, methods, templates, or agent resources;
* narrating generic execution bookkeeping;
* discussing the local workspace rather than the task;
* announcing actions without explaining their task relevance.

Do not preserve operational chatter merely because it appears inside a `<thought>` block.

Rewrite retained thoughts so that they express only:

* the current task-relevant state;
* the reason for the next action;
* the interpretation of an observation;
* the basis for a decision;
* a concise limitation when relevant.

### 6.2 Remove repetition and incomplete prose

Remove or rewrite:

* repeated restatements of the same plan;
* repeated summaries of unchanged information;
* repetitive transition phrases;
* unfinished sentences;
* truncated thoughts;
* duplicated reasoning;
* conclusions that are stated multiple times without adding information.

Each retained thought must be complete, concise, and useful.

### 6.3 Remove unproductive retry loops

A failure can be useful when it motivates a meaningful change of strategy.

Preserve a failed interaction only when it contributes to understanding the subsequent recovery or limitation.

Remove redundant failed retries when they:

* repeat essentially the same call;
* fail for the same reason;
* introduce no new information;
* do not cause a meaningful strategy change;
* do not contribute to the final conclusion.

When removing a failed interaction:

* remove the corresponding tool call and its matching observation together;
* preserve chronological consistency;
* preserve any later reasoning that remains necessary;
* do not remove successful tool evidence.

Do not turn an unproductive retry loop into a successful trajectory by inventing results.

### 6.4 Align stated plans with executed actions

Reasoning must describe actions that actually occur in the trajectory.

Remove or rewrite statements that:

* promise an action that is never executed;
* claim that an action succeeded when no supporting observation exists;
* describe a result before it is observed;
* claim that a file, structure, score, or analysis exists when it does not;
* continue referring to an abandoned plan as if it determined the answer.

The cleaned reasoning should reflect the actions and observations that actually occurred.

### 6.5 Ground all scientific claims

Every decisive scientific claim must be grounded in one or more of:

* explicit user constraints;
* real successful observations;
* values directly present in the trajectory;
* already-established facts explicitly stated within the sample.

Do not introduce external scientific knowledge merely to strengthen the answer.

Do not infer the meaning, direction, unit, or transformation of a metric unless it is explicitly established in the sample.

Do not convert or reinterpret numerical values unless the conversion rule is explicitly available in the trajectory.

When the meaning of a metric is uncertain:

* describe only the directly observed comparison;
* avoid unsupported interpretation;
* skip the sample if the final decision requires that interpretation.

### 6.6 Repair final-answer grounding

The final answer must agree with the user objective, retained reasoning, and available observations.

For every decisive result in the final answer:

* verify that it appears in or follows unambiguously from the trajectory;
* verify that exact strings and ordering are preserved where required;
* verify that the explanation uses only real evidence;
* remove unsupported assertions.

If the existing final answer is wrong but the correct answer is unambiguously determined by the existing observations, correct it.

If the correct answer is not unambiguous, skip the trajectory.

If the final-answer schema contains an evidence field and relevant successful observations exist, the evidence field must not remain empty.

Evidence added to the final answer must be:

* concise;
* directly copied or faithfully summarized from observations;
* limited to information relevant to the final decision;
* free of invented values or interpretations.

If the schema contains a reason or rationale field, it must state the actual evidence-based decision rule rather than a generic statement that an answer was selected or extracted.

Do not add fields that would make the final-answer schema incompatible with the source format.

### 6.7 Repair deterministic upstream corruption

The source may contain corruption introduced by earlier mechanical processing.

Repair such corruption only when the intended content is deterministically recoverable from the sample itself.

Examples of corruption categories include:

* malformed ReAct or XML-like tags;
* altered ordinary prose;
* altered scientific units;
* malformed artifact references;
* broken JSON escaping;
* a field copied from the wrong neighboring field;
* duplicated values that are structurally impossible;
* invalid wrapper syntax.

General rules:

1. Never treat arbitrary slash-containing text as a filesystem path.
2. Preserve normal prose, scientific notation, units, tags, and identifiers.
3. Artifact references must remain stable plain-text references, not malformed markup.
4. If a corrupted nonessential field cannot be recovered reliably, remove that field.
5. If a corrupted essential field cannot be recovered reliably, skip the trajectory.
6. Never invent a replacement numerical value.
7. Never silently convert a corrupted value into a plausible-looking value.

### 6.8 Preserve useful ReAct behavior

Preserve task-relevant interactions that demonstrate:

* appropriate tool selection;
* interpretation of successful observations;
* meaningful recovery from a failure;
* evidence-based decision making;
* limitations caused by unavailable evidence.

Keep the ReAct sequence coherent after cleanup.

Do not reorder tool calls or observations.

Do not move observations between roles.

Do not fabricate a cleaner action sequence than the one that actually occurred.

---

## 7. Minimality requirement

The cleaned trajectory must be a minimal sufficient trajectory.

After cleanup, each retained component must satisfy at least one of these purposes:

* defines the user objective;
* performs a relevant action;
* provides relevant evidence;
* explains a meaningful decision;
* explains a meaningful recovery;
* states the grounded final answer.

Remove content that serves none of these purposes.

Minimality does not mean removing necessary reasoning or evidence. It means removing content that does not improve the training signal.

---

## 8. Final-answer requirements

The cleaned final answer must:

1. satisfy the user's requested output semantics;
2. preserve exact molecular strings and ordering when required;
3. be supported by retained observations;
4. contain no invented values;
5. contain no unsupported metric interpretation;
6. contain a concise evidence-based reason when the schema supports one;
7. contain concise non-empty evidence when the schema supports evidence and relevant observations exist;
8. agree with all retained reasoning;
9. avoid generic statements that merely say the result was selected, ranked, predicted, or extracted;
10. preserve the existing `<final_answer>...</final_answer>` wrapper when present.

---

## 9. Structure-preservation requirements

Unless removal is explicitly allowed by this prompt, preserve:

* top-level `schema_version`, `id`, and `messages`;
* message roles;
* message chronology;
* `step_loss_mask`;
* valid `<thought>` blocks;
* valid `<tool_call>` blocks;
* valid `<observation>` blocks;
* the final-answer wrapper;
* tool names;
* successful observation values;
* exact user-provided molecular strings.

Do not broadly reformat the sample.

Do not rewrite user instructions merely for style.

Do not modify successful observations except to repair deterministic corruption.

---

## 10. Mandatory validation before completion

Do not finish until all applicable checks pass.

### File checks

* `{{SOURCE_FILENAME}}` is unchanged.
* A kept sample has exactly one new JSON file: `{{CLEANED_FILENAME}}`.
* A skipped sample has no `{{CLEANED_FILENAME}}`.
* `{{CLEANED_FILENAME}}` is valid JSON.
* The cleaned file is the complete sample, not a wrapper or report.

### Structural checks

* The top-level object remains compatible with the source schema.
* Message roles and chronology are coherent.
* ReAct tags and JSON blocks are valid.
* Tool calls and observations remain correctly paired after any removal.
* No deterministic structural corruption remains.

### Semantic checks

* No operational or coding-agent chatter remains.
* No incomplete or truncated thought remains.
* No redundant unproductive retry loop remains.
* No thought claims an unexecuted action or nonexistent result.
* No unsupported metric meaning, direction, or unit remains.
* Every decisive final-answer claim is grounded in retained evidence.
* The final answer agrees with the cleaned reasoning and observations.
* Relevant evidence is not omitted from an available evidence field.
* No value, result, tool output, or scientific fact has been fabricated.

#### Hard forbidden thought phrases

Inspect assistant `<thought>` blocks only. A kept sample must not contain any of:

```text
.claude
skills directory
skill directory
run_log
todo list
update todo
Phase 0
Task Type Triage
Type A
L1
L2
L3
methodology files
workflow skills
read the relevant skills
read the methodology
checking the directory
current working directory
```

Continue editing until they are removed. If they cannot be removed safely, skip.

#### Evidence hard check

If the final-answer schema contains `evidence` and the retained trajectory contains a successful observation relevant to the final answer, `evidence` must be a non-empty list grounded only in observations or user constraints. If safe evidence cannot be supplied, skip.

#### Unsupported metric interpretation hard check

Unless explicitly defined by the user task or observations, remove unsupported interpretations such as `log10(IC50)`, `IC50 ≈`, `Ki ≈`, `converted to`, or `corresponds to IC50`. If the final decision depends on such an interpretation, skip.

#### Exact molecular strings hard check

For benchmark tasks, every molecular string in tool calls and the final answer must exactly match the user-provided candidate strings:

* AC: only the exact Molecule A or Molecule B strings.
* PF: only exact strings from the user's SMILES list.
* VS: only exact strings from the candidate list.

Never correct, complete, canonicalize, or guess a user-provided molecular string. Skip if a tool call or final answer uses a different string, if an invalid input was silently corrected, or if an artifact marker corrupts a molecular string.

#### Unexecuted-plan hard check

Remove plans for actions that were never executed or whose results do not support the final answer, including promised docking, rescoring, cross-validation, or irrelevant file/report generation.

#### Unproductive retry-loop hard check

You may delete repeated failures only when they use the same tool or purpose, nearly identical arguments, the same failure reason, cause no strategy change, and do not support the final answer. Always remove the assistant tool call and matching user observation together. Never rewrite failure as success, and preserve failures that demonstrate meaningful recovery.

#### Mechanical postprocess boundary

Do not preserve or introduce tool metadata such as `tool_use_id`, `raw_event_index`, `raw_pointer`, or local path strings in rewritten assistant text.

Do not introduce local absolute or relative filesystem paths. Use existing `<artifact:...>` placeholders if an artifact reference is needed.

Mechanical path cleanup and final observation metadata removal are handled by postprocess. Do not remove observation metadata before using `raw_status` or `raw_is_error` when they are needed to resolve a status conflict.

#### VS ranking semantic repair

For VS / ranking tasks, check whether the final `ranked_smiles` is consistent with the tool evidence.

If docking scores are the stated ranking criterion, rank molecules by docking affinity from most favorable to least favorable. More negative docking affinity means stronger predicted binding.

Use only scores actually present in tool observations. Do not invent missing scores.

If some candidates have failed or missing docking scores, place scored candidates first according to the available ranking criterion, then place failed/missing-score candidates at the end in their original candidate order, unless the trajectory explicitly states another rule.

If rescoring metrics are explicitly used as the final ranking criterion, follow the trajectory's stated criterion and explain it.

If the ranking criterion is ambiguous or the observations cannot be aligned to candidate SMILES, skip the sample by not creating `{{CLEANED_FILENAME}}`. You may print `vs_ranking_ambiguous` to stdout, but do not create a report or add a warning field to the sample.

The final-answer `ranked_smiles` must contain exactly the candidate SMILES required by the task, use original strings exactly, contain no duplicates, be consistent with available evidence, and include concise evidence for the top-ranked molecules.

#### Observation status semantic repair

Check every tool observation for status consistency.

If an observation has outer `ok=true` / `status=success` but its content clearly contains an execution error, failed request, missing service, timeout, traceback, or connection failure, rewrite only the observation status wrapper so it is consistently represented as an error.

Allowed changes:

* `ok: true` to `false`;
* `status: "success"` to `"error"`;
* `content.status` to `"error"` when content is JSON;
* preserve `content.error` or the existing error message;
* optionally shorten a very long error message without changing its meaning.

Forbidden changes:

* do not fabricate successful outputs;
* do not remove an error by pretending the tool succeeded;
* do not invent metrics, docking scores, structures, or artifacts;
* do not change valid successful observations;
* do not rewrite scientific observation content except for status normalization.

If uncertain whether an observation is truly failed, do not normalize the status. Leave the observation unchanged. If that uncertainty is important for the final answer or prevents a grounded final answer, skip the trajectory by not creating `{{CLEANED_FILENAME}}`. The downstream validator will flag unresolved status conflicts.

### Minimality checks

* Every retained message contributes to the task.
* Every retained thought is concise and task-relevant.
* The trajectory contains no avoidable repetition.
* Removing any additional retained evidence would weaken the training signal.

If any required semantic or structural check cannot be satisfied safely, delete `{{CLEANED_FILENAME}}` and skip the trajectory.

Now inspect `{{SOURCE_FILENAME}}`, decide whether to keep or skip it, and perform the required local edit.
