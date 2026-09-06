---
name: clean-drug-trajectory
description: Produce a restricted semantic reasoning patch without changing trajectory facts.
---

# Clean semantic trajectory reasoning

Read `source_trajectory.json`, `cleaning_context.json`, and `editable_reasoning.json` completely.

The semantic trajectory is the authority. Only propose replacements for the `reasoning` field of an existing `assistant_decision`, identified by its exact `source_message_id`. Never change, synthesize, reorder, merge, or delete decisions, tool calls, arguments, observations, final responses, task text, resource references, or provenance.

Remove:

- L2/L3/LR/auto-generated skill orchestration, CLAUDE.md, and teacher-runtime narration;
- references to collection sidecars or transcript inspection;
- consecutive prose that repeats an action, observation, or conclusion without adding information.

Preserve scientific intent, evidence, parameters, failures, uncertainty, alternatives, replanning, and path references already grounded in the user task or a tool observation. A replacement may be empty only when the original reasoning is pure removable scaffolding. Do not invent new paths or resource identifiers, and do not introduce protocol tags, new facts, or facts learned only from the final answer.

Provide a concise task-level plan in `high_level_plan`, targeting the first assistant decision. Describe ordered scientific subgoals without leaking concrete results. The materializer prepends it to that decision's cleaned reasoning; do not create a separate decision.

Write exactly one JSON file named `semantic_reasoning_patch.json`:

```json
{
  "schema_version": "semantic_reasoning_patch_v1",
  "sample_id": "the source id",
  "high_level_plan": {
    "decision_id": "first source_message_id",
    "text": "concise task-level plan"
  },
  "reasoning_replacements": [
    {"decision_id": "source_message_id", "replacement": "cleaned prose"}
  ]
}
```

Use an empty `reasoning_replacements` array when no reasoning needs editing. Do not modify any input file and do not write conversational output.
