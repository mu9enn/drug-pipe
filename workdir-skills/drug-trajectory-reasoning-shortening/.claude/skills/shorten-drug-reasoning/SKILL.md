---
name: shorten-drug-reasoning
description: Compress only explicitly selected oversized reasoning fields in a scientific tool trajectory.
---

# Shorten Drug Reasoning

Read `shortening_input.json`. Its `targets` contain only reasoning available at that decision; do not seek or infer later trajectory information.

For every target:

1. Preserve the operative scientific plan, molecule or entity identifiers, numeric parameters, tool-result evidence already cited in the reasoning, uncertainty, failure diagnosis, and the reason for any changed plan.
2. Remove repeated restatements, circular comparisons, exhaustive scratch work, narration about the agent/runtime, and redundant confirmations.
3. Do not introduce facts, results, or conclusions absent from that target's original reasoning.
4. Aim to make the replacement comfortably shorter than `target_reasoning_tokens`; this is a one-pass compression task, not a summary of the whole trajectory.
5. Keep any initial `High-level plan:` concise if it is present.

Write only `reasoning_shorten_patch.json` using the schema at `schema/reasoning_shorten_patch_v1.schema.json`. Include every target decision exactly once and no other decision.
