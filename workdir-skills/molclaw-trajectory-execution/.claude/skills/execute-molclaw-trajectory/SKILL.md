---
name: execute-molclaw-trajectory
description: Execute one evidence-grounded MolClaw drug-discovery task with the bundled methodology, workflow, tool, logging, and result contracts. Use for MolBench or KG trajectory workdirs containing question.json and the canonical MolClaw L3/L2/L1 skill hierarchy.
---

# Execute a MolClaw trajectory

Read `references/execution_protocol.md` completely, then execute the task in `question.json` or the user prompt. Treat that reference as the workflow and artifact contract for this scene. `references/question.example.json` documents the representative sidecar shape; the workdir's `question.json` is always authoritative.

Keep triage, skill selection, dependency planning, and self-check internal. Read only the L3/L2/L1 resources selected by triage, except that independent skill files may be read continuously or in parallel without confirmation narration.

Do not emit routine progress phrases or restate plans, tool results, logs, file inventories, self-checks, or the final conclusion. Before a tool call, explain only a substantive scientific decision, in at most two sentences. After a tool result that does not alter the execution path, proceed directly.

Write the required `run_log.md` and `result.md` artifacts exactly as specified by the execution protocol. Return the scientific conclusion once.
