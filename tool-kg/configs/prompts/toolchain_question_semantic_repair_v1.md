You are repairing a rejected or non-rolloutable MolClaw Stage3 question.

Read `simple_context.json`, `previous_output.json`, `semantic_feedback.json`, and `output_schema.json`. The sampled hidden toolchain is inspiration, not a required execution contract. Preserve the same grounded scientific entities, but make the question closed, useful, and executable.

Recover in this order:
1. Query the read-only Science-KB for the full record of the already selected protein or compound when its sequence, PDB cross-references, or complete SMILES can supply the missing starting value. Do not switch to an unrelated target or compound.
2. Use an existing PDB in place of sequence-to-structure prediction, or an existing complete sequence in place of a missing PDB, when scientifically appropriate.
3. When a real inline input can start a standard enabling MolClaw capability, add that prerequisite even if it is absent from the sampled blueprint. In particular, a complete protein sequence can support sequence-to-structure prediction before PDB-dependent repair, packing, or pocket analysis.
4. Omit an infeasible blueprint prefix and formulate the question around a useful feasible subset of the capabilities and the real facts that remain.

Never invent or derive unsupported warhead fragments, dummy attachment points, file paths, Base64, trajectories, BioEmu outputs, docking poses, or other run artifacts. Never ask the future user to provide a missing value. Standard method or tool names may appear naturally, but never expose internal tool IDs, pair IDs, or the hidden blueprint itself.
A `success` question must contain every required starting value inline; do not use conditional clauses such as “if unavailable”, “if not available”, or “provide X before proceeding”.

Return exactly one compact JSON object matching `output_schema.json`:

```json
{
  "status": "success",
  "public_question_text": "...",
  "question_payload": {
    "task": "...",
    "inputs": {},
    "expected_output": "..."
  },
  "rationale": "..."
}
```

Return `reject` only when no scientifically useful, self-contained task can be formed after all four recovery steps. Print JSON only and do not write files.
Do not start subagents or background work. Do not use Bash, Write, or another tool to emit the answer: the final assistant response itself must be the JSON object.
