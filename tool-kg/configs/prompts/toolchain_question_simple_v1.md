You generate realistic user questions for a MolClaw drug-discovery and computational-biology agent.

Read `simple_context.json` and `output_schema.json`.

The context contains:
1. a hidden toolchain blueprint sampled from valid Tool-KG edges;
2. compact tool cards for the blueprint tools;
3. compact edge evidence;
4. one randomly selected grounding seed and a small set of related real facts from the local Science-KB.

The hidden toolchain is inspiration, not a required execution contract. Prefer its scientifically useful capabilities, but you may omit an infeasible prefix or use a feasible subset when the supplied facts cannot start every blueprint tool. The public question does not need to reproduce the hidden toolchain. Standard method or tool names may appear when they make the scientific request natural; never expose internal tool IDs, pair IDs, or the hidden blueprint itself.

Write one natural user question that would benefit from capabilities similar to the blueprint. Prefer a closed and executable task:
- include concrete SMILES, UniProt IDs, PDB IDs, complete protein sequences, compound names, constraints, or other values when available;
- use only identifiers and scientific values present in `grounding_facts` or returned by the read-only Science-KB MCP for the same grounded protein/compound;
- never invent biological facts, molecule identifiers, structure files, paths, or base64;
- never use placeholders such as `user_provided_file`, `target_protein`, `ligand_smiles`, `/path/to/file`, or `to_be_specified`;
- do not claim that unavailable files exist.

The rollout has no human available for follow-up. Never ask the execution agent to request missing information from the user.
A `success` question must contain every required starting value inline; do not use conditional clauses such as “if unavailable”, “if not available”, or “provide X before proceeding”.

Before rejecting, try these recovery steps in order:
1. query the read-only Science-KB for the full record of the already selected protein or compound when a sequence, PDB cross-reference, or complete SMILES is missing;
2. use an available PDB instead of sequence-to-structure prediction, or an available complete sequence instead of a missing PDB, when scientifically appropriate;
3. when a real inline input can start a standard enabling MolClaw capability, add that prerequisite even if it is absent from the sampled blueprint (for example, predict a structure from a complete protein sequence before PDB-dependent analysis);
4. omit an infeasible blueprint prefix and formulate a useful task around the remaining feasible capabilities and real facts.

Do not invent warhead fragments, attachment points, file paths, Base64, trajectories, BioEmu outputs, docking poses, or other run artifacts. If no useful task remains after recovery, return `reject`.

The public question must request action and results. Prefer verbs such as analyze, predict, evaluate, compare, rank, screen, identify, generate, and return. Do not ask to design, describe, or set up a workflow.

Output exactly one compact JSON object with this shape:

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

For reject, use an empty public question and payload, plus a non-empty rationale.
Do not write `output.json` or any other file. Print the JSON object as your final answer.
Do not start subagents or background work. Do not use Bash, Write, or another tool to emit the answer: the final assistant response itself must be the JSON object.
Do not output a workflow graph, trajectory, edge claims, tool list, or chain-of-thought. Do not put internal tool IDs or a blueprint inside `question_payload`.
