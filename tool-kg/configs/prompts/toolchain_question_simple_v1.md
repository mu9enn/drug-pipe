You generate realistic user questions for a MolClaw drug-discovery and computational-biology agent.

Read `simple_context.json` and `output_schema.json`.

The context contains:
1. a hidden toolchain blueprint sampled from valid Tool-KG edges;
2. compact tool cards for the blueprint tools;
3. compact edge evidence;
4. one randomly selected grounding seed and a small set of related real facts from the local Science-KB.

The hidden toolchain is only an internal blueprint. Never expose tool IDs, tool/product names derived from the blueprint, the hidden chain, or explicit tool order in the public question. Do not use sequencing words such as first, then, next, finally, afterwards, or subsequently to narrate execution steps.

Write one natural user question that would benefit from capabilities similar to the blueprint. Prefer a closed and executable task:
- include concrete SMILES, UniProt IDs, PDB IDs, complete protein sequences, compound names, constraints, or other values when available;
- use only identifiers and scientific values present in `grounding_facts`;
- never invent biological facts, molecule identifiers, structure files, paths, or base64;
- never use placeholders such as `user_provided_file`, `target_protein`, `ligand_smiles`, `/path/to/file`, or `to_be_specified`;
- do not claim that unavailable files exist.

The rollout has no human available for follow-up. Never ask the execution agent to request missing information from the user. If the provided facts are insufficient for a self-contained, executable task, return `reject`.

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
Do not output a workflow graph, trajectory, edge claims, tool list, or chain-of-thought. Do not put tool IDs or a blueprint inside `question_payload`.
