---
name: molclaw-visualize-molecule
description: >
  Render a molecule from a SMILES string or a server-side molecular structure
  file with the MolClaw MCP tool `visualize_molecule`.
license: MIT license
metadata:
  skill-author: PJLab
  skill-level: L1-Tool
  version: 1.0
---

# MolClaw Molecule Visualization

Use the live MCP tool `visualize_molecule` when the task needs a simple molecular
structure image.

This tool produces a depiction only. It does not calculate molecular properties
or protein–ligand interactions. Use `interaction_visualizer` instead when the
task requires residue-level interaction analysis.

## Input

The live schema has one required field:

| Field | Type | Meaning |
|---|---|---|
| `input` | string | A SMILES string or a server-side `.sdf`, `.smi`, `.smiles`, or `.mol` path |

For a local molecular file, upload it with the MolClaw file-transfer tool first
and pass the returned server-side artifact path. Do not invent a server path.

## MCP call

From a SMILES string:

```xml
<tool_call>{"tool_name":"visualize_molecule","arguments":{"input":"CCO"}}</tool_call>
```

From a server-side artifact:

```xml
<tool_call>{"tool_name":"visualize_molecule","arguments":{"input":"<artifact:molecule/candidate.sdf>"}}</tool_call>
```

## Output

On success, the result contains:

- `status: "success"`
- `msg`
- `image_path`: server-generated PNG path

Treat the returned image as the authoritative artifact. In Drug-Pipe online
inference, the raw server path is converted to a canonical artifact reference
before it is shown to the model or used in the final answer.

If the tool returns an error, preserve the observation and revise the input; do
not claim that an image was created.
