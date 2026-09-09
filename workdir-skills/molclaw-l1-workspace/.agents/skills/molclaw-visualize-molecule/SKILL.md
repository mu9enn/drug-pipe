---
name: molclaw-visualize-molecule
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__visualize_molecule`. Render a molecule from a SMILES string or a server-side molecular structure file with the MolClaw MCP tool `visualize_molecule`.\n"
license: MIT license
metadata:
  skill-author: PJLab
  skill-level: L1-Tool
  version: 1.0
---

# MolClaw Molecule Visualization

## Invocation

Use the exact tool name and parameter schema exposed by the current harness.
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

## Arguments

From a SMILES string:

```json
{"input":"CCO"}
```

From a server-side artifact:

```json
{"input":"/server/path/candidate.sdf"}
```

## Output

On success, the result contains:

- `status: "success"`
- `msg`
- `image_path`: server-generated PNG path

Treat the returned image path as the authoritative artifact and preserve it
exactly for downstream calls.

If the tool returns an error, preserve the observation and revise the input; do
not claim that an image was created.
