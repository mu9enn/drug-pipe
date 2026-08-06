---
name: molclaw-visualize-protein
description: >
  Render a server-side PDB protein structure as a PNG with the MolClaw MCP tool
  `visualize_protein`.
license: MIT license
metadata:
  skill-author: PJLab
  skill-level: L1-Tool
  version: 1.0
---

# MolClaw Protein Visualization

Use the live MCP tool `visualize_protein` when the task needs an image of a
protein structure from a PDB file.

This tool renders the structure only. It does not repair a PDB or calculate
protein–ligand interactions. Use `fix_pdb` before visualization when structural
cleanup is required, and use `interaction_visualizer` for residue-level
interaction analysis.

## Input

The live schema has one required field:

| Field | Type | Meaning |
|---|---|---|
| `pdb_file_path` | string | Server-side path to a PDB file |

Use the path returned by protein retrieval, prediction, or `fix_pdb`. For a
local PDB, upload it with the MolClaw file-transfer tool first. Do not pass a
local workspace path or fabricate a server path.

## MCP call

```xml
<tool_call>{"tool_name":"visualize_protein","arguments":{"pdb_file_path":"<artifact:structure/protein.pdb>"}}</tool_call>
```

## Output

On success, the result contains:

- `status: "success"`
- `msg`
- `image_path`: server-generated PNG path

Treat the returned PNG as the authoritative visualization artifact. In
Drug-Pipe online inference, its raw server path is converted to a canonical
artifact reference before entering model-visible context or the final answer.

If the tool reports a missing or invalid PDB, preserve the error observation,
obtain or repair a valid server-side PDB, and retry only when justified.
