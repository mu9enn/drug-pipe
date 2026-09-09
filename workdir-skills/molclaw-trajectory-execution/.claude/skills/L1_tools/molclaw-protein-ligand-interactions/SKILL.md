---
name: molclaw-protein-ligand-interactions
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__analyze_protein_ligand_interactions`. Analyze protein-small-molecule contacts, interaction types, pocket composition, and steric clashes from a complex PDB."
license: MIT license
metadata:
    skill-author: PJLab
---

# Protein-Ligand Interaction Analysis

## Invocation

Use the exact tool name and parameter schema exposed by the current harness.

## Tool: `analyze_protein_ligand_interactions`

Inputs:

- `complex_file` (required): server-visible PDB containing protein and a HETATM small-molecule ligand.
- `ligand_identifier` (required): ligand residue name or `auto` to select the largest non-protein HETATM group.

The result reports hydrogen bonds, hydrophobic contacts, pi interactions, salt and halogen bonds, contacted residues, pocket composition, and steric clashes. Verify `status` before interpreting them.

This tool is for HETATM small molecules. For protein or peptide partners represented as ATOM records, use `interaction_visualizer` in its appropriate mode instead.
