---
name: molclaw-protein-metrics
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__calculate_pdb_basic_info`, `mcp__molclaw-scp__calculate_pdb_composition_info`, `mcp__molclaw-scp__calculate_pdb_quality_metrics`, `mcp__molclaw-scp__calculate_pdb_structural_geometry`, `mcp__molclaw-scp__calculate_protein_sequence_properties`. Compute sequence properties and basic, geometry, quality, and composition metrics from protein sequences or PDB files."
license: MIT license
metadata:
    skill-author: PJLab
---

# Protein Metrics

## Invocation

Use the exact tool name and parameter schema exposed by the current harness.

## Tools

- `calculate_protein_sequence_properties`: accepts `sequence` and returns length, molecular weight, pI, extinction coefficients, stability, aliphatic index, hydrophilicity, and amino-acid composition.
- `calculate_pdb_basic_info`: accepts `pdb_file_path` and returns atom, residue, chain, element, and approximate molecular-weight statistics.
- `calculate_pdb_structural_geometry`: accepts `pdb_file_path` and returns coordinate ranges, centroid, radius of gyration, and maximum C-alpha distance.
- `calculate_pdb_quality_metrics`: accepts `pdb_file_path` and returns average B-factor, occupancy, and residues per chain.
- `calculate_pdb_composition_info`: accepts `pdb_file_path` and returns atom-name, residue-name, and per-chain composition counts.

Use server-visible paths for PDB inputs. Check `status` and `msg` before using returned metrics as evidence.
