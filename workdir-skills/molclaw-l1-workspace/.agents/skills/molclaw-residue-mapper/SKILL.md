---
name: molclaw-residue-mapper
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__residue_mapper`. Map residue indices among UniProt, PDB author numbering, and tool-internal numbering."
license: MIT license
metadata:
    skill-author: PJLab
---

# Residue Mapper

## Invocation

Use the exact tool name and parameter schema exposed by the current harness.

## Tool: `residue_mapper`

Required input:

- `pdb_path`: server-visible protein PDB path.

Optional sequence sources are `uniprot_id`, `uniprot_fasta`, or `uniprot_seq`; `chain` limits mapping to one chain. Use `predicted` with `input_seq_start` only for predicted-structure arithmetic mapping. `query` can request selected residues, and `output_format` may be `csv` or `json`.

Use `mapping_file` and `query_results` as the residue-numbering authority. Report mismatched or unmapped residues instead of assuming a constant offset.
