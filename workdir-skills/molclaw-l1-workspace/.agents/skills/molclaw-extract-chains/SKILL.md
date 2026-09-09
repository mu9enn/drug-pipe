---
name: molclaw-extract-chains
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__extract_and_save_chains`, `mcp__molclaw-scp__extract_pdb_chains`. Extract protein sequence of each chain from the protein structure file (pdb format)."
license: MIT license
metadata:
    skill-author: PJLab
---

# Extract Protein Chains

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

Use tool *extract_pdb_chains* to extract protein chains from the repaired pdb file

Tool description:

```tex
Extract the amino acid sequence of each chain from the PDB file.
Args:
    pdb_file_path (str): Path to input pdb file
Return:
    status (str): success/error
    msg (str): message
    chains (List[dict]): List of dict, each containing the keys 'chain' and 'sequence'.
        --chain (str): Chain ID
        --sequence (str): Sequence string
```

Tool usage:

```python
response = await client.session.call_tool(
    "extract_pdb_chains",
    arguments={
        "pdb_file_path": fixed_pdb_path
    }
)
result = client.parse_result(response)
protein_chains = result['chains']
```
