---
name: molclaw-mol2mol-sampling
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__reinvent_mol2mol_sampling`. Generate new molecules sampling from the input molecule."
license: MIT license
metadata:
    skill-author: PJLab
---

# Mol2Mol Molecule Generation

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

The description of tool *reinvent_mol2mol_sampling*.

```tex
Generate new molecules sampling from the input molecule using different priors ('similarity': broad exploration, 'medium_similarity': balanced exploration, 'high_similarity': conservative optimization, 'scaffold': strict scaffold preservation, 'scaffold_generic': generic scaffold preservation, 'mmp': MMP-style local modifications).
Args:
    smiles (str): Input SMILES string
    n (int): Number of molecules for sampling
    min_similarity (float): Required minimum similarity threshold (commonly 0.6)
    prior_type (str): Required prior type; options: ['scaffold_generic', 'scaffold', 'mmp', 'similarity', 'high_similarity', 'medium_similarity'] (commonly 'similarity')
    lipinski (bool): Required flag controlling Lipinski filtering (commonly True)
    filter_preset (str): Required filter preset; options: ['none', 'minimal', 'default', 'strict'] (commonly 'default')
Return:
    status (str): success/error
    msg (str): message
    save_smiles_file (str): Path to the saved SMILES file
    output_smiles_list (List[str]): List of generated SMILES strings
```

How to use tool *reinvent_mol2mol_sampling*:

```python
response = await client.session.call_tool(
    "reinvent_mol2mol_sampling",
    arguments={
        "smiles": smiles,
        "n": n,
        "min_similarity": min_similarity,
        "prior_type": prior_type,
        "lipinski": True,
        "filter_preset": filter_type
    }
)
result = client.parse_result(response)
output_smiles_list = result["output_smiles_list"]
```
