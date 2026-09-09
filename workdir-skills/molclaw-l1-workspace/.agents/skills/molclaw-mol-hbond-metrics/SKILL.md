---
name: molclaw-mol-hbond-metrics
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__calculate_mol_hbond`. Compute hydrogen bonding-related properties for a list of SMILES strings, specifically determining the number of hydrogen bond donors and acceptors for each input molecule."
license: MIT license
metadata:
    skill-author: PJLab
---

# Molecular Hydrogen Bonding-related Properties Calculation

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

The description of tool *calculate_mol_hbond*.

```tex
Compute hydrogen bonding-related properties for each SMILES.
Args:
    smiles_list (List[str]): List of input SMILES strings, (e.g., ["N[C@@H](Cc1ccc(O)cc1)C(=O)O", "CC(C)C1=CC=CC=C1"])
Return:
    status (str): success/error
    msg (str): message
    metrics (List[dict]): List of dict, each containing several feature keys.
        --smiles (str): A SMILES string of smiles_list
        --num_h_donors (int): Number of hydrogen bond donors
        --num_h_acceptors (int): Number of hydrogen bond acceptors
```

How to use tool *calculate_mol_hbond*:

```python
response = await client.session.call_tool(
    "calculate_mol_hbond",			
    arguments={
        "smiles_list": smiles_list
    }
)
result = client.parse_result(response)
metrics = result["metrics"]
```
