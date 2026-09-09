---
name: molclaw-mol-complexity-metrics
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__calculate_mol_complexity`. Compute custom molecular complexity-related descriptors for a given list of SMILES strings, returning the molecular complexity score, aromatic proportion, and asphericity value for each input molecule."
license: MIT license
metadata:
    skill-author: PJLab
---

# Molecular Complexity-related Properties Calculation

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

The description of tool *calculate_mol_complexity*.

```tex
Compute custom molecular complexity-related descriptors for each SMILES.
Args:
    smiles_list (List[str]): List of input SMILES strings, (e.g., ["N[C@@H](Cc1ccc(O)cc1)C(=O)O", "CC(C)C1=CC=CC=C1"])
Return:
    status (str): success/error
    msg (str): message
    metrics (List[dict]): List of dict, each containing feature keys.
        --smiles (str): A SMILES string of smiles_list
        --molecular_complexity (int): Molecular complexity
        --aromatic_proportion (float): Aromatic proportion 
        --asphericity (float): Asphericity
```

How to use tool *calculate_mol_complexity*:

```python
response = await client.session.call_tool(
    "calculate_mol_complexity",			
    arguments={
        "smiles_list": smiles_list
    }
)
result = client.parse_result(response)
metrics = result["metrics"]
```
