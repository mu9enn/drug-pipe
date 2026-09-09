---
name: molclaw-smiles-valid-check
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__is_valid_smiles`. Check if the input molecule SMILES string is valid."
license: MIT license
metadata:
    skill-author: PJLab
---

# Molecule SMILES Valid Check

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

The description of tool *is_valid_smiles*.

```tex
Check if the input SMILES string is valid
Args:
    smiles_list (List[str]): List of input SMILES strings, (e.g., ["N[C@@H](Cc1ccc(O)cc1)C(=O)O", "CC(C)C1=CC=CC=C1"])
Return:
    status (str): success/partial_success/error
    msg (str): message
    valid_res (List[dict]): List of dict, each containing the keys 'smiles' and 'is_valid'. 
        --smiles (str): A SMILES string of smiles_list
        --is_valid (bool): Is the SMILES valid or not
```

How to use tool *is_valid_smiles* :

```python
response = await client.session.call_tool(
    "is_valid_smiles",
    arguments={
        "smiles_list": smiles_list
    }
)
result = client.parse_result(response)
valid_res = result["valid_res"]
```
