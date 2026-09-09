---
name: molclaw-compound-retrieve
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__retrieve_smiles_by_compoundname`. Retrieve SMILES strings by compound name using PubChem with an NCI resolver fallback."
license: MIT license
metadata:
    skill-author: PJLab
---

# Retrieve Compound SMILES

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

The description of tool *retrieve_smiles_by_compoundname*.

```tex
Retrieve SMILES strings by compound name. The service queries PubChem first and falls back to the NCI resolver when PubChem is busy or unavailable.
Args:
    compound_names (List[str]): List of input compound names (e.g., ["aspirin", "caffeine"])
Return:
    status (str): success/partial_success/error
    msg (str): message
    retrieve_smiles (List[dict]): List of dict, each containing the keys 'compound_name' and 'smiles'.
        --compound_name (str): A compound name of compound_names 
        --smiles (str): The retrieved SMILES string, if it exists; otherwise, None.
```

How to use tool *retrieve_smiles_by_compoundname* :

```python
response = await client.session.call_tool(
    "retrieve_smiles_by_compoundname",
    arguments={
        "compound_names": compound_names
    }
)
result = client.parse_result(response)
retrieve_smiles = result["retrieve_smiles"]
```
