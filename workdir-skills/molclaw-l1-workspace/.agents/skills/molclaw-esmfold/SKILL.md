---
name: molclaw-esmfold
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__pred_protein_structure_esmfold`. Use ESMFold model to predict 3D structure of the input protein sequence."
license: MIT license
metadata:
    skill-author: PJLab
---

# Protein Structure Prediction

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

The description of tool *pred_protein_structure_esmfold*.

```tex
Use the ESMFold model for protein 3D structure prediction.
Args:
    sequence (str): Protein sequence
Return:
    status: success/error
    msg: message
    pdb_path (str): The predicted pdb file path
```

How to use tool *pred_protein_structure_esmfold* :

```python
response = await client.session.call_tool(
    "pred_protein_structure_esmfold",
    arguments={
        "sequence": sequence
    }
)
result = client.parse_result(response)
pred_protein_structure = result["pdb_path"]
```
