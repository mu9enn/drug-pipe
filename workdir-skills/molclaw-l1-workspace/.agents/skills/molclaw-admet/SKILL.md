---
name: molclaw-admet
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__pred_mol_admet`. Predict the ADMET (absorption, distribution, metabolism, excretion, and toxicity) properties of the input molecules."
license: MIT license
metadata:
    skill-author: PJLab
---

# ADMET Properties Prediction

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

The description of tool *pred_mol_admet*.

```tex
Predict the ADMET (absorption, distribution, metabolism, excretion, and toxicity) properties of the input molecules from smiles list or file.
Args:
    smiles_list (List[str]): Required list of input SMILES strings; pass [] when using smiles_file
    smiles_file (str): Required path to a TXT/CSV SMILES file; pass '' when using smiles_list
Return:
    status (str): success/error
    msg (str): message
    json_content (List[Dcit]): List of dict, each containing the keys 'smiles', 'physicochemical', 'druglikeness' and 'admet_predictions', where 'admet_predictions' includes over 90 key-value pairs representing various molecular properties 
    json_file (str): Path to the json file saving the ADMET prediction results
```

How to use tool *pred_mol_admet* :

```python
response = await client.session.call_tool(
    "pred_mol_admet",
    arguments={
        "smiles_list": smiles_list,
        "smiles_file": ''
    }
)
result = client.parse_result(response)
admet_predictions = result["json_content"]
```
