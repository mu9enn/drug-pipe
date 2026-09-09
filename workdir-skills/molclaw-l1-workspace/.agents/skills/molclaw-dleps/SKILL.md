---
name: molclaw-dleps
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__calculate_dleps_score`. Calculate disease reversal scores for the provided molecules relative to a specific disease."
license: MIT license
metadata:
    skill-author: PJLab
---

# DLEPS Score Calculation

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

The description of tool *calculate_dleps_score*.

```tex
Enter a list of candidate small molecules. Based on the input disease name, identify upregulated and downregulated genes associated with the disease state, and predict a reversal score for each small molecule. Generally, a score above 0.2 indicates effectiveness, with higher scores being better.
Args:
    smiles_list (List[str]): List of input SMILES strings, (e.g., ["N[C@@H](Cc1ccc(O)cc1)C(=O)O", "CC(C)C1=CC=CC=C1"])
    disease_name (str): Supportes diseases, e.g., "Aging", "Gout", "Pulmonary fibrosis", "Non-alcoholic fatty liver disease", "Obesity" 
Return:
    status (str): success/error
    msg (str): message
    pred_scores (List[dict]): List of dict, each containing the keys 'smiles' and 'cs_score'. 
        --smiles (str): A SMILES string of smiles_list 
        --cs_score (float): Predicted reverse score
```

How to use tool *calculate_dleps_score* :

```python
response = await client.session.call_tool(
    "calculate_dleps_score",
    arguments={
        "smiles_list": smiles_list,
        "disease_name": disease_name
    }
)
result = client.parse_result(response)
pred_scores = result["pred_scores"]
```
