---
name: molclaw-drug-likeness
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__calculate_mol_drug_chemistry`. Compute the drug-likeness metrics (QED score and Number of violations of Lipinski's Rule of Five) of the input candidate molecules (SMILES format)."
license: MIT license
metadata:
    skill-author: PJLab
---

# Molecular Drug-likeness Metrics Calculation

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

The description of tool *calculate_mol_drug_chemistry*.

```tex
Compute key drug-likeness metrics for each SMILES.
Args:
    smiles_list (List[str]): List of input SMILES strings, (e.g., ["N[C@@H](Cc1ccc(O)cc1)C(=O)O", "CC(C)C1=CC=CC=C1"])
Return:
    status (str): success/error
    msg (str): message
    metrics (List[dict]): List of dict, each containing feature keys.
        --smiles (str): A SMILES string of smiles_list
        --qed (float): Quantitative Estimate of Drug-likeness (QED) score
        --lipinski_rule_of_5_violations (int): Number of violations of Lipinski's Rule of Five
```

How to use tool *calculate_mol_drug_chemistry* :

```python
response = await client.session.call_tool(
    "calculate_mol_drug_chemistry",
    arguments={
        "smiles_list": smiles_list
    }
)
result = client.parse_result(response)
druglikeness_metrics = result["metrics"]
```
