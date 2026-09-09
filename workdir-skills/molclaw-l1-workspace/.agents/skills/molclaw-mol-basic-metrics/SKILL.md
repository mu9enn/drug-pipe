---
name: molclaw-mol-basic-metrics
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__calculate_mol_basic_info`. Compute a set of basic molecular properties for a given list of SMILES strings, returning the molecular formula, exact and average molecular weights, counts of heavy and total atoms, number of bonds, valence electrons, and formal charge for each input molecule."
license: MIT license
metadata:
    skill-author: PJLab
---

# Molecular Basic Properties Calculation

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

The description of tool *calculate_mol_basic_info*.

```tex
Compute a set of basic molecular properties for each SMILES.
Args:
    smiles_list (List[str]): List of input SMILES strings, (e.g., ["N[C@@H](Cc1ccc(O)cc1)C(=O)O", "CC(C)C1=CC=CC=C1"])
Return:
    status (str): success/error
    msg (str): message
    metrics (List[dict]): List of dict, each containing feature keys.
        --smiles (str): A SMILES string of smiles_list
        --molecular_formula (str): Molecular formula, e.g. "C9H11NO3"
        --exact_molecular_weight (float): Exact molecular weight
        --molecular_weight (float): Average molecular weight
        --num_heavy_atoms (int): Number of heavy atoms
        --num_atoms (int): Number of total atoms
        --num_bonds (int): Number of bonds
        --num_valence_electrons (int): Number of valence electrons
        --formal_charge (int): Number of formal charge
```

How to use tool *calculate_mol_basic_info*:

```python
response = await client.session.call_tool(
    "calculate_mol_basic_info",			
    arguments={
        "smiles_list": smiles_list
    }
)
result = client.parse_result(response)
metrics = result["metrics"]
```
