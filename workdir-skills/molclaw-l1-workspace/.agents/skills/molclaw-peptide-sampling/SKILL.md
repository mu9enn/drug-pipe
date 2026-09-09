---
name: molclaw-peptide-sampling
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__get_pepinvent_info`, `mcp__molclaw-scp__pepinvent_peptide_sampling_by_peptide`, `mcp__molclaw-scp__pepinvent_peptide_sampling_by_template`. Generate new peptide molecules sampling from the input peptide sequence."
license: MIT license
metadata:
    skill-author: PJLab
---

# Peptide Molecule Generation

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

The description of tool *pepinvent_peptide_sampling_by_peptide*.

```tex
Generate new peptide molecules sampling from the input peptide sequence.
Args:
    peptide (str): SMILES representation of a peptide sequence, with amino acid residues separated by '|?|', e.g., 'N[C@@H](CCCCN)C(=O)|?|N[C@@H](CC(C)C)C(=O)|?|N[C@@H](CCCNC(=N)N)C(=O)' 
    n (int): Number of molecules for sampling
    filter_preset (str): Required filter preset; options: ['none', 'minimal', 'default', 'strict'] (commonly 'default')
    mw_min (float): Required minimum molecular weight (use 0.0 for no lower bound)
    mw_max (float): Required maximum molecular weight (use 0.0 for no upper bound)
Return:
    status (str): success/error
    msg (str): message
    save_smiles_file (str): Path to the saved SMILES file
    output_smiles_list (List[str]): List of generated SMILES strings
```

How to use tool *pepinvent_peptide_sampling_by_peptide* :

```python
response = await client.session.call_tool(
    "pepinvent_peptide_sampling_by_peptide",
    arguments={
        "peptide": smiles,
        "n": n,
        "filter_preset": filter_type,
        "mw_min": mw_min,
        "mw_max": mw_max
    }
)
result = client.parse_result(response)
output_smiles_list = result["output_smiles_list"]
```
