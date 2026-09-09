---
name: molclaw-rgroup-sampling
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__libinvent_rgroup_sampling_by_scaffold`, `mcp__molclaw-scp__libinvent_rgroup_sampling_by_scaffold_name`. Generate new molecules sampling from the input scaffold."
license: MIT license
metadata:
    skill-author: PJLab
---

# Molecule Generation from Scaffold

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

The description of tool *libinvent_rgroup_sampling_by_scaffold*.

```tex
Generate new molecules sampling from the input scaffold.
Args:
    scaffold (str): Input scaffold SMILES string containing R-group position markers such as [*:1], [*:2], etc. e.g., 'c1ccc([*:1])cc1C(=O)N[*:2]'
    n (int): Number of molecules for sampling
    lipinski (bool): Required flag controlling Lipinski filtering (commonly True)
    filter_preset (str): Required filter preset; options: ['none', 'minimal', 'default', 'strict'] (commonly 'default')
Return:
    status (str): success/error
    msg (str): message
    save_smiles_file (str): Path to the saved SMILES file
    output_smiles_list (List[str]): List of generated SMILES strings
```

How to use tool *libinvent_rgroup_sampling_by_scaffold* :

```python
response = await client.session.call_tool(
    "libinvent_rgroup_sampling_by_scaffold",
    arguments={
        "scaffold": scaffold,
        "n": n,
        "lipinski": True,
        "filter_preset": filter_type
    }
)
result = client.parse_result(response)
output_smiles_list = result["output_smiles_list"]
```
