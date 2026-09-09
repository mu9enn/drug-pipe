---
name: molclaw-denovo-sampling
description: "Use when calling these MolClaw MCP tools: `mcp__molclaw-scp__reinvent_denovo_sampling`. Generate new molecules de novo."
license: MIT license
metadata:
    skill-author: PJLab
---

# Molecule Generation De Novo

## Invocation

Use the exact tool name and parameter schema exposed by the current harness. Tool names in SDK examples below are raw MCP names, not model-facing aliases. Local files must be uploaded with `molclaw-file-transfer` before a server tool can use them. Repair PDB inputs first when the workflow requires it.

The description of tool *reinvent_denovo_sampling*.

```tex
Generate new molecules de novo.
Args:
    n (int): Number of molecules for sampling
    lipinski (bool): Required flag controlling Lipinski filtering (commonly True)
    filter_preset (str): Required filter preset; options: ['none', 'minimal', 'default', 'strict', 'druglike', 'all'] (commonly 'druglike')
Return:
    status (str): success/error
    msg (str): message
    save_smiles_file (str): Path to the saved SMILES file
    output_smiles_list (List[str]): List of generated SMILES strings
```

How to use tool *reinvent_denovo_sampling* :

```python
response = await client.session.call_tool(
    "reinvent_denovo_sampling",
    arguments={
        "n": n,
        "lipinski": True,
        "filter_preset": filter_type
    }
)
result = client.parse_result(response)
output_smiles_list = result["output_smiles_list"]
```
